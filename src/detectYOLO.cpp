#include "detectYOLO.h"
#include <jetson-utils/logging.h>

//----------------------------------------------------------
// 创建检测器（工厂方法）
//----------------------------------------------------------
DetectYOLO* DetectYOLO::create(const char* enginePath, float threshold)
{
    DetectYOLO* detector = new DetectYOLO();
    detector->setThreshold(threshold);

    if (!detector->loadModel(enginePath))
    {
        LogError("DetectYOLO: 加载模型失败 %s\n", enginePath);
        delete detector;
        return nullptr;
    }

    return detector;
}

//----------------------------------------------------------
// 构造/析构
//----------------------------------------------------------
DetectYOLO::DetectYOLO()
{
}

DetectYOLO::~DetectYOLO()
{
}

//----------------------------------------------------------
// 初始化模型参数
//----------------------------------------------------------
bool DetectYOLO::initModelParams()
{
    // 从引擎获取输入维度
    // 假设输入名为 "images"，格式 [1, 3, H, W]
    auto inputDims = mEngine->getTensorShape("images");
    if (inputDims.nbDims != 4)
    {
        LogError("DetectYOLO: 输入维度错误，期望4维，实际%d维\n", inputDims.nbDims);
        return false;
    }

    mInputHeight = inputDims.d[2];  // H
    mInputWidth = inputDims.d[3];   // W

    // 从引擎获取输出维度
    // 假设输出名为 "output0"，格式 [1, max_det, 6]
    auto outputDims = mEngine->getTensorShape("output0");
    if (outputDims.nbDims != 3)
    {
        LogError("DetectYOLO: 输出维度错误，期望3维，实际%d维\n", outputDims.nbDims);
        return false;
    }

    mMaxDetections = outputDims.d[1];  // max_det (300)
    int numValues = outputDims.d[2];   // 6 (x1,y1,x2,y2,conf,cls)

    if (numValues != 6)
    {
        LogWarning("DetectYOLO: 输出每检测值数量=%d，期望6\n", numValues);
    }

    // 计算输出大小（字节）
    mOutputSize = 1 * mMaxDetections * numValues * sizeof(float);

    LogInfo("DetectYOLO: 输入 %dx%d, 最大检测数 %d\n",
            mInputWidth, mInputHeight, mMaxDetections);

    return true;
}

//----------------------------------------------------------
// 预处理
//----------------------------------------------------------
bool DetectYOLO::preProcess(void* image, uint32_t width, uint32_t height)
{
    // 调用 CUDA LetterBox 预处理
    cudaError_t err = cudaLetterBoxPreprocess(
        image,
        width,
        height,
        (float*)mInputDevice,
        mInputWidth,
        mInputHeight,
        &mLetterBoxInfo,
        mStream
    );

    if (err != cudaSuccess)
    {
        LogError("DetectYOLO: 预处理失败 - %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
}

//----------------------------------------------------------
// 后处理
//----------------------------------------------------------
int DetectYOLO::postProcess(uint32_t width, uint32_t height)
{
    // 端到端输出格式: [1, 300, 6]
    // 每个检测: [x1, y1, x2, y2, confidence, class_id]
    // 坐标是相对于模型输入尺寸(640x640)的

    const float* output = mOutputHost;

    for (int i = 0; i < mMaxDetections; i++)
    {
        const float* det = output + i * 6;

        float x1 = det[0];
        float y1 = det[1];
        float x2 = det[2];
        float y2 = det[3];
        float conf = det[4];
        int classId = (int)det[5];

        // 过滤低置信度和无效检测
        // 端到端模型中，无效检测的置信度通常为0或负数
        if (conf < mThreshold || conf <= 0)
            continue;

        // 坐标还原: 模型输出 -> 原始图像
        // 1. 去除 padding
        x1 = (x1 - mLetterBoxInfo.padLeft) / mLetterBoxInfo.scale;
        y1 = (y1 - mLetterBoxInfo.padTop) / mLetterBoxInfo.scale;
        x2 = (x2 - mLetterBoxInfo.padLeft) / mLetterBoxInfo.scale;
        y2 = (y2 - mLetterBoxInfo.padTop) / mLetterBoxInfo.scale;

        // 2. 裁剪到图像边界
        x1 = fmaxf(0.0f, fminf(x1, (float)width));
        y1 = fmaxf(0.0f, fminf(y1, (float)height));
        x2 = fmaxf(0.0f, fminf(x2, (float)width));
        y2 = fmaxf(0.0f, fminf(y2, (float)height));

        // 过滤无效框
        if (x2 <= x1 || y2 <= y1)
            continue;

        // 添加检测结果
        Detection det_result;
        det_result.x1 = x1;
        det_result.y1 = y1;
        det_result.x2 = x2;
        det_result.y2 = y2;
        det_result.confidence = conf;
        det_result.classId = classId;

        mDetections.push_back(det_result);
    }

    // 端到端模型已经做过NMS，这里不需要再做

    return (int)mDetections.size();
}
