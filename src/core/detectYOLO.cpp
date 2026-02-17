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


DetectYOLO::DetectYOLO()
{
}

DetectYOLO::~DetectYOLO()
{
}

bool DetectYOLO::preProcess(void* image, uint32_t width, uint32_t height)
{
    cudaError_t err = cudaLetterBoxPreprocess(
        image,
        width,
        height,
        (float*)mInputDevice,
        inputW,
        inputH,
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

int DetectYOLO::postProcess(uint32_t width, uint32_t height)
{
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

        if (conf < mThreshold || conf <= 0)
            continue;

        // 坐标还原: 去除 padding，恢复到原始图像尺寸
        x1 = (x1 - mLetterBoxInfo.padLeft) / mLetterBoxInfo.scale;
        y1 = (y1 - mLetterBoxInfo.padTop) / mLetterBoxInfo.scale;
        x2 = (x2 - mLetterBoxInfo.padLeft) / mLetterBoxInfo.scale;
        y2 = (y2 - mLetterBoxInfo.padTop) / mLetterBoxInfo.scale;

        // 裁剪到图像边界
        x1 = fmaxf(0.0f, fminf(x1, (float)width));
        y1 = fmaxf(0.0f, fminf(y1, (float)height));
        x2 = fmaxf(0.0f, fminf(x2, (float)width));
        y2 = fmaxf(0.0f, fminf(y2, (float)height));

        if (x2 <= x1 || y2 <= y1)
            continue;

        Detection det_result;
        det_result.x1 = x1;
        det_result.y1 = y1;
        det_result.x2 = x2;
        det_result.y2 = y2;
        det_result.confidence = conf;
        det_result.classId = classId;

        mDetections.push_back(det_result);
    }

    // 端到端模型已做过 NMS，此处无需再做

    return (int)mDetections.size();
}
