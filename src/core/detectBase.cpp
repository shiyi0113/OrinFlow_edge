#include "detectBase.h"
#include <jetson-utils/logging.h>
#include <jetson-utils/filesystem.h>
#include <fstream>
#include <algorithm>
#include <sstream>

// TensorRT Logger
class TRTLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            LogWarning("[TensorRT] %s\n", msg);
    }
} gLogger;

//----------------------------------------------------------
// 构造/析构
//----------------------------------------------------------
DetectBase::DetectBase()
{
}

DetectBase::~DetectBase()
{
    release();
}

//----------------------------------------------------------
// 加载模型
//----------------------------------------------------------
bool DetectBase::loadModel(const char* enginePath, int deviceId)
{
    cudaSetDevice(deviceId);

    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        LogError("DetectBase: 无法打开引擎文件 %s\n", enginePath);
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);
    file.close();

    mRuntime = nvinfer1::createInferRuntime(gLogger);
    if (!mRuntime) {
        LogError("DetectBase: 创建 TensorRT Runtime 失败\n");
        return false;
    }

    mEngine = mRuntime->deserializeCudaEngine(engineData.data(), fileSize);
    if (!mEngine) {
        LogError("DetectBase: 反序列化引擎失败\n");
        return false;
    }

    mContext = mEngine->createExecutionContext();
    if (!mContext) {
        LogError("DetectBase: 创建执行上下文失败\n");
        return false;
    }

    cudaStreamCreate(&mStream);

    if (!initModelParams()) {
        LogError("DetectBase: 初始化模型参数失败\n");
        return false;
    }

    if (!allocBuffers()) {
        LogError("DetectBase: 分配显存失败\n");
        return false;
    }

    mModelLoaded = true;
    LogInfo("DetectBase: 模型加载成功 (%dx%d, %d类)\n",
            mInputWidth, mInputHeight, mNumClasses);

    return true;
}

//----------------------------------------------------------
// 分配显存
//----------------------------------------------------------
bool DetectBase::allocBuffers()
{
    mInputSize = mInputWidth * mInputHeight * 3 * sizeof(float);
    if (cudaMalloc(&mInputDevice, mInputSize) != cudaSuccess) {
        LogError("DetectBase: 分配输入显存失败\n");
        return false;
    }

    if (cudaMalloc(&mOutputDevice, mOutputSize) != cudaSuccess) {
        LogError("DetectBase: 分配输出显存失败\n");
        return false;
    }

    mOutputHost = new float[mOutputSize / sizeof(float)];

    return true;
}

//----------------------------------------------------------
// 执行检测（模板方法）
//----------------------------------------------------------
int DetectBase::detect(void* image, uint32_t width, uint32_t height,
                       Detection** detections)
{
    if (!mModelLoaded) {
        LogError("DetectBase: 模型未加载\n");
        return -1;
    }

    mDetections.clear();

    if (!preProcess(image, width, height)) {
        LogError("DetectBase: 预处理失败\n");
        return -1;
    }

    if (!inference()) {
        LogError("DetectBase: 推理失败\n");
        return -1;
    }

    int count = postProcess(width, height);

    if (detections)
        *detections = mDetections.data();

    return count;
}

//----------------------------------------------------------
// TensorRT 推理
//----------------------------------------------------------
bool DetectBase::inference()
{
    mContext->setTensorAddress("images", mInputDevice);
    mContext->setTensorAddress("output0", mOutputDevice);

    bool success = mContext->enqueueV3(mStream);

    cudaStreamSynchronize(mStream);

    cudaMemcpy(mOutputHost, mOutputDevice, mOutputSize, cudaMemcpyDeviceToHost);

    return success;
}

//----------------------------------------------------------
// 通用 NMS
//----------------------------------------------------------
void DetectBase::nms(float iouThreshold)
{
    std::sort(mDetections.begin(), mDetections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<Detection> result;
    std::vector<bool> suppressed(mDetections.size(), false);

    for (size_t i = 0; i < mDetections.size(); i++) {
        if (suppressed[i]) continue;

        result.push_back(mDetections[i]);

        for (size_t j = i + 1; j < mDetections.size(); j++) {
            if (suppressed[j]) continue;

            if (mDetections[i].classId == mDetections[j].classId) {
                if (mDetections[i].iou(mDetections[j]) > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    mDetections = std::move(result);
}

//----------------------------------------------------------
// 标签相关
//----------------------------------------------------------
bool DetectBase::loadLabels(const char* labelPath)
{
    std::string content = readFile(labelPath);
    if (content.empty()) {
        LogWarning("DetectBase: 无法加载标签文件 %s\n", labelPath);
        return false;
    }

    std::istringstream stream(content);
    std::string line;
    mLabels.clear();

    while (std::getline(stream, line)) {
        if (!line.empty())
            mLabels.push_back(line);
    }
    mNumClasses = (int)mLabels.size();
    LogInfo("DetectBase: 加载 %zu 个类别标签\n", mLabels.size());
    return true;
}

const char* DetectBase::getClassName(int classId) const
{
    if (classId >= 0 && classId < (int)mLabels.size())
        return mLabels[classId].c_str();
    return "unknown";
}

//----------------------------------------------------------
// 释放资源
//----------------------------------------------------------
void DetectBase::release()
{
    if (mOutputHost)   { delete[] mOutputHost; mOutputHost = nullptr; }
    if (mInputDevice)  { cudaFree(mInputDevice); mInputDevice = nullptr; }
    if (mOutputDevice) { cudaFree(mOutputDevice); mOutputDevice = nullptr; }
    if (mStream)       { cudaStreamDestroy(mStream); mStream = nullptr; }
    if (mContext)      { delete mContext; mContext = nullptr; }
    if (mEngine)       { delete mEngine; mEngine = nullptr; }
    if (mRuntime)      { delete mRuntime; mRuntime = nullptr; }

    mModelLoaded = false;
}
