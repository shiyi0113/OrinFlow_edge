#include "DetectBase.h"
#include "../utils/common.h"
#include <fstream>
#include <jetson-utils/filesystem.h>
#include <sstream>

DetectBase::DetectBase()
    : mLogger(TRTLogger::getInstance())
{
}

DetectBase::~DetectBase()
{
    if (mOutputHost)   { delete[] mOutputHost; mOutputHost = nullptr; }
    if (mInputDevice)  { cudaFree(mInputDevice); mInputDevice = nullptr; }
    if (mOutputDevice) { cudaFree(mOutputDevice); mOutputDevice = nullptr; }
    if (mStream)       { cudaStreamDestroy(mStream); mStream = nullptr; }
    
    mModelLoaded = false;
}

bool DetectBase::loadModel(const char* enginePath)
{
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LogError("DetectBase: 无法打开引擎文件 %s\n", enginePath);
        return false;
    }
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);
    file.close();

    mRuntime.reset(nvinfer1::createInferRuntime(mLogger));
    if (!mRuntime) {
        LogError("DetectBase: 创建 TensorRT Runtime 失败\n");
        return false;
    }

    mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!mEngine) {
        LogError("DetectBase: 反序列化引擎失败\n");
        return false;
    }

    engineData.clear();
    engineData.shrink_to_fit();

    mContext.reset(mEngine->createExecutionContext());
    if (!mContext) {
        LogError("DetectBase: 创建执行上下文失败\n");
        return false;
    }
    CUDA_CHECK(cudaStreamCreate(&mStream));

    if (!allocBuffers()) {
        LogError("DetectBase: 分配显存失败\n");
        return false;
    }
    mModelLoaded = true;
    LogInfo("DetectBase: 模型加载成功\n");
    return true;
}

bool DetectBase::allocBuffers()
{
    int nbIOTensors = mEngine->getNbIOTensors();
    for(int i = 0; i < nbIOTensors; ++i)
    {
        const char* name = mEngine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = mEngine->getTensorIOMode(name);
        nvinfer1::Dims dims = mEngine->getTensorShape(name);
        nvinfer1::DataType dtype = mEngine->getTensorDataType(name);

        size_t volume = 1;
        for(int d = 0; d < dims.nbDims; ++d)
        {
            volume *= dims.d[d];
        }
        size_t elementSize = 0;
        switch(dtype)
        {
            case nvinfer1::DataType::kFLOAT: elementSize = 4; break;
            case nvinfer1::DataType::kINT8 : elementSize = 1; break;
            case nvinfer1::DataType::kHALF:  elementSize = 2; break;
            default: elementSize = 4;
        }
        size_t totalSize = volume * elementSize;

        void* allocatedPtr = nullptr;
        CUDA_CHECK(cudaMalloc(&allocatedPtr, totalSize));
        if(mode == nvinfer1::TensorIOMode::kINPUT)
        {
            if(mInputDevice) cudaFree(mInputDevice);
            mInputDevice = allocatedPtr;
            mContext->setTensorAddress(name, mInputDevice);
            inputH = dims.d[2];
            inputW = dims.d[3];
        }
        else if(mode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            if(mOutputDevice) cudaFree(mOutputDevice);
            mOutputDevice = allocatedPtr;
            mContext->setTensorAddress(name, mOutputDevice);
            mOutputSize = totalSize;

            if(mOutputHost) delete[] mOutputHost;
            mOutputHost = new float[volume];
        }
    }
    return true;
}

int DetectBase::detect(void* image, uint32_t width, uint32_t height,Detection** detections)
{
    if(!mModelLoaded)
    {
        LogError("DetectBase: 模型未加载\n");
        return -1;
    }
    mDetections.clear();
    // 预处理
    if (!preProcess(image, width, height)) {
        LogError("DetectBase: 预处理失败\n");
        return -1;
    }
    // 推理（enqueueV3 是异步提交，失败表示入队本身出错）
    if (!mContext->enqueueV3(mStream)) {
        LogError("DetectBase: 推理入队失败\n");
        return -1;
    }
    // D2H 拷贝与同步（detect() 返回 int，无法用 CUDA_CHECK 宏，手动内联检查）
    cudaError_t err = cudaMemcpyAsync(mOutputHost, mOutputDevice, mOutputSize,
                                      cudaMemcpyDeviceToHost, mStream);
    if (err != cudaSuccess) {
        LogError("DetectBase: cudaMemcpyAsync 失败 - %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaStreamSynchronize(mStream);
    if (err != cudaSuccess) {
        LogError("DetectBase: cudaStreamSynchronize 失败 - %s\n", cudaGetErrorString(err));
        return -1;
    }
    // 后处理
    int count = postProcess(width, height);
    // 返回结果
    if (detections)
        *detections = mDetections.data();
    
    return count;
}

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

    LogInfo("DetectBase: 加载 %zu 个类别标签\n", mLabels.size());
    return true;
}

const char* DetectBase::getClassName(int classId) const
{
    if (classId >= 0 && classId < (int)mLabels.size())
        return mLabels[classId].c_str();
    return "unknown";
}