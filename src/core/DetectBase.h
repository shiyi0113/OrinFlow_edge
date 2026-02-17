#pragma once

#include "detection.h"
#include "TRTLogger.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <string>
#include <memory>
#include <vector>

class DetectBase
{
public:
    DetectBase();
    virtual ~DetectBase();

    bool loadModel(const char* enginePath);
    int detect(void* image, uint32_t width, uint32_t height, Detection** detections);

    const char* getClassName(int classId) const;
    bool loadLabels(const char* labelPath);

protected:
    bool mModelLoaded = false;
    
    TRTLogger&                                   mLogger;
    std::unique_ptr<nvinfer1::IRuntime>          mRuntime  = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine>       mEngine   = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext  = nullptr;

    cudaStream_t mStream  = nullptr;
    void* mInputDevice  = nullptr;
    void* mOutputDevice = nullptr;
    float* mOutputHost  = nullptr;

    std::vector<Detection> mDetections;
    std::vector<std::string> mLabels;

    bool allocBuffers();

    int mInputSize = 0;
    int mOutputSize = 0;
    int inputH = 0;
    int inputW = 0;
    int numDets = 0;

    virtual bool preProcess(void* image, uint32_t width, uint32_t height) = 0;
    virtual int postProcess(uint32_t width, uint32_t height) = 0;
};