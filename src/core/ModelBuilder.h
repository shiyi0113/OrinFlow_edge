#pragma once

#include "TRTLogger.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <string>
#include <memory>

enum class Precision
{
    FP32,
    FP16,
    INT8  
};

class ModelBuilder
{
public:
    ~ModelBuilder();
    static ModelBuilder* create(const char* onnxPath,Precision precision = Precision::FP16);
    bool build(const char* enginePath);
    void setMaxWorkspaceSize(size_t bytes) {mMaxWorkspaceSize = bytes;}

private:
    ModelBuilder();
    bool createBuilderAndNetwork();
    bool parseONNX(const char* onnxPath);
    bool configurePrecision();
    bool buildAndSerialize(const char* enginePath);

    TRTLogger& mLogger;
    std::unique_ptr<nvinfer1::IBuilder>           mBuilder = nullptr;
    std::unique_ptr<nvinfer1::INetworkDefinition> mNetwork = nullptr;
    std::unique_ptr<nvinfer1::IBuilderConfig>     mConfig  = nullptr;
    std::unique_ptr<nvonnxparser::IParser>        mParser  = nullptr;
    
    Precision  mPrecision = Precision::FP16;
    size_t     mMaxWorkspaceSize = 1ULL << 28;  // 默认 256MB
};