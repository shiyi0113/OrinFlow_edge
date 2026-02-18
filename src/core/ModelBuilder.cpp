#include "ModelBuilder.h"

#include <fstream>

ModelBuilder::ModelBuilder()
    : mLogger(TRTLogger::getInstance())
{}

ModelBuilder::~ModelBuilder()
{}

ModelBuilder* ModelBuilder::create(const char* onnxPath, Precision precision)
{
    ModelBuilder* builder = new ModelBuilder();
    builder->mPrecision = precision;

    if (!builder->createBuilderAndNetwork()) {
        delete builder;
        return nullptr;
    }

    if (!builder->parseONNX(onnxPath)) {
        delete builder;
        return nullptr;
    }

    return builder;
}

bool ModelBuilder::build(const char* enginePath)
{
    LogInfo("ModelBuilder: 开始构建引擎...\n");

    if (!configurePrecision())
        return false;

    if (!buildAndSerialize(enginePath))
        return false;

    LogInfo("ModelBuilder: 引擎构建完成 → %s\n", enginePath);
    return true;
}

bool ModelBuilder::createBuilderAndNetwork()
{
    mBuilder.reset(nvinfer1::createInferBuilder(mLogger));
    if(!mBuilder)
    {
        LogError("ModelBuilder: 创建 Builder 失败\n");
        return false;
    }
    
    mNetwork.reset(mBuilder->createNetworkV2(0));
    if(!mNetwork)
    {
        LogError("ModelBuilder: 创建 Network 失败\n");
        return false;
    }

    mConfig.reset(mBuilder->createBuilderConfig());
    if(!mConfig)
    {
        LogError("ModelBuilder: 创建 BuilderConfig 失败\n");
        return false;
    }

    mConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, mMaxWorkspaceSize);
    return true;
}

bool ModelBuilder::parseONNX(const char* onnxPath)
{
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*mNetwork,mLogger)
    );
    if(!parser)
    {
        LogError("ModelBuilder: 创建 ONNX Parser 失败\n");
        return false;
    }

    bool isOk = parser->parseFromFile(onnxPath,
        static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    if(!isOk)
    {
        LogError("ModelBuilder: 解析 ONNX 模型失败\n");
        return false;
    }
    LogInfo("ModelBuilder: ONNX 解析成功 (%d 个输入, %d 个输出)\n",
            mNetwork->getNbInputs(), mNetwork->getNbOutputs());
    return true;
}

bool ModelBuilder::configurePrecision()
{
    switch(mPrecision){
        case Precision::FP32:
            LogInfo("ModelBuilder: 使用 FP32 精度\n");
            break;
        case Precision::FP16:
            if (!mBuilder->platformHasFastFp16()) {
                LogWarning("ModelBuilder: 当前平台不支持快速 FP16，回退到 FP32\n");
                break;
            }
            mConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
            LogInfo("ModelBuilder: 使用 FP16 精度\n");
            break;
        case Precision::INT8:
            if (!mBuilder->platformHasFastInt8()) {
                LogError("ModelBuilder: 当前平台不支持 INT8\n");
                return false;
            }
            mConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
            mConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
            LogInfo("ModelBuilder: 使用 INT8 精度\n");
            break;
        default: 
            break;
    }
    return true;
}

bool ModelBuilder::buildAndSerialize(const char* enginePath)
{
    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        mBuilder->buildSerializedNetwork(*mNetwork,*mConfig)
    );
    if(!serialized | serialized->size() == 0)
    {
        LogError("ModelBuilder: 构建引擎失败\n");
        return false;
    }

    std::ofstream file(enginePath,std::ios::binary);
    if(!file.is_open())
    {
        LogError("ModelBuilder: 无法写入文件 %s\n", enginePath);
        return false;
    }
    file.write(static_cast<const char*>(serialized->data()),serialized->size());
    file.close();

    LogInfo("ModelBuilder: 引擎已保存 (%.1f MB)\n",
            serialized->size() / (1024.0 * 1024.0));
    return true;
}