#include "ModelBuilder.h"

#include <fstream>
#include <iostream>
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

    hasDynamicShape = false;
    std::cout << "=== ONNX模型解析成功 ===" << std::endl;
    std::cout << "网格输入:" << std::endl;
    for(int i = 0; i < mNetwork->getNbInputs(); ++i)
    {
        nvinfer1::ITensor* input = mNetwork->getInput(i);
        std::cout << " [" << i << "] name=\"" << input->getName() << "\" shape=";
        nvinfer1::Dims dims = input->getDimensions();
        std::cout << "(";
        for(int d = 0; d < dims.nbDims; ++d)
        {
            if(d > 0) std::cout << ", ";
            std::cout << dims.d[d];
            if(dims.d[d] == -1)
            {
                hasDynamicShape = true;
            }
        } 
        std::cout << ")" << std::endl;
    }
    std::cout << "网格输出:" << std::endl;
    for(int i = 0; i < mNetwork->getNbOutputs(); ++i)
    {
        nvinfer1::ITensor* output = mNetwork->getOutput(i);
        std::cout << " [" << i << "] name=\"" << output->getName() << "\" shape=";
        nvinfer1::Dims dims = output->getDimensions();
        std::cout << "(";
        for(int d = 0; d < dims.nbDims; ++d)
        {
            if(d > 0) std::cout << ", ";
            std::cout << dims.d[d];
        } 
        std::cout << ")" << std::endl;
    }
    if(hasDynamicShape){
        nvinfer1::IOptimizationProfile* profile = mBuilder->createOptimizationProfile();
        for(int i = 0; i < mNetwork->getNbInputs(); ++i)
        {
            nvinfer1::ITensor* input = mNetwork->getInput(i);
            const char* name = input->getName();
            nvinfer1::Dims dims = input->getDimensions();
            nvinfer1::Dims minDims = dims, optDims = dims,maxDims = dims;
            for(int d = 0; d < dims.nbDims; ++d)
            {
                if(dims.d[d] == -1){
                    if(d == 0)
                    {
                        minDims.d[d] = 1;
                        optDims.d[d] = 1;
                        maxDims.d[d] = 1;
                    }
                    else
                    {
                        minDims.d[d] = 640;
                        optDims.d[d] = 640;
                        maxDims.d[d] = 640;
                    }
                }
            }
            profile->setDimensions(name,nvinfer1::OptProfileSelector::kMIN,minDims);
            profile->setDimensions(name,nvinfer1::OptProfileSelector::kOPT,optDims);
            profile->setDimensions(name,nvinfer1::OptProfileSelector::kMAX,maxDims);

            std::cout << "设置 Profile: \"" << name << "\" "
                      << "batch=[" << minDims.d[0] << "," << optDims.d[0]
                      << "," << maxDims.d[0] << "]" << std::endl;
        }
        mConfig->addOptimizationProfile(profile);
    }
    else
    {
        std::cout << "模型输入为静态 shape，无需配置 Optimization Profile." << std::endl;
    }

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