#pragma once

#include <NvInfer.h>
#include <jetson-utils/logging.h>
#include <string>
#include <mutex>

class TRTLogger : public nvinfer1::ILogger
{
public:
    static TRTLogger& getInstance()
    {
        static TRTLogger instance;
        return instance;
    }

    TRTLogger(Severity level = Severity::kWARNING,
              const char* tag = "TensorRT")
        : mLevel(level),mTag(tag ? tag : "TensorRT") {}
    
    void setLevel(Severity level) { mLevel = level; }
    Severity getLevel() const { return mLevel; }
    void setTag(const char* tag) { mTag = tag ? tag : "TensorRT"; }
    const std::string& getTag() const { return mTag; }

    void log(Severity severity, const char* msg) noexcept override
    {
        if(severity > mLevel) return;
        std::lock_guard<std::mutex> lock(mMutex);
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            LogError("[%s] INTERNAL_ERROR: %s\n", mTag.c_str(), msg);
            break;
        case Severity::kERROR:
            LogError("[%s] ERROR: %s\n", mTag.c_str(), msg);
            break;
        case Severity::kWARNING:
            LogWarning("[%s] WARNING: %s\n", mTag.c_str(), msg);
            break;
        case Severity::kINFO:
            LogInfo("[%s] INFO: %s\n", mTag.c_str(), msg);
            break;
        case Severity::kVERBOSE:
            LogVerbose("[%s] VERBOSE: %s\n", mTag.c_str(), msg);
            break;
        default:
            break;
        }
    }
    
private:
    Severity  mLevel;
    std::string mTag;
    std::mutex mMutex;
};
