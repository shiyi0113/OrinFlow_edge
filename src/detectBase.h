#ifndef __DETECT_BASE_H__
#define __DETECT_BASE_H__

#include "detection.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

/**
 * 检测器基类
 * 封装 TensorRT 引擎管理和推理流程
 */
class DetectBase
{
public:
    DetectBase();
    virtual ~DetectBase();
    
    /**
     * 加载 TensorRT 引擎
     * @param enginePath .engine 文件路径
     * @param deviceId GPU设备ID
     * @return 成功返回 true
     */
    bool loadModel(const char* enginePath, int deviceId = 0);
    
    /**
     * 执行检测（模板方法）
     * @param image 输入图像（GPU内存，uchar3格式）
     * @param width 图像宽度
     * @param height 图像高度
     * @param detections 输出检测结果数组
     * @return 检测到的目标数量
     */
    int detect(void* image, uint32_t width, uint32_t height, Detection** detections);
    
    /**
     * 获取类别名称
     */
    const char* getClassName(int classId) const;
    
    /**
     * 加载类别标签文件
     */
    bool loadLabels(const char* labelPath);
    
    /**
     * 获取模型输入尺寸
     */
    inline int getInputWidth() const  { return mInputWidth; }
    inline int getInputHeight() const { return mInputHeight; }
    inline int getNumClasses() const  { return mNumClasses; }

protected:
    /**
     * 预处理：将输入图像转换为模型输入张量
     * @param image 输入图像（GPU内存）
     * @param width 图像宽度
     * @param height 图像高度
     * @return 成功返回 true
     */
    virtual bool preProcess(void* image, uint32_t width, uint32_t height) = 0;
    
    /**
     * 后处理：解析模型输出为检测结果
     * @param width 原始图像宽度（用于坐标还原）
     * @param height 原始图像高度
     * @return 检测到的目标数量
     */
    virtual int postProcess(uint32_t width, uint32_t height) = 0;
    
    /**
     * 初始化模型参数
     * @return 成功返回 true
     */
    virtual bool initModelParams() = 0;

protected:
    /**
     * 执行 TensorRT 推理
     */
    bool inference();
    
    /**
     * 通用 NMS（子类可调用）
     */
    void nms(float iouThreshold = 0.45f);
    
    /**
     * 分配 GPU 显存
     */
    bool allocBuffers();
    
    /**
     * 释放资源
     */
    void release();

protected:
    // ========== TensorRT 相关 ==========
    nvinfer1::IRuntime*          mRuntime  = nullptr;
    nvinfer1::ICudaEngine*       mEngine   = nullptr;
    nvinfer1::IExecutionContext* mContext  = nullptr;
    
    // ========== CUDA 相关 ==========
    cudaStream_t mStream  = nullptr;
    void*        mInputDevice  = nullptr;  // 模型输入显存
    void*        mOutputDevice = nullptr;  // 模型输出显存
    float*       mOutputHost   = nullptr;  // 模型输出（主机内存，用于后处理）
    
    // ========== 模型参数 ==========
    int mInputWidth   = 0;
    int mInputHeight  = 0;
    int mInputSize    = 0;   // 输入张量字节数
    int mOutputSize   = 0;   // 输出张量字节数
    int mNumClasses   = 0;
    int mMaxDetections = 100;
    
    // ========== 检测结果 ==========
    std::vector<Detection> mDetections;
    std::vector<std::string> mLabels;
    
    // ========== 状态 ==========
    bool mModelLoaded = false;
};

#endif
