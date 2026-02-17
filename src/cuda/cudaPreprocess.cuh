#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/**
 * LetterBox 预处理参数
 * 记录缩放和填充信息，用于后处理坐标还原
 */
struct LetterBoxInfo
{
    float scale;      // 缩放比例
    float padLeft;    // 左侧填充像素
    float padTop;     // 顶部填充像素
    int   newWidth;   // 缩放后宽度（不含padding）
    int   newHeight;  // 缩放后高度（不含padding）
};

/**
 * CUDA LetterBox 预处理
 *
 * 将输入图像（uchar3, RGB, HWC）转换为模型输入（float, RGB, CHW）
 *
 * 处理流程:
 *   1. 保持宽高比缩放到目标尺寸
 *   2. 灰色填充 (114, 114, 114)
 *   3. HWC -> CHW
 *   4. 归一化到 [0, 1]
 *
 * 注意: jetson-utils 提供的图像已经是 RGB 格式，无需通道交换
 */
cudaError_t cudaLetterBoxPreprocess(
    void* input,
    int inputWidth,
    int inputHeight,
    float* output,
    int outputWidth,
    int outputHeight,
    LetterBoxInfo* info,
    cudaStream_t stream = nullptr
);
