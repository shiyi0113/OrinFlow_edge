#include "cudaPreprocess.cuh"

// 填充颜色 (114, 114, 114) 归一化后的值
static constexpr float FILL_VALUE = 114.0f / 255.0f;

/**
 * LetterBox 预处理内核
 *
 * 每个线程处理输出图像的一个像素
 * 计算该像素对应的输入图像位置，进行双线性插值或填充
 */
__global__ void letterBoxPreprocessKernel(
    const uchar3* __restrict__ input,
    float* __restrict__ output,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight,
    float scale,
    float padLeft,
    float padTop
)
{
    // 计算输出像素坐标
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;

    if (outX >= outputWidth || outY >= outputHeight)
        return;

    // 输出平面大小（用于 CHW 布局）
    int planeSize = outputWidth * outputHeight;

    // 计算对应的输入坐标（去除padding后反向映射）
    float srcX = (outX - padLeft) / scale;
    float srcY = (outY - padTop) / scale;

    float r, g, b;

    // 判断是否在有效区域内
    if (srcX >= 0 && srcX < inputWidth && srcY >= 0 && srcY < inputHeight)
    {
        // 双线性插值
        int x0 = (int)floorf(srcX);
        int y0 = (int)floorf(srcY);
        int x1 = min(x0 + 1, inputWidth - 1);
        int y1 = min(y0 + 1, inputHeight - 1);

        float dx = srcX - x0;
        float dy = srcY - y0;

        // 获取四个邻域像素 (BGR格式)
        uchar3 p00 = input[y0 * inputWidth + x0];
        uchar3 p01 = input[y0 * inputWidth + x1];
        uchar3 p10 = input[y1 * inputWidth + x0];
        uchar3 p11 = input[y1 * inputWidth + x1];

        // 双线性插值 + 归一化 + BGR->RGB
        // input是BGR，输出需要RGB，所以 r=z(B位置), b=x(R位置)
        r = ((1 - dx) * (1 - dy) * p00.z + dx * (1 - dy) * p01.z +
             (1 - dx) * dy * p10.z + dx * dy * p11.z) / 255.0f;

        g = ((1 - dx) * (1 - dy) * p00.y + dx * (1 - dy) * p01.y +
             (1 - dx) * dy * p10.y + dx * dy * p11.y) / 255.0f;

        b = ((1 - dx) * (1 - dy) * p00.x + dx * (1 - dy) * p01.x +
             (1 - dx) * dy * p10.x + dx * dy * p11.x) / 255.0f;
    }
    else
    {
        // 填充区域，使用灰色
        r = g = b = FILL_VALUE;
    }

    // 写入输出（CHW格式：R平面, G平面, B平面）
    int outIdx = outY * outputWidth + outX;
    output[0 * planeSize + outIdx] = r;  // R channel
    output[1 * planeSize + outIdx] = g;  // G channel
    output[2 * planeSize + outIdx] = b;  // B channel
}

cudaError_t cudaLetterBoxPreprocess(
    void* input,
    int inputWidth,
    int inputHeight,
    float* output,
    int outputWidth,
    int outputHeight,
    LetterBoxInfo* info,
    cudaStream_t stream
)
{
    // 计算缩放比例（保持宽高比）
    float scaleW = (float)outputWidth / inputWidth;
    float scaleH = (float)outputHeight / inputHeight;
    float scale = fminf(scaleW, scaleH);

    // 缩放后的尺寸
    int newWidth = (int)roundf(inputWidth * scale);
    int newHeight = (int)roundf(inputHeight * scale);

    // 计算居中padding
    float padLeft = (outputWidth - newWidth) / 2.0f;
    float padTop = (outputHeight - newHeight) / 2.0f;

    // 填充输出信息
    if (info)
    {
        info->scale = scale;
        info->padLeft = padLeft;
        info->padTop = padTop;
        info->newWidth = newWidth;
        info->newHeight = newHeight;
    }

    // 启动内核
    dim3 blockDim(32, 32);
    dim3 gridDim(
        (outputWidth + blockDim.x - 1) / blockDim.x,
        (outputHeight + blockDim.y - 1) / blockDim.y
    );

    letterBoxPreprocessKernel<<<gridDim, blockDim, 0, stream>>>(
        (const uchar3*)input,
        output,
        inputWidth,
        inputHeight,
        outputWidth,
        outputHeight,
        scale,
        padLeft,
        padTop
    );

    return cudaGetLastError();
}
