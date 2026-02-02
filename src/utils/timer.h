#ifndef __TIMER_H__
#define __TIMER_H__

#include <cuda_runtime.h>

/**
 * CUDA Event 计时器
 * 精确测量 GPU 操作耗时（毫秒）
 */
class CudaTimer
{
public:
    CudaTimer()
    {
        cudaEventCreate(&mStart);
        cudaEventCreate(&mStop);
    }

    ~CudaTimer()
    {
        cudaEventDestroy(mStart);
        cudaEventDestroy(mStop);
    }

    /** 开始计时（插入到指定 CUDA 流） */
    void start(cudaStream_t stream = nullptr)
    {
        cudaEventRecord(mStart, stream);
    }

    /** 停止计时并返回耗时（毫秒） */
    float stop(cudaStream_t stream = nullptr)
    {
        cudaEventRecord(mStop, stream);
        cudaEventSynchronize(mStop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, mStart, mStop);
        return ms;
    }

private:
    cudaEvent_t mStart;
    cudaEvent_t mStop;
};

#endif
