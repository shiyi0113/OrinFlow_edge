#pragma once

#include "DetectBase.h"
#include "cuda/cudaPreprocess.cuh"

/**
 * YOLO26 检测器
 *
 * 支持 YOLO 端到端模型，输出格式: [batch, max_det, 6]
 * 其中 6 = [x1, y1, x2, y2, confidence, class_id]
 *
 * 模型规格:
 *   - 输入: [1, 3, 640, 640] float32, RGB, CHW, 归一化到[0,1]
 *   - 输出: [1, 300, 6] float32
 */
class DetectYOLO : public DetectBase
{
public:
    /**
     * 创建 YOLO 检测器
     * @param enginePath TensorRT 引擎文件路径
     * @param threshold  置信度阈值，默认0.25
     * @return 成功返回检测器指针，失败返回 nullptr
     */
    static DetectYOLO* create(const char* enginePath, float threshold = 0.25f);

    DetectYOLO();
    virtual ~DetectYOLO();

    void setThreshold(float threshold) { mThreshold = threshold; }
    float getThreshold() const { return mThreshold; }

protected:
    bool preProcess(void* image, uint32_t width, uint32_t height) override;
    int postProcess(uint32_t width, uint32_t height) override;

protected:
    float mThreshold = 0.75f;
    int   mMaxDetections = 300;

    LetterBoxInfo mLetterBoxInfo;
};