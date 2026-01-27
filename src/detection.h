#ifndef __DETECTION_H__
#define __DETECTION_H__

#include <vector>
#include <string>
#include <cmath>
/**
 * 单个检测结果
 */
struct Detection
{
    float x1, y1, x2, y2;  // 边界框坐标（像素）
    float confidence;      // 置信度 [0,1]
    int   classId;         // 类别ID
    
    // 计算框的宽高和面积
    inline float width() const  { return x2 - x1; }
    inline float height() const { return y2 - y1; }
    inline float area() const   { return width() * height(); }
    
    // 计算IoU
    inline float iou(const Detection& other) const {
        float interX1 = fmaxf(x1, other.x1);
        float interY1 = fmaxf(y1, other.y1);
        float interX2 = fminf(x2, other.x2);
        float interY2 = fminf(y2, other.y2);
        
        float interArea = fmaxf(0.0f, interX2 - interX1) * 
                          fmaxf(0.0f, interY2 - interY1);
        float unionArea = area() + other.area() - interArea;
        
        return (unionArea > 0) ? (interArea / unionArea) : 0.0f;
    }
};

#endif
