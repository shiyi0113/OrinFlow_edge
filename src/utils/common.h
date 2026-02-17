#pragma once

#include <string>
#include <unistd.h>
#include <linux/limits.h>
#include <vector_types.h>   // float4

inline std::string getProjectRoot()
{
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return "";
    buf[len] = '\0';

    std::string exePath(buf);
    // 向上三级
    for (int i = 0; i < 3; i++) {
        size_t pos = exePath.rfind('/');
        if (pos == std::string::npos) return "";
        exePath = exePath.substr(0, pos);
    }
    return exePath;
}

inline std::string resolvePath(const char* path, const std::string& root)
{
    std::string s(path);
    if (s.empty() || s[0] == '/' || root.empty()) return s;
    if (s.find("://") != std::string::npos) return s;   // csi://0, rtsp://... 等
    return root + "/" + s;
}

inline const float4* getColorTable(int* count = nullptr)
{
    static const float4 COLORS[] = {
        {255, 0, 0, 255},     // 红
        {0, 255, 0, 255},     // 绿
        {0, 0, 255, 255},     // 蓝
        {255, 255, 0, 255},   // 黄
        {255, 0, 255, 255},   // 紫
        {0, 255, 255, 255},   // 青
        {255, 128, 0, 255},   // 橙
        {128, 255, 0, 255},   // 黄绿
        {0, 128, 255, 255},   // 天蓝
        {255, 0, 128, 255},   // 玫红
    };
    static const int NUM_COLORS = sizeof(COLORS) / sizeof(COLORS[0]);

    if (count) *count = NUM_COLORS;
    return COLORS;
}

// 按 classId 循环取色
inline const float4& getClassColor(int classId)
{
    int count = 0;
    const float4* colors = getColorTable(&count);
    return colors[classId % count];
}
