#pragma once

#include <string>
#include <cstdlib>               // getenv
#include <unistd.h>              // readlink, access
#include <linux/limits.h>
#include <cuda_runtime.h>        // cudaError_t, cudaGetErrorString
#include <jetson-utils/logging.h> // LogError
#include <vector_types.h>        // float4

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            LogError("CUDA error [%s:%d] `%s`: %s\n",                      \
                     __FILE__, __LINE__, #call, cudaGetErrorString(_err));  \
            return false;                                                   \
        }                                                                   \
    } while (0)

// 返回项目根目录的绝对路径。
inline std::string getProjectRoot()
{
    // 1. 环境变量覆盖（最高优先级）
    const char* envRoot = getenv("ORINFLOW_ROOT");
    if (envRoot && envRoot[0] != '\0')
        return std::string(envRoot);

    // 2. 从 exe 向上查找项目专属锚点文件
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        std::string path(buf);
        for (int i = 0; i < 10; i++) {
            size_t pos = path.rfind('/');
            if (pos == std::string::npos) break;
            path = path.substr(0, pos);
            if (path.empty()) break;
            if (access((path + "/data/labels/coco.txt").c_str(), F_OK) == 0)
                return path;
        }
    }

    // 3. CMake 编译期注入的绝对路径
#ifdef ORINFLOW_COMPILE_TIME_ROOT
    if (access(ORINFLOW_COMPILE_TIME_ROOT "/data/labels/coco.txt", F_OK) == 0)
        return ORINFLOW_COMPILE_TIME_ROOT;
#endif

    return "";
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
