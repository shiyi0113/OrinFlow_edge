#ifndef __LOGGER_H__
#define __LOGGER_H__

#include <fstream>
#include <cstdio>

/**
 * CSV 性能日志
 * 每帧记录: 帧号, FPS, 推理耗时(ms), 检测数
 */
class PerfLogger
{
public:
    PerfLogger() = default;

    ~PerfLogger() { close(); }

    /**
     * 打开日志文件并写入 CSV 表头
     * @param path 输出文件路径
     * @return 成功返回 true
     */
    bool open(const char* path)
    {
        mFile.open(path, std::ios::out | std::ios::trunc);
        if (!mFile.is_open())
            return false;
        mFile << "frame,fps,inference_ms,detections" << std::endl;
        return true;
    }

    /** 写入一行记录 */
    void log(int frame, float fps, float inferenceMs, int detections)
    {
        if (mFile.is_open())
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "%d,%.1f,%.2f,%d", frame, fps, inferenceMs, detections);
            mFile << buf << std::endl;
        }
    }

    void close()
    {
        if (mFile.is_open())
            mFile.close();
    }

private:
    std::ofstream mFile;
};

#endif
