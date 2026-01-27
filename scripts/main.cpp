#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <jetson-utils/cudaFont.h>
#include <jetson-utils/cudaDraw.h>
#include <signal.h>
#include "detectYOLO.h"

// 预定义颜色（用于不同类别）
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

bool gSignalReceived = false;

void signalHandler(int sig) {
    gSignalReceived = true;
}

void printUsage() {
    printf("\nUsage: yolo26_detect [options]\n\n");
    printf("Options:\n");
    printf("  --input=URI       Input video source (file, csi, v4l2, rtsp)\n");
    printf("  --output=URI      Output video destination\n");
    printf("  --model=PATH      Path to TensorRT engine file\n");
    printf("  --labels=PATH     Path to class labels file\n");
    printf("  --threshold=N     Detection threshold (default: 0.25)\n");
    printf("\nExamples:\n");
    printf("  yolo26_detect --input=video.mp4 --output=result.mp4\n");
    printf("  yolo26_detect --input=csi://0 --output=display://0\n");
    printf("\n");
}

/**
 * 绘制检测结果
 */
void drawDetections(void* image, uint32_t width, uint32_t height,
                    Detection* detections, int count,
                    DetectYOLO* detector, cudaFont* font)
{
    for (int i = 0; i < count; i++)
    {
        const Detection& det = detections[i];

        // 根据类别选择颜色
        const float4& color = COLORS[det.classId % NUM_COLORS];

        // 绘制边界框
        cudaDrawRect(image, width, height, IMAGE_RGB8,
             (int)det.x1, (int)det.y1, (int)det.x2, (int)det.y2, 
             make_float4(0, 0, 0, 0), color, 2.0f);

        // 绘制标签文字
        if (font)
        {
            char label[128];
            snprintf(label, sizeof(label), "%s %.0f%%",
                     detector->getClassName(det.classId),
                     det.confidence * 100.0f);

            font->OverlayText(image, IMAGE_RGB8, width, height,
                              label, det.x1 + 2, det.y1 + 2,
                              make_float4(255, 255, 255, 255), color);
        }
    }
}

int main(int argc, char** argv) {
    // 注册信号处理
    signal(SIGINT, signalHandler);

    // 解析命令行参数
    commandLine cmdLine(argc, argv);

    if (cmdLine.GetFlag("help")) {
        printUsage();
        return 0;
    }

    // 获取参数
    const char* inputUri = cmdLine.GetString("input", "video.mp4");
    const char* outputUri = cmdLine.GetString("output", "result.mp4");
    const char* modelPath = cmdLine.GetString("model", "yolo26.engine");
    const char* labelsPath = cmdLine.GetString("labels", "coco_labels.txt");
    float threshold = cmdLine.GetFloat("threshold", 0.25f);

    printf("=== YOLO26 Detector ===\n");
    printf("Input:     %s\n", inputUri);
    printf("Output:    %s\n", outputUri);
    printf("Model:     %s\n", modelPath);
    printf("Labels:    %s\n", labelsPath);
    printf("Threshold: %.2f\n", threshold);
    printf("\n");


    // 创建检测器
    DetectYOLO* detector = DetectYOLO::create(modelPath, threshold);
    if (!detector) {
        printf("Failed to create detector: %s\n", modelPath);
        return -1;
    }

    // 加载类别标签
    detector->loadLabels(labelsPath);

    // 创建输入源
    videoSource* input = videoSource::Create(inputUri);
    if (!input) {
        printf("Failed to create input: %s\n", inputUri);
        return -1;
    }

    // 创建输出
    videoOutput* output = videoOutput::Create(outputUri);
    if (!output) {
        printf("Failed to create output: %s\n", outputUri);
        return -1;
    }

    // 创建字体（用于绘制标签）
    cudaFont* font = cudaFont::Create();

    printf("Processing started...\n");
    int frameCount = 0;
    std::ofstream logFile("performance_log.txt");
        logFile << "Frame, FPS, DetectionCount" << std::endl;
    // 主循环
    while (!gSignalReceived) {
        // 1. 获取图像
        int status = 0;
        uchar3* imgInput = nullptr;
        if (!input->Capture(&imgInput, &status)) {
            if (status == videoSource::TIMEOUT) {
                printf("Capture timeout, retrying...\n");
                continue;
            }
            if (status == videoSource::EOS) {
                printf("End of stream\n");
                break;
            }
            printf("Capture error\n");
            break;
        }
        // 2. 检测
        Detection* detections = NULL;
        int count = detector->detect(imgInput, 
                                    input->GetWidth(), 
                                    input->GetHeight(), 
                                    &detections);
        // 3. 后处理
        if (count > 0) {
            drawDetections(imgInput, input->GetWidth(), input->GetHeight(),
                           detections, count, detector, font);
        }
        // 4. 保存结果
        output->Render(imgInput, input->GetWidth(), input->GetHeight());
        // 状态显示
        char statusStr[128];
        snprintf(statusStr, sizeof(statusStr), "YOLO26 | %d detections | %.1f FPS",
                 count, output->GetFrameRate());
        output->SetStatus(statusStr);
        if (frameCount % 30 == 0) { // 每30帧打印一次
            printf("Frame: %d, FPS: %.2f, Detections: %d\n", frameCount, currentFPS, count);
        }
        frameCount++;
        if (!output->IsStreaming())
            break;
    }

    // 清理
    printf("Processed %d frames\n", frameCount);

    // 清理资源
    SAFE_DELETE(font);
    SAFE_DELETE(detector);
    SAFE_DELETE(input);
    SAFE_DELETE(output);
    return 0;
}