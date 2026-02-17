#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <jetson-utils/cudaFont.h>
#include <jetson-utils/cudaDraw.h>
#include <signal.h>

#include "core/detectYOLO.h"

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
    printf("\nUsage: detectnet [options]\n\n");
    printf("Options:\n");
    printf("  --input=URI       Input video source (file, csi, v4l2, rtsp)\n");
    printf("  --output=URI      Output video destination\n");
    printf("  --model=PATH      Path to TensorRT engine file\n");
    printf("  --labels=PATH     Path to class labels file\n");
    printf("  --threshold=N     Detection threshold (default: 0.25)\n");
    printf("  --log=PATH        Performance log output (default: none)\n");
    printf("\nExamples:\n");
    printf("  detectnet --input=video.mp4 --output=result.mp4\n");
    printf("  detectnet --input=csi://0 --output=display://0\n");
    printf("  detectnet --input=video.mp4 --log=perf.csv\n");
    printf("\n");
}

void drawDetections(void* image, uint32_t width, uint32_t height,
                    Detection* detections, int count,
                    DetectYOLO* detector, cudaFont* font)
{
    for (int i = 0; i < count; i++)
    {
        const Detection& det = detections[i];
        const float4& color = COLORS[det.classId % NUM_COLORS];

        cudaDrawRect(image, width, height, IMAGE_RGB8,
             (int)det.x1, (int)det.y1, (int)det.x2, (int)det.y2,
             make_float4(0, 0, 0, 0), color, 2.0f);

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
    signal(SIGINT, signalHandler);

    commandLine cmdLine(argc, argv);

    if (cmdLine.GetFlag("help")) {
        printUsage();
        return 0;
    }

    const char* inputUri   = cmdLine.GetString("input", "video.mp4");
    const char* outputUri  = cmdLine.GetString("output", "result.mp4");
    const char* modelPath  = cmdLine.GetString("model", "yolo26.engine");
    const char* labelsPath = cmdLine.GetString("labels", "coco.txt");
    float threshold        = cmdLine.GetFloat("threshold", 0.25f);

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

    detector->loadLabels(labelsPath);

    // 创建输入/输出
    videoSource* input = videoSource::Create(inputUri);
    if (!input) {
        printf("Failed to create input: %s\n", inputUri);
        return -1;
    }

    videoOutput* output = videoOutput::Create(outputUri);
    if (!output) {
        printf("Failed to create output: %s\n", outputUri);
        return -1;
    }

    cudaFont* font = cudaFont::Create();

    printf("Processing started...\n");

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

        // 3. 绘制
        if (count > 0) {
            drawDetections(imgInput, input->GetWidth(), input->GetHeight(),
                           detections, count, detector, font);
        }

        // 4. 输出
        output->Render(imgInput, input->GetWidth(), input->GetHeight());

        float fps = output->GetFrameRate();

        // 状态栏
        char statusStr[128];
        snprintf(statusStr, sizeof(statusStr), "YOLO26 | %d detections | %.1f FPS",
                 count, fps);
        output->SetStatus(statusStr);

        if (!output->IsStreaming())
            break;
    }

    SAFE_DELETE(font);
    SAFE_DELETE(detector);
    SAFE_DELETE(input);
    SAFE_DELETE(output);
    return 0;
}
