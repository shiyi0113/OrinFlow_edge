#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <jetson-utils/commandLine.h>
#include <jetson-utils/cudaFont.h>
#include <jetson-utils/cudaDraw.h>
#include <signal.h>

#include "core/DetectYOLO.h"
#include "core/ModelBuilder.h"
#include "utils/common.h"

bool gSignalReceived = false;

void signalHandler(int sig) {
    gSignalReceived = true;
}

void printUsage() {
    printf("\nUsage: detectnet [options]\n\n");
    printf("Options:\n");
    printf("  --input=URI       Input video source (file, csi, v4l2, rtsp)\n");
    printf("  --output=URI      Output video destination\n");
    printf("  --model=PATH      Path to Model file\n");
    printf("  --labels=PATH     Path to class labels file\n");
    printf("  --threshold=N     Detection threshold (default: 0.75)\n");
    printf("  --precision=MODE  Build precision: fp32, fp16, int8 (default: fp16)\n");
    printf("\nNote: Relative paths are resolved from the project root directory.\n");
    printf("      Absolute paths and protocol URIs (csi://, rtsp://) are used as-is.\n");
    printf("\nExamples:\n");
    printf("  detectnet --input=data/videos/test.mp4 --output=output/result.mp4\n");
    printf("  detectnet --input=csi://0 --output=display://0\n");
    printf("  detectnet --input=/home/user/video.mp4\n");
    printf("\n");
}

void drawDetections(void* image, uint32_t width, uint32_t height,
                    Detection* detections, int count,
                    DetectYOLO* detector, cudaFont* font)
{
    for (int i = 0; i < count; i++)
    {
        const Detection& det = detections[i];
        const float4& color = getClassColor(det.classId);

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

    std::string root = getProjectRoot();

    std::string inputUri   = resolvePath(cmdLine.GetString("input",  "data/videos/test_1.mp4"), root);
    std::string outputUri  = resolvePath(cmdLine.GetString("output", "output/output.mp4"), root);
    std::string modelPath  = resolvePath(cmdLine.GetString("model",  "models/yolo26x_INT8.engine"), root);
    std::string labelsPath = resolvePath(cmdLine.GetString("labels", "data/labels/coco.txt"), root);
    float threshold        = cmdLine.GetFloat("threshold", 0.75f);
    const char* precisionStr = cmdLine.GetString("precision", "fp16");

    Precision precision = Precision::FP16;
    if (strcmp(precisionStr, "fp32") == 0)
        precision = Precision::FP32;
    else if(strcmp(precisionStr, "int8") == 0)
        precision = Precision::INT8;
    else if (strcmp(precisionStr, "fp16") != 0)
        printf("Warning: unknown precision '%s', using fp16\n", precisionStr);

    printf("=== YOLO26 Detector ===\n");
    printf("Root:      %s\n", root.c_str());
    printf("Input:     %s\n", inputUri.c_str());
    printf("Output:    %s\n", outputUri.c_str());
    printf("Model:     %s\n", modelPath.c_str());
    printf("Labels:    %s\n", labelsPath.c_str());
    printf("Threshold: %.2f\n", threshold);
    printf("Precision: %s\n", precisionStr);
    printf("\n");

    std::string enginePath = modelPath;
    if(modelPath.size()>=5 && modelPath.substr(modelPath.size() - 5) == ".onnx")
    {
        enginePath = modelPath.substr(0,modelPath.size() - 5) + ".engine";
        printf("检测到 ONNX 模型，开始构建引擎...\n");

        ModelBuilder* builder = ModelBuilder::create(modelPath.c_str(), precision);
        if (!builder) {
            printf("Failed to create model builder: %s\n", modelPath.c_str());
            return -1;
        }

        if (!builder->build(enginePath.c_str())) {
            printf("Failed to build engine: %s\n", enginePath.c_str());
            delete builder;
            return -1;
        }

        delete builder;
        printf("引擎构建完成: %s\n\n", enginePath.c_str());
    }
    // 创建检测器
    DetectYOLO* detector = DetectYOLO::create(enginePath.c_str(), threshold);
    if (!detector) {
        printf("Failed to create detector: %s\n", enginePath.c_str());
        return -1;
    }

    // 创建输入/输出
    videoSource* input = videoSource::Create(inputUri.c_str());
    if (!input) {
        printf("Failed to create input: %s\n", inputUri.c_str());
        return -1;
    }

    videoOutput* output = videoOutput::Create(outputUri.c_str());
    if (!output) {
        printf("Failed to create output: %s\n", outputUri.c_str());
        return -1;
    }

    detector->loadLabels(labelsPath.c_str());
    cudaFont* font = cudaFont::Create();

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
        int count = detector->detect(imgInput, input->GetWidth(), input->GetHeight(), &detections);

        // 3. 绘制
        if (count > 0) {
            drawDetections(imgInput, input->GetWidth(), input->GetHeight(), detections, count, detector, font);
        }

        // 4. 输出
        output->Render(imgInput, input->GetWidth(), input->GetHeight());

        if (!output->IsStreaming())
            break;
    }

    SAFE_DELETE(font);
    SAFE_DELETE(detector);
    SAFE_DELETE(input);
    SAFE_DELETE(output);
    return 0;
}
