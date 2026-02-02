# OrinFlow Edge

OrinFlow 的边缘端子项目。在 Jetson Orin Nano Super 上通过 TensorRT 部署 YOLO26 检测模型，实现实时视频目标检测。

## 架构

```
.engine ──▶ TensorRT 加载 ──▶ GPU 预处理 ──▶ 推理 ──▶ 后处理 ──▶ 可视化输出
                                (LetterBox)              (坐标还原)    (画框+标签)
```

- **src/core/** — TensorRT 引擎管理与推理流程（DetectBase / DetectYOLO）
- **src/cuda/** — CUDA 预处理算子（LetterBox + BGR→RGB + HWC→CHW + 归一化）
- **src/utils/** — 工具类（CUDA 计时器、CSV 性能日志）
- **app/** — 可执行程序（视频输入 → 检测 → 渲染输出）

## 环境要求

- Jetson Orin Nano Super (JetPack 6.x)
- CUDA Toolkit
- TensorRT 10.x
- [jetson-utils](https://github.com/dusty-nv/jetson-utils)

## 构建

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

编译产物：`build/bin/detectnet`

## 模型准备

使用 [OrinFlow Lab](https://github.com/your-repo/OrinFlow_lab) 在 PC 端优化模型，然后用 `trtexec` 在 Jetson 上构建 engine：

```bash
# INT8 量化模型（带 QDQ 节点）
trtexec --onnx=yolo26x_qat_int8.onnx --saveEngine=yolo26x.engine --int8 --fp16

# FP16 模型
trtexec --onnx=yolo26x.onnx --saveEngine=yolo26x.engine --fp16
```

## 使用

```bash
# 视频文件检测
./build/bin/detectnet --input=video.mp4 --output=result.mp4 --model=yolo26x.engine --labels=data/labels/coco.txt

# CSI 摄像头实时检测
./build/bin/detectnet --input=csi://0 --output=display://0 --model=yolo26x.engine

# 带性能日志
./build/bin/detectnet --input=video.mp4 --model=yolo26x.engine --log=perf.csv
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | video.mp4 | 输入源（文件 / csi / v4l2 / rtsp） |
| `--output` | result.mp4 | 输出目标（文件 / display） |
| `--model` | yolo26.engine | TensorRT engine 路径 |
| `--labels` | coco.txt | 类别标签文件 |
| `--threshold` | 0.25 | 置信度阈值 |
| `--log` | 无 | 性能日志输出路径 (CSV) |

## 许可证

MIT
