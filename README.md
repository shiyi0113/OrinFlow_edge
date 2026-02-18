# OrinFlow Edge

OrinFlow 的边缘端子项目。在 Jetson Orin Nano Super 上通过 TensorRT 部署 YOLO26 检测模型，实现实时视频目标检测。

## 架构

```
ONNX / .engine ──▶ TensorRT 加载 ──▶ GPU 预处理 ──▶ 推理 ──▶ 后处理 ──▶ 可视化输出
```

- **src/core/** — TensorRT 引擎管理与推理流程（DetectBase / DetectYOLO）
- **src/cuda/** — CUDA 预处理算子（LetterBox + HWC→CHW + 归一化）
- **src/utils/** — 工具函数（项目路径解析、类别颜色映射）
- **app/** — 可执行程序（视频输入 → 检测 → 渲染输出）

## 环境

- Jetson Orin Nano Super (JetPack 6.2)
- CUDA Toolkit
- TensorRT 10.3
- [jetson-utils](https://github.com/dusty-nv/jetson-utils)

## 构建

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

编译产物：`build/bin/detectnet`

## 模型准备

`--model` 参数同时接受 `.onnx` 和 `.engine` 文件。传入 ONNX 时程序会自动调用 `ModelBuilder` 构建引擎并保存为同名 `.engine` 文件。

也可以提前用 `trtexec` 手动构建：

```bash
# INT8 量化模型（带 QDQ 节点）
trtexec --onnx=yolo26x_qat_int8.onnx --saveEngine=yolo26x.engine --int8 --fp16

# FP16 模型
trtexec --onnx=yolo26x.onnx --saveEngine=yolo26x.engine --fp16
```

## 使用

```bash
# 视频文件检测
./build/bin/detectnet --input=data/videos/test.mp4 --output=output/result.mp4 --model=models/yolo26x.engine --labels=data/labels/coco.txt

# CSI 摄像头实时检测
./build/bin/detectnet --input=csi://0 --output=display://0 --model=models/yolo26x.engine

# 传入 ONNX，自动构建引擎后检测
./build/bin/detectnet --input=data/videos/test.mp4 --model=models/yolo26x.onnx --precision=int8
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `data/videos/test_1.mp4` | 输入源（文件 / csi / v4l2 / rtsp） |
| `--output` | `output/output.mp4` | 输出目标（文件 / display） |
| `--model` | `models/yolo26x_INT8.engine` | TensorRT engine 或 ONNX 模型路径 |
| `--labels` | `data/labels/coco.txt` | 类别标签文件 |
| `--threshold` | `0.75` | 置信度阈值 |
| `--precision` | `fp16` | 构建精度（仅传入 ONNX 时生效）：`fp32` / `fp16` / `int8` |

> 相对路径以项目根目录为基准；协议 URI（`csi://`、`rtsp://`）直接使用。

## 许可证

MIT
