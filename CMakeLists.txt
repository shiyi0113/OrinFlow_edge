cmake_minimum_required(VERSION 3.10)
project(OrinFlow_edge LANGUAGES CXX CUDA)

# CUDA 架构 (Orin Nano Super = SM 8.7)
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 依赖
find_package(CUDAToolkit REQUIRED)
find_package(jetson-utils REQUIRED)

# TensorRT 路径
set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")

# 子目录
add_subdirectory(src)
add_subdirectory(app)
