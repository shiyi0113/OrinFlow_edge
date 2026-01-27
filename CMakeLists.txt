cmake_minimum_required(VERSION 3.10)
project(OrinFlow_edge LANGUAGES CXX CUDA)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 87)  
# C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 查找依赖
find_package(CUDAToolkit REQUIRED)
find_package(jetson-utils REQUIRED)

# TensorRT 路径
set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")

# 添加子目录
add_subdirectory(src)
add_subdirectory(scripts)
