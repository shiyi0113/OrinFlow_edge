cmake_minimum_required(VERSION 3.10)
project(yolo-inference LANGUAGES CXX CUDA)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 87)  
# C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 查找依赖
find_package(CUDA REQUIRED)
find_package(jetson-utils REQUIRED)

# TensorRT 路径（根据你的安装位置调整）
set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu" CACHE PATH "TensorRT include")
set(TENSORRT_LIB_DIR "/usr/lib/aarch64-linux-gnu" CACHE PATH "TensorRT lib")

# 包含目录
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${jetson-utils_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

# 链接目录
link_directories(${TENSORRT_LIB_DIR})

# 添加子目录
add_subdirectory(src)
add_subdirectory(scripts)
