# --- 1. 定义源文件 ---
file(GLOB_RECURSE CUDA_SOURCES *.cu)
file(GLOB_RECURSE CPP_SOURCES *.cpp)

# --- 2. 创建库目标 ---
add_library(yolo26_detect SHARED
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

# --- 配置包含路径 ---
target_include_directories(yolo26_detect PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}           # 当前 src 目录
    ${PROJECT_SOURCE_DIR}                 # 项目根目录
    ${TENSORRT_INCLUDE_DIR}               # TensorRT 目录
    ${jetson-utils_INCLUDE_DIRS}          # Jetson-utils 目录
    ${CUDAToolkit_INCLUDE_DIRS}           # CUDA Toolkit 目录
)

# --- 链接库 ---
target_link_libraries(yolo26_detect
    PRIVATE
    CUDA::cudart
    CUDA::cublas
    nvinfer
    nvinfer_plugin
    jetson-utils
    -L${TENSORRT_LIB_DIR}
)