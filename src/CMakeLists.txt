# 源文件
set(CORE_SOURCES
    core/DetectBase.cpp
    core/DetectYOLO.cpp
    core/ModelBuilder.cpp
)

set(CUDA_SOURCES
    cuda/cudaPreprocess.cu
)

# 共享库
add_library(orinflow_edge SHARED
    ${CORE_SOURCES}
    ${CUDA_SOURCES}
)

target_include_directories(orinflow_edge PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}             
    ${TENSORRT_INCLUDE_DIR}
    ${jetson-utils_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(orinflow_edge PRIVATE
    CUDA::cudart
    CUDA::cublas
    nvonnxparser
    nvinfer
    nvinfer_plugin
    jetson-utils
    -L${TENSORRT_LIB_DIR}
)
