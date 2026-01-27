add_executable(detectnet main.cpp)

# 包含目录
target_include_directories(detectnet PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${jetson-utils_INCLUDE_DIRS}
)
# 链接库
target_link_libraries(detectnet PRIVATE
    yolo26_detect
    jetson-utils
)

# 输出到 bin 目录
set_target_properties(detectnet PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
