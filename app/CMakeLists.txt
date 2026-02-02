add_executable(detectnet main.cpp)

target_include_directories(detectnet PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${jetson-utils_INCLUDE_DIRS}
)

target_link_libraries(detectnet PRIVATE
    orinflow_edge
    jetson-utils
)

set_target_properties(detectnet PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
