cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(visualisierung LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w -g -G")

include_directories(include)
include_directories(vendor/imgui)

add_executable(visualisierung src/main.cu)

set_property(TARGET visualisierung PROPERTY CUDA_ARCHITECTURES OFF)

target_link_libraries(visualisierung PRIVATE X11 GL)
