cmake_minimum_required(VERSION 3.28)
project(hello_cuda)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
endif()
enable_language(CUDA)

add_executable(hello_cuda main.cu)

set_target_properties(hello_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)