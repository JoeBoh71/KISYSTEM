cmake_minimum_required(VERSION 3.20)
project(U3DAW LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4070

# Find packages
find_package(CUDAToolkit 12.3 REQUIRED)

# Paths (adjust as needed)
set(ASIO_SDK_PATH "C:/SDKs/ASIO_SDK_2.3.3" CACHE PATH "ASIO SDK path")
set(RME_DRIVER_PATH "C:/Program Files/RME" CACHE PATH "RME driver path")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${ASIO_SDK_PATH}/common
    ${CUDAToolkit_INCLUDE_DIRS}
)

# TEP Engine library
add_library(u3daw_tep STATIC
    src/tep/pqmf/linkwitz_riley.cu
    src/tep/pqmf/pqmf_engine.cu
    src/tep/operators/amplitude_gain.cu
    src/tep/operators/phase_shift.cu
    src/tep/operators/psychoacoustic.cu
    src/tep/tep_engine.cu
)

target_link_libraries(u3daw_tep
    CUDA::cufft
    CUDA::cublas
    CUDA::cudart
)

# Tests
enable_testing()
add_subdirectory(tests)

# Main executable (placeholder)
add_executable(u3daw_test src/main_test.cpp)
target_link_libraries(u3daw_test u3daw_tep)
