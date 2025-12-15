cmake_minimum_required(VERSION 3.10)
project(financeARLM)

# Require C++17
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Include Eigen (adjust this path to where you put Eigen in WSL)
include_directories("~/eigen/eigen-5.0.0")

# Add executable
add_executable(financeARLM 
    src/main.cpp
    src/autograd.cpp
    src/tensor.cpp
    src/MatrixMultiply.cpp
    src/nditerator.cpp
)

target_link_libraries(financeARLM PRIVATE OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})