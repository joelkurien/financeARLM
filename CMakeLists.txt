cmake_minimum_required(VERSION 3.10)
project(financeARLM)

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include Eigen (adjust this path to where you put Eigen in WSL)
include_directories("/home/susan/eigen/eigen-5.0.0")

# Add executable
add_executable(financeARLM src/main.cpp)