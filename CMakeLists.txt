cmake_minimum_required(VERSION 3.15)
project(PINN_Semiconductor_Simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include header files
include_directories(include)

# Collect source files
file(GLOB SOURCES src/*.cpp)

add_executable(pinn_semiconductor ${SOURCES})
