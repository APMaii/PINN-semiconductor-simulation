cmake_minimum_required(VERSION 3.15)
project(PINN_Semiconductor_Simulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include header files
include_directories(include)

# Collect source files (exclude main.cpp for library)
file(GLOB SOURCES src/*.cpp)
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

# Main executable
file(GLOB MAIN_SOURCE src/main.cpp)
add_executable(pinn_semiconductor ${MAIN_SOURCE} ${SOURCES})

# Test executable - test_network
add_executable(test_network tests/test_network.cpp ${SOURCES})

# Test executable - test_pinn
add_executable(test_pinn tests/test_pinn.cpp ${SOURCES})

# Test executable - test_fd_solver
add_executable(test_fd_solver tests/test_fd_solver.cpp ${SOURCES})

# Enable testing
enable_testing()
add_test(NAME NetworkTest COMMAND test_network)
add_test(NAME PINNTest COMMAND test_pinn)
add_test(NAME FDSolverTest COMMAND test_fd_solver)
