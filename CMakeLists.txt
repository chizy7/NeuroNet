cmake_minimum_required(VERSION 3.10)
project(NeuroNet)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(include)
include_directories(third_party/eigen-3.4.0)

# Add core source files
set(SRC
    src/main.cpp
    src/core/NeuralNetwork.cpp
    src/core/Layer.cpp
    src/core/Optimizer.cpp
    src/core/Trainer.cpp
    src/core/Loss.cpp
    src/utils/DataLoader.cpp
    src/utils/Logger.cpp
)

# Create executable
add_executable(NeuroNet ${SRC})
