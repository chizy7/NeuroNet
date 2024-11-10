cmake_minimum_required(VERSION 3.10)
project(NeuroNet)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)

# Find MPI package
find_package(MPI REQUIRED)

# Find CPPREST package
find_package(cpprestsdk REQUIRED)

# AWS
find_package(AWSSDK REQUIRED COMPONENTS s3)
include_directories(${AWSSDK_INCLUDE_DIRS})
link_directories(${AWSSDK_LIBRARY_DIRS})

# Specify the macOS SDK path for CMake
if(APPLE)
    set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
endif()

# Include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories
include_directories(include)
include_directories(third_party/eigen-3.4.0)
include_directories(${MPI_INCLUDE_PATH})

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

# Link libraries
target_link_libraries(NeuroNet
    PRIVATE
    ${MPI_LIBRARIES}
    cpprestsdk::cpprest
    ${AWSSDK_LINK_LIBRARIES}
    ${OpenCV_LIBS}
)