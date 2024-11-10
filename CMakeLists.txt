cmake_minimum_required(VERSION 3.10)
project(NeuroNet)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)

# Find MPI package
find_package(MPI REQUIRED)

# Find CPPREST package
find_package(cpprestsdk REQUIRED)

# Specify the macOS SDK path for CMake
if(APPLE)
    set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
endif()

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

# Link MPI libraries
target_link_libraries(NeuroNet PRIVATE ${MPI_LIBRARIES})

# Specify the library path for cpprestsdk
link_directories(/opt/homebrew/Cellar/cpprestsdk/2.10.19/lib)

# Include the header files for cpprestsdk
include_directories(/opt/homebrew/Cellar/cpprestsdk/2.10.19/include)

# Use cpprestsdk's CMake alias
target_link_libraries(NeuroNet PRIVATE cpprestsdk::cpprest)