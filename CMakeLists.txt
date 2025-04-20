cmake_minimum_required(VERSION 3.10)
project(NeuroNet)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)

# Enable testing
enable_testing()

# Define the debug variable globally
add_definitions(-DDEBUG_MODE)

# Find MPI package
find_package(MPI REQUIRED)

# Find CPPREST package
find_package(cpprestsdk REQUIRED)

# AWS
find_package(AWSSDK REQUIRED COMPONENTS s3)
include_directories(${AWSSDK_INCLUDE_DIRS})
link_directories(${AWSSDK_LIBRARY_DIRS})

# Specify the macOS SDK path dynamically
if(APPLE)
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for macOS")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS deployment version")
endif()

# Try to use the system Google Test first
find_package(GTest QUIET)

if(NOT GTEST_FOUND)
    # If system Google Test is not found, download it
    message(STATUS "System Google Test not found, downloading it...")
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    set(GTEST_LIBRARIES gtest gtest_main)
    set(GTEST_INCLUDE_DIRS "${googletest_SOURCE_DIR}/googletest/include")
else()
    message(STATUS "Using system Google Test")
endif()

# Include OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories
include_directories(include)
include_directories(third_party/eigen-3.4.0)
include_directories(${MPI_INCLUDE_PATH})
include_directories(${GTEST_INCLUDE_DIRS})

# Add core source files
set(SRC
    src/core/NeuralNetwork.cpp
    src/core/Layer.cpp
    src/core/Optimizer.cpp
    src/core/Trainer.cpp
    src/core/Loss.cpp
    src/utils/DataLoader.cpp
    src/utils/Logger.cpp
    src/utils/global_flags.cpp
)

# Create main executable
add_executable(NeuroNet src/main.cpp)

# Link libraries to main executable
target_link_libraries(NeuroNet
    PRIVATE
    NeuroNetLib
    ${MPI_LIBRARIES}
    cpprestsdk::cpprest
    ${AWSSDK_LINK_LIBRARIES}
    ${OpenCV_LIBS}
)

# Create a library target for the core code to be used in tests
add_library(NeuroNetLib STATIC ${SRC} ${GLOBAL_FLAGS})
target_link_libraries(NeuroNetLib
    PRIVATE
    ${MPI_LIBRARIES}
    cpprestsdk::cpprest
    ${AWSSDK_LINK_LIBRARIES}
    ${OpenCV_LIBS}
)

# Unit Tests
add_executable(UnitTests
    tests/unit/test_main.cpp
    tests/unit/test_layer.cpp
    tests/unit/test_optimizer.cpp
    tests/unit/test_loss.cpp
)
target_link_libraries(UnitTests
    PRIVATE
    NeuroNetLib
    ${GTEST_LIBRARIES}
    ${MPI_LIBRARIES}
    cpprestsdk::cpprest
    ${AWSSDK_LINK_LIBRARIES}
    ${OpenCV_LIBS}
)
add_test(NAME UnitTests COMMAND UnitTests)

# Integration Tests
add_executable(IntegrationTests
    tests/integration/test_main.cpp
    tests/integration/test_neural_network.cpp
)
target_link_libraries(IntegrationTests
    PRIVATE
    NeuroNetLib
    ${GTEST_LIBRARIES}
    ${MPI_LIBRARIES}
    cpprestsdk::cpprest
    ${AWSSDK_LINK_LIBRARIES}
    ${OpenCV_LIBS}
)
add_test(NAME IntegrationTests COMMAND IntegrationTests)

# System Tests
add_executable(SystemTests
    tests/system/test_main.cpp
    tests/system/test_end_to_end.cpp
)
target_link_libraries(SystemTests
    PRIVATE
    NeuroNetLib
    ${GTEST_LIBRARIES}
    ${MPI_LIBRARIES}
    cpprestsdk::cpprest
    ${AWSSDK_LINK_LIBRARIES}
    ${OpenCV_LIBS}
)
add_test(NAME SystemTests COMMAND SystemTests)