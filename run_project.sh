#!/bin/bash

# Run NeuroNet Project Script

# Check if build directory exists
if [ ! -d "build" ]; then
  echo "Build directory not found. Building the project..."
  mkdir build
fi

# Navigate to build directory
cd build || exit

# Run CMake to configure the build
cmake ..

# Build the project
make

# Run the main executable
echo "Running NeuroNet..."
./NeuroNet