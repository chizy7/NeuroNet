#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}NeuroNet Test Runner${NC}"
echo "----------------------------------------"

# Check for Xcode Command Line Tools
if ! xcode-select -p &> /dev/null; then
  echo -e "${RED}Xcode Command Line Tools not found.${NC}"
  echo "Installing Xcode Command Line Tools (this may take a while)..."
  xcode-select --install
  echo "After installation completes, please run this script again."
  exit 1
fi

# Check for correct SDK path
SDK_PATH=$(xcrun --show-sdk-path 2>/dev/null)
if [ $? -ne 0 ] || [ -z "$SDK_PATH" ]; then
  echo -e "${RED}Could not determine proper SDK path.${NC}"
  echo "Please make sure Xcode and Command Line Tools are properly installed."
  exit 1
fi
echo -e "${GREEN}Using SDK path: $SDK_PATH${NC}"

# Use proper path for CMake
CMAKE_PATH=$(which cmake)
if [ -z "$CMAKE_PATH" ]; then
  echo -e "${RED}CMake not found in PATH.${NC}"
  
  # Look for alternative CMake locations
  BREW_CMAKE="/opt/homebrew/bin/cmake"
  if [ -f "$BREW_CMAKE" ]; then
    CMAKE_PATH="$BREW_CMAKE"
    echo -e "${YELLOW}Found CMake at $CMAKE_PATH${NC}"
    echo "Consider adding /opt/homebrew/bin to your PATH."
  else
    echo "Please install CMake using Homebrew: brew install cmake"
    exit 1
  fi
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
  echo "Creating build directory..."
  mkdir -p build
fi

# Navigate to build directory
cd build || exit 1

# Generate CMake files with proper macOS configuration
echo "Configuring CMake..."
$CMAKE_PATH .. \
  -DCMAKE_OSX_SYSROOT="$SDK_PATH" \
  -DCMAKE_OSX_ARCHITECTURES="arm64" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="13.0"

if [ $? -ne 0 ]; then
  echo -e "${RED}CMake configuration failed.${NC}"
  echo "Please check the error messages above."
  exit 1
fi

# Compile the project
echo "Compiling the project..."
$CMAKE_PATH --build .

if [ $? -ne 0 ]; then
  echo -e "${RED}Compilation failed.${NC}"
  echo "Please check the error messages above."
  exit 1
fi

# Check if test executables exist
TEST_EXECUTABLES=0

# Run all tests using CTest
echo -e "\n${YELLOW}Running all tests with CTest...${NC}"
$CMAKE_PATH --build . --target test

# Check for specific test executables
if [ -f "./UnitTests" ]; then
  TEST_EXECUTABLES=$((TEST_EXECUTABLES + 1))
  echo -e "\n${YELLOW}Running unit tests...${NC}"
  ./UnitTests
fi

if [ -f "./IntegrationTests" ]; then
  TEST_EXECUTABLES=$((TEST_EXECUTABLES + 1))
  echo -e "\n${YELLOW}Running integration tests...${NC}"
  ./IntegrationTests
fi

if [ -f "./SystemTests" ]; then
  TEST_EXECUTABLES=$((TEST_EXECUTABLES + 1))
  echo -e "\n${YELLOW}Running system tests...${NC}"
  ./SystemTests
fi

if [ $TEST_EXECUTABLES -eq 0 ]; then
  echo -e "${YELLOW}No test executables found.${NC}"
  echo "Make sure test targets are defined in CMakeLists.txt and compiled successfully."
  echo "Example targets should include: UnitTests, IntegrationTests, SystemTests"
  
  # Check if CTest found any tests
  TEST_COUNT=$($CMAKE_PATH --build . --target test | grep "No tests were found" | wc -l)
  if [ $TEST_COUNT -gt 0 ]; then
    echo -e "${YELLOW}CTest also didn't find any tests.${NC}"
    echo "You may need to add tests using add_test() in your CMakeLists.txt"
  fi
fi

echo -e "\n${GREEN}Test run completed.${NC}"