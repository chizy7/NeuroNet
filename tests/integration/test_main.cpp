#include "gtest/gtest.h"
#include "Logger.h"
#include "GlobalFlags.h"

int main(int argc, char **argv) {
    // Initialize debug flag to false for tests
    debug = false;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}