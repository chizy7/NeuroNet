#include "Logger.h"
#include <iostream>

void Logger::info(const std::string& message) {
    if (debug) {
        std::cout << "[INFO]: " << message << std::endl;
    }
}

void Logger::error(const std::string& message) {
    if (debug) {
        std::cerr << "[ERROR]: " << message << std::endl;
    }
}