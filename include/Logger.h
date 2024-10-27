#pragma once
#include <string>

extern bool debug; // Debug flag for global access

class Logger {
public:
    static void info(const std::string& message);
    static void error(const std::string& message);
};