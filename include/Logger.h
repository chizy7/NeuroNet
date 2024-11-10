#pragma once
#include <string>

extern bool debug; 

class Logger {
public:
    static void info(const std::string& message);
    static void error(const std::string& message);
};