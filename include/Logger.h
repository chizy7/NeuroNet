#pragma once
#include <string>

class Logger {
public:
    static void info(const std::string& message);
    static void error(const std::string& message);
};