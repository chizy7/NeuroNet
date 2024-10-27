#pragma once
#include <string>

extern bool debug; // Declare debug as extern to avoid multiple definitions

class Logger {
public:
    static void info(const std::string& message);
    static void error(const std::string& message);
};