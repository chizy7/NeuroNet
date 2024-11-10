#include "ModelVersioning.h"
#include <iostream>
#include <fstream>

ModelVersioning::ModelVersioning() {}

void ModelVersioning::save_version(const std::string& version, const std::string& model_path) {
    version_history[version] = model_path;
    std::cout << "Model saved as version " << version << " at " << model_path << std::endl;
}

void ModelVersioning::rollback_version(const std::string& version, const std::string& model_path) {
    if (version_history.find(version) != version_history.end()) {
        std::ifstream src(version_history[version], std::ios::binary);
        std::ofstream dst(model_path, std::ios::binary);
        dst << src.rdbuf();
        std::cout << "Rolled back to version " << version << " at " << model_path << std::endl;
    } else {
        std::cerr << "Version not found: " << version << std::endl;
    }
}

void ModelVersioning::list_versions() const {
    std::cout << "Available versions:" << std::endl;
    for (const auto& version : version_history) {
        std::cout << version.first << " -> " << version.second << std::endl;
    }
}