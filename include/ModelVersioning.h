#pragma once
#include <string>
#include <map>

class ModelVersioning {
public:
    ModelVersioning();

    void save_version(const std::string& version, const std::string& model_path);
    void rollback_version(const std::string& version, const std::string& model_path);
    void list_versions() const;

private:
    std::map<std::string, std::string> version_history;
};