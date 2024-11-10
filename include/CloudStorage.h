#pragma once
#include <string>

class CloudStorage {
public:
    static bool upload_to_s3(const std::string& local_file, const std::string& s3_path);
    static bool download_from_s3(const std::string& s3_path, const std::string& local_file);
};