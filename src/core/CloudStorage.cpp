#include "CloudStorage.h"
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <fstream>
#include <iostream>
#include <cstdlib>

bool CloudStorage::upload_to_s3(const std::string& local_file, const std::string& s3_path) {
    // Get bucket name from environment variable
    const char* bucket_name = std::getenv("AWS_BUCKET_NAME");
    if (!bucket_name) {
        std::cerr << "Error: AWS_BUCKET_NAME environment variable not set" << std::endl;
        return false;
    }

    Aws::SDKOptions options;
    Aws::InitAPI(options);

    // Configure the client
    Aws::Client::ClientConfiguration config;
    const char* region = std::getenv("AWS_REGION");
    if (!region) {
        std::cerr << "Error: AWS_REGION environment variable not set" << std::endl;
        return false;
    }
    config.region = region;
    
    Aws::S3::S3Client s3_client(config);

    // Read the local file
    std::ifstream file(local_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << local_file << std::endl;
        return false;
    }

    Aws::S3::Model::PutObjectRequest request;
    request.WithBucket(bucket_name).WithKey(s3_path);

    std::shared_ptr<Aws::IOStream> input_data = std::make_shared<std::stringstream>();
    *input_data << file.rdbuf();
    request.SetBody(input_data);

    auto outcome = s3_client.PutObject(request);
    if (!outcome.IsSuccess()) {
        std::cerr << "Failed to upload to S3: " << outcome.GetError().GetMessage() << std::endl;
        return false;
    }

    Aws::ShutdownAPI(options);
    return true;
}

bool CloudStorage::download_from_s3(const std::string& s3_path, const std::string& local_file) {
    const char* bucket_name = std::getenv("AWS_BUCKET_NAME");
    if (!bucket_name) {
        std::cerr << "Error: AWS_BUCKET_NAME environment variable not set" << std::endl;
        return false;
    }

    Aws::SDKOptions options;
    Aws::InitAPI(options);

    Aws::Client::ClientConfiguration config;
    const char* region = std::getenv("AWS_REGION");
    if (!region) {
        std::cerr << "Error: AWS_REGION environment variable not set" << std::endl;
        return false;
    }
    config.region = region;
    
    Aws::S3::S3Client s3_client(config);

    Aws::S3::Model::GetObjectRequest request;
    request.WithBucket(bucket_name).WithKey(s3_path);

    auto outcome = s3_client.GetObject(request);
    if (!outcome.IsSuccess()) {
        std::cerr << "Failed to download from S3: " << outcome.GetError().GetMessage() << std::endl;
        return false;
    }

    std::ofstream file(local_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << local_file << std::endl;
        return false;
    }

    file << outcome.GetResultWithOwnership().GetBody().rdbuf();
    Aws::ShutdownAPI(options);
    return true;
}