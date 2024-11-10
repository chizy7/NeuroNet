#include "MultiModalDataLoader.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

Eigen::MatrixXd MultiModalDataLoader::load_image_data(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return Eigen::MatrixXd();
    }
    Eigen::MatrixXd img_data(image.rows, image.cols);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            img_data(i, j) = image.at<uchar>(i, j) / 255.0;
        }
    }
    return img_data;
}

Eigen::MatrixXd MultiModalDataLoader::load_text_data(const std::string& text_path) {
    // Dummy implementation for text data
    Eigen::MatrixXd text_data(1, 10);  // Assume the text is tokenized into 10 tokens
    std::ifstream file(text_path);
    std::string token;
    int i = 0;
    while (file >> token && i < 10) {
        text_data(0, i++) = std::hash<std::string>{}(token) % 10000;  // Convert text to some numerical representation
    }
    return text_data;
}

Eigen::MatrixXd MultiModalDataLoader::load_audio_data(const std::string& audio_path) {
    // Placeholder for audio data loading (e.g., using libsndfile)
    Eigen::MatrixXd audio_data(1, 16000);  // Assume 1-second audio at 16kHz
    // Load audio into audio_data...
    return audio_data;
}

Eigen::MatrixXd MultiModalDataLoader::augment_image(const Eigen::MatrixXd& img_data) {
    // Example: Apply random flipping and rotation
    Eigen::MatrixXd augmented_img = img_data;
    if (rand() % 2 == 0) {
        augmented_img = img_data.colwise().reverse();  // Horizontal flip
    }
    int angle = rand() % 360;
    // Rotate the image by "angle" degrees (implement rotation here...)
    return augmented_img;
}