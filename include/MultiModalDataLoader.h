#pragma once
#include <Eigen/Dense>
#include <string>

class MultiModalDataLoader {
public:
    static Eigen::MatrixXd load_image_data(const std::string& image_path);
    static Eigen::MatrixXd load_text_data(const std::string& text_path);
    static Eigen::MatrixXd load_audio_data(const std::string& audio_path);
};