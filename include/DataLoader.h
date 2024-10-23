#pragma once
#include <Eigen/Dense>
#include <string>

class DataLoader {
public:
    static Eigen::MatrixXd load_csv(const std::string& file_path);
};