#pragma once
#include <Eigen/Dense>
#include <string>

class DataLoader {
public:
    static void load_mnist_csv(const std::string& file_path, Eigen::MatrixXd& data, Eigen::VectorXi& labels);
};