#pragma once
#include <Eigen/Dense>
#include <string>

class DataLoader {
public:
    // Loads MNIST data from a CSV file into data and labels
    static void load_mnist_csv(const std::string& file_path, Eigen::MatrixXd& data, Eigen::VectorXi& labels);

    // Optional virtual destructor for extensibility in the future
    virtual ~DataLoader() = default;
};