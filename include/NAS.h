#pragma once
#include "NeuralNetwork.h"

class NAS {
public:
    void search(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const std::vector<int>& layer_options, int epochs, int batch_size);

private:
    void evaluate_architecture(const std::vector<int>& architecture, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs, int batch_size);
};