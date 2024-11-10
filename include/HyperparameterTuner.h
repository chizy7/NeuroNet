#pragma once
#include "NeuralNetwork.h"
#include "Optimizer.h"

class HyperparameterTuner {
public:
    void grid_search(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const std::vector<double>& learning_rates, const std::vector<int>& batch_sizes, int epochs);

private:
    void evaluate_model(double learning_rate, int batch_size, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs);
};