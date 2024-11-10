#include "HyperparameterTuner.h"
#include "MSE.h"
#include <iostream>

void HyperparameterTuner::grid_search(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const std::vector<double>& learning_rates, const std::vector<int>& batch_sizes, int epochs) {
    for (double lr : learning_rates) {
        for (int batch_size : batch_sizes) {
            evaluate_model(lr, batch_size, X, Y, epochs);
        }
    }
}

void HyperparameterTuner::evaluate_model(double learning_rate, int batch_size, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs) {
    NeuralNetwork nn;
    nn.add_layer(new DenseLayer(X.rows(), 128));  // Example architecture
    nn.add_layer(new DenseLayer(128, Y.rows()));

    Optimizer* opt = new Adam(learning_rate);
    nn.compile(opt, new MSE());
    
    nn.train(X, Y, epochs, batch_size);
    
    std::cout << "Learning rate: " << learning_rate << ", Batch size: " << batch_size << std::endl;
}