#include "NAS.h"
#include "MSE.h"
#include <iostream>

void NAS::search(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, const std::vector<int>& layer_options, int epochs, int batch_size) {
    // Iterate over possible architectures
    for (int hidden_layers : layer_options) {
        for (int nodes_per_layer : layer_options) {
            std::vector<int> architecture = {hidden_layers, nodes_per_layer};
            evaluate_architecture(architecture, X, Y, epochs, batch_size);
        }
    }
}

void NAS::evaluate_architecture(const std::vector<int>& architecture, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs, int batch_size) {
    NeuralNetwork nn;
    
    // Build the architecture
    nn.add_layer(new DenseLayer(X.rows(), architecture[0]));  // First hidden layer
    for (int i = 1; i < architecture.size(); ++i) {
        nn.add_layer(new DenseLayer(architecture[i-1], architecture[i]));  // Hidden layers
    }
    nn.add_layer(new DenseLayer(architecture.back(), Y.rows()));  // Output layer

    nn.compile(new Adam(0.001), new MSE());
    
    // Train and evaluate the model
    nn.train(X, Y, epochs, batch_size);
    
    std::cout << "Tested architecture: ";
    for (int layer_size : architecture) {
        std::cout << layer_size << " ";
    }
    std::cout << std::endl;
}