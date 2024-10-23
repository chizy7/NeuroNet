#include <iostream>
#include "NeuralNetwork.h"
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"
#include "Logger.h"
#include "DataLoader.h"

int main() {
    Logger::info("Initializing Neural Network");

    // Define a neural network with 2 layers
    NeuralNetwork nn;
    nn.add_layer(new DenseLayer(2, 5));  // Input layer
    nn.add_layer(new DenseLayer(5, 1));  // Output layer

    // Compile the model with SGD optimizer and MSE loss
    nn.compile(new SGD(0.01), new MSE());

    // Load dataset
    Eigen::MatrixXd X = DataLoader::load_csv("data/train_X.csv");
    Eigen::MatrixXd Y = DataLoader::load_csv("data/train_Y.csv");

    // Train the model
    nn.train(X, Y, 100);

    return 0;
}