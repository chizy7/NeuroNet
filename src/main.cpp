#include <iostream>
#include "NeuralNetwork.h"
#include "DataLoader.h"
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"
#include "Logger.h"
#include <Eigen/Dense>

bool debug = false; // Set to true to enable detailed output

int main() {
    Logger::info("Initializing Neural Network");

    Eigen::MatrixXd train_data;
    Eigen::VectorXi train_labels;
    Eigen::MatrixXd test_data;
    Eigen::VectorXi test_labels;

    // Load MNIST training and testing datasets
    DataLoader::load_mnist_csv("../data/mnist_train.csv", train_data, train_labels);
    DataLoader::load_mnist_csv("../data/mnist_test.csv", test_data, test_labels);

    // Define the neural network structure for MNIST
    NeuralNetwork nn;
    nn.add_layer(new DenseLayer(784, 128));   // First hidden layer with input size 784
    nn.add_layer(new DenseLayer(128, 64));    // Second hidden layer
    nn.add_layer(new DenseLayer(64, 10));     // Output layer with 10 units for 10 classes

    // Compile the model with an optimizer and loss function suitable for classification
    nn.compile(new SGD(0.01), new CrossEntropyLoss());

    // Train the model on the MNIST training data
    nn.train(train_data, train_labels, 10, 64);  // 10 epochs, batch size of 64

    // Evaluate model accuracy on the test data
    double accuracy = nn.evaluate(test_data, test_labels);
    std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;

    return 0;
}