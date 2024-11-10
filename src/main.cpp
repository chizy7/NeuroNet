#include <iostream>
#include <filesystem>
#include "NeuralNetwork.h"
#include "DataLoader.h"
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"
#include "Logger.h"
#include <Eigen/Dense>

bool debug = false; // debug flag

int main(int argc, char* argv[]) {
    // Enable debug mode if "--debug" is passed as an argument
    if (argc > 1 && std::string(argv[1]) == "--debug") {
        debug = true;
    }

    Logger::info("Initializing Neural Network");

    Eigen::MatrixXd train_data;
    Eigen::VectorXi train_labels;
    Eigen::MatrixXd test_data;
    Eigen::VectorXi test_labels;

    // Ensure the models directory exists before proceeding
    const std::string models_dir = "../models";
    if (!std::filesystem::exists(models_dir)) {
        if (std::filesystem::create_directories(models_dir)) {
            Logger::info("Created models directory successfully.");
        } else {
            Logger::error("Failed to create models directory. Exiting...");
            return -1; // Exit if directory creation fails
        }
    }

    // Load MNIST training and testing datasets
    DataLoader::load_mnist_csv("../data/mnist_train.csv", train_data, train_labels);
    DataLoader::load_mnist_csv("../data/mnist_test.csv", test_data, test_labels);

    // Define the neural network structure for MNIST
    NeuralNetwork nn;
    nn.add_layer(new DenseLayer(784, 128));   // First hidden layer with input size 784
    nn.add_layer(new DenseLayer(128, 64));    // Second hidden layer
    nn.add_layer(new DenseLayer(64, 10));     // Output layer with 10 units for 10 classes

    // Compile the model with an optimizer and loss function suitable for classification
    nn.compile(new Adam(0.001), new CrossEntropyLoss());

    /*
 * Compile the model with an optimizer and loss function suitable for classification:
 * Use Adam optimizer -> Advanced optimizer with adaptive learning rates and momentum
 *    nn.compile(new Adam(0.001), new CrossEntropyLoss());
 *
 * Alternatively, switch to SGD -> Simple optimizer with a fixed learning rate:
 *    nn.compile(new SGD(0.01), new CrossEntropyLoss());
 *
 * Test alternative approaches:
 *    bool use_sgd = false; // Toggle optimizer
 *    if (use_sgd) {
 *        nn.compile(new SGD(0.01), new CrossEntropyLoss());
 *    } else {
 *        nn.compile(new Adam(0.001), new CrossEntropyLoss());
 *    }
 *
 * Adjust the training loss and save directory as needed.
 */

    // Attempt to load the model or train a new one if it doesn't exist
    std::string model_path = models_dir + "/neuro_net.model";
    if (std::filesystem::exists(model_path)) {
        Logger::info("Model file found. Loading the trained model...");
        nn.load_model(model_path);
    } else {
        Logger::info("Model file not found. Training a new model...");
        nn.train(train_data, train_labels, 10, 64);  // Train the model
        Logger::info("Saving the newly trained model...");
        nn.save_model(model_path); // Save the trained model
    }

    // Save the training loss history to a file
    std::string loss_history_path = std::filesystem::absolute(std::filesystem::path(models_dir) / "training_loss.csv").string();
    std::cout << "Full path for loss history: " << loss_history_path << std::endl;
    Logger::info("Saving loss history to: " + loss_history_path);
    nn.save_loss_history(loss_history_path);

    // Evaluate model accuracy on the test data
    double accuracy = nn.evaluate(test_data, test_labels);
    Logger::info("Evaluation completed.");
    std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;

    return 0;
}