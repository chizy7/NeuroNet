#include <mpi.h>
#include "NeuralNetwork.h"
#include "Logger.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <thread>

NeuralNetwork::NeuralNetwork() : optimizer(nullptr), loss_function(nullptr) {}

NeuralNetwork::~NeuralNetwork() {
    // Smart pointers automatically handle memory deallocation, so no manual deletes are needed
    layers.clear();
    optimizer.reset();
    loss_function.reset();
}

void NeuralNetwork::add_layer(Layer* layer) {
    layers.emplace_back(std::unique_ptr<Layer>(layer)); 
}

Eigen::MatrixXd NeuralNetwork::forward(const Eigen::MatrixXd& input) {
    Eigen::MatrixXd output = input;

    // Sequential forward pass
    for (const auto& layer : layers) {
        if (debug) {
            std::cout << "Input to layer: " << output.rows() << "x" << output.cols() << std::endl;
        }
        output = layer->forward(output);
        if (debug) {
            std::cout << "Output of layer: " << output.rows() << "x" << output.cols() << std::endl;
        }
    }
    return output;
}

void NeuralNetwork::add_dynamic_layer(Layer* layer) {
    dynamic_layers.push_back(layer);
}

Eigen::MatrixXd NeuralNetwork::forward_dynamic(const Eigen::MatrixXd& input) {
    Eigen::MatrixXd output = input;
    for (Layer* layer : dynamic_layers) {
        output = layer->forward(output);
    }
    return output;
}

void NeuralNetwork::backward(const Eigen::MatrixXd& output, const Eigen::MatrixXd& labels) {
    Eigen::MatrixXd grad_loss = loss_function->gradient(output, labels);

    // Reverse order for backpropagation
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad_loss = (*it)->backward(grad_loss);
    }

    // Update weights using the optimizer
    optimizer->update(layers);  // Pass unique_ptr vector
}

void NeuralNetwork::compile(Optimizer* opt, Loss* loss_fn) {
    optimizer.reset(opt);  // Transfer ownership to unique_ptr
    loss_function.reset(loss_fn);  // Transfer ownership to unique_ptr
}

void NeuralNetwork::train(const Eigen::MatrixXd& X, const Eigen::VectorXi& Y, int epochs, int batch_size) {
    int num_samples = X.rows();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (int i = 0; i < num_samples; i += batch_size) {
            int end = std::min(i + batch_size, num_samples);
            Eigen::MatrixXd X_batch = X.middleRows(i, end - i);
            Eigen::MatrixXd Y_batch = Eigen::MatrixXd::Zero(end - i, 10);

            for (int j = 0; j < end - i; ++j) {
                Y_batch(j, Y(i + j)) = 1;
            }

            Eigen::MatrixXd output = forward(X_batch);
            total_loss += loss_function->calculate(output, Y_batch);
            backward(output, Y_batch);
        }

        double avg_loss = total_loss / (num_samples / batch_size);
        loss_history.push_back(avg_loss);  // Track average loss for the epoch

        if (debug) {
            std::cout << "Epoch " << epoch + 1 << " completed. Average Loss: " << avg_loss << std::endl;
        }
    }
}

void NeuralNetwork::save_model(const std::string& file_path) {
    // Ensure the directory exists before saving
    std::filesystem::path directory = std::filesystem::path(file_path).parent_path();
    if (!directory.empty() && !std::filesystem::exists(directory)) {
        std::filesystem::create_directories(directory);
    }

    // Open the file and save the model
    std::ofstream file(file_path, std::ios::binary);
    if (file.is_open()) {
        for (const auto& layer : layers) { // Use const auto& to access the unique_ptr
            layer->save(file); // Dereference the unique_ptr to access the Layer
        }
        file.close();
        std::cout << "Model saved to " << file_path << std::endl;
    } else {
        std::cerr << "Failed to open file for saving." << std::endl;
    }
}

void NeuralNetwork::load_model(const std::string& file_path) {
    // Ensure the file exists before attempting to load
    if (!std::filesystem::exists(file_path)) {
        std::cerr << "Model file does not exist: " << file_path << std::endl;
        return; // Early exit if the file does not exist
    }

    std::ifstream file(file_path, std::ios::binary);
    if (file.is_open()) {
        for (auto& layer : layers) { // Use auto& for unique_ptr compatibility
            layer->load(file); // Call the load method for each layer
        }
        file.close();
        std::cout << "Model loaded from " << file_path << std::endl;
    } else {
        std::cerr << "Failed to open model file for loading: " << file_path << std::endl;
    }
}

void NeuralNetwork::save_loss_history(const std::string& file_path) {
    std::ofstream file(file_path);
    if (file.is_open()) {
        for (double loss : loss_history) {
            file << loss << "\n";
        }
        file.close();
        std::cout << "Loss history saved to " << file_path << std::endl;
    } else {
        std::cerr << "Failed to open file for saving loss history. Path: " << file_path << std::endl;
        std::cerr << "Check directory existence and write permissions." << std::endl;
    }
}

void NeuralNetwork::train_distributed(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs, int batch_size, MPI_Comm comm) {
    int world_size, rank;
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &rank);

    // Divide the dataset among nodes
    int num_samples = X.cols();
    int local_batch_size = num_samples / world_size;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < num_samples; i += batch_size) {
            int batch_end = std::min(i + local_batch_size, num_samples);

            Eigen::MatrixXd X_batch = X.middleCols(rank * local_batch_size, local_batch_size);
            Eigen::MatrixXd Y_batch = Y.middleCols(rank * local_batch_size, local_batch_size);

            Eigen::MatrixXd output = forward(X_batch);
            backward(output, Y_batch);

            // Average gradients across all nodes
            for (const auto& layer : layers) {  // Use const auto& to access the unique_ptr
                MPI_Allreduce(MPI_IN_PLACE, layer->get_grad_weights().data(), layer->get_grad_weights().size(), MPI_DOUBLE, MPI_SUM, comm);
                MPI_Allreduce(MPI_IN_PLACE, layer->get_grad_bias().data(), layer->get_grad_bias().size(), MPI_DOUBLE, MPI_SUM, comm);
            }

            optimizer->update(layers);
        }
        if (rank == 0) {
            Eigen::MatrixXd output = forward(X);
            double loss = loss_function->calculate(output, Y);
            std::cout << "Epoch " << epoch + 1 << " Loss: " << loss << std::endl;
        }
    }
}

double NeuralNetwork::evaluate(const Eigen::MatrixXd& X, const Eigen::VectorXi& Y) {
    int num_samples = X.rows();
    int correct = 0;

    Eigen::MatrixXd output = forward(X);

    for (int i = 0; i < num_samples; ++i) {
        int predicted_class;
        output.row(i).maxCoeff(&predicted_class);
        if (predicted_class == Y(i)) {
            correct++;
        }
    }

    double accuracy = (static_cast<double>(correct) / num_samples) * 100.0;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    return accuracy;
}