#include "NeuralNetwork.h"
#include "Logger.h"
#include <iostream>
#include <algorithm>
#include <numeric>

NeuralNetwork::NeuralNetwork() : optimizer(nullptr), loss_function(nullptr) {}

NeuralNetwork::~NeuralNetwork() {
    // Free dynamically allocated layers
    for (Layer* layer : layers) {
        delete layer;
    }
    // Free dynamically allocated optimizer and loss function if they exist
    delete optimizer;
    delete loss_function;
}

void NeuralNetwork::add_layer(Layer* layer) {
    layers.push_back(layer);
}

Eigen::MatrixXd NeuralNetwork::forward(const Eigen::MatrixXd& input) {
    Eigen::MatrixXd output = input;
    for (Layer* layer : layers) {
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

void NeuralNetwork::backward(const Eigen::MatrixXd& output, const Eigen::MatrixXd& labels) {
    Eigen::MatrixXd grad_loss = loss_function->gradient(output, labels);
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad_loss = (*it)->backward(grad_loss);
    }
    optimizer->update(layers);
}

void NeuralNetwork::compile(Optimizer* opt, Loss* loss_fn) {
    this->optimizer = opt;
    this->loss_function = loss_fn;
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

        if (debug) {
            std::cout << "Epoch " << epoch + 1 << " completed. Average Loss: " << total_loss / (num_samples / batch_size) << std::endl;
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