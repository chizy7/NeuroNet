#include "NeuralNetwork.h"
#include <iostream>

NeuralNetwork::NeuralNetwork() : optimizer(nullptr), loss_function(nullptr) {}

void NeuralNetwork::add_layer(Layer* layer) {
    layers.push_back(layer);
}

Eigen::MatrixXd NeuralNetwork::forward(const Eigen::MatrixXd& input) {
    Eigen::MatrixXd output = input;
    for (Layer* layer : layers) {
        output = layer->forward(output);
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

void NeuralNetwork::train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs) {
    for (int i = 0; i < epochs; ++i) {
        Eigen::MatrixXd output = forward(X);
        backward(output, Y);
        std::cout << "Epoch " << i+1 << " Loss: " << loss_function->calculate(output, Y) << std::endl;
    }
}