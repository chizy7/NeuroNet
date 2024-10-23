#include "Layer.h"
#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size) {
    weights = Eigen::MatrixXd::Random(output_size, input_size);
    bias = Eigen::MatrixXd::Random(output_size, 1);
}

Eigen::MatrixXd DenseLayer::forward(const Eigen::MatrixXd& input) {
    input_cache = input;
    // return (weights * input).colwise() + bias;
    return (weights * input).colwise() + bias.col(0);
}

Eigen::MatrixXd DenseLayer::backward(const Eigen::MatrixXd& grad_output) {
    grad_weights = grad_output * input_cache.transpose();
    grad_bias = grad_output.rowwise().sum();
    return weights.transpose() * grad_output;
}

void DenseLayer::update_weights(double learning_rate) {
    weights -= learning_rate * grad_weights;
    bias -= learning_rate * grad_bias;
}