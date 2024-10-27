#include "Layer.h"
#include "Logger.h"
#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size) {
    weights = Eigen::MatrixXd::Random(output_size, input_size);
    bias = Eigen::VectorXd::Random(output_size); // Use Eigen::VectorXd for bias
}

Eigen::MatrixXd DenseLayer::forward(const Eigen::MatrixXd& input) {
    if (debug) {
        std::cout << "[DenseLayer Forward] Input size: " << input.rows() << "x" << input.cols() << std::endl;
        std::cout << "[DenseLayer Forward] Weights size: " << weights.rows() << "x" << weights.cols() << std::endl;
        std::cout << "[DenseLayer Forward] Bias size: " << bias.size() << "x1" << std::endl;
    }

    input_cache = input;
    Eigen::MatrixXd output = input * weights.transpose();

    // Add bias vector to each row of output
    output.rowwise() += bias.transpose();

    if (debug) {
        std::cout << "[DenseLayer Forward] Output size: " << output.rows() << "x" << output.cols() << std::endl;
    }

    return output;
}

Eigen::MatrixXd DenseLayer::backward(const Eigen::MatrixXd& grad_output) {
    if (grad_output.cols() != weights.rows()) {
        Logger::error("Backward pass dimension mismatch: grad_output.cols() = " + std::to_string(grad_output.cols()) +
                      ", weights.rows() = " + std::to_string(weights.rows()));
        exit(1);
    }

    grad_weights = grad_output.transpose() * input_cache;
    grad_bias = grad_output.colwise().sum();

    Eigen::MatrixXd grad_input = grad_output * weights;

    if (debug) {
        std::cout << "[DenseLayer Backward] grad_weights size: " << grad_weights.rows() << "x" << grad_weights.cols() << std::endl;
        std::cout << "[DenseLayer Backward] grad_bias size: " << grad_bias.size() << "x1" << std::endl;
        std::cout << "[DenseLayer Backward] grad_input size: " << grad_input.rows() << "x" << grad_input.cols() << std::endl;
    }

    return grad_input;
}

void DenseLayer::update_weights(double learning_rate) {
    weights -= learning_rate * grad_weights;
    bias -= learning_rate * grad_bias;
}