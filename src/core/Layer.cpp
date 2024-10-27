#include "Layer.h"
#include "Logger.h"
#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size) {
    weights = Eigen::MatrixXd::Random(output_size, input_size);
    bias = Eigen::MatrixXd::Random(output_size, 1);
}

Eigen::MatrixXd DenseLayer::forward(const Eigen::MatrixXd& input) {
    if (debug) {
        std::cout << "[DenseLayer Forward] Input size: " << input.rows() << "x" << input.cols() << std::endl;
        std::cout << "[DenseLayer Forward] Weights size: " << weights.rows() << "x" << weights.cols() << std::endl;
        std::cout << "[DenseLayer Forward] Bias size: " << bias.rows() << "x" << bias.cols() << std::endl;
    }
    
    input_cache = input;
    Eigen::MatrixXd output = input * weights.transpose();

    // Create a matrix of the same size as output, where each row is a copy of the bias vector
    Eigen::MatrixXd bias_expanded = bias.transpose().replicate(output.rows(), 1);

    if (debug) {
        std::cout << "[DenseLayer Forward] Output before bias addition size: " << output.rows() << "x" << output.cols() << std::endl;
        std::cout << "[DenseLayer Forward] Expanded bias size: " << bias_expanded.rows() << "x" << bias_expanded.cols() << std::endl;
    }
    
    output += bias_expanded;

    if (debug) {
        std::cout << "[DenseLayer Forward] Output after bias addition size: " << output.rows() << "x" << output.cols() << std::endl;
    }
    
    return output;
}

Eigen::MatrixXd DenseLayer::backward(const Eigen::MatrixXd& grad_output) {
    // Verify that the dimensions of grad_output are compatible with weights
    if (grad_output.cols() != weights.rows()) {
        std::cerr << "Backward pass dimension mismatch: grad_output.cols() = "
                  << grad_output.cols() << ", weights.rows() = " << weights.rows()
                  << std::endl;
        exit(1);
    }

    // Compute gradients
    grad_weights = grad_output.transpose() * input_cache;
    grad_bias = grad_output.colwise().sum().transpose();  // Ensure grad_bias is a column vector

    // Compute grad_input
    Eigen::MatrixXd grad_input = grad_output * weights;

    // Log dimensions for debugging
    if (debug) {
        std::cout << "[DenseLayer Backward] grad_weights size: " << grad_weights.rows() << "x" << grad_weights.cols() << std::endl;
        std::cout << "[DenseLayer Backward] grad_bias size: " << grad_bias.rows() << "x" << grad_bias.cols() << std::endl;
        std::cout << "[DenseLayer Backward] grad_input size: " << grad_input.rows() << "x" << grad_input.cols() << std::endl;
    }

    return grad_input;
}

void DenseLayer::update_weights(double learning_rate) {
    weights -= learning_rate * grad_weights;
    bias -= learning_rate * grad_bias;
}