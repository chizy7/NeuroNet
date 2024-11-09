#include "Layer.h"
#include <iostream>

DenseLayer::DenseLayer(int input_size, int output_size, double l2_lambda)
    : l2_lambda(l2_lambda) {
    weights = Eigen::MatrixXd::Random(output_size, input_size);
    bias = Eigen::VectorXd::Random(output_size);
}

Eigen::MatrixXd DenseLayer::forward(const Eigen::MatrixXd& input) {
    input_cache = input;
    Eigen::MatrixXd output = input * weights.transpose();
    output.rowwise() += bias.transpose();
    return output;
}

Eigen::MatrixXd DenseLayer::backward(const Eigen::MatrixXd& grad_output) {
    grad_weights = grad_output.transpose() * input_cache;
    grad_bias = grad_output.colwise().sum();
    return grad_output * weights;
}

double DenseLayer::get_regularization_loss() const {
    return l2_lambda * weights.squaredNorm();
}

void DenseLayer::update_weights(const Eigen::MatrixXd& grad_weights_update, const Eigen::VectorXd& grad_bias_update) {
    weights -= grad_weights_update + l2_lambda * weights;
    bias -= grad_bias_update;
}

void DenseLayer::save(std::ofstream& out) const {
    if (out.is_open()) {
        int rows = weights.rows(), cols = weights.cols();
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        out.write(reinterpret_cast<const char*>(weights.data()), sizeof(double) * rows * cols);

        int bias_size = bias.size();
        out.write(reinterpret_cast<const char*>(&bias_size), sizeof(bias_size));
        out.write(reinterpret_cast<const char*>(bias.data()), sizeof(double) * bias_size);
    }
}

void DenseLayer::load(std::ifstream& in) {
    if (in.is_open()) {
        int rows, cols;
        in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        weights.resize(rows, cols);
        in.read(reinterpret_cast<char*>(weights.data()), sizeof(double) * rows * cols);

        int bias_size;
        in.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));
        bias.resize(bias_size);
        in.read(reinterpret_cast<char*>(bias.data()), sizeof(double) * bias_size);
    }
}