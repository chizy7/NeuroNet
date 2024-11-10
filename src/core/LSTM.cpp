#include "LSTM.h"
#include <iostream>

LSTM::LSTM(int input_size, int hidden_size, int output_size) {
    Wf = Eigen::MatrixXd::Random(hidden_size, input_size);
    Wi = Eigen::MatrixXd::Random(hidden_size, input_size);
    Wc = Eigen::MatrixXd::Random(hidden_size, input_size);
    Wo = Eigen::MatrixXd::Random(hidden_size, input_size);

    bf = Eigen::MatrixXd::Random(hidden_size, 1);
    bi = Eigen::MatrixXd::Random(hidden_size, 1);
    bc = Eigen::MatrixXd::Random(hidden_size, 1);
    bo = Eigen::MatrixXd::Random(hidden_size, 1);
}

Eigen::MatrixXd LSTM::forward(const Eigen::MatrixXd& input) {
    // Apply LSTM forward pass (simplified for brevity)
    hidden_state = Wf * input + bf;
    cell_state = Wi * input + bi;
    Eigen::MatrixXd output = Wo * hidden_state + bo;
    return output;
}