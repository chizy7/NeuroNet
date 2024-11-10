#pragma once
#include <Eigen/Dense>

class LSTM {
public:
    LSTM(int input_size, int hidden_size, int output_size);
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input);

private:
    Eigen::MatrixXd Wf, Wi, Wc, Wo;  // Forget, input, cell, output gates
    Eigen::MatrixXd bf, bi, bc, bo;  // Biases
    Eigen::MatrixXd hidden_state;
    Eigen::MatrixXd cell_state;
};