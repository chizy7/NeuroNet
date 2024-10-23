#include "Loss.h"
#include <iostream>

double MSE::calculate(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) {
    return (Y_pred - Y_true).array().square().mean();
}

Eigen::MatrixXd MSE::gradient(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) {
    return 2 * (Y_pred - Y_true) / Y_true.rows();
}