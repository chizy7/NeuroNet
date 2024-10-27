#include "Loss.h"
#include <iostream>

double CrossEntropyLoss::calculate(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) {
    constexpr double epsilon = 1e-12;  // Small constant to prevent log(0)
    return -(Y_true.array() * (Y_pred.array() + epsilon).log()).sum() / Y_true.rows();
}

Eigen::MatrixXd CrossEntropyLoss::gradient(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) {
    return -Y_true.array() / Y_pred.array();
}