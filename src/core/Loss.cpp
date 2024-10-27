#include "Loss.h"
#include <iostream>

double CrossEntropyLoss::calculate(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) {
    return -(Y_true.array() * Y_pred.array().log()).sum() / Y_true.rows();
}

Eigen::MatrixXd CrossEntropyLoss::gradient(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) {
    return -Y_true.array() / Y_pred.array();
}