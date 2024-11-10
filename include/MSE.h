#pragma once

#include "Loss.h"
#include <Eigen/Dense>

class MSE : public Loss {
public:
    double calculate(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) override {
        return (Y_pred - Y_true).squaredNorm() / Y_true.rows();
    }

    Eigen::MatrixXd gradient(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) override {
        return 2.0 * (Y_pred - Y_true) / Y_true.rows();
    }
};