#pragma once
#include <Eigen/Dense>

class Loss {
public:
    virtual ~Loss() = default;  // Virtual destructor for safe polymorphic use
    virtual double calculate(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) = 0;
    virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) = 0;
};

class CrossEntropyLoss : public Loss {
public:
    double calculate(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) override;
    Eigen::MatrixXd gradient(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) override;
};