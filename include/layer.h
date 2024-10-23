#pragma once
#include <Eigen/Dense>

class Layer {
public:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) = 0;
    virtual void update_weights(double learning_rate) = 0;
};

class DenseLayer : public Layer {
public:
    DenseLayer(int input_size, int output_size);
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;
    void update_weights(double learning_rate) override;

private:
    Eigen::MatrixXd weights;
    Eigen::MatrixXd bias;
    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd grad_weights;
    Eigen::MatrixXd grad_bias;
};