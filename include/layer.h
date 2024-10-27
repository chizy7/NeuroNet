#pragma once
#include <Eigen/Dense>

class Layer {
public:
    virtual ~Layer() = default;  // Virtual destructor for safe polymorphic use
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
    Eigen::VectorXd bias;            // Declared as a vector for better bias handling
    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd grad_weights;
    Eigen::VectorXd grad_bias;
};