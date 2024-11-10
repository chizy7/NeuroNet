#pragma once
#include <Eigen/Dense>
#include <fstream>

class Layer {
public:
    virtual ~Layer() = default;  // Virtual destructor for safe polymorphic use

    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
    virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) = 0;
    virtual void update_weights(const Eigen::MatrixXd& grad_weights_update, const Eigen::VectorXd& grad_bias_update) = 0;

    virtual Eigen::MatrixXd get_weights() const = 0;
    virtual void set_weights(const Eigen::MatrixXd& new_weights) = 0;

    virtual Eigen::MatrixXd get_grad_weights() const = 0; 
    virtual Eigen::VectorXd get_grad_bias() const = 0;    

    // Serialization methods
    virtual void save(std::ofstream& out) const = 0;
    virtual void load(std::ifstream& in) = 0;
};

class DenseLayer : public Layer {
public:
    DenseLayer(int input_size, int output_size, double l2_lambda = 0.01);
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;
    void update_weights(const Eigen::MatrixXd& grad_weights_update, const Eigen::VectorXd& grad_bias_update) override;

    Eigen::MatrixXd get_grad_weights() const override { return grad_weights; }
    Eigen::VectorXd get_grad_bias() const override { return grad_bias; }

    Eigen::MatrixXd get_weights() const override;
    void set_weights(const Eigen::MatrixXd& new_weights) override;

    void set_l2_lambda(double lambda) { l2_lambda = lambda; }
    double get_regularization_loss() const;

    void save(std::ofstream& out) const override;
    void load(std::ifstream& in) override;

private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd bias;
    Eigen::MatrixXd input_cache;
    Eigen::MatrixXd grad_weights;
    Eigen::VectorXd grad_bias;
    double l2_lambda;
};