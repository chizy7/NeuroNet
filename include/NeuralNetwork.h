#pragma once
#include <vector>
#include <Eigen/Dense>
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"

class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();  // Destructor for cleanup

    void add_layer(Layer* layer);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input);
    void backward(const Eigen::MatrixXd& output, const Eigen::MatrixXd& labels);
    void compile(Optimizer* optimizer, Loss* loss_fn);
    
    // Training with batch size
    void train(const Eigen::MatrixXd& X, const Eigen::VectorXi& Y, int epochs, int batch_size);
    
    // Evaluate the model accuracy on test data
    double evaluate(const Eigen::MatrixXd& X, const Eigen::VectorXi& Y);

private:
    std::vector<Layer*> layers;
    Optimizer* optimizer;
    Loss* loss_function;
};