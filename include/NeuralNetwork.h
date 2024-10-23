#pragma once
#include <vector>
#include <Eigen/Dense>
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"

class NeuralNetwork {
public:
    NeuralNetwork();

    void add_layer(Layer* layer);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input);
    void backward(const Eigen::MatrixXd& output, const Eigen::MatrixXd& labels);
    void compile(Optimizer* optimizer, Loss* loss_fn);
    void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs);

private:
    std::vector<Layer*> layers;
    Optimizer* optimizer;
    Loss* loss_function;
};