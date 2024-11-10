#pragma once
#include <vector>
#include <memory> 
#include <Eigen/Dense>
#include <mpi.h>
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"

class NeuralNetwork {
public:
    NeuralNetwork();
    ~NeuralNetwork();  // Destructor for cleanup

    std::vector<Layer*> dynamic_layers;
    
    void add_dynamic_layer(Layer* layer);
    Eigen::MatrixXd forward_dynamic(const Eigen::MatrixXd& input);

    void add_layer(Layer* layer);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input);
    void backward(const Eigen::MatrixXd& output, const Eigen::MatrixXd& labels);
    void compile(Optimizer* optimizer, Loss* loss_fn);
    
    // Training with batch size
    void train(const Eigen::MatrixXd& X, const Eigen::VectorXi& Y, int epochs, int batch_size);
    
    // Evaluate the model accuracy on test data
    double evaluate(const Eigen::MatrixXd& X, const Eigen::VectorXi& Y);

    // Model saving and loading
    void save_model(const std::string& file_path);
    void load_model(const std::string& file_path);

    // Loss history tracking
    void save_loss_history(const std::string& file_path);

    // Distributed Training
    void train_distributed(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs, int batch_size, MPI_Comm comm);

private:
    std::vector<std::unique_ptr<Layer>> layers;  // Use smart pointers
    std::unique_ptr<Optimizer> optimizer;
    std::unique_ptr<Loss> loss_function;
    std::vector<double> loss_history;  // Track loss across epochs
};