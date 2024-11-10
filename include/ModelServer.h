#pragma once
#include "NeuralNetwork.h"
#include <string>

class ModelServer {
public:
    ModelServer(const std::string& model_path);
    void start_server(int port);
    Eigen::MatrixXd predict(const Eigen::MatrixXd& input);

private:
    NeuralNetwork nn;
};