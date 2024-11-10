#pragma once
#include <vector>
#include <memory>
#include "Layer.h"

class Optimizer {
public:
    virtual void update(const std::vector<std::unique_ptr<Layer>>& layers) = 0;  // Accept unique_ptr
    virtual ~Optimizer() = default;
};

class SGD : public Optimizer {
public:
    SGD(double learning_rate);
    void update(const std::vector<std::unique_ptr<Layer>>& layers) override;

private:
    double learning_rate;
};

class Adam : public Optimizer {
public:
    Adam(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

    void update(const std::vector<std::unique_ptr<Layer>>& layers) override;

private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    std::vector<Eigen::MatrixXd> m; // First moment est
    std::vector<Eigen::MatrixXd> v; // Second moment est
    int t; // Time step
};

class Lookahead : public Optimizer {
public:
    Lookahead(Optimizer* base_optimizer, double alpha = 0.5, int k = 5);
    void update(const std::vector<std::unique_ptr<Layer>>& layers) override;

private:
    Optimizer* base_optimizer;
    double alpha;
    int k;
    std::vector<Eigen::MatrixXd> slow_weights;
};