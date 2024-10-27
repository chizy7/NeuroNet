#pragma once
#include <vector>
#include "Layer.h"

class Optimizer {
public:
    virtual void update(const std::vector<Layer*>& layers) = 0;
    virtual ~Optimizer() = default;  // Add virtual destructor
};

class SGD : public Optimizer {
public:
    SGD(double learning_rate);
    void update(const std::vector<Layer*>& layers) override;

private:
    double learning_rate;
};