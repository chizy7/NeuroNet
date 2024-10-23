#include "Optimizer.h"
#include <iostream>

SGD::SGD(double lr) : learning_rate(lr) {}

void SGD::update(const std::vector<Layer*>& layers) {
    for (Layer* layer : layers) {
        layer->update_weights(learning_rate);
    }
}