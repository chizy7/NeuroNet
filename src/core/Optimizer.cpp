#include "Optimizer.h"
#include "Logger.h"
#include <iostream>

SGD::SGD(double lr) : learning_rate(lr) {}

void SGD::update(const std::vector<Layer*>& layers) {
    for (Layer* layer : layers) {
        layer->update_weights(learning_rate);
        
        // Optional debug statement to log after each layer update
        if (debug) {
            std::cout << "[DEBUG]: Updated weights for layer with learning rate: " << learning_rate << std::endl;
        }
    }
}