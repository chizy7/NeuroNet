#include "Optimizer.h"
#include <iostream>

SGD::SGD(double lr) : learning_rate(lr) {}

void SGD::update(const std::vector<std::unique_ptr<Layer>>& layers) {
    for (const auto& layer : layers) {
        // Update weights with the learning rate
        layer->update_weights(learning_rate * Eigen::MatrixXd::Ones(layer->get_grad_weights().rows(), layer->get_grad_weights().cols()), 
                              learning_rate * Eigen::VectorXd::Ones(layer->get_grad_bias().size()));
    }
}

Adam::Adam(double lr, double b1, double b2, double eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void Adam::update(const std::vector<std::unique_ptr<Layer>>& layers) {
    t += 1;

    if (m.empty()) {
        m.resize(layers.size());
        v.resize(layers.size());
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];

        Eigen::MatrixXd grad_weights = layer->get_grad_weights();
        Eigen::VectorXd grad_bias = layer->get_grad_bias();

        // Initialize moment matrices if not already done
        if (m[i].size() == 0) {
            m[i] = Eigen::MatrixXd::Zero(grad_weights.rows(), grad_weights.cols());
            v[i] = Eigen::MatrixXd::Zero(grad_weights.rows(), grad_weights.cols());
        }

        // Update biased first moment estimate
        m[i] = beta1 * m[i] + (1 - beta1) * grad_weights;

        // Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1 - beta2) * grad_weights.cwiseProduct(grad_weights);

        // Correct bias in first moment
        Eigen::MatrixXd m_hat = m[i] / (1 - std::pow(beta1, t));

        // Correct bias in second moment
        Eigen::MatrixXd v_hat = v[i] / (1 - std::pow(beta2, t));

        // // Compute weight and bias updates
        Eigen::MatrixXd weight_update = learning_rate * m_hat.array().cwiseQuotient(v_hat.array().sqrt() + epsilon).matrix();
        Eigen::VectorXd bias_update = learning_rate * grad_bias;

        // Apply updates to weights and biases
        layer->update_weights(weight_update, bias_update);
    }
}

Lookahead::Lookahead(Optimizer* base_optimizer, double alpha, int k)
    : base_optimizer(base_optimizer), alpha(alpha), k(k) {}

void Lookahead::update(const std::vector<std::unique_ptr<Layer>>& layers) {
    if (slow_weights.empty()) {
        for (const auto& layer : layers) {
            slow_weights.push_back(layer->get_weights()); // Access layer through unique_ptr
        }
    }

    base_optimizer->update(layers);

    if (--k == 0) {
        for (size_t i = 0; i < layers.size(); ++i) {
            Eigen::MatrixXd fast_weights = layers[i]->get_weights(); // Dereference unique_ptr
            Eigen::MatrixXd new_weights = slow_weights[i] + alpha * (fast_weights - slow_weights[i]);
            layers[i]->set_weights(new_weights); // Dereference unique_ptr
            slow_weights[i] = new_weights;
        }
        k = 5;  // Reset step counter
    }
}