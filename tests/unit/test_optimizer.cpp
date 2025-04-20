#include "gtest/gtest.h"
#include "Optimizer.h"
#include "Layer.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>

class OptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test layer
        layers.push_back(std::make_unique<DenseLayer>(5, 3));
    }

    void TearDown() override {
        layers.clear();
    }

    std::vector<std::unique_ptr<Layer>> layers;
};

TEST_F(OptimizerTest, SGDCreation) {
    // Test that SGD can be created with a learning rate
    SGD sgd(0.01);
    
    // No explicit verification needed - just check that creation doesn't throw
    SUCCEED();
}

TEST_F(OptimizerTest, SGDUpdate) {
    // Create SGD optimizer
    SGD sgd(0.01);
    
    // Store original weights
    Eigen::MatrixXd original_weights = layers[0]->get_weights();
    
    // Set up grad_weights by doing a forward and backward pass
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(2, 5);
    Eigen::MatrixXd output = layers[0]->forward(input);
    Eigen::MatrixXd grad_output = Eigen::MatrixXd::Ones(2, 3);
    layers[0]->backward(grad_output);
    
    // Update weights
    sgd.update(layers);
    
    // Check that weights have been updated (they should be different)
    Eigen::MatrixXd updated_weights = layers[0]->get_weights();
    
    // The weights should be different after update
    EXPECT_FALSE((updated_weights.array() == original_weights.array()).all());
}

TEST_F(OptimizerTest, AdamCreation) {
    // Test that Adam can be created with default parameters
    Adam adam();
    
    // Test with custom parameters
    Adam adam_custom(0.001, 0.9, 0.999, 1e-8);
    
    // No explicit verification needed - just check that creation doesn't throw
    SUCCEED();
}

TEST_F(OptimizerTest, AdamUpdate) {
    // Create Adam optimizer
    Adam adam(0.001);
    
    // Store original weights
    Eigen::MatrixXd original_weights = layers[0]->get_weights();
    
    // Set up grad_weights by doing a forward and backward pass
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(2, 5);
    Eigen::MatrixXd output = layers[0]->forward(input);
    Eigen::MatrixXd grad_output = Eigen::MatrixXd::Ones(2, 3);
    layers[0]->backward(grad_output);
    
    // Update weights
    adam.update(layers);
    
    // Check that weights have been updated (they should be different)
    Eigen::MatrixXd updated_weights = layers[0]->get_weights();
    
    // The weights should be different after update
    EXPECT_FALSE((updated_weights.array() == original_weights.array()).all());
}

TEST_F(OptimizerTest, LookaheadCreation) {
    // Create a base optimizer
    SGD* sgd = new SGD(0.01);
    
    // Test that Lookahead can wrap another optimizer
    Lookahead lookahead(sgd, 0.5, 5);
    
    // No explicit verification needed - just check that creation doesn't throw
    SUCCEED();
}

TEST_F(OptimizerTest, LookaheadUpdate) {
    // Create base optimizer and Lookahead
    SGD* sgd = new SGD(0.01);
    Lookahead lookahead(sgd, 0.5, 1); // k=1 for immediate update
    
    // Store original weights
    Eigen::MatrixXd original_weights = layers[0]->get_weights();
    
    // Set up grad_weights by doing a forward and backward pass
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(2, 5);
    Eigen::MatrixXd output = layers[0]->forward(input);
    Eigen::MatrixXd grad_output = Eigen::MatrixXd::Ones(2, 3);
    layers[0]->backward(grad_output);
    
    // Update weights
    lookahead.update(layers);
    
    // Check that weights have been updated (they should be different)
    Eigen::MatrixXd updated_weights = layers[0]->get_weights();
    
    // The weights should be different after update
    EXPECT_FALSE((updated_weights.array() == original_weights.array()).all());
    
    // Clean up
    delete sgd;
}