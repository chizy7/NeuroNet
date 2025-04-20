#include "gtest/gtest.h"
#include "Layer.h"
#include <Eigen/Dense>
#include <memory>

class DenseLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_size = 5;
        output_size = 3;
        layer = new DenseLayer(input_size, output_size);
    }

    void TearDown() override {
        delete layer;
    }

    int input_size;
    int output_size;
    DenseLayer* layer;
};

TEST_F(DenseLayerTest, ForwardPassDimensions) {
    // Create test input
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(2, input_size);
    
    // Perform forward pass
    Eigen::MatrixXd output = layer->forward(input);
    
    // Check dimensions
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), output_size);
}

TEST_F(DenseLayerTest, BackwardPassDimensions) {
    // Create test input and run forward pass first
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(2, input_size);
    layer->forward(input);
    
    // Create gradient output
    Eigen::MatrixXd grad_output = Eigen::MatrixXd::Random(2, output_size);
    
    // Run backward pass
    Eigen::MatrixXd grad_input = layer->backward(grad_output);
    
    // Check dimensions
    EXPECT_EQ(grad_input.rows(), 2);
    EXPECT_EQ(grad_input.cols(), input_size);
}

TEST_F(DenseLayerTest, WeightsAndBiasAccess) {
    Eigen::MatrixXd weights = layer->get_weights();
    
    // Check dimensions of weights
    EXPECT_EQ(weights.rows(), output_size);
    EXPECT_EQ(weights.cols(), input_size);
    
    // Check if weights can be set
    Eigen::MatrixXd new_weights = Eigen::MatrixXd::Ones(output_size, input_size) * 0.1;
    layer->set_weights(new_weights);
    
    // Verify the weights were set correctly
    Eigen::MatrixXd retrieved_weights = layer->get_weights();
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            EXPECT_NEAR(retrieved_weights(i, j), 0.1, 1e-10);
        }
    }
}

TEST_F(DenseLayerTest, RegularizationLoss) {
    // Set weights to known values
    Eigen::MatrixXd new_weights = Eigen::MatrixXd::Ones(output_size, input_size);
    layer->set_weights(new_weights);
    
    // Set L2 lambda
    layer->set_l2_lambda(0.1);
    
    // Calculate expected regularization loss: l2_lambda * sum(weights^2)
    double expected_loss = 0.1 * output_size * input_size; // Each weight is 1.0
    
    // Get actual regularization loss
    double actual_loss = layer->get_regularization_loss();
    
    // Check if the regularization loss is calculated correctly
    EXPECT_NEAR(actual_loss, expected_loss, 1e-10);
}