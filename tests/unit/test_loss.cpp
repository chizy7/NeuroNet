#include "gtest/gtest.h"
#include "Loss.h"
#include "MSE.h"
#include <Eigen/Dense>

class LossTest : public ::testing::Test {
protected:
    void SetUp() override {
        cross_entropy = new CrossEntropyLoss();
        mse = new MSE();
    }

    void TearDown() override {
        delete cross_entropy;
        delete mse;
    }

    CrossEntropyLoss* cross_entropy;
    MSE* mse;
};

TEST_F(LossTest, CrossEntropyCalculation) {
    // Create test data
    Eigen::MatrixXd y_pred(2, 3);
    y_pred << 0.7, 0.2, 0.1,
              0.3, 0.6, 0.1;
    
    Eigen::MatrixXd y_true(2, 3);
    y_true << 1.0, 0.0, 0.0,
              0.0, 1.0, 0.0;
    
    // Calculate loss
    double loss = cross_entropy->calculate(y_pred, y_true);
    
    // Expected loss: -1/2 * (log(0.7) + log(0.6))
    double expected_loss = -0.5 * (std::log(0.7) + std::log(0.6));
    
    // Check that loss is calculated correctly
    EXPECT_NEAR(loss, expected_loss, 1e-5);
}

TEST_F(LossTest, CrossEntropyGradient) {
    // Create test data
    Eigen::MatrixXd y_pred(2, 3);
    y_pred << 0.7, 0.2, 0.1,
              0.3, 0.6, 0.1;
    
    Eigen::MatrixXd y_true(2, 3);
    y_true << 1.0, 0.0, 0.0,
              0.0, 1.0, 0.0;
    
    // Calculate gradient
    Eigen::MatrixXd grad = cross_entropy->gradient(y_pred, y_true);
    
    // Expected gradients
    Eigen::MatrixXd expected_grad(2, 3);
    expected_grad << -1.0/0.7, 0.0, 0.0,
                     0.0, -1.0/0.6, 0.0;
    
    // Check dimensions
    EXPECT_EQ(grad.rows(), 2);
    EXPECT_EQ(grad.cols(), 3);
    
    // Check gradient values
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(grad(i, j), expected_grad(i, j), 1e-5);
        }
    }
}

TEST_F(LossTest, MSECalculation) {
    // Create test data
    Eigen::MatrixXd y_pred(2, 2);
    y_pred << 0.7, 0.2,
              0.3, 0.6;
    
    Eigen::MatrixXd y_true(2, 2);
    y_true << 1.0, 0.0,
              0.0, 1.0;
    
    // Calculate loss
    double loss = mse->calculate(y_pred, y_true);
    
    // Expected loss: ((0.7-1.0)^2 + (0.2-0.0)^2 + (0.3-0.0)^2 + (0.6-1.0)^2) / 2
    double expected_loss = (std::pow(0.7-1.0, 2) + std::pow(0.2-0.0, 2) +
                           std::pow(0.3-0.0, 2) + std::pow(0.6-1.0, 2)) / 2.0;
    
    // Check that loss is calculated correctly
    EXPECT_NEAR(loss, expected_loss, 1e-5);
}

TEST_F(LossTest, MSEGradient) {
    // Create test data
    Eigen::MatrixXd y_pred(2, 2);
    y_pred << 0.7, 0.2,
              0.3, 0.6;
    
    Eigen::MatrixXd y_true(2, 2);
    y_true << 1.0, 0.0,
              0.0, 1.0;
    
    // Calculate gradient
    Eigen::MatrixXd grad = mse->gradient(y_pred, y_true);
    
    // Expected gradients: 2*(y_pred - y_true) / n
    Eigen::MatrixXd expected_grad(2, 2);
    expected_grad << 2.0*(0.7-1.0)/2.0, 2.0*(0.2-0.0)/2.0,
                     2.0*(0.3-0.0)/2.0, 2.0*(0.6-1.0)/2.0;
    
    // Check dimensions
    EXPECT_EQ(grad.rows(), 2);
    EXPECT_EQ(grad.cols(), 2);
    
    // Check gradient values
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_NEAR(grad(i, j), expected_grad(i, j), 1e-5);
        }
    }
}