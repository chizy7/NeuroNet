#include "gtest/gtest.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"
#include "MSE.h"
#include <Eigen/Dense>
#include <memory>
#include <filesystem>

class NeuralNetworkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up a simple network architecture for testing
        nn.add_layer(new DenseLayer(4, 8));
        nn.add_layer(new DenseLayer(8, 2));
        
        // Compile with SGD optimizer and MSE loss
        nn.compile(new SGD(0.1), new MSE());
        
        // Create simple test data (XOR problem)
        X.resize(4, 4);
        X << 0, 0, 0, 1,
             0, 1, 0, 0,
             1, 0, 0, 0,
             1, 1, 0, 0;
        
        Y.resize(4);
        Y << 0, 1, 1, 0;
    }

    void TearDown() override {
        // Clean up temporary files
        std::filesystem::path model_path = "test_model.bin";
        if (std::filesystem::exists(model_path)) {
            std::filesystem::remove(model_path);
        }
    }

    NeuralNetwork nn;
    Eigen::MatrixXd X;
    Eigen::VectorXi Y;
};

TEST_F(NeuralNetworkTest, ForwardPass) {
    // Test forward pass dimensions
    Eigen::MatrixXd output = nn.forward(X);
    
    // Check output dimensions
    EXPECT_EQ(output.rows(), 4);
    EXPECT_EQ(output.cols(), 2);
}

TEST_F(NeuralNetworkTest, Training) {
    // Train for a few epochs
    nn.train(X, Y, 50, 4);
    
    // Check if the model can fit the training data
    Eigen::MatrixXd output = nn.forward(X);
    
    // Get predicted classes
    Eigen::VectorXi predicted_classes(4);
    for (int i = 0; i < 4; i++) {
        output.row(i).maxCoeff(&predicted_classes(i));
    }
    
    // Check that at least some predictions are correct
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        if (predicted_classes(i) == Y(i)) {
            correct++;
        }
    }
    
    // Expect at least 2 correct predictions after training
    // (might not fit perfectly with just 50 epochs)
    EXPECT_GE(correct, 2);
}

TEST_F(NeuralNetworkTest, ModelSaveLoad) {
    // Train the model a bit
    nn.train(X, Y, 10, 4);
    
    // Get predictions before saving
    Eigen::MatrixXd before_save = nn.forward(X);
    
    // Save the model
    std::string model_path = "test_model.bin";
    nn.save_model(model_path);
    
    // Create a new network with the same architecture
    NeuralNetwork nn2;
    nn2.add_layer(new DenseLayer(4, 8));
    nn2.add_layer(new DenseLayer(8, 2));
    
    // Load the saved weights
    nn2.load_model(model_path);
    
    // Get predictions after loading
    Eigen::MatrixXd after_load = nn2.forward(X);
    
    // Predictions should be the same (within a small epsilon)
    for (int i = 0; i < before_save.rows(); i++) {
        for (int j = 0; j < before_save.cols(); j++) {
            EXPECT_NEAR(before_save(i, j), after_load(i, j), 1e-5);
        }
    }
}

TEST_F(NeuralNetworkTest, Evaluation) {
    // Train for a few epochs
    nn.train(X, Y, 50, 4);
    
    // Evaluate the model
    double accuracy = nn.evaluate(X, Y);
    
    // Check that accuracy is a valid percentage
    EXPECT_GE(accuracy, 0.0);
    EXPECT_LE(accuracy, 100.0);
    
    // With enough training, we should get decent accuracy on this simple dataset
    EXPECT_GE(accuracy, 50.0);
}