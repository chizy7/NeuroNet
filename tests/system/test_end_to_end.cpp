#include "gtest/gtest.h"
#include "NeuralNetwork.h"
#include "Layer.h"
#include "Optimizer.h"
#include "Loss.h"
#include "DataLoader.h"
#include <Eigen/Dense>
#include <memory>
#include <filesystem>
#include <random>

class EndToEndTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a small synthetic dataset for MNIST-like testing
        createSyntheticData();
        
        // Set up a network architecture
        nn.add_layer(new DenseLayer(input_size, 32));
        nn.add_layer(new DenseLayer(32, 16));
        nn.add_layer(new DenseLayer(16, num_classes));
        
        // Compile with Adam optimizer and CrossEntropy loss
        nn.compile(new Adam(0.01), new CrossEntropyLoss());
    }

    void TearDown() override {
        // Clean up temporary files
        if (std::filesystem::exists("test_model_e2e.bin")) {
            std::filesystem::remove("test_model_e2e.bin");
        }
        if (std::filesystem::exists("test_loss_history.csv")) {
            std::filesystem::remove("test_loss_history.csv");
        }
    }
    
    void createSyntheticData() {
        // Small synthetic dataset that mimics MNIST structure but is much smaller
        num_samples = 20;
        input_size = 16; // 4x4 "images" instead of 28x28
        num_classes = 3; // Just 3 classes instead of 10
        
        // Random number generation
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        // Generate random data
        train_data = Eigen::MatrixXd::Zero(num_samples, input_size);
        for (int i = 0; i < num_samples; i++) {
            for (int j = 0; j < input_size; j++) {
                train_data(i, j) = dis(gen);
            }
        }
        
        // Generate random labels
        std::uniform_int_distribution<> label_dis(0, num_classes - 1);
        train_labels = Eigen::VectorXi::Zero(num_samples);
        for (int i = 0; i < num_samples; i++) {
            train_labels(i) = label_dis(gen);
        }
        
        // Use same data for testing in this synthetic case
        test_data = train_data;
        test_labels = train_labels;
    }

    NeuralNetwork nn;
    Eigen::MatrixXd train_data;
    Eigen::VectorXi train_labels;
    Eigen::MatrixXd test_data;
    Eigen::VectorXi test_labels;
    int num_samples;
    int input_size;
    int num_classes;
};

TEST_F(EndToEndTest, TrainEvaluateSave) {
    // Train the model
    nn.train(train_data, train_labels, 10, 5);
    
    // Evaluate the model
    double accuracy = nn.evaluate(test_data, test_labels);
    
    // Check that accuracy is a valid percentage
    EXPECT_GE(accuracy, 0.0);
    EXPECT_LE(accuracy, 100.0);
    
    // Save the model
    nn.save_model("test_model_e2e.bin");
    
    // Save loss history
    nn.save_loss_history("test_loss_history.csv");
    
    // Check that files exist
    EXPECT_TRUE(std::filesystem::exists("test_model_e2e.bin"));
    EXPECT_TRUE(std::filesystem::exists("test_loss_history.csv"));
}

TEST_F(EndToEndTest, LoadEvaluate) {
    // First train and save the model
    nn.train(train_data, train_labels, 10, 5);
    nn.save_model("test_model_e2e.bin");
    
    // Create a new network with the same architecture
    NeuralNetwork nn2;
    nn2.add_layer(new DenseLayer(input_size, 32));
    nn2.add_layer(new DenseLayer(32, 16));
    nn2.add_layer(new DenseLayer(16, num_classes));
    
    // Load the saved model
    nn2.load_model("test_model_e2e.bin");
    
    // Get predictions from both networks
    Eigen::MatrixXd pred1 = nn.forward(test_data);
    Eigen::MatrixXd pred2 = nn2.forward(test_data);
    
    // Predictions should be identical
    for (int i = 0; i < pred1.rows(); i++) {
        for (int j = 0; j < pred1.cols(); j++) {
            EXPECT_NEAR(pred1(i, j), pred2(i, j), 1e-5);
        }
    }
    
    // Evaluate loaded model
    double accuracy = nn2.evaluate(test_data, test_labels);
    
    // Should get same accuracy as the original model
    double orig_accuracy = nn.evaluate(test_data, test_labels);
    EXPECT_NEAR(accuracy, orig_accuracy, 1e-5);
}

TEST_F(EndToEndTest, FullPipeline) {
    // This test covers the full pipeline of:
    // 1. Creating a model
    // 2. Training it
    // 3. Evaluating it
    // 4. Saving it
    // 5. Loading it in a new instance
    // 6. Making predictions
    
    // 1-2. Create and train model (already done in SetUp)
    nn.train(train_data, train_labels, 15, 5);
    
    // 3. Evaluate model
    double orig_accuracy = nn.evaluate(test_data, test_labels);
    
    // 4. Save model
    nn.save_model("test_model_e2e.bin");
    
    // 5. Create new model and load weights
    NeuralNetwork new_nn;
    new_nn.add_layer(new DenseLayer(input_size, 32));
    new_nn.add_layer(new DenseLayer(32, 16));
    new_nn.add_layer(new DenseLayer(16, num_classes));
    new_nn.compile(new Adam(0.01), new CrossEntropyLoss());
    
    new_nn.load_model("test_model_e2e.bin");
    
    // 6. Make predictions with new model
    double new_accuracy = new_nn.evaluate(test_data, test_labels);
    
    // Accuracy should be the same for both models
    EXPECT_NEAR(orig_accuracy, new_accuracy, 1e-5);
    
    // The model should perform reasonably well on this synthetic data
    // Since we're using the same data for training and testing
    EXPECT_GT(new_accuracy, 35.0); // At least 35% accuracy
}