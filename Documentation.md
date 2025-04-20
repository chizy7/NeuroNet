# NeuroNet: C++ Neural Network Library

NeuroNet is a comprehensive neural network implementation in C++ leveraging the Eigen library for efficient matrix operations. This project provides a flexible framework for building, training, and evaluating neural networks with support for various optimization algorithms, loss functions, and distributed training capabilities.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Dependencies](#dependencies)
4. [Architecture](#architecture)
5. [Components](#components)
6. [Usage Examples](#usage-examples)
7. [Testing](#testing)
8. [Advanced Features](#advanced-features)
9. [Performance Monitoring](#performance-monitoring)
10. [Model Management](#model-management)
11. [Debugging](#debugging)
12. [Known Issues and Future Improvements](#known-issues-and-future-improvements)
13. [License](#license)

## Features

- **Modular Architecture**: Extensible design with clear separation of components
- **Multiple Layer Types**:
  - Dense (Fully Connected) Layers
  - LSTM Layers for sequence modeling
- **Optimization Algorithms**:
  - Stochastic Gradient Descent (SGD)
  - Adam optimizer with adaptive learning rates
  - Lookahead optimizer for improved convergence
- **Loss Functions**:
  - Mean Squared Error (MSE)
  - Cross-Entropy Loss for classification
- **Training Methods**:
  - Standard batch training
  - Distributed training using MPI
  - Federated learning capabilities
  - Asynchronous training with multi-threading
- **Testing Framework**:
  - Unit tests for individual components
  - Integration tests for component interactions
  - System tests for end-to-end validation
- **Model Management**:
  - Save and load model weights
  - Model versioning system
  - Cloud storage integration with AWS S3
- **Hyperparameter Optimization**:
  - Grid search for hyperparameter tuning
  - Neural Architecture Search (NAS)
- **Data Handling**:
  - CSV data loading with preprocessing
  - Multi-modal data support (images, text, audio)
- **Production Deployment**:
  - Model serving via REST API
- **Performance Monitoring**:
  - Training metrics tracking
  - Inference latency monitoring

## Installation

### Dependencies

NeuroNet relies on the following libraries:

- **Eigen**: C++ template library for linear algebra (version 3.4.0 or newer)
- **MPI**: For distributed training capabilities
- **cpprestsdk**: For model serving via HTTP
- **AWS SDK for C++**: For cloud storage integration
- **OpenCV**: For image processing in multi-modal data loading
- **Google Test**: For running the test suite

### Building from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/chizy7/NeuroNet.git
   cd NeuroNet
   ```

2. Ensure all dependencies are installed. On Ubuntu/Debian:
   ```bash
   sudo apt-get install libeigen3-dev libopenmpi-dev libcpprest-dev libaws-sdk-cpp-all-dev libopencv-dev libgtest-dev
   ```

3. Create build directory and compile:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

4. Set up environment variables (optional, for AWS integration):
   ```bash
   cp ../.env.template ../.env
   # Edit .env file with your AWS credentials
   source ../load_env.sh
   ```

5. Run the executable:
   ```bash
   ./NeuroNet
   ```

## Architecture

NeuroNet follows an object-oriented design with the following core components:

- **Layer**: Abstract base class for all neural network layers
- **NeuralNetwork**: Controls the overall network architecture and training process
- **Optimizer**: Implements weight update algorithms
- **Loss**: Calculates loss and gradients for backpropagation
- **DataLoader**: Handles data ingestion and preprocessing

The system is designed with polymorphism to allow easy extension of components.

## Components

### Layers

#### DenseLayer
A fully connected layer implementing forward and backward propagation with L2 regularization.

```cpp
DenseLayer layer(input_size, output_size, l2_lambda);
```

The DenseLayer performs the following operations:
- **Forward pass**: `output = input * weights.transpose() + bias`
- **Backward pass**: Computes gradients for weights and propagates gradients to previous layers
- **L2 Regularization**: Applies weight decay proportional to the L2 norm of weights

#### LSTM Layer
Long Short-Term Memory layer for sequence modeling.

```cpp
LSTM lstm(input_size, hidden_size, output_size);
```

The LSTM implementation provides:
- Gates: forget, input, cell, and output gates
- State management: hidden state and cell state
- Forward pass implementation
- Note: The current implementation is simplified for demonstration purposes

### Optimizers

#### SGD (Stochastic Gradient Descent)
Basic gradient descent with configurable learning rate.

```cpp
SGD sgd(learning_rate);
```

#### Adam
Advanced optimizer with adaptive learning rates and momentum.

```cpp
Adam adam(learning_rate, beta1, beta2, epsilon);
```

#### Lookahead
Meta-optimizer that can wrap any base optimizer for improved convergence.

```cpp
Lookahead lookahead(base_optimizer, alpha, k);
```

### Loss Functions

#### MSE (Mean Squared Error)
For regression problems.

```cpp
MSE mse;
```

#### CrossEntropyLoss
For classification problems.

```cpp
CrossEntropyLoss cross_entropy;
```

### Data Handling

#### DataLoader
Handles loading and preprocessing of CSV data, particularly for MNIST.

```cpp
DataLoader::load_mnist_csv(file_path, data, labels);
```

#### MultiModalDataLoader
Supports loading and processing different data types.

```cpp
MultiModalDataLoader::load_image_data(image_path);
MultiModalDataLoader::load_text_data(text_path);
MultiModalDataLoader::load_audio_data(audio_path);
```

## Usage Examples

### Basic Neural Network Configuration

```cpp
// Create a neural network
NeuralNetwork nn;

// Add layers
nn.add_layer(new DenseLayer(784, 128));   // Input layer to hidden layer
nn.add_layer(new DenseLayer(128, 64));    // Hidden layer to hidden layer
nn.add_layer(new DenseLayer(64, 10));     // Hidden layer to output layer

// Compile with optimizer and loss function
nn.compile(new Adam(0.001), new CrossEntropyLoss());

// Train the network
nn.train(train_data, train_labels, 10, 64);  // epochs = 10, batch_size = 64

// Evaluate on test data
double accuracy = nn.evaluate(test_data, test_labels);

// Save the trained model
nn.save_model("models/my_model.model");
```

### Loading a Pre-trained Model

```cpp
NeuralNetwork nn;
// Set up the same architecture as during training
nn.add_layer(new DenseLayer(784, 128));
nn.add_layer(new DenseLayer(128, 64));
nn.add_layer(new DenseLayer(64, 10));

// Load the weights
nn.load_model("models/my_model.model");

// Use for prediction
Eigen::MatrixXd output = nn.forward(input_data);
```

### Distributed Training with MPI

```cpp
// Initialize MPI
MPI_Init(&argc, &argv);

// Create neural network
NeuralNetwork nn;
// Add layers...
nn.compile(new Adam(0.001), new CrossEntropyLoss());

// Train in distributed mode
nn.train_distributed(X, Y, epochs, batch_size, MPI_COMM_WORLD);

// Finalize MPI
MPI_Finalize();
```

### Hyperparameter Tuning

```cpp
HyperparameterTuner tuner;

// Define parameter ranges
std::vector<double> learning_rates = {0.001, 0.01, 0.1};
std::vector<int> batch_sizes = {32, 64, 128};

// Perform grid search
tuner.grid_search(X, Y, learning_rates, batch_sizes, 5); // 5 epochs per configuration
```

### Neural Architecture Search

```cpp
NAS nas;

// Define layer options (nodes per layer)
std::vector<int> layer_options = {64, 128, 256};

// Perform architecture search
nas.search(X, Y, layer_options, 5, 64); // 5 epochs, batch size 64
```

## Testing

NeuroNet includes a comprehensive testing framework to ensure code quality and reliability. The testing suite is built using Google Test and is organized into three categories:

### Test Structure

1. **Unit Tests**: Test individual components in isolation
   - DenseLayer tests for forward and backward propagation
   - Optimizer tests for weight updates
   - Loss function tests for calculation and gradients

2. **Integration Tests**: Test interactions between components
   - NeuralNetwork tests for training and evaluation
   - Model saving and loading tests
   - Forward and backward pass through multiple layers

3. **System Tests**: Test end-to-end workflows
   - Full training and evaluation pipeline
   - Model persistence and restoration
   - Performance on synthetic datasets

### Running Tests

The tests can be run using the provided script:

```bash
./run_tests.sh
```

This script will:
1. Configure the environment for testing
2. Build the project with test targets
3. Run all test categories
4. Report test results

For macOS users, the script includes special handling for Apple Silicon (M1/M2/M3) and proper SDK path detection.

### Writing New Tests

To add a new test, follow these steps:

1. Create a new test file in the appropriate directory (unit, integration, or system)
2. Include the necessary headers and Google Test framework:
   ```cpp
   #include "gtest/gtest.h"
   #include "ComponentToTest.h"
   ```
3. Write test cases using the `TEST` or `TEST_F` macros:
   ```cpp
   TEST(ComponentNameTest, FunctionalityName) {
       // Test setup
       // Actions
       // Assertions using EXPECT_* or ASSERT_*
   }
   ```
4. Add the new test file to CMakeLists.txt in the appropriate executable target

Example of a simple unit test:

```cpp
#include "gtest/gtest.h"
#include "Layer.h"

TEST(DenseLayerTest, ForwardPassDimensions) {
    DenseLayer layer(10, 5);  // 10 inputs, 5 outputs
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(2, 10);  // Batch of 2 samples
    
    Eigen::MatrixXd output = layer.forward(input);
    
    // Check output dimensions
    EXPECT_EQ(output.rows(), 2);  // Same number of samples
    EXPECT_EQ(output.cols(), 5);  // Correct output dimension
}
```

### Test Troubleshooting

If you encounter issues with tests:

1. Check for dependency conflicts (especially with Google Test)
2. Ensure proper include paths in your test files
3. Verify that the debug variable is properly defined
4. For macOS users, make sure the correct SDK path is being used

## Advanced Features

### Model Serving

NeuroNet includes a REST API server for deploying models in production:

```cpp
// Load a model
ModelServer server("models/my_model.model");

// Start the server
server.start_server(8080);
```

Clients can then send prediction requests via HTTP POST:

```bash
curl -X POST -H "Content-Type: application/json" -d '[[0.1, 0.2, ...]]' http://localhost:8080
```

### Cloud Storage Integration

Save and load models from AWS S3:

```cpp
// Upload a model to S3
CloudStorage::upload_to_s3("models/my_model.model", "models/my_model_v1.model");

// Download a model from S3
CloudStorage::download_from_s3("models/my_model_v1.model", "models/my_model_local.model");
```

### Model Versioning

Track and manage different model versions:

```cpp
ModelVersioning versioning;

// Save a model version
versioning.save_version("v1.0", "models/my_model_v1.model");

// Rollback to a previous version
versioning.rollback_version("v1.0", "models/current_model.model");

// List available versions
versioning.list_versions();
```

## Performance Monitoring

Track training metrics and inference performance:

```cpp
Monitoring monitor;

// Log training metrics
monitor.log_training_metrics(epoch, loss, accuracy, precision, recall);

// Log inference latency
monitor.log_inference_latency(latency);

// Export metrics to file
monitor.export_metrics_to_file("metrics.csv");
```

## Debugging

NeuroNet includes a debugging system to help with development and troubleshooting:

```cpp
// Enable debug mode
bool debug = true;

// Use logging
Logger::info("Starting training");
Logger::error("Failed to load file");
```

To enable debug mode when running the application:

```bash
./NeuroNet --debug
```

## Known Issues and Future Improvements

- **Convolutional Layers**: Currently not implemented, planned for future releases
- **Attention Mechanisms**: Support for transformer-based architectures
- **GPU Acceleration**: Integration with CUDA for faster training
- **Quantization**: Support for model quantization for deployment on edge devices
- **AutoML**: Extend NAS capabilities with more sophisticated search algorithms
- **Reinforcement Learning**: Add support for RL algorithms and environments
- **macOS Build Issues**: Some users may encounter SDK path and library linking issues on macOS

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 Chizaram Chibueze

## API Reference

### NeuralNetwork Class

#### Constructors

```cpp
NeuralNetwork();
~NeuralNetwork();
```

#### Methods

```cpp
void add_layer(Layer* layer);
Eigen::MatrixXd forward(const Eigen::MatrixXd& input);
void backward(const Eigen::MatrixXd& output, const Eigen::MatrixXd& labels);
void compile(Optimizer* optimizer, Loss* loss_fn);
void train(const Eigen::MatrixXd& X, const Eigen::VectorXi& Y, int epochs, int batch_size);
double evaluate(const Eigen::MatrixXd& X, const Eigen::VectorXi& Y);
void save_model(const std::string& file_path);
void load_model(const std::string& file_path);
void save_loss_history(const std::string& file_path);
void train_distributed(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs, int batch_size, MPI_Comm comm);
void train_federated(const Eigen::MatrixXd& X_local, const Eigen::MatrixXd& Y_local, int epochs, int batch_size, MPI_Comm comm);
void train_async(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, int epochs, int batch_size);
```

### Layer Class

#### Methods

```cpp
virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& input) = 0;
virtual Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) = 0;
virtual void update_weights(const Eigen::MatrixXd& grad_weights_update, const Eigen::VectorXd& grad_bias_update) = 0;
virtual Eigen::MatrixXd get_weights() const = 0;
virtual void set_weights(const Eigen::MatrixXd& new_weights) = 0;
virtual Eigen::MatrixXd get_grad_weights() const = 0;
virtual Eigen::VectorXd get_grad_bias() const = 0;
virtual void save(std::ofstream& out) const = 0;
virtual void load(std::ifstream& in) = 0;
```

### DenseLayer Class

#### Constructor

```cpp
DenseLayer(int input_size, int output_size, double l2_lambda = 0.01);
```

#### Methods

```cpp
Eigen::MatrixXd forward(const Eigen::MatrixXd& input) override;
Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output) override;
void update_weights(const Eigen::MatrixXd& grad_weights_update, const Eigen::VectorXd& grad_bias_update) override;
Eigen::MatrixXd get_weights() const override;
void set_weights(const Eigen::MatrixXd& new_weights) override;
void set_l2_lambda(double lambda);
double get_regularization_loss() const;
```

### Optimizer Classes

#### SGD

```cpp
SGD(double learning_rate);
void update(const std::vector<std::unique_ptr<Layer>>& layers) override;
```

#### Adam

```cpp
Adam(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
void update(const std::vector<std::unique_ptr<Layer>>& layers) override;
```

#### Lookahead

```cpp
Lookahead(Optimizer* base_optimizer, double alpha = 0.5, int k = 5);
void update(const std::vector<std::unique_ptr<Layer>>& layers) override;
```

### Loss Classes

#### MSE

```cpp
double calculate(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) override;
Eigen::MatrixXd gradient(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) override;
```

#### CrossEntropyLoss

```cpp
double calculate(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) override;
Eigen::MatrixXd gradient(const Eigen::MatrixXd& Y_pred, const Eigen::MatrixXd& Y_true) override;
```