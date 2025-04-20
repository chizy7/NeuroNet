# NeuroNet

NeuroNet is a neural network implementation in C++ leveraging the Eigen library for matrix operations. This project is structured to support custom neural network layers and optimized backpropagation, with an example classifier for the MNIST dataset.

## Features

- **Layered Architecture**: Utilizes a DenseLayer structure, supporting forward and backward propagation.
- **Optimization and Loss Functions**:
    - **Optimizer**: Stochastic Gradient Descent (SGD) with weight updates. 
    - **Adam Optimizer**: Advanced optimizer with adaptive learning rates and momentum.
    - **Loss Function**: Cross-Entropy loss, ideal for classification tasks.
    - **MSE Loss**: Mean Squared Error loss for regression problems.
- **Debug Logging**: Debug statements log matrix dimensions and provide insights into forward and backward passes for troubleshooting.
- **Eigen Matrix Broadcasting**: Customized solution for bias broadcasting issues with Eigen, using manual replication for compatibility.
- **Comprehensive Test Suite**: Unit, integration, and system tests to ensure code correctness.

## Installation

### Prerequisites

- **Eigen Library**: Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. Ensure Eigen is available in the third_party directory or download Eigen from the [official website](http://eigen.tuxfamily.org/index.php?title=Main_Page) or install via your package manager.
- **CMake and Make**: Required to build the project.
- **Google Test**: For running the test suite. It will be automatically downloaded if not found on your system.
- **MPI**: For distributed training capabilities.
- **cpprestsdk**: For model serving capabilities.
- **AWS SDK for C++**: For cloud storage integration.
- **OpenCV**: For image processing capabilities.

### Building NeuroNet

1. Clone the repository:

```bash
git clone https://github.com/chizy7/NeuroNet.git
cd NeuroNet
```

2. Build with CMake:

```bash
mkdir build && cd build
cmake ..
make
```

3. Run the executable:

```bash
./NeuroNet
```

## Testing

NeuroNet includes a comprehensive test suite to validate functionality and prevent regressions. The tests are organized into three categories:

1. **Unit Tests**: Test individual components (Layer, Optimizer, Loss) in isolation.
2. **Integration Tests**: Test interactions between components (NeuralNetwork).
3. **System Tests**: Test end-to-end workflows.

To run the tests:

```bash
./run_tests.sh
```

This script will configure the environment, build the project, and run all test categories.

## Dataset

To run the example MNIST classifier, download the MNIST dataset in CSV format from [this Kaggle dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_test.csv).

1. Download both mnist_train.csv and mnist_test.csv.
2. Place them in the data/ directory of the project.

## Usage

1. **Train the Model**: NeuroNet loads MNIST data, trains for specified epochs, and prints accuracy.
2. **Debugging Mode**: Enable debug by passing the `--debug` flag when running the executable to get detailed layer-wise output of matrix sizes during the forward and backward passes.
3. **Save and Load Models**: Models can be saved after training and loaded for later use.

## Code Structure

- **src/core/**: Core neural network components
  - **Layer.cpp**: Contains layer definitions and forward/backward functions.
  - **NeuralNetwork.cpp**: Core neural network functionality, including training and evaluation.
  - **Optimizer.cpp**: Implementation of optimization algorithms.
  - **Loss.cpp**: Implementation of loss functions.
- **include/**: Headers for neural network components and utilities.
- **data/**: MNIST CSV files.
- **tests/**: Test suite
  - **unit/**: Unit tests for individual components.
  - **integration/**: Integration tests for component interactions.
  - **system/**: End-to-end system tests.

## Known Issues and Improvements

- **Additional Layers**: Currently, only DenseLayer is implemented. Other layer types (e.g., convolutional) could be added. 
- **Evaluation Metrics**: Extend beyond accuracy for multi-class tasks. 
- **Memory Optimization**: Explore more efficient memory handling for large datasets.
- **GPU Acceleration**: Add support for GPU acceleration using CUDA or similar technologies.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.