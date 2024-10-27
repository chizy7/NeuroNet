# NeuroNet

NeuroNet is a neural network implementation in C++ leveraging the Eigen library for matrix operations. This project is structured to support custom neural network layers and optimized backpropagation, with an example classifier for the MNIST dataset.

## Features

- **Layered Architecture**: Utilizes a DenseLayer structure, supporting forward and backward propagation.
- **Optimization and Loss Functions**:
    - **Optimizer**: Stochastic Gradient Descent (SGD) with weight updates. 
    - **Loss Function**: Cross-Entropy loss, ideal for classification tasks.
- **Debug Logging**: Debug statements log matrix dimensions and provide insights into forward and backward passes for troubleshooting.
- **Eigen Matrix Broadcasting**: Customized solution for bias broacasting issues with Eigen, using manual replication for compatibility.

## Installation

### Prerequisites

- **Eigen Library**: Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms. Ensure Eigen is available in the third_party directory or download Eigen from the [official website](http://eigen.tuxfamily.org/index.php?title=Main_Page) or install via your package manager.
- **CMake and Make**: Required to build the project. 

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

## Dataset

To run the example MNIST classifier, download the MNIST dataset in CSV format from [this Kaggle dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_test.csv).

1. Download both mnist_train.csv and mnist_test.csv.
2. Place them in the data/ directory of the project.

## Usage

1. **Train the Model**: NeuroNet loads MNIST data, trains for specified epochs, and prints accuracy.
2. **Debugging Mode**: Enable debug in `logger.h` to get detailed layer-wise output of matrix sizes during the forward and backward passes.

## Code Structure

- **src/core/Layer.cpp**: Contains layer definitions and forward/backward functions.
- **src/core/NeuralNet.cpp**: Core neural network functionality, including training and evaluation.
- **include/:** Headers for neural network components and utilities. 
- **data/:** MNIST CSV files.

## Known Issues and Improvements

- **Additional Layers**: Currently, only DenseLayer is implemented. Other layer types (e.g., convolutional) could be added. 
- **Evaluation Metrics**: Extend beyond accuracy for multi-class tasks. 
- **Memory Optimization**: Explore more efficient memory handling for large datasets.

## License

This project is licenced under the MIT License. See the [LICENSE](LICENSE) file for details.