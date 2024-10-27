#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

void DataLoader::load_mnist_csv(const std::string& file_path, Eigen::MatrixXd& data, Eigen::VectorXi& labels) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return;
    }

    std::string line;
    std::vector<std::vector<double>> data_rows;
    std::vector<int> label_values;

    // Read the first line to check if it's a header
    if (std::getline(file, line)) {
        // Check if the first line contains any non-numeric characters
        if (line.find_first_not_of("0123456789,") != std::string::npos) {
            std::cout << "Detected header row, skipping it." << std::endl;
            // Skip to the next line
            std::getline(file, line);
        }
    }

    // Process the rest of the lines
    do {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        // Read the label (first column)
        if (!std::getline(ss, value, ',')) {
            std::cerr << "Error: missing label in line: " << line << std::endl;
            continue;  // Skip to the next line
        }

        try {
            int label = std::stoi(value);
            label_values.push_back(label);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: invalid label '" << value << "' in line: " << line << std::endl;
            continue;  // Skip to the next line
        }

        // Read the pixel values
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value) / 255.0);  // Normalize to [0, 1] for improved model performance
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: invalid pixel value '" << value << "' in line: " << line << std::endl;
                row.clear();  // Clear the row to avoid incomplete entries
                break;
            }
        }

        // Only add rows that have the expected 784 pixel values
        if (row.size() == 784) {
            data_rows.push_back(row);
        } else {
            std::cerr << "Error: row does not have 784 pixel values in line: " << line << std::endl;
            label_values.pop_back();  // Remove the label if row is invalid
        }
    } while (std::getline(file, line));

    // Convert vectors to Eigen matrices
    int num_samples = data_rows.size();
    int num_features = data_rows[0].size();

    data.resize(num_samples, num_features);
    labels.resize(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        labels(i) = label_values[i];
        for (int j = 0; j < num_features; ++j) {
            data(i, j) = data_rows[i][j];
        }
    }

    std::cout << "Loaded " << num_samples << " samples from " << file_path << std::endl;
}