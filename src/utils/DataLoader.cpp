#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <vector>

Eigen::MatrixXd DataLoader::load_csv(const std::string& file_path) {
    std::ifstream file(file_path);
    std::string line;
    std::vector<std::vector<double>> data;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }
        data.push_back(row);
    }

    Eigen::MatrixXd matrix(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}