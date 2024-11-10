#include "ModelServer.h"
#include <iostream>
#include <cpprest/http_listener.h>
#include <cpprest/json.h>

// #include <thread>
// #include <chrono>

using namespace web;
using namespace http;
using namespace utility;
using namespace http::experimental::listener;

ModelServer::ModelServer(const std::string& model_path) {
    nn.load_model(model_path);  // Load the pre-trained model
}

void ModelServer::start_server(int port) {
    http_listener listener(U("http://localhost:" + std::to_string(port)));
    // http_listener listener(U("http://localhost:" + std::to_string(port) + "/predict"));
    // http_listener listener(U("http://0.0.0.0:" + std::to_string(port)));

    listener.support(methods::POST, [this](http_request request) {
        std::cout << "Received a request..." << std::endl;
        request.extract_json().then([this, &request](pplx::task<json::value> task) {
            try {
                auto input_json = task.get();
                Eigen::MatrixXd input = json_to_eigen(input_json);

                auto output = predict(input);

                // Convert output to JSON and send response
                json::value response = eigen_to_json(output);
                std::cout << "Sending response..." << std::endl;
                request.reply(status_codes::OK, response);
            } catch (const std::exception& e) {
                std::cerr << "Error processing request: " << e.what() << std::endl;
                request.reply(status_codes::BadRequest, "Invalid JSON input");
            }
        });
    });

    try {
        listener.open().wait();
        std::cout << "Model server running on port " << port << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to start the server: " << e.what() << std::endl;
    }
    // listener.open().wait();
    // std::cout << "Model server running on port " << port << std::endl;

    // std::cout << "Press Ctrl+C to exit..." << std::endl;

    // // Keep the server alive
    // while (true) {
    //     std::this_thread::sleep_for(std::chrono::hours(24));  // Keep the thread alive indefinitely
    // }
}

Eigen::MatrixXd ModelServer::predict(const Eigen::MatrixXd& input) {
    return nn.forward(input);
}

Eigen::MatrixXd json_to_eigen(const json::value& json_obj) {
    // Ensure the JSON object is an array
    if (!json_obj.is_array()) {
        throw std::invalid_argument("Provided JSON object is not an array");
    }

    // Get the array
    const auto& arr = json_obj.as_array();
    int rows = static_cast<int>(arr.size());
    if (rows == 0) {
        throw std::invalid_argument("Empty JSON array provided");
    }

    // Get the number of columns from the first row
    const auto& first_row = arr.at(0).as_array();
    int cols = static_cast<int>(first_row.size());

    // Initialize Eigen matrix
    Eigen::MatrixXd matrix(rows, cols);

    // Populate the Eigen matrix
    for (int i = 0; i < rows; ++i) {
        const auto& row = arr.at(i).as_array(); // Access row as JSON array
        if (static_cast<int>(row.size()) != cols) {
            throw std::invalid_argument("Inconsistent number of columns in JSON rows");
        }

        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = row.at(j).as_double(); // Access element and convert to double
        }
    }

    return matrix;
}

json::value eigen_to_json(const Eigen::MatrixXd& matrix) {
    json::value result = json::value::array();

    for (int i = 0; i < matrix.rows(); ++i) {
        json::value row = json::value::array();
        for (int j = 0; j < matrix.cols(); ++j) {
            row[j] = json::value::number(matrix(i, j));
        }
        result[i] = std::move(row);
    }

    return result;
}