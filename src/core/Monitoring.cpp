#include "Monitoring.h"
#include <iostream>
#include <fstream>

Monitoring::Monitoring() {}

void Monitoring::log_training_metrics(int epoch, double loss, double accuracy, double precision, double recall) {
    training_metrics.push_back({epoch, loss, accuracy, precision, recall});
    std::cout << "Epoch " << epoch << " - Loss: " << loss << ", Accuracy: " << accuracy << std::endl;
}

void Monitoring::log_inference_latency(double latency) {
    inference_latency.push_back(latency);
    std::cout << "Inference Latency: " << latency << " ms" << std::endl;
}

void Monitoring::export_metrics_to_file(const std::string& file_path) {
    std::ofstream file(file_path);
    for (const auto& metric : training_metrics) {
        file << metric.epoch << "," << metric.loss << "," << metric.accuracy << "," << metric.precision << "," << metric.recall << std::endl;
    }
    for (const double& latency : inference_latency) {
        file << "Latency," << latency << std::endl;
    }
    file.close();
    std::cout << "Metrics exported to " << file_path << std::endl;
}