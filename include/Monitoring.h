#pragma once
#include <string>
#include <vector>

class Monitoring {
public:
    Monitoring();
    
    void log_training_metrics(int epoch, double loss, double accuracy, double precision, double recall);
    void log_inference_latency(double latency);
    void export_metrics_to_file(const std::string& file_path);

private:
    struct TrainingMetrics {
        int epoch;
        double loss;
        double accuracy;
        double precision;
        double recall;
    };

    std::vector<TrainingMetrics> training_metrics;
    std::vector<double> inference_latency;
};