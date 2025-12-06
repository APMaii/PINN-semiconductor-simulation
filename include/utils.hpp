#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace utils {

// Random number generator utility
class RandomGenerator {
public:
    explicit RandomGenerator(unsigned int seed = 12345)
        : generator_(seed), uniform_(-1.0, 1.0), normal_(0.0, 1.0) {}

    double uniform() { return uniform_(generator_); }
    double normal() { return normal_(generator_); }
    double uniform(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(generator_);
    }

private:
    std::mt19937 generator_;
    std::uniform_real_distribution<double> uniform_;
    std::normal_distribution<double> normal_;
};

// Initialize weights using Xavier initialization
inline double xavier_init(int fan_in, int fan_out) {
    static RandomGenerator rng;
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    return rng.uniform(-limit, limit);
}

// Initialize weights using He initialization
inline double he_init(int fan_in) {
    static RandomGenerator rng;
    double stddev = std::sqrt(2.0 / fan_in);
    return rng.normal() * stddev;
}

// Save vector to file
inline bool save_vector(const std::vector<double>& vec, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    for (const auto& v : vec) {
        file << std::scientific << std::setprecision(15) << v << "\n";
    }
    file.close();
    return true;
}

// Load vector from file
inline bool load_vector(std::vector<double>& vec, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    vec.clear();
    double value;
    while (file >> value) {
        vec.push_back(value);
    }
    file.close();
    return true;
}

// Save 2D data to CSV
inline bool save_csv(const std::vector<std::vector<double>>& data, 
                     const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << std::scientific << std::setprecision(15) << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    file.close();
    return true;
}

// Generate random points in domain
inline std::vector<std::pair<double, double>> 
generate_random_points(double x_min, double x_max, 
                       double y_min, double y_max, 
                       int n_points) {
    static RandomGenerator rng;
    std::vector<std::pair<double, double>> points;
    points.reserve(n_points);
    for (int i = 0; i < n_points; ++i) {
        double x = rng.uniform(x_min, x_max);
        double y = rng.uniform(y_min, y_max);
        points.emplace_back(x, y);
    }
    return points;
}

// Linspace: generate evenly spaced values
inline std::vector<double> linspace(double start, double end, int n) {
    std::vector<double> result;
    result.reserve(n);
    if (n == 1) {
        result.push_back(start);
        return result;
    }
    double step = (end - start) / (n - 1);
    for (int i = 0; i < n; ++i) {
        result.push_back(start + i * step);
    }
    return result;
}

// Meshgrid: create coordinate matrices
inline void meshgrid(const std::vector<double>& x, 
                     const std::vector<double>& y,
                     std::vector<std::vector<double>>& X,
                     std::vector<std::vector<double>>& Y) {
    size_t nx = x.size();
    size_t ny = y.size();
    
    X.resize(ny);
    Y.resize(ny);
    
    for (size_t j = 0; j < ny; ++j) {
        X[j].resize(nx);
        Y[j].resize(nx);
        for (size_t i = 0; i < nx; ++i) {
            X[j][i] = x[i];
            Y[j][i] = y[j];
        }
    }
}

} // namespace utils

#endif // UTILS_HPP

