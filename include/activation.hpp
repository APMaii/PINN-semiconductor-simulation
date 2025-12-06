#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <cmath>
#include <functional>
#include <vector>

namespace activation {

// Sigmoid activation function
inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// Tanh activation function
inline double tanh_activation(double x) {
    return std::tanh(x);
}

inline double tanh_derivative(double x) {
    double t = std::tanh(x);
    return 1.0 - t * t;
}

// ReLU activation function
inline double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

inline double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// Swish activation function
inline double swish(double x) {
    return x * sigmoid(x);
}

inline double swish_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 + x * (1.0 - s));
}

// Activation function type
enum class ActivationType {
    SIGMOID,
    TANH,
    RELU,
    SWISH
};

// Activation function wrapper
class Activation {
public:
    explicit Activation(ActivationType type = ActivationType::TANH)
        : type_(type) {}

    double operator()(double x) const {
        switch (type_) {
            case ActivationType::SIGMOID:
                return sigmoid(x);
            case ActivationType::TANH:
                return tanh_activation(x);
            case ActivationType::RELU:
                return relu(x);
            case ActivationType::SWISH:
                return swish(x);
            default:
                return tanh_activation(x);
        }
    }

    double derivative(double x) const {
        switch (type_) {
            case ActivationType::SIGMOID:
                return sigmoid_derivative(x);
            case ActivationType::TANH:
                return tanh_derivative(x);
            case ActivationType::RELU:
                return relu_derivative(x);
            case ActivationType::SWISH:
                return swish_derivative(x);
            default:
                return tanh_derivative(x);
        }
    }

    void apply(std::vector<double>& vec) const {
        for (auto& v : vec) {
            v = (*this)(v);
        }
    }

private:
    ActivationType type_;
};

} // namespace activation

#endif // ACTIVATION_HPP

