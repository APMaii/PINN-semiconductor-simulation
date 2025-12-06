#include "network.hpp"
#include "utils.hpp"
#include <cmath>
#include <cassert>

namespace network {

Layer::Layer(int input_size, int output_size, activation::ActivationType activation_type)
    : input_size_(input_size), output_size_(output_size), activation_(activation_type) {
    
    // Initialize weights and biases
    weights_.resize(output_size_);
    for (int i = 0; i < output_size_; ++i) {
        weights_[i].resize(input_size_);
        for (int j = 0; j < input_size_; ++j) {
            weights_[i][j] = utils::xavier_init(input_size_, output_size_);
        }
    }
    
    bias_.resize(output_size_, 0.0);
    
    activation_input_.resize(output_size_);
    activation_output_.resize(output_size_);
}

std::vector<double> Layer::forward(const std::vector<double>& input) {
    assert(input.size() == static_cast<size_t>(input_size_));
    
    input_ = input;
    activation_output_.resize(output_size_);
    
    // Compute weighted sum + bias
    for (int i = 0; i < output_size_; ++i) {
        activation_input_[i] = bias_[i];
        for (int j = 0; j < input_size_; ++j) {
            activation_input_[i] += weights_[i][j] * input[j];
        }
        activation_output_[i] = activation_(activation_input_[i]);
    }
    
    return activation_output_;
}

std::vector<double> Layer::forward_with_grad(const std::vector<double>& input,
                                             std::vector<std::vector<double>>& grad_weights,
                                             std::vector<double>& grad_bias) {
    auto output = forward(input);
    
    // Initialize gradients
    grad_weights.resize(output_size_);
    for (int i = 0; i < output_size_; ++i) {
        grad_weights[i].resize(input_size_, 0.0);
    }
    grad_bias.resize(output_size_, 0.0);
    
    return output;
}

void Layer::set_weights(const std::vector<std::vector<double>>& weights) {
    assert(weights.size() == static_cast<size_t>(output_size_));
    for (size_t i = 0; i < weights.size(); ++i) {
        assert(weights[i].size() == static_cast<size_t>(input_size_));
    }
    weights_ = weights;
}

void Layer::set_bias(const std::vector<double>& bias) {
    assert(bias.size() == static_cast<size_t>(output_size_));
    bias_ = bias;
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes,
                             activation::ActivationType activation_type) {
    assert(layer_sizes.size() >= 2);
    
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        layers_.emplace_back(layer_sizes[i], layer_sizes[i + 1], activation_type);
    }
    
    layer_outputs_.resize(layers_.size());
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    std::vector<double> current = input;
    
    for (size_t i = 0; i < layers_.size(); ++i) {
        current = layers_[i].forward(current);
        layer_outputs_[i] = current;
    }
    
    return current;
}

std::vector<double> NeuralNetwork::forward_with_intermediates(const std::vector<double>& input) {
    return forward(input);
}

std::vector<double> NeuralNetwork::compute_input_gradient(const std::vector<double>& input,
                                                          const std::vector<double>& output_gradient) {
    // Simplified gradient computation using finite differences for now
    // Full automatic differentiation would require storing more intermediate values
    double h = 1e-6;
    std::vector<double> grad(input.size());
    
    auto base_output = forward(input);
    
    for (size_t i = 0; i < input.size(); ++i) {
        std::vector<double> input_perturbed = input;
        input_perturbed[i] += h;
        auto perturbed_output = forward(input_perturbed);
        
        double dot_product = 0.0;
        for (size_t j = 0; j < output_gradient.size(); ++j) {
            dot_product += output_gradient[j] * (perturbed_output[j] - base_output[j]) / h;
        }
        grad[i] = dot_product;
    }
    
    return grad;
}

std::vector<double> NeuralNetwork::compute_input_hessian(const std::vector<double>& input,
                                                         int output_index) {
    // Simplified implementation - compute second derivatives
    double h = 1e-6;
    std::vector<double> hessian(input.size());
    
    std::vector<double> input_plus, input_minus;
    for (size_t i = 0; i < input.size(); ++i) {
        input_plus = input;
        input_minus = input;
        input_plus[i] += h;
        input_minus[i] -= h;
        
        auto output_plus = forward(input_plus);
        auto output_minus = forward(input_minus);
        
        hessian[i] = (output_plus[output_index] - 2.0 * forward(input)[output_index] + output_minus[output_index]) / (h * h);
    }
    
    return hessian;
}

std::vector<double> NeuralNetwork::get_parameters() const {
    std::vector<double> params;
    
    for (const auto& layer : layers_) {
        const auto& weights = layer.get_weights();
        const auto& bias = layer.get_bias();
        
        // Add weights
        for (const auto& w_row : weights) {
            params.insert(params.end(), w_row.begin(), w_row.end());
        }
        
        // Add bias
        params.insert(params.end(), bias.begin(), bias.end());
    }
    
    return params;
}

void NeuralNetwork::set_parameters(const std::vector<double>& params) {
    size_t idx = 0;
    
    for (auto& layer : layers_) {
        auto& weights = const_cast<std::vector<std::vector<double>>&>(layer.get_weights());
        auto& bias = const_cast<std::vector<double>&>(layer.get_bias());
        
        // Set weights
        for (auto& w_row : weights) {
            for (auto& w : w_row) {
                if (idx < params.size()) {
                    w = params[idx++];
                }
            }
        }
        
        // Set bias
        for (auto& b : bias) {
            if (idx < params.size()) {
                b = params[idx++];
            }
        }
    }
}

int NeuralNetwork::get_num_parameters() const {
    int count = 0;
    for (const auto& layer : layers_) {
        count += layer.get_input_size() * layer.get_output_size() + layer.get_output_size();
    }
    return count;
}

int NeuralNetwork::get_input_size() const {
    if (layers_.empty()) return 0;
    return layers_[0].get_input_size();
}

int NeuralNetwork::get_output_size() const {
    if (layers_.empty()) return 0;
    return layers_.back().get_output_size();
}

} // namespace network

