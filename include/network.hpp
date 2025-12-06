#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include <memory>
#include <functional>
#include "activation.hpp"

namespace network {

class Layer {
public:
    Layer(int input_size, int output_size, 
          activation::ActivationType activation_type = activation::ActivationType::TANH);
    
    // Forward pass
    std::vector<double> forward(const std::vector<double>& input);
    
    // Forward pass with gradient computation
    std::vector<double> forward_with_grad(const std::vector<double>& input,
                                          std::vector<std::vector<double>>& grad_weights,
                                          std::vector<double>& grad_bias);
    
    // Get parameters
    const std::vector<std::vector<double>>& get_weights() const { return weights_; }
    const std::vector<double>& get_bias() const { return bias_; }
    
    // Set parameters
    void set_weights(const std::vector<std::vector<double>>& weights);
    void set_bias(const std::vector<double>& bias);
    
    // Get activation output (for gradient computation)
    const std::vector<double>& get_activation_input() const { return activation_input_; }
    const std::vector<double>& get_activation_output() const { return activation_output_; }
    
    int get_input_size() const { return input_size_; }
    int get_output_size() const { return output_size_; }

private:
    int input_size_;
    int output_size_;
    activation::Activation activation_;
    
    std::vector<std::vector<double>> weights_;  // [output_size][input_size]
    std::vector<double> bias_;                  // [output_size]
    
    // Store intermediate values for backprop
    std::vector<double> input_;
    std::vector<double> activation_input_;
    std::vector<double> activation_output_;
};

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layer_sizes,
                  activation::ActivationType activation_type = activation::ActivationType::TANH);
    
    // Forward pass
    std::vector<double> forward(const std::vector<double>& input);
    
    // Forward pass with all intermediate values stored
    std::vector<double> forward_with_intermediates(const std::vector<double>& input);
    
    // Compute gradient with respect to input (for automatic differentiation)
    std::vector<double> compute_input_gradient(const std::vector<double>& input,
                                               const std::vector<double>& output_gradient);
    
    // Compute second derivative with respect to input
    std::vector<double> compute_input_hessian(const std::vector<double>& input,
                                              int output_index);
    
    // Get all parameters as flat vector
    std::vector<double> get_parameters() const;
    
    // Set parameters from flat vector
    void set_parameters(const std::vector<double>& params);
    
    // Get number of parameters
    int get_num_parameters() const;
    
    // Get layers
    const std::vector<Layer>& get_layers() const { return layers_; }
    std::vector<Layer>& get_layers() { return layers_; }
    
    // Get input and output sizes
    int get_input_size() const;
    int get_output_size() const;

private:
    std::vector<Layer> layers_;
    std::vector<std::vector<double>> layer_outputs_;  // Store outputs of each layer
};

} // namespace network

#endif // NETWORK_HPP

