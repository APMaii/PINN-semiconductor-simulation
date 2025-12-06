#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "../include/network.hpp"
#include "../include/activation.hpp"

void test_layer_forward() {
    std::cout << "Testing Layer::forward()...\n";
    
    network::Layer layer(2, 3, activation::ActivationType::TANH);
    
    std::vector<double> input = {1.0, 2.0};
    std::vector<double> output = layer.forward(input);
    
    assert(output.size() == 3);
    std::cout << "  Output size: " << output.size() << " ✓\n";
    
    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << "  Output[" << i << "] = " << output[i] << "\n";
    }
    
    std::cout << "Layer forward test passed!\n\n";
}

void test_network_forward() {
    std::cout << "Testing NeuralNetwork::forward()...\n";
    
    std::vector<int> architecture = {2, 10, 5, 3};
    network::NeuralNetwork network(architecture);
    
    std::vector<double> input = {0.5, 0.3};
    std::vector<double> output = network.forward(input);
    
    assert(output.size() == 3);
    std::cout << "  Input size: " << input.size() << "\n";
    std::cout << "  Output size: " << output.size() << " ✓\n";
    
    for (size_t i = 0; i < output.size(); ++i) {
        std::cout << "  Output[" << i << "] = " << output[i] << "\n";
    }
    
    std::cout << "Network forward test passed!\n\n";
}

void test_network_parameters() {
    std::cout << "Testing NeuralNetwork parameters...\n";
    
    std::vector<int> architecture = {2, 5, 3};
    network::NeuralNetwork network(architecture);
    
    int num_params = network.get_num_parameters();
    std::cout << "  Number of parameters: " << num_params << "\n";
    
    auto params = network.get_parameters();
    assert(params.size() == static_cast<size_t>(num_params));
    std::cout << "  Parameters vector size: " << params.size() << " ✓\n";
    
    // Modify parameters and set them back
    params[0] += 0.1;
    network.set_parameters(params);
    auto params2 = network.get_parameters();
    assert(std::abs(params2[0] - params[0]) < 1e-10);
    std::cout << "  Parameter set/get: ✓\n";
    
    std::cout << "Network parameters test passed!\n\n";
}

void test_network_gradient() {
    std::cout << "Testing NeuralNetwork gradient computation...\n";
    
    std::vector<int> architecture = {2, 5, 3};
    network::NeuralNetwork network(architecture);
    
    std::vector<double> input = {0.5, 0.3};
    std::vector<double> output_grad = {1.0, 0.5, 0.2};
    
    std::vector<double> input_grad = network.compute_input_gradient(input, output_grad);
    
    assert(input_grad.size() == input.size());
    std::cout << "  Input gradient size: " << input_grad.size() << " ✓\n";
    
    std::cout << "  Input gradient: [" << input_grad[0] << ", " << input_grad[1] << "]\n";
    
    std::cout << "Network gradient test passed!\n\n";
}

int main() {
    std::cout << "=== Neural Network Tests ===\n\n";
    
    try {
        test_layer_forward();
        test_network_forward();
        test_network_parameters();
        test_network_gradient();
        
        std::cout << "All tests passed! ✓\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
}

