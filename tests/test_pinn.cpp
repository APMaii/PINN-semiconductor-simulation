#include <iostream>
#include <cassert>
#include <functional>
#include "../include/pinn.hpp"
#include "../include/domain.hpp"
#include "../include/optimizers.hpp"

double test_doping_profile(double x, double y) {
    return 1.0e16;  // Uniform doping
}

void test_pinn_creation() {
    std::cout << "Testing PINN creation...\n";
    
    domain::Domain domain(0.0, 1e-6, 0.0, 1e-6);
    std::function<double(double, double)> doping = test_doping_profile;
    std::vector<int> architecture = {2, 10, 10, 3};
    
    optimizers::Adam* optimizer = new optimizers::Adam(0.001);
    pinn::PINN pinn_model(architecture, domain, doping, optimizer);
    
    std::cout << "  Network created: ✓\n";
    std::cout << "  Parameters: " << pinn_model.get_network().get_num_parameters() << "\n";
    
    std::cout << "PINN creation test passed!\n\n";
}

void test_pinn_predict() {
    std::cout << "Testing PINN prediction...\n";
    
    domain::Domain domain(0.0, 1e-6, 0.0, 1e-6);
    std::function<double(double, double)> doping = test_doping_profile;
    std::vector<int> architecture = {2, 10, 10, 3};
    
    optimizers::Adam* optimizer = new optimizers::Adam(0.001);
    pinn::PINN pinn_model(architecture, domain, doping, optimizer);
    
    std::vector<double> output = pinn_model.predict(0.5e-6, 0.5e-6);
    
    assert(output.size() == 3);
    std::cout << "  Prediction output size: " << output.size() << " ✓\n";
    std::cout << "  psi = " << output[0] << "\n";
    std::cout << "  n = " << output[1] << "\n";
    std::cout << "  p = " << output[2] << "\n";
    
    std::cout << "PINN prediction test passed!\n\n";
}

void test_pinn_evaluate_grid() {
    std::cout << "Testing PINN grid evaluation...\n";
    
    domain::Domain domain(0.0, 1e-6, 0.0, 1e-6);
    std::function<double(double, double)> doping = test_doping_profile;
    std::vector<int> architecture = {2, 10, 10, 3};
    
    optimizers::Adam* optimizer = new optimizers::Adam(0.001);
    pinn::PINN pinn_model(architecture, domain, doping, optimizer);
    
    std::vector<std::vector<double>> psi_grid, n_grid, p_grid;
    pinn_model.evaluate_on_grid(10, 10, psi_grid, n_grid, p_grid);
    
    assert(psi_grid.size() == 10);
    assert(psi_grid[0].size() == 10);
    std::cout << "  Grid size: " << psi_grid.size() << " x " << psi_grid[0].size() << " ✓\n";
    
    std::cout << "PINN grid evaluation test passed!\n\n";
}

int main() {
    std::cout << "=== PINN Tests ===\n\n";
    
    try {
        test_pinn_creation();
        test_pinn_predict();
        test_pinn_evaluate_grid();
        
        std::cout << "All tests passed! ✓\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
}

