#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <fstream>
#include <iomanip>
#include "pinn.hpp"
#include "fd_solver.hpp"
#include "domain.hpp"
#include "optimizers.hpp"
#include "semiconductor_params.hpp"

// Example doping profile: step junction
double step_doping_profile(double x, double y) {
    double device_length = 1.0e-6;  // 1 micron
    double mid_point = device_length / 2.0;
    
    if (x < mid_point) {
        return 1.0e18;  // N-type (donors) in left half
    } else {
        return -1.0e18;  // P-type (acceptors) in right half
    }
}

// Uniform doping profile
double uniform_doping_profile(double x, double y) {
    return 1.0e16;  // Lightly doped N-type
}

int main(int argc, char* argv[]) {
    std::cout << "PINN Semiconductor Device Simulation\n";
    std::cout << "====================================\n\n";
    
    // Define domain
    double x_min = 0.0;
    double x_max = 1.0e-6;  // 1 micron
    double y_min = 0.0;
    double y_max = 1.0e-6;  // 1 micron
    
    domain::Domain domain(x_min, x_max, y_min, y_max);
    
    // Set boundary conditions
    domain::BoundaryCondition bc_left = {domain::BoundaryType::DIRICHLET, 0.0, 0.0, "left"};
    domain::BoundaryCondition bc_right = {domain::BoundaryType::DIRICHLET, 1.0, 0.0, "right"};  // 1V applied
    domain::BoundaryCondition bc_bottom = {domain::BoundaryType::NEUMANN, 0.0, 0.0, "bottom"};
    domain::BoundaryCondition bc_top = {domain::BoundaryType::NEUMANN, 0.0, 0.0, "top"};
    
    domain.add_boundary_condition("left", bc_left);
    domain.add_boundary_condition("right", bc_right);
    domain.add_boundary_condition("bottom", bc_bottom);
    domain.add_boundary_condition("top", bc_top);
    
    // Create doping profile
    std::function<double(double, double)> doping = step_doping_profile;
    
    // Neural network architecture: [2 input (x,y), hidden layers, 3 output (psi, n, p)]
    std::vector<int> architecture = {2, 50, 50, 50, 3};
    
    // Create optimizer
    optimizers::Adam* optimizer = new optimizers::Adam(0.001);
    
    // Create PINN
    pinn::PINN pinn_model(architecture, domain, doping, optimizer);
    
    std::cout << "Network architecture: ";
    for (size_t i = 0; i < architecture.size(); ++i) {
        std::cout << architecture[i];
        if (i < architecture.size() - 1) std::cout << " -> ";
    }
    std::cout << "\n";
    std::cout << "Total parameters: " << pinn_model.get_network().get_num_parameters() << "\n\n";
    
    // Train the model
    std::cout << "Starting training...\n";
    pinn_model.train(5000, 2000, 50);  // 5000 epochs, 2000 interior points, 50 per edge
    
    std::cout << "\nTraining completed. Final loss: " << pinn_model.get_current_loss() << "\n\n";
    
    // Save model
    std::string model_file = "data/results/pinn_model.dat";
    if (pinn_model.save_model(model_file)) {
        std::cout << "Model saved to " << model_file << "\n";
    }
    
    // Evaluate on grid
    int nx = 50;
    int ny = 50;
    std::vector<std::vector<double>> psi_grid, n_grid, p_grid;
    pinn_model.evaluate_on_grid(nx, ny, psi_grid, n_grid, p_grid);
    
    // Save results to CSV
    std::ofstream output_file("data/results/pinn_solution.csv");
    if (output_file.is_open()) {
        output_file << "x,y,psi,n,p\n";
        
        double dx = (x_max - x_min) / (nx - 1);
        double dy = (y_max - y_min) / (ny - 1);
        
        for (int j = 0; j < ny; ++j) {
            double y = y_min + j * dy;
            for (int i = 0; i < nx; ++i) {
                double x = x_min + i * dx;
                output_file << std::scientific << std::setprecision(15)
                           << x << "," << y << ","
                           << psi_grid[j][i] << "," << n_grid[j][i] << "," << p_grid[j][i] << "\n";
            }
        }
        output_file.close();
        std::cout << "Results saved to data/results/pinn_solution.csv\n";
    }
    
    // Optional: Compare with finite difference solver
    std::cout << "\nRunning finite difference solver for comparison...\n";
    fd_solver::FDSolver fd_solver(nx, ny, x_min, x_max, y_min, y_max);
    fd_solver.set_doping_profile(doping);
    fd_solver.set_potential_boundary("left", 0.0);
    fd_solver.set_potential_boundary("right", 1.0);
    
    if (fd_solver.solve(1000, 1e-6)) {
        std::cout << "FD solver converged.\n";
        fd_solver.save_solution("data/results/fd_solution.csv");
        std::cout << "FD solution saved to data/results/fd_solution.csv\n";
    } else {
        std::cout << "FD solver did not converge.\n";
    }
    
    std::cout << "\nSimulation completed successfully!\n";
    
    return 0;
}

