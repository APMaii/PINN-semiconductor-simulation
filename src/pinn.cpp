#include "pinn.hpp"
#include "optimizers.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

namespace pinn {

PINN::PINN(const std::vector<int>& network_architecture,
           const domain::Domain& domain,
           const std::function<double(double, double)>& doping_profile,
           optimizers::Optimizer* optimizer)
    : network_(network_architecture), domain_(domain), 
      doping_profile_(doping_profile),
      loss_function_(domain, doping_profile),
      current_loss_(0.0) {
    
    if (optimizer) {
        optimizer_ = std::unique_ptr<optimizers::Optimizer>(optimizer);
    } else {
        // Default optimizer: Adam
        optimizer_ = std::make_unique<optimizers::Adam>(0.001);
    }
}

PINN::~PINN() = default;

void PINN::set_optimizer(optimizers::Optimizer* optimizer) {
    optimizer_ = std::unique_ptr<optimizers::Optimizer>(optimizer);
}

void PINN::regenerate_collocation_points(int n_interior, int n_boundary_per_edge) {
    collocation_points_ = domain_.generate_collocation_points(n_interior, n_boundary_per_edge);
}

void PINN::train(int epochs, int n_interior_points, int n_boundary_points_per_edge) {
    regenerate_collocation_points(n_interior_points, n_boundary_points_per_edge);
    
    std::cout << "Training PINN for " << epochs << " epochs...\n";
    std::cout << "Collocation points: " << collocation_points_.size() << "\n";
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        train_step();
        
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                     << " - Loss: " << current_loss_ << "\n";
        }
        
        // Adaptive collocation point regeneration
        if ((epoch + 1) % 1000 == 0) {
            regenerate_collocation_points(n_interior_points, n_boundary_points_per_edge);
        }
    }
}

void PINN::train_step() {
    // Compute loss
    current_loss_ = loss_function_.compute_loss(network_, collocation_points_);
    
    // Compute gradients
    std::vector<double> gradients;
    loss_function_.compute_gradients(network_, collocation_points_, gradients);
    
    // Update parameters
    auto params = network_.get_parameters();
    optimizer_->step(gradients, params);
    network_.set_parameters(params);
}

std::vector<double> PINN::predict(double x, double y) const {
    std::vector<double> input = {x, y};
    return network_.forward(input);
}

void PINN::evaluate_on_grid(int nx, int ny,
                            std::vector<std::vector<double>>& psi_grid,
                            std::vector<std::vector<double>>& n_grid,
                            std::vector<std::vector<double>>& p_grid) const {
    psi_grid.resize(ny);
    n_grid.resize(ny);
    p_grid.resize(ny);
    
    double x_min = domain_.get_x_min();
    double x_max = domain_.get_x_max();
    double y_min = domain_.get_y_min();
    double y_max = domain_.get_y_max();
    
    double dx = (x_max - x_min) / (nx - 1);
    double dy = (y_max - y_min) / (ny - 1);
    
    for (int j = 0; j < ny; ++j) {
        psi_grid[j].resize(nx);
        n_grid[j].resize(nx);
        p_grid[j].resize(nx);
        
        double y = y_min + j * dy;
        
        for (int i = 0; i < nx; ++i) {
            double x = x_min + i * dx;
            auto output = predict(x, y);
            
            if (output.size() >= 3) {
                psi_grid[j][i] = output[0];
                n_grid[j][i] = output[1];
                p_grid[j][i] = output[2];
            } else {
                psi_grid[j][i] = 0.0;
                n_grid[j][i] = 0.0;
                p_grid[j][i] = 0.0;
            }
        }
    }
}

bool PINN::save_model(const std::string& filename) const {
    auto params = network_.get_parameters();
    return utils::save_vector(params, filename);
}

bool PINN::load_model(const std::string& filename) {
    std::vector<double> params;
    if (!utils::load_vector(params, filename)) {
        return false;
    }
    
    if (static_cast<int>(params.size()) != network_.get_num_parameters()) {
        return false;
    }
    
    network_.set_parameters(params);
    return true;
}

} // namespace pinn

