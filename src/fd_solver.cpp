#include "fd_solver.hpp"
#include "semiconductor_params.hpp"
#include "utils.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>

namespace fd_solver {

FDSolver::FDSolver(int nx, int ny, 
                   double x_min, double x_max,
                   double y_min, double y_max)
    : nx_(nx), ny_(ny), x_min_(x_min), x_max_(x_max), 
      y_min_(y_min), y_max_(y_max) {
    
    dx_ = (x_max_ - x_min_) / (nx_ - 1);
    dy_ = (y_max_ - y_min_) / (ny_ - 1);
    
    // Initialize grids
    psi_grid_.resize(ny_, std::vector<double>(nx_, 0.0));
    n_grid_.resize(ny_, std::vector<double>(nx_, semiconductor::n_i));
    p_grid_.resize(ny_, std::vector<double>(nx_, semiconductor::n_i));
    
    // Initialize boundary conditions with default values
    psi_left_.resize(ny_, 0.0);
    psi_right_.resize(ny_, 0.0);
    psi_bottom_.resize(nx_, 0.0);
    psi_top_.resize(nx_, 0.0);
    
    n_left_.resize(ny_, semiconductor::n_i);
    n_right_.resize(ny_, semiconductor::n_i);
    n_bottom_.resize(nx_, semiconductor::n_i);
    n_top_.resize(nx_, semiconductor::n_i);
    
    p_left_.resize(ny_, semiconductor::n_i);
    p_right_.resize(ny_, semiconductor::n_i);
    p_bottom_.resize(nx_, semiconductor::n_i);
    p_top_.resize(nx_, semiconductor::n_i);
}

void FDSolver::set_doping_profile(const std::function<double(double, double)>& doping) {
    doping_profile_ = doping;
}

void FDSolver::set_potential_boundary(const std::string& side, double value) {
    if (side == "left") {
        std::fill(psi_left_.begin(), psi_left_.end(), value);
    } else if (side == "right") {
        std::fill(psi_right_.begin(), psi_right_.end(), value);
    } else if (side == "bottom") {
        std::fill(psi_bottom_.begin(), psi_bottom_.end(), value);
    } else if (side == "top") {
        std::fill(psi_top_.begin(), psi_top_.end(), value);
    }
}

void FDSolver::set_carrier_boundary(const std::string& side, double n_value, double p_value) {
    if (side == "left") {
        std::fill(n_left_.begin(), n_left_.end(), n_value);
        std::fill(p_left_.begin(), p_left_.end(), p_value);
    } else if (side == "right") {
        std::fill(n_right_.begin(), n_right_.end(), n_value);
        std::fill(p_right_.begin(), p_right_.end(), p_value);
    } else if (side == "bottom") {
        std::fill(n_bottom_.begin(), n_bottom_.end(), n_value);
        std::fill(p_bottom_.begin(), p_bottom_.end(), p_value);
    } else if (side == "top") {
        std::fill(n_top_.begin(), n_top_.end(), n_value);
        std::fill(p_top_.begin(), p_top_.end(), p_value);
    }
}

void FDSolver::initialize_guess() {
    // Initialize with equilibrium values
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            double N_net = get_doping(i, j);
            // Simple equilibrium approximation
            psi_grid_[j][i] = 0.0;
            n_grid_[j][i] = semiconductor::n_i;
            p_grid_[j][i] = semiconductor::n_i;
        }
    }
    apply_boundary_conditions();
}

double FDSolver::get_doping(int i, int j) const {
    if (doping_profile_) {
        return doping_profile_(get_x(i), get_y(j));
    }
    return 0.0;
}

void FDSolver::apply_boundary_conditions() {
    // Apply boundary conditions
    for (int j = 0; j < ny_; ++j) {
        psi_grid_[j][0] = psi_left_[j];
        psi_grid_[j][nx_ - 1] = psi_right_[j];
        n_grid_[j][0] = n_left_[j];
        n_grid_[j][nx_ - 1] = n_right_[j];
        p_grid_[j][0] = p_left_[j];
        p_grid_[j][nx_ - 1] = p_right_[j];
    }
    
    for (int i = 0; i < nx_; ++i) {
        psi_grid_[0][i] = psi_bottom_[i];
        psi_grid_[ny_ - 1][i] = psi_top_[i];
        n_grid_[0][i] = n_bottom_[i];
        n_grid_[ny_ - 1][i] = n_top_[i];
        p_grid_[0][i] = p_bottom_[i];
        p_grid_[ny_ - 1][i] = p_top_[i];
    }
}

double FDSolver::compute_poisson_residual(int i, int j) const {
    if (i == 0 || i == nx_ - 1 || j == 0 || j == ny_ - 1) {
        return 0.0;  // Boundary points
    }
    
    // Laplacian: (d²/dx² + d²/dy²)psi
    double d2psi_dx2 = (psi_grid_[j][i+1] - 2.0 * psi_grid_[j][i] + psi_grid_[j][i-1]) / (dx_ * dx_);
    double d2psi_dy2 = (psi_grid_[j+1][i] - 2.0 * psi_grid_[j][i] + psi_grid_[j-1][i]) / (dy_ * dy_);
    double laplacian = d2psi_dx2 + d2psi_dy2;
    
    // RHS: (q/ε) * (p - n + N_D - N_A)
    double N_net = get_doping(i, j);
    double N_D = (N_net > 0) ? N_net : 0.0;
    double N_A = (N_net < 0) ? -N_net : 0.0;
    double rhs = (semiconductor::q / semiconductor::eps) * 
                 (p_grid_[j][i] - n_grid_[j][i] + N_D - N_A);
    
    return -laplacian - rhs;
}

double FDSolver::compute_electron_residual(int i, int j) const {
    // Simplified - full implementation would include current continuity
    return 0.0;
}

double FDSolver::compute_hole_residual(int i, int j) const {
    // Simplified - full implementation would include current continuity
    return 0.0;
}

bool FDSolver::solve(int max_iterations, double tolerance) {
    initialize_guess();
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        double max_residual = 0.0;
        
        // Gauss-Seidel iteration for Poisson equation
        for (int j = 1; j < ny_ - 1; ++j) {
            for (int i = 1; i < nx_ - 1; ++i) {
                double N_net = get_doping(i, j);
                double N_D = (N_net > 0) ? N_net : 0.0;
                double N_A = (N_net < 0) ? -N_net : 0.0;
                double rhs = (semiconductor::q / semiconductor::eps) * 
                            (p_grid_[j][i] - n_grid_[j][i] + N_D - N_A);
                
                double coeff = 2.0 / (dx_ * dx_) + 2.0 / (dy_ * dy_);
                double update = (1.0 / coeff) * (
                    (psi_grid_[j][i+1] + psi_grid_[j][i-1]) / (dx_ * dx_) +
                    (psi_grid_[j+1][i] + psi_grid_[j-1][i]) / (dy_ * dy_) +
                    rhs
                );
                
                double residual = std::abs(update - psi_grid_[j][i]);
                max_residual = std::max(max_residual, residual);
                psi_grid_[j][i] = update;
            }
        }
        
        apply_boundary_conditions();
        
        if (max_residual < tolerance) {
            return true;
        }
    }
    
    return false;
}

void FDSolver::get_solution(double x, double y, double& psi, double& n, double& p) const {
    // Bilinear interpolation
    int i = static_cast<int>((x - x_min_) / dx_);
    int j = static_cast<int>((y - y_min_) / dy_);
    
    i = std::max(0, std::min(nx_ - 2, i));
    j = std::max(0, std::min(ny_ - 2, j));
    
    double x1 = get_x(i);
    double x2 = get_x(i + 1);
    double y1 = get_y(j);
    double y2 = get_y(j + 1);
    
    double wx = (x - x1) / (x2 - x1);
    double wy = (y - y1) / (y2 - y1);
    
    psi = (1 - wx) * (1 - wy) * psi_grid_[j][i] +
          wx * (1 - wy) * psi_grid_[j][i + 1] +
          (1 - wx) * wy * psi_grid_[j + 1][i] +
          wx * wy * psi_grid_[j + 1][i + 1];
    
    n = (1 - wx) * (1 - wy) * n_grid_[j][i] +
        wx * (1 - wy) * n_grid_[j][i + 1] +
        (1 - wx) * wy * n_grid_[j + 1][i] +
        wx * wy * n_grid_[j + 1][i + 1];
    
    p = (1 - wx) * (1 - wy) * p_grid_[j][i] +
        wx * (1 - wy) * p_grid_[j][i + 1] +
        (1 - wx) * wy * p_grid_[j + 1][i] +
        wx * wy * p_grid_[j + 1][i + 1];
}

bool FDSolver::save_solution(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << "x,y,psi,n,p\n";
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            file << std::scientific << std::setprecision(15)
                 << get_x(i) << "," << get_y(j) << ","
                 << psi_grid_[j][i] << "," << n_grid_[j][i] << "," << p_grid_[j][i] << "\n";
        }
    }
    
    file.close();
    return true;
}

} // namespace fd_solver

