#ifndef FD_SOLVER_HPP
#define FD_SOLVER_HPP

#include <vector>
#include <functional>
#include <string>

namespace fd_solver {

// Finite difference solver for drift-diffusion equations
class FDSolver {
public:
    FDSolver(int nx, int ny, 
             double x_min, double x_max,
             double y_min, double y_max);
    
    // Set doping profile
    void set_doping_profile(const std::function<double(double, double)>& doping);
    
    // Set boundary conditions
    void set_potential_boundary(const std::string& side, double value);
    void set_carrier_boundary(const std::string& side, double n_value, double p_value);
    
    // Solve drift-diffusion equations
    bool solve(int max_iterations = 1000, double tolerance = 1e-6);
    
    // Get solution at point
    void get_solution(double x, double y, double& psi, double& n, double& p) const;
    
    // Get full solution grids
    const std::vector<std::vector<double>>& get_psi_grid() const { return psi_grid_; }
    const std::vector<std::vector<double>>& get_n_grid() const { return n_grid_; }
    const std::vector<std::vector<double>>& get_p_grid() const { return p_grid_; }
    
    // Save solution to file
    bool save_solution(const std::string& filename) const;

private:
    int nx_, ny_;
    double x_min_, x_max_, y_min_, y_max_;
    double dx_, dy_;
    
    std::function<double(double, double)> doping_profile_;
    
    std::vector<std::vector<double>> psi_grid_;  // Electrostatic potential
    std::vector<std::vector<double>> n_grid_;    // Electron density
    std::vector<std::vector<double>> p_grid_;    // Hole density
    
    // Boundary conditions
    std::vector<double> psi_left_, psi_right_, psi_bottom_, psi_top_;
    std::vector<double> n_left_, n_right_, n_bottom_, n_top_;
    std::vector<double> p_left_, p_right_, p_bottom_, p_top_;
    
    // Helper functions
    void initialize_guess();
    double compute_poisson_residual(int i, int j) const;
    double compute_electron_residual(int i, int j) const;
    double compute_hole_residual(int i, int j) const;
    void apply_boundary_conditions();
    double get_doping(int i, int j) const;
    
    // Grid indexing
    double get_x(int i) const { return x_min_ + i * dx_; }
    double get_y(int j) const { return y_min_ + j * dy_; }
};

} // namespace fd_solver

#endif // FD_SOLVER_HPP

