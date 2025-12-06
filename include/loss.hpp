#ifndef LOSS_HPP
#define LOSS_HPP

#include <vector>
#include <functional>
#include "domain.hpp"
#include "network.hpp"
#include "semiconductor_params.hpp"

namespace loss {

// Loss computation for PINN
class PhysicsInformedLoss {
public:
    PhysicsInformedLoss(const domain::Domain& domain,
                       const std::function<double(double, double)>& doping_profile);
    
    // Compute total loss (PDE residuals + boundary conditions)
    double compute_loss(network::NeuralNetwork& network,
                       const std::vector<domain::Point2D>& collocation_points);
    
    // Compute PDE residual loss
    double compute_pde_loss(network::NeuralNetwork& network,
                           const std::vector<domain::Point2D>& interior_points);
    
    // Compute boundary condition loss
    double compute_boundary_loss(network::NeuralNetwork& network,
                                const std::vector<domain::Point2D>& boundary_points);
    
    // Compute gradients for optimization
    void compute_gradients(network::NeuralNetwork& network,
                          const std::vector<domain::Point2D>& collocation_points,
                          std::vector<double>& gradients);
    
    // Set loss weights
    void set_pde_weight(double weight) { pde_weight_ = weight; }
    void set_boundary_weight(double weight) { boundary_weight_ = weight; }
    
    double get_pde_weight() const { return pde_weight_; }
    double get_boundary_weight() const { return boundary_weight_; }

private:
    const domain::Domain& domain_;
    std::function<double(double, double)> doping_profile_;
    
    double pde_weight_;
    double boundary_weight_;
    
    // Helper functions for computing PDE residuals
    double compute_poisson_residual(const std::vector<double>& psi,
                                   const std::vector<double>& n,
                                   const std::vector<double>& p,
                                   double x, double y);
    
    double compute_electron_continuity_residual(const std::vector<double>& psi,
                                               const std::vector<double>& n,
                                               double x, double y);
    
    double compute_hole_continuity_residual(const std::vector<double>& psi,
                                           const std::vector<double>& p,
                                           double x, double y);
    
    // Compute derivatives using finite differences
    std::vector<double> compute_gradient_2d(network::NeuralNetwork& network,
                                            const domain::Point2D& point,
                                            int output_index);
    
    double compute_laplacian_2d(network::NeuralNetwork& network,
                                const domain::Point2D& point,
                                int output_index);
    
    // Compute boundary condition error
    double compute_boundary_error(const std::vector<double>& output,
                                 const domain::Point2D& point);
};

} // namespace loss

#endif // LOSS_HPP

