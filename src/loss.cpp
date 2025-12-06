#include "loss.hpp"
#include <cmath>
#include <algorithm>

namespace loss {

PhysicsInformedLoss::PhysicsInformedLoss(const domain::Domain& domain,
                                         const std::function<double(double, double)>& doping_profile)
    : domain_(domain), doping_profile_(doping_profile), 
      pde_weight_(1.0), boundary_weight_(1.0) {
}

double PhysicsInformedLoss::compute_loss(network::NeuralNetwork& network,
                                        const std::vector<domain::Point2D>& collocation_points) {
    if (collocation_points.empty()) {
        return 0.0;
    }
    
    // Separate interior and boundary points
    std::vector<domain::Point2D> interior_points;
    std::vector<domain::Point2D> boundary_points;
    
    for (const auto& point : collocation_points) {
        if (domain_.is_on_boundary(point)) {
            boundary_points.push_back(point);
        } else {
            interior_points.push_back(point);
        }
    }
    
    double pde_loss = compute_pde_loss(network, interior_points);
    double bc_loss = compute_boundary_loss(network, boundary_points);
    
    return pde_weight_ * pde_loss + boundary_weight_ * bc_loss;
}

double PhysicsInformedLoss::compute_pde_loss(network::NeuralNetwork& network,
                                            const std::vector<domain::Point2D>& interior_points) {
    if (interior_points.empty()) {
        return 0.0;
    }
    
    double total_loss = 0.0;
    
    for (const auto& point : interior_points) {
        std::vector<double> input = {point.x, point.y};
        std::vector<double> output = network.forward(input);
        
        // Output should be [psi, n, p] (electrostatic potential, electron density, hole density)
        if (output.size() < 3) {
            continue;
        }
        
        double psi = output[0];
        double n_val = output[1];
        double p_val = output[2];
        
        // Compute Laplacian of psi
        double laplacian_psi = compute_laplacian_2d(network, point, 0);
        
        // Compute doping
        double N_net = doping_profile_(point.x, point.y);
        double N_D = (N_net > 0) ? N_net : 0.0;
        double N_A = (N_net < 0) ? -N_net : 0.0;
        
        // Poisson equation residual: -∇²ψ - (q/ε) * (p - n + N_D - N_A) = 0
        double poisson_rhs = (semiconductor::q / semiconductor::eps) * (p_val - n_val + N_D - N_A);
        double poisson_res = -laplacian_psi - poisson_rhs;
        
        // Simplified continuity equation residuals (full implementation would compute current divergence)
        double electron_res = 0.0;  // Placeholder
        double hole_res = 0.0;      // Placeholder
        
        total_loss += poisson_res * poisson_res + 
                     electron_res * electron_res + 
                     hole_res * hole_res;
    }
    
    return total_loss / interior_points.size();
}

double PhysicsInformedLoss::compute_boundary_loss(network::NeuralNetwork& network,
                                                  const std::vector<domain::Point2D>& boundary_points) {
    if (boundary_points.empty()) {
        return 0.0;
    }
    
    double total_loss = 0.0;
    
    for (const auto& point : boundary_points) {
        std::vector<double> input = {point.x, point.y};
        std::vector<double> output = network.forward(input);
        
        double error = compute_boundary_error(output, point);
        total_loss += error * error;
    }
    
    return total_loss / boundary_points.size();
}

double PhysicsInformedLoss::compute_poisson_residual(const std::vector<double>& psi,
                                                     const std::vector<double>& n,
                                                     const std::vector<double>& p,
                                                     double x, double y) {
    // Poisson equation: -∇²ψ = (q/ε) * (p - n + N_D - N_A)
    // Using simplified 2D version
    
    double N_D = 0.0;  // Donor concentration
    double N_A = 0.0;  // Acceptor concentration
    double N_net = doping_profile_(x, y);  // Net doping (N_D - N_A)
    
    if (N_net > 0) {
        N_D = N_net;
    } else {
        N_A = -N_net;
    }
    
    // Get psi value
    double psi_val = psi[0];
    double n_val = n[0];
    double p_val = p[0];
    
    // Simplified: assume we compute laplacian separately
    // For now, return a simplified residual
    double rhs = (semiconductor::q / semiconductor::eps) * (p_val - n_val + N_D - N_A);
    
    // This will be combined with laplacian computation
    return rhs;
}

double PhysicsInformedLoss::compute_electron_continuity_residual(const std::vector<double>& psi,
                                                                 const std::vector<double>& n,
                                                                 double x, double y) {
    // Electron continuity equation
    // Simplified version: div(J_n) = 0 for steady state
    // J_n = q * μ_n * n * E - q * D_n * ∇n
    // where E = -∇ψ
    
    double n_val = n[0];
    double psi_val = psi[0];
    
    // Simplified residual (full implementation would compute gradients)
    return 0.0;  // Placeholder - needs gradient computation
}

double PhysicsInformedLoss::compute_hole_continuity_residual(const std::vector<double>& psi,
                                                             const std::vector<double>& p,
                                                             double x, double y) {
    // Hole continuity equation
    // Similar to electron continuity but for holes
    
    double p_val = p[0];
    
    return 0.0;  // Placeholder - needs gradient computation
}

std::vector<double> PhysicsInformedLoss::compute_gradient_2d(network::NeuralNetwork& network,
                                                             const domain::Point2D& point,
                                                             int output_index) {
    double h = 1e-6;
    std::vector<double> grad(2);
    
    // Compute ∂u/∂x
    std::vector<double> input_x = {point.x + h, point.y};
    std::vector<double> output_x = network.forward(input_x);
    std::vector<double> input_x0 = {point.x - h, point.y};
    std::vector<double> output_x0 = network.forward(input_x0);
    grad[0] = (output_x[output_index] - output_x0[output_index]) / (2.0 * h);
    
    // Compute ∂u/∂y
    std::vector<double> input_y = {point.x, point.y + h};
    std::vector<double> output_y = network.forward(input_y);
    std::vector<double> input_y0 = {point.x, point.y - h};
    std::vector<double> output_y0 = network.forward(input_y0);
    grad[1] = (output_y[output_index] - output_y0[output_index]) / (2.0 * h);
    
    return grad;
}

double PhysicsInformedLoss::compute_laplacian_2d(network::NeuralNetwork& network,
                                                 const domain::Point2D& point,
                                                 int output_index) {
    double h = 1e-6;
    
    // Compute ∂²u/∂x²
    std::vector<double> input_xp = {point.x + h, point.y};
    std::vector<double> output_xp = network.forward(input_xp);
    std::vector<double> input_xm = {point.x - h, point.y};
    std::vector<double> output_xm = network.forward(input_xm);
    std::vector<double> input_x0 = {point.x, point.y};
    std::vector<double> output_x0 = network.forward(input_x0);
    double d2u_dx2 = (output_xp[output_index] - 2.0 * output_x0[output_index] + output_xm[output_index]) / (h * h);
    
    // Compute ∂²u/∂y²
    std::vector<double> input_yp = {point.x, point.y + h};
    std::vector<double> output_yp = network.forward(input_yp);
    std::vector<double> input_ym = {point.x, point.y - h};
    std::vector<double> output_ym = network.forward(input_ym);
    double d2u_dy2 = (output_yp[output_index] - 2.0 * output_x0[output_index] + output_ym[output_index]) / (h * h);
    
    return d2u_dx2 + d2u_dy2;
}

double PhysicsInformedLoss::compute_boundary_error(const std::vector<double>& output,
                                                   const domain::Point2D& point) {
    // Get boundary condition for this point
    std::string label = domain_.get_boundary_label(point);
    auto bc = domain_.get_boundary_condition(label);
    
    if (bc.type == domain::BoundaryType::DIRICHLET) {
        // For potential (psi), check boundary value
        if (output.size() > 0) {
            return output[0] - bc.value;
        }
    }
    
    return 0.0;
}

void PhysicsInformedLoss::compute_gradients(network::NeuralNetwork& network,
                                            const std::vector<domain::Point2D>& collocation_points,
                                            std::vector<double>& gradients) {
    // Simplified gradient computation using finite differences
    // Full implementation would use automatic differentiation
    
    auto params = network.get_parameters();
    gradients.resize(params.size(), 0.0);
    
    double h = 1e-6;
    double base_loss = compute_loss(network, collocation_points);
    
    for (size_t i = 0; i < params.size(); ++i) {
        auto params_perturbed = params;
        params_perturbed[i] += h;
        network.set_parameters(params_perturbed);
        
        double perturbed_loss = compute_loss(network, collocation_points);
        gradients[i] = (perturbed_loss - base_loss) / h;
        
        // Restore original parameters
        network.set_parameters(params);
    }
}

} // namespace loss

