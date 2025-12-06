#ifndef PINN_HPP
#define PINN_HPP

#include <vector>
#include <memory>
#include <functional>
#include <string>
#include "network.hpp"
#include "optimizers.hpp"
#include "domain.hpp"
#include "loss.hpp"
#include "utils.hpp"

namespace pinn {

// Main PINN class for semiconductor simulation
class PINN {
public:
    PINN(const std::vector<int>& network_architecture,
         const domain::Domain& domain,
         const std::function<double(double, double)>& doping_profile,
         optimizers::Optimizer* optimizer = nullptr);
    
    ~PINN();
    
    // Training
    void train(int epochs, int n_interior_points = 1000, int n_boundary_points_per_edge = 50);
    void train_step();
    
    // Evaluate solution
    std::vector<double> predict(double x, double y) const;
    
    // Evaluate on grid
    void evaluate_on_grid(int nx, int ny, 
                         std::vector<std::vector<double>>& psi_grid,
                         std::vector<std::vector<double>>& n_grid,
                         std::vector<std::vector<double>>& p_grid) const;
    
    // Save/load model
    bool save_model(const std::string& filename) const;
    bool load_model(const std::string& filename);
    
    // Get current loss
    double get_current_loss() const { return current_loss_; }
    
    // Set optimizer
    void set_optimizer(optimizers::Optimizer* optimizer);
    
    // Get network
    const network::NeuralNetwork& get_network() const { return network_; }
    network::NeuralNetwork& get_network() { return network_; }

private:
    network::NeuralNetwork network_;
    domain::Domain domain_;
    std::function<double(double, double)> doping_profile_;
    std::unique_ptr<optimizers::Optimizer> optimizer_;
    
    loss::PhysicsInformedLoss loss_function_;
    
    std::vector<domain::Point2D> collocation_points_;
    
    double current_loss_;
    
    // Regenerate collocation points
    void regenerate_collocation_points(int n_interior, int n_boundary_per_edge);
};

} // namespace pinn

#endif // PINN_HPP

