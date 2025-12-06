#include "optimizers.hpp"
#include <cmath>
#include <algorithm>

namespace optimizers {

void SGD::step(const std::vector<double>& gradients, 
               std::vector<double>& parameters) {
    if (gradients.size() != parameters.size()) {
        return;  // Error handling
    }
    
    // Initialize velocity if needed
    if (velocity_.size() != parameters.size()) {
        velocity_.resize(parameters.size(), 0.0);
    }
    
    // Update parameters
    for (size_t i = 0; i < parameters.size(); ++i) {
        if (momentum_ > 0.0) {
            velocity_[i] = momentum_ * velocity_[i] - learning_rate_ * gradients[i];
            parameters[i] += velocity_[i];
        } else {
            parameters[i] -= learning_rate_ * gradients[i];
        }
    }
}

void Adam::step(const std::vector<double>& gradients, 
                std::vector<double>& parameters) {
    if (gradients.size() != parameters.size()) {
        return;  // Error handling
    }
    
    // Initialize moment estimates if needed
    if (m_.size() != parameters.size()) {
        m_.resize(parameters.size(), 0.0);
        v_.resize(parameters.size(), 0.0);
    }
    
    t_++;
    
    // Update biased moment estimates
    for (size_t i = 0; i < parameters.size(); ++i) {
        m_[i] = beta1_ * m_[i] + (1.0 - beta1_) * gradients[i];
        v_[i] = beta2_ * v_[i] + (1.0 - beta2_) * gradients[i] * gradients[i];
        
        // Compute bias-corrected estimates
        double m_hat = m_[i] / (1.0 - std::pow(beta1_, t_));
        double v_hat = v_[i] / (1.0 - std::pow(beta2_, t_));
        
        // Update parameters
        parameters[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
    }
}

} // namespace optimizers

