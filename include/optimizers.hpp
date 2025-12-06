#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include <vector>
#include <functional>
#include <memory>

namespace optimizers {

// Base optimizer class
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(const std::vector<double>& gradients, 
                     std::vector<double>& parameters) = 0;
    virtual void reset() {}
};

// Stochastic Gradient Descent
class SGD : public Optimizer {
public:
    SGD(double learning_rate = 0.01, double momentum = 0.0)
        : learning_rate_(learning_rate), momentum_(momentum) {}
    
    void step(const std::vector<double>& gradients, 
              std::vector<double>& parameters) override;
    
    void reset() override {
        velocity_.clear();
    }
    
    void set_learning_rate(double lr) { learning_rate_ = lr; }
    double get_learning_rate() const { return learning_rate_; }

private:
    double learning_rate_;
    double momentum_;
    std::vector<double> velocity_;
};

// Adam optimizer
class Adam : public Optimizer {
public:
    Adam(double learning_rate = 0.001, 
         double beta1 = 0.9, 
         double beta2 = 0.999,
         double epsilon = 1e-8)
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}
    
    void step(const std::vector<double>& gradients, 
              std::vector<double>& parameters) override;
    
    void reset() override {
        m_.clear();
        v_.clear();
        t_ = 0;
    }
    
    void set_learning_rate(double lr) { learning_rate_ = lr; }
    double get_learning_rate() const { return learning_rate_; }

private:
    double learning_rate_;
    double beta1_, beta2_, epsilon_;
    int t_;
    std::vector<double> m_;  // First moment estimate
    std::vector<double> v_;  // Second moment estimate
};

// Learning rate scheduler
class LearningRateScheduler {
public:
    virtual ~LearningRateScheduler() = default;
    virtual double get_learning_rate(int epoch) = 0;
};

// Exponential decay scheduler
class ExponentialDecayScheduler : public LearningRateScheduler {
public:
    ExponentialDecayScheduler(double initial_lr, double decay_rate, int decay_steps)
        : initial_lr_(initial_lr), decay_rate_(decay_rate), decay_steps_(decay_steps) {}
    
    double get_learning_rate(int epoch) override {
        return initial_lr_ * std::pow(decay_rate_, epoch / decay_steps_);
    }

private:
    double initial_lr_, decay_rate_;
    int decay_steps_;
};

// Step decay scheduler
class StepDecayScheduler : public LearningRateScheduler {
public:
    StepDecayScheduler(double initial_lr, double decay_factor, int step_size)
        : initial_lr_(initial_lr), decay_factor_(decay_factor), step_size_(step_size) {}
    
    double get_learning_rate(int epoch) override {
        return initial_lr_ * std::pow(decay_factor_, epoch / step_size_);
    }

private:
    double initial_lr_, decay_factor_;
    int step_size_;
};

} // namespace optimizers

#endif // OPTIMIZERS_HPP

