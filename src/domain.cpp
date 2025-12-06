#include "domain.hpp"
#include "utils.hpp"
#include <cmath>
#include <random>
#include <algorithm>

namespace domain {

Domain::Domain(double x_min, double x_max, double y_min, double y_max)
    : x_min_(x_min), x_max_(x_max), y_min_(y_min), y_max_(y_max) {
}

bool Domain::is_inside(const Point2D& p) const {
    return is_inside(p.x, p.y);
}

bool Domain::is_inside(double x, double y) const {
    return (x >= x_min_ && x <= x_max_ && y >= y_min_ && y <= y_max_);
}

bool Domain::is_on_boundary(const Point2D& p, double tolerance) const {
    return (std::abs(p.x - x_min_) < tolerance || std::abs(p.x - x_max_) < tolerance ||
            std::abs(p.y - y_min_) < tolerance || std::abs(p.y - y_max_) < tolerance);
}

std::string Domain::get_boundary_label(const Point2D& p, double tolerance) const {
    if (std::abs(p.x - x_min_) < tolerance) return "left";
    if (std::abs(p.x - x_max_) < tolerance) return "right";
    if (std::abs(p.y - y_min_) < tolerance) return "bottom";
    if (std::abs(p.y - y_max_) < tolerance) return "top";
    return "interior";
}

std::vector<Point2D> Domain::generate_interior_points(int n_points) const {
    utils::RandomGenerator rng;
    std::vector<Point2D> points;
    points.reserve(n_points);
    
    for (int i = 0; i < n_points; ++i) {
        double x = rng.uniform(x_min_, x_max_);
        double y = rng.uniform(y_min_, y_max_);
        points.emplace_back(x, y);
    }
    
    return points;
}

std::vector<Point2D> Domain::generate_boundary_points(int n_points_per_edge) const {
    std::vector<Point2D> points;
    points.reserve(4 * n_points_per_edge);
    
    double dx = (x_max_ - x_min_) / (n_points_per_edge + 1);
    double dy = (y_max_ - y_min_) / (n_points_per_edge + 1);
    
    // Bottom edge
    for (int i = 1; i <= n_points_per_edge; ++i) {
        points.emplace_back(x_min_ + i * dx, y_min_);
    }
    
    // Top edge
    for (int i = 1; i <= n_points_per_edge; ++i) {
        points.emplace_back(x_min_ + i * dx, y_max_);
    }
    
    // Left edge
    for (int i = 1; i <= n_points_per_edge; ++i) {
        points.emplace_back(x_min_, y_min_ + i * dy);
    }
    
    // Right edge
    for (int i = 1; i <= n_points_per_edge; ++i) {
        points.emplace_back(x_max_, y_min_ + i * dy);
    }
    
    // Corners
    points.emplace_back(x_min_, y_min_);
    points.emplace_back(x_max_, y_min_);
    points.emplace_back(x_min_, y_max_);
    points.emplace_back(x_max_, y_max_);
    
    return points;
}

std::vector<Point2D> Domain::generate_collocation_points(int n_interior, int n_boundary_per_edge) const {
    auto interior = generate_interior_points(n_interior);
    auto boundary = generate_boundary_points(n_boundary_per_edge);
    
    std::vector<Point2D> collocation;
    collocation.reserve(interior.size() + boundary.size());
    
    collocation.insert(collocation.end(), interior.begin(), interior.end());
    collocation.insert(collocation.end(), boundary.begin(), boundary.end());
    
    return collocation;
}

void Domain::add_boundary_condition(const std::string& label, const BoundaryCondition& bc) {
    // Remove existing condition with same label
    boundary_conditions_.erase(
        std::remove_if(boundary_conditions_.begin(), boundary_conditions_.end(),
                      [&label](const auto& p) { return p.first == label; }),
        boundary_conditions_.end()
    );
    
    boundary_conditions_.emplace_back(label, bc);
}

BoundaryCondition Domain::get_boundary_condition(const std::string& label) const {
    for (const auto& [lbl, bc] : boundary_conditions_) {
        if (lbl == label) {
            return bc;
        }
    }
    // Return default Dirichlet with zero value
    return {BoundaryType::DIRICHLET, 0.0, 0.0, label};
}

Point2D Domain::generate_random_point() const {
    utils::RandomGenerator rng;
    return Point2D(rng.uniform(x_min_, x_max_), rng.uniform(y_min_, y_max_));
}

} // namespace domain

