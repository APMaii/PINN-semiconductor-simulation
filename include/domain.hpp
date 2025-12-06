#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include <vector>
#include <utility>
#include <string>

namespace domain {

// Point in 2D domain
struct Point2D {
    double x;
    double y;
    
    Point2D(double x_ = 0.0, double y_ = 0.0) : x(x_), y(y_) {}
};

// Boundary condition types
enum class BoundaryType {
    DIRICHLET,      // Fixed value
    NEUMANN,        // Fixed derivative
    ROBIN           // Mixed
};

// Boundary condition specification
struct BoundaryCondition {
    BoundaryType type;
    double value;           // For Dirichlet or Robin
    double derivative;      // For Neumann
    std::string label;      // Boundary label (e.g., "left", "right")
};

// Domain definition
class Domain {
public:
    Domain(double x_min, double x_max, double y_min, double y_max);
    
    // Get domain bounds
    double get_x_min() const { return x_min_; }
    double get_x_max() const { return x_max_; }
    double get_y_min() const { return y_min_; }
    double get_y_max() const { return y_max_; }
    
    // Check if point is inside domain
    bool is_inside(const Point2D& p) const;
    bool is_inside(double x, double y) const;
    
    // Generate interior points
    std::vector<Point2D> generate_interior_points(int n_points) const;
    
    // Generate boundary points
    std::vector<Point2D> generate_boundary_points(int n_points_per_edge) const;
    
    // Generate collocation points (interior + boundary)
    std::vector<Point2D> generate_collocation_points(int n_interior, int n_boundary_per_edge) const;
    
    // Add boundary condition
    void add_boundary_condition(const std::string& label, const BoundaryCondition& bc);
    
    // Get boundary condition
    BoundaryCondition get_boundary_condition(const std::string& label) const;
    
    // Check if point is on boundary
    bool is_on_boundary(const Point2D& p, double tolerance = 1e-6) const;
    std::string get_boundary_label(const Point2D& p, double tolerance = 1e-6) const;

private:
    double x_min_, x_max_, y_min_, y_max_;
    std::vector<std::pair<std::string, BoundaryCondition>> boundary_conditions_;
    
    Point2D generate_random_point() const;
};

} // namespace domain

#endif // DOMAIN_HPP

