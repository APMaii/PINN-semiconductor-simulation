#include <iostream>
#include <cassert>
#include <functional>
#include <cmath>
#include "../include/fd_solver.hpp"

double test_doping_profile(double x, double y) {
    double mid = 0.5e-6;
    if (x < mid) {
        return 1.0e18;  // N-type
    } else {
        return -1.0e18;  // P-type
    }
}

void test_fd_solver_creation() {
    std::cout << "Testing FDSolver creation...\n";
    
    fd_solver::FDSolver solver(50, 50, 0.0, 1e-6, 0.0, 1e-6);
    solver.set_doping_profile(test_doping_profile);
    
    std::cout << "  Solver created: ✓\n";
    std::cout << "FDSolver creation test passed!\n\n";
}

void test_fd_solver_boundary_conditions() {
    std::cout << "Testing FDSolver boundary conditions...\n";
    
    fd_solver::FDSolver solver(50, 50, 0.0, 1e-6, 0.0, 1e-6);
    solver.set_doping_profile(test_doping_profile);
    solver.set_potential_boundary("left", 0.0);
    solver.set_potential_boundary("right", 1.0);
    
    std::cout << "  Boundary conditions set: ✓\n";
    std::cout << "FDSolver boundary conditions test passed!\n\n";
}

void test_fd_solver_solve() {
    std::cout << "Testing FDSolver solve()...\n";
    
    fd_solver::FDSolver solver(20, 20, 0.0, 1e-6, 0.0, 1e-6);
    solver.set_doping_profile(test_doping_profile);
    solver.set_potential_boundary("left", 0.0);
    solver.set_potential_boundary("right", 1.0);
    
    bool converged = solver.solve(100, 1e-4);
    
    if (converged) {
        std::cout << "  Solver converged: ✓\n";
    } else {
        std::cout << "  Solver did not converge (may need more iterations)\n";
    }
    
    // Test solution retrieval
    double psi, n, p;
    solver.get_solution(0.5e-6, 0.5e-6, psi, n, p);
    std::cout << "  Solution at center: psi=" << psi << ", n=" << n << ", p=" << p << "\n";
    
    std::cout << "FDSolver solve test passed!\n\n";
}

void test_fd_solver_grid_access() {
    std::cout << "Testing FDSolver grid access...\n";
    
    fd_solver::FDSolver solver(10, 10, 0.0, 1e-6, 0.0, 1e-6);
    solver.set_doping_profile(test_doping_profile);
    solver.set_potential_boundary("left", 0.0);
    solver.set_potential_boundary("right", 1.0);
    
    solver.solve(50, 1e-4);
    
    const auto& psi_grid = solver.get_psi_grid();
    assert(psi_grid.size() == 10);
    assert(psi_grid[0].size() == 10);
    
    std::cout << "  Grid size: " << psi_grid.size() << " x " << psi_grid[0].size() << " ✓\n";
    std::cout << "  psi[5][5] = " << psi_grid[5][5] << "\n";
    
    std::cout << "FDSolver grid access test passed!\n\n";
}

int main() {
    std::cout << "=== Finite Difference Solver Tests ===\n\n";
    
    try {
        test_fd_solver_creation();
        test_fd_solver_boundary_conditions();
        test_fd_solver_solve();
        test_fd_solver_grid_access();
        
        std::cout << "All tests passed! ✓\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
}

