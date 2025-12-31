# PINN Semiconductor Device Simulation

A Physics-Informed Neural Network implementation for solving semiconductor drift-diffusion equations. This project grew out of frustration with traditional TCAD solvers being slow and difficult to integrate into optimization workflows. If you've ever wanted to simulate a PN junction but didn't want to wait minutes for each run, this might be what you're looking for.

## What This Does
Traditional finite element and finite difference solvers for semiconductor devices work great, but they're computationally expensive. Physics-Informed Neural Networks (PINNs) offer a different approach: train a neural network that inherently satisfies the underlying physics equations. Once trained, you get near-instant predictions that respect the drift-diffusion equations.

This implementation solves the coupled system of:
- **Poisson's equation** for the electrostatic potential
- **Electron continuity equation** for electron transport
- **Hole continuity equation** for hole transport

All written from scratch in C++17—no TensorFlow or PyTorch dependencies. Just pure C++ and some Python scripts for visualization.

## Features

- **Self-contained neural network**: Built from the ground up with backpropagation, automatic differentiation, and multiple activation functions
- **Physics-informed training**: Loss function includes both PDE residuals and boundary conditions
- **Reference FD solver**: Included finite difference solver for validation and comparison
- **Flexible architecture**: Easy to modify network structure, activation functions, and optimizers
- **Visualization tools**: Python scripts to plot results, compare with FD solutions, and generate doping profiles

## Building the Project

### Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, or MSVC 2017+)
- CMake 3.15 or higher
- Python 3.6+ with NumPy and Matplotlib (for visualization scripts)

### Compilation

```bash
mkdir build
cd build
cmake ..
make
```

This will create the main executable `pinn_semiconductor` and three test executables.

### Running Tests

```bash
cd build
./test_network
./test_pinn
./test_fd_solver
```


Or run all tests through CMake:

```bash
cd build
ctest
```

## Quick Start

The main executable trains a PINN on a simple step junction device:

```bash
./pinn_semiconductor
```

This will:
1. Train a neural network to solve the drift-diffusion equations
2. Save the model to `data/results/pinn_model.dat`
3. Output solutions to `data/results/pinn_solution.csv`
4. Run the finite difference solver for comparison

By default, it simulates a 1μm × 1μm device with a step junction (N-type on the left, P-type on the right) with 1V applied bias.

### Visualizing Results

After running the simulation, plot the results:

```bash
cd scripts
python plot_results.py ../data/results/pinn_solution.csv ../data/results/fd_solution.csv
```

This creates contour plots of the electrostatic potential, carrier densities, and compares PINN vs FD solutions.

## Project Structure

```
pinn-semiconductor-simulation/
├── CMakeLists.txt          # Build configuration
├── README.md
├── LICENSE
│
├── data/                   # Output directory
│   ├── doping_profiles/   # Generated doping profiles
│   ├── results/           # Simulation outputs
│   └── plots/             # Generated plots
│
├── docs/                   # Documentation (add your PDFs/images here)
│
├── include/                # Header files
│   ├── activation.hpp     # Activation functions (tanh, sigmoid, ReLU, etc.)
│   ├── domain.hpp         # Domain definition and boundary conditions
│   ├── fd_solver.hpp      # Finite difference reference solver
│   ├── loss.hpp           # Physics-informed loss functions
│   ├── network.hpp        # Neural network implementation
│   ├── optimizers.hpp     # SGD and Adam optimizers
│   ├── pinn.hpp           # Main PINN class
│   ├── semiconductor_params.hpp  # Physical constants (q, ε, μ, etc.)
│   └── utils.hpp          # Utility functions
│
├── src/                    # Implementation files
│   ├── activation.cpp
│   ├── domain.cpp
│   ├── fd_solver.cpp
│   ├── loss.cpp
│   ├── main.cpp           # Example usage
│   ├── network.cpp
│   ├── optimizers.cpp
│   ├── pinn.cpp
│   └── utils.cpp
│
├── scripts/                # Python utilities
│   ├── plot_results.py    # Visualize simulation results
│   ├── convert_results_to_csv.py
│   └── generate_doping_profile.py  # Create doping profiles
│
└── tests/                  # Unit tests
    ├── test_network.cpp
    ├── test_pinn.cpp
    └── test_fd_solver.cpp
```

## Customizing the Simulation

### Changing the Device Geometry

Edit `main.cpp` to modify the domain:


```cpp
double x_min = 0.0;
double x_max = 2.0e-6;  // Change device length
double y_min = 0.0;
double y_max = 1.0e-6;
domain::Domain domain(x_min, x_max, y_min, y_max);
```

### Modifying the Doping Profile

The doping profile is defined as a function in `main.cpp`. For example, a Gaussian junction:

```cpp
double gaussian_doping(double x, double y) {
    double center = 0.5e-6;
    double width = 0.2e-6;
    return 1e18 * exp(-pow((x - center) / width, 2));
}
```

Or use the provided script:

```bash
cd scripts
python generate_doping_profile.py
```

### Adjusting Network Architecture

Change the network structure in `main.cpp`:

```cpp
std::vector<int> architecture = {2, 64, 64, 64, 3};  // Deeper/wider network
```

The first number (2) is the input dimension (x, y), the last (3) is the output (ψ, n, p), and the middle numbers are hidden layer sizes.

### Training Parameters

Adjust training in `main.cpp`:

```cpp
pinn_model.train(10000, 3000, 100);  // epochs, interior_points, boundary_points_per_edge
```

More interior points and longer training generally improve accuracy at the cost of computation time.

## Understanding the Physics

The drift-diffusion model consists of three coupled equations:


1. **Poisson's equation**: Relates the electric field to charge density
   ```
   -∇²ψ = (q/ε) × (p - n + N_D - N_A)
   ```

2. **Electron continuity**: Conservation of electrons
   ```
   ∇·J_n = q × R
   ```
   where J_n = qμ_n n E - qD_n ∇n

3. **Hole continuity**: Conservation of holes
   ```
   ∇·J_p = -q × R
   ```

The PINN learns to satisfy these equations everywhere in the domain while respecting boundary conditions.

## Performance Notes

Training time depends heavily on:
- Number of collocation points
- Network size
- Number of epochs
- Your hardware

On a modern CPU, expect 5-10 minutes for the default configuration (5000 epochs, ~4000 collocation points). The network size is intentionally kept modest for faster iteration—feel free to experiment with larger networks if you need higher accuracy.

Once trained, prediction is nearly instant (microseconds per point), which makes this approach attractive for parameter sweeps and optimization.

## Limitations & Future Work

This is a working implementation, but there's always room for improvement:

- The continuity equations are simplified—full current continuity would require more careful handling of carrier transport
- No recombination/generation models beyond basic lifetimes
- 2D only (though extending to 3D is straightforward)
- Training can be finicky—learning rate and network architecture matter a lot

Potential improvements:
- Adaptive collocation point sampling
- Better automatic differentiation (currently using finite differences)
- Support for time-dependent simulations
- GPU acceleration
- More sophisticated loss weighting strategies

## Contributing

Found a bug? Have an idea for improvement? Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Inspired by the work on Physics-Informed Neural Networks by Raissi et al. and adapted for semiconductor device simulation. The finite difference solver serves as a sanity check to validate PINN results.

---

*If you find this useful or have questions, feel free to open an issue. Happy simulating!*
