# PINN-semiconductor-simulation

# Physics-Informed Neural Network (PINN) for Semiconductor Device Simulation

This repository contains a C++17 implementation of a **Physics-Informed Neural Network (PINN)** for solving the **Drift–Diffusion semiconductor equations**. The goal is to create a fast, differentiable, ML-based alternative to classical TCAD-style solvers.

## 🚀 Features
- Pure C++ neural network implementation (no TensorFlow/PyTorch)
- Solves:
- Poisson equation
- Electron continuity equation
- Hole continuity equation
- Physics-informed loss (PDE residuals + boundary conditions)
- Extensible architecture
- Python scripts for visualization

## 📂 Folder Structure

pinn-semiconductor-simulation/
│
├── CMakeLists.txt
├── README.md
├── LICENSE
│
├── data/
│   ├── doping_profiles/
│   ├── results/
│   └── plots/
│
├── docs/
│   ├── drift_diffusion_equations.pdf
│   ├── device_geometry.png
│   └── pinn_architecture.png
│
├── include/
│   ├── activation.hpp
│   ├── loss.hpp
│   ├── network.hpp
│   ├── optimizers.hpp
│   ├── pinn.hpp
│   ├── domain.hpp
│   ├── utils.hpp
│   ├── fd_solver.hpp
│   └── semiconductor_params.hpp
│
├── src/
│   ├── activation.cpp
│   ├── loss.cpp
│   ├── network.cpp
│   ├── optimizers.cpp
│   ├── pinn.cpp
│   ├── domain.cpp
│   ├── utils.cpp
│   ├── fd_solver.cpp
│   └── main.cpp
│
├── scripts/
│   ├── plot_results.py
│   ├── convert_results_to_csv.py
│   └── generate_doping_profile.py
│
└── tests/
    ├── test_network.cpp
    ├── test_pinn.cpp
    └── test_fd_solver.cpp
    
---

## 📄 **LICENSE** (MIT License)
```text
MIT License

Copyright (c) 2025 YOUR NAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
