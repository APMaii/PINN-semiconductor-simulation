#!/usr/bin/env python3
"""
Generate doping profiles for semiconductor device simulation
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def generate_step_junction(x, y, x_mid, N_left, N_right):
    """Generate step junction doping profile"""
    profile = np.zeros_like(x)
    profile[x < x_mid] = N_left
    profile[x >= x_mid] = N_right
    return profile

def generate_gaussian_junction(x, y, x_center, width, N_peak):
    """Generate Gaussian doping profile"""
    return N_peak * np.exp(-((x - x_center) / width) ** 2)

def generate_linearly_graded_junction(x, y, x_min, x_max, N_min, N_max):
    """Generate linearly graded junction"""
    return N_min + (N_max - N_min) * (x - x_min) / (x_max - x_min)

def save_doping_profile(x, y, doping, filename):
    """Save doping profile to file"""
    data = np.column_stack([x.flatten(), y.flatten(), doping.flatten()])
    np.savetxt(filename, data, header="x,y,doping", comments="", delimiter=",")
    print(f"Doping profile saved to {filename}")

def plot_doping_profile(x, y, doping, title="Doping Profile"):
    """Plot doping profile"""
    plt.figure(figsize=(10, 6))
    if x.ndim == 2:
        plt.contourf(x, y, doping, levels=50, cmap='RdBu_r')
        plt.colorbar(label='Doping (cm⁻³)')
    else:
        plt.plot(x, doping, linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.yscale('symlog')
        plt.xlabel('Position (m)')
        plt.ylabel('Doping (cm⁻³)')
    
    plt.title(title)
    plt.tight_layout()
    
    output_dir = "../data/doping_profiles"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/doping_profile.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("Doping Profile Generator")
    print("=" * 50)
    
    # Default parameters
    x_min, x_max = 0.0, 1e-6  # 1 micron
    y_min, y_max = 0.0, 1e-6
    nx, ny = 100, 100
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # Generate step junction by default
    x_mid = (x_max - x_min) / 2.0
    doping = generate_step_junction(X, Y, x_mid, 1e18, -1e18)
    
    # Save profile
    output_dir = "../data/doping_profiles"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/step_junction.csv"
    save_doping_profile(X, Y, doping, filename)
    
    # Plot profile
    plot_doping_profile(X, Y, doping, "Step Junction Doping Profile")
    
    print("\nExample profiles:")
    print("- Step junction: N_left = 1e18, N_right = -1e18")
    print("- Gaussian: peak at center, width = 0.2e-6")
    print("- Linear: graded from N_min to N_max")

if __name__ == "__main__":
    main()

