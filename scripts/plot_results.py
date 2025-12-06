#!/usr/bin/env python3
"""
Plot results from PINN semiconductor simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def load_solution(filename):
    """Load solution from CSV file"""
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        return None
    
    df = pd.read_csv(filename)
    return df

def plot_solution(df, title="PINN Solution"):
    """Plot 2D solution fields"""
    if df is None:
        return
    
    # Get unique coordinates
    x_unique = sorted(df['x'].unique())
    y_unique = sorted(df['y'].unique())
    
    nx = len(x_unique)
    ny = len(y_unique)
    
    # Create grids
    X = df['x'].values.reshape(ny, nx)
    Y = df['y'].values.reshape(ny, nx)
    psi = df['psi'].values.reshape(ny, nx)
    n = df['n'].values.reshape(ny, nx)
    p = df['p'].values.reshape(ny, nx)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot electrostatic potential
    im1 = axes[0, 0].contourf(X, Y, psi, levels=50, cmap='viridis')
    axes[0, 0].set_title('Electrostatic Potential (V)')
    axes[0, 0].set_xlabel('x (m)')
    axes[0, 0].set_ylabel('y (m)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot electron density
    im2 = axes[0, 1].contourf(X, Y, n, levels=50, cmap='plasma')
    axes[0, 1].set_title('Electron Density (cm⁻³)')
    axes[0, 1].set_xlabel('x (m)')
    axes[0, 1].set_ylabel('y (m)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot hole density
    im3 = axes[1, 0].contourf(X, Y, p, levels=50, cmap='inferno')
    axes[1, 0].set_title('Hole Density (cm⁻³)')
    axes[1, 0].set_xlabel('x (m)')
    axes[1, 0].set_ylabel('y (m)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot charge density
    rho = p - n  # Simplified charge density
    im4 = axes[1, 1].contourf(X, Y, rho, levels=50, cmap='RdBu_r')
    axes[1, 1].set_title('Charge Density (cm⁻³)')
    axes[1, 1].set_xlabel('x (m)')
    axes[1, 1].set_ylabel('y (m)')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig

def compare_solutions(pinn_file, fd_file=None):
    """Compare PINN and FD solutions"""
    pinn_df = load_solution(pinn_file)
    
    if pinn_df is None:
        return
    
    fig = plot_solution(pinn_df, "PINN Solution")
    
    if fd_file and os.path.exists(fd_file):
        fd_df = load_solution(fd_file)
        if fd_df is not None:
            fig2 = plot_solution(fd_df, "Finite Difference Solution")
            
            # Compute difference
            if len(pinn_df) == len(fd_df):
                diff_psi = pinn_df['psi'] - fd_df['psi']
                diff_n = pinn_df['n'] - fd_df['n']
                diff_p = pinn_df['p'] - fd_df['p']
                
                x_unique = sorted(pinn_df['x'].unique())
                y_unique = sorted(pinn_df['y'].unique())
                nx = len(x_unique)
                ny = len(y_unique)
                
                X = pinn_df['x'].values.reshape(ny, nx)
                Y = pinn_df['y'].values.reshape(ny, nx)
                diff_psi_grid = diff_psi.values.reshape(ny, nx)
                
                fig3, ax = plt.subplots(figsize=(10, 8))
                im = ax.contourf(X, Y, diff_psi_grid, levels=50, cmap='RdBu_r')
                ax.set_title('Difference: PINN - FD (Potential)')
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                
                print(f"\nError statistics:")
                print(f"Max |psi_diff|: {np.abs(diff_psi).max():.6e}")
                print(f"Mean |psi_diff|: {np.abs(diff_psi).mean():.6e}")
                print(f"Max |n_diff|: {np.abs(diff_n).max():.6e}")
                print(f"Mean |n_diff|: {np.abs(diff_n).mean():.6e}")
    
    return fig

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <pinn_solution.csv> [fd_solution.csv]")
        sys.exit(1)
    
    pinn_file = sys.argv[1]
    fd_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    fig = compare_solutions(pinn_file, fd_file)
    
    # Save plots
    output_dir = "../data/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    if fig:
        fig.savefig(f"{output_dir}/pinn_solution.png", dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {output_dir}/pinn_solution.png")
    
    plt.show()

if __name__ == "__main__":
    main()

