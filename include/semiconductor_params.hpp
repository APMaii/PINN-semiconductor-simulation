#ifndef SEMICONDUCTOR_PARAMS_HPP
#define SEMICONDUCTOR_PARAMS_HPP

#include <cmath>

namespace semiconductor {

// Physical constants
constexpr double k_B = 1.380649e-23;        // Boltzmann constant (J/K)
constexpr double q = 1.602176634e-19;       // Elementary charge (C)
constexpr double eps_0 = 8.8541878128e-12;  // Vacuum permittivity (F/m)
constexpr double T = 300.0;                 // Temperature (K)
constexpr double V_T = k_B * T / q;         // Thermal voltage (V)

// Silicon material parameters
constexpr double eps_r = 11.7;              // Relative permittivity of Si
constexpr double eps = eps_0 * eps_r;       // Permittivity of Si (F/m)
constexpr double mu_n = 0.135;              // Electron mobility (m²/V·s) at 300K
constexpr double mu_p = 0.048;              // Hole mobility (m²/V·s) at 300K
constexpr double D_n = mu_n * V_T;          // Electron diffusivity (m²/s)
constexpr double D_p = mu_p * V_T;          // Hole diffusivity (m²/s)
constexpr double n_i = 1.0e10;              // Intrinsic carrier concentration (cm⁻³)

// Unit conversions
constexpr double cm_to_m = 1e-2;
constexpr double m_to_cm = 1e2;

// Device parameters (default values)
constexpr double device_length = 1.0e-6;    // Device length (m)
constexpr double device_width = 1.0e-6;     // Device width (m)

// Default boundary conditions
constexpr double V_applied = 1.0;           // Applied voltage (V)

// Recombination parameters
constexpr double tau_n = 1e-6;              // Electron lifetime (s)
constexpr double tau_p = 1e-6;              // Hole lifetime (s)

} // namespace semiconductor

#endif // SEMICONDUCTOR_PARAMS_HPP

