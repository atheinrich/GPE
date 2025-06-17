#####################################################################################
# Gross-Pitaevskii equation
#
# I use the split-step Fourier method with imaginary-time evolution to find the
# ground state after assuming some ansatz, like a Gaussian.
#####################################################################################

#####################################################################################
# Imports
import numpy as np
import matplotlib.pyplot as plt

#####################################################################################
# Parameters
## Manual settings
### Static parameters
m = 1.443 * 10**(-25) # mass in kg; rubidium-87
a = 5.290 * 10**(-9)  # scattering length in m
λ = 804.1 * 10**(-9)  # wavelength in m; rubidium-87 D2
ω = 2*np.pi * 50      # harmonic trap frequency in Hz

### Dynamic parameters
n_steps = 1000  # number of x values in the domain
x_max   = 10    # width of spatial grid
dt      = 0.001 # time step

## Automatic rescaling
### With units
ℏ   = 1.05 * 10**-34     # Planck's constant
π   = np.pi              # pi
x_0 = np.sqrt(ℏ / (m*ω)) # trap size
g   = 2 * ℏ**2 * a / (m * x_0**2) # coefficient of nonlinear term in 1D
k_s = 2*π/(λ*np.sqrt(2)) # recoil momentum
E_s = (ℏ*k_s)**2 / (2*m) # energy scale
x_s = 1/k_s              # length scale
t_s = ℏ/E_s              # time scale
g_s = g / (E_s * x_s)    # coefficient of nonlinear term

### Without units
ℏ = 1
m = 1
g = g_s

#####################################################################################
# Functions
def build_bases():
    """ Builds position and momentum values as 1D arrays. """
    
    x  = np.linspace(-x_max, x_max, n_steps, endpoint=False)
    dx = x[1]-x[0]
    p  = 2*π * ℏ * np.fft.fftfreq(n_steps, d=dx)
    return x, dx, p

def build_kinetic_and_potential():
    """ Defines potential and calculates operators as 1D arrays. """
    
    U = (1/2) * (m * ω**2) * x**2
    K = p**2 / (2 * m)
    return U, K

def build_ansatz():
    """ Defines and constructs a guess for the ground state as a 1D array. """
    
    ψ_up_x   = np.exp(x**2)
    ψ_down_x = np.exp(x**2)
    ψ_up_x   /= np.sqrt(np.sum(np.abs(ψ_up_x)**2) * dx)
    ψ_down_x /= np.sqrt(np.sum(np.abs(ψ_down_x)**2) * dx)
    return np.stack([ψ_up_x, ψ_down_x])

def optimize_ansatz(ψ_x):
    """ Optimizes the ansatz with the split-step Fourier method and imaginary-time
        evolution via second-order Strang splitting. """
    
    for _ in range(n_steps):
        ψ_up_x, ψ_down_x = ψ_x
        
        # Half of a step in momentum space
        ψ_up_k = np.fft.fft(ψ_up_x)
        ψ_up_k *= np.exp(-(K/ℏ) * (dt/2))
        ψ_up_x = np.fft.ifft(ψ_up_k)
        ψ_down_k = np.fft.fft(ψ_down_x)
        ψ_down_k *= np.exp(-(K/ℏ) * (dt/2))
        ψ_down_x = np.fft.ifft(ψ_down_k)
        
        
        # Full step in position space
        U_g = U + g * (np.abs(ψ_up_x)**2 + np.abs(ψ_down_x)**2)
        ψ_up_x *= np.exp(-(U_g/ℏ) * dt)
        ψ_down_x *= np.exp(-(U_g/ℏ) * dt)
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(ψ_up_x)**2 + np.abs(ψ_down_x)**2) * dx)
        ψ_up_x   /= norm
        ψ_down_x /= norm
        ψ_x = np.stack([ψ_up_x, ψ_down_x])
    
    return ψ_x

#####################################################################################
# Generate data
x, dx, p = build_bases()
U, K     = build_kinetic_and_potential()

ψ_x      = build_ansatz()
ψ_x      = optimize_ansatz(ψ_x)

#####################################################################################
# Plot data
plt.plot(x, np.abs(ψ_x[0])**2, label=r'$|\psi_\uparrow(x)|^2$')
plt.plot(x, np.abs(ψ_x[1])**2, label=r'$|\psi_\downarrow(x)|^2$')
plt.plot(x, U / np.max(U) * np.max(np.abs(ψ_x[0])**2), '--', label='U(x) scaled')

plt.title('Spinor BEC Ground State (No SOC yet)')
plt.xlabel(r'$x/\sqrt{ℏ/mω}$')
plt.ylabel(r'Probability Density')

plt.legend(loc='upper right')
plt.grid()
plt.show()

#####################################################################################