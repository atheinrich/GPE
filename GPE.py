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
from tqdm import tqdm

#####################################################################################
# Parameters
## Manual settings
### Static parameters
m = 1.443 * 10**(-25) # mass in kg; rubidium-87
a = 5.290 * 10**(-9)  # scattering length in m
λ = 804.1 * 10**(-9)  # wavelength in m; rubidium-87 D2
ω = 2*np.pi * 50      # harmonic trap frequency in Hz
δ = -0.5               # detuning
Ω_0 = 0.5               # Raman coupling strength

### Dynamic parameters
n_steps = 512  # number of x values in the domain
x_max   = 1    # width of spatial grid
dt      = 0.01 # time step

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
def _build_bases():
    """ Builds position and momentum values as 1D arrays. """
    
    x  = np.linspace(-x_max, x_max, n_steps, endpoint=False)
    dx = x[1]-x[0]
    p  = 2*π * ℏ * np.fft.fftfreq(n_steps, d=dx)
    return [x, dx, p]

def _build_kinetic_and_potential(sys):
    """ Defines potential and calculates operators as 1D arrays. """
    
    # Harmonic trap
    U_0    = (1/2) * (m * ω**2) * sys.x**2
    
    # Zeeman detuning
    U_up   = U_0 + δ/2
    U_down = U_0 - δ/2
    
    # Kinetic energy
    K = sys.p**2 / (2 * m)
    
    return [U_0, U_up, U_down, K]

def _build_ansatz(sys):
    """ Defines and constructs a guess for the ground state as a 1D array. """
    
    # Gaussian ansatz
    ψ_up_x   = np.exp(sys.x**2)
    ψ_down_x = np.exp(sys.x**2)
    
    # Normalize
    ψ_up_x   /= np.sqrt(np.sum(np.abs(ψ_up_x)**2) * sys.dx)
    ψ_down_x /= np.sqrt(np.sum(np.abs(ψ_down_x)**2) * sys.dx)
    
    return np.stack([ψ_up_x, ψ_down_x])

def _optimize_ansatz(sys, Ω=Ω_0):
    """ Optimizes the ansatz with the split-step Fourier method and imaginary-time
        evolution via second-order Strang splitting.
        
        The spin-mixing potential needs to loop through each position, because numpy
        doesn't know how to handle an exponential of a 2D array with 1D arrays
        as elements. It can technically be worked with an exponential matrix, but
        the hyperbolic expansion is more efficient for a 2×2 matrix. """
    
    ψ_x = sys.ψ_x
    
    # Refine over n_steps
    for _ in tqdm(range(n_steps), desc=f"{'refining ansatz':<35}"):
        ψ_up_x, ψ_down_x = ψ_x
        
        # Half step in momentum space
        ψ_up_k = np.fft.fft(ψ_up_x)
        ψ_up_k *= np.exp(-(sys.K/ℏ) * (dt/2))
        ψ_up_x = np.fft.ifft(ψ_up_k)
        ψ_down_k = np.fft.fft(ψ_down_x)
        ψ_down_k *= np.exp(-(sys.K/ℏ) * (dt/2))
        ψ_down_x = np.fft.ifft(ψ_down_k)
        
        # Full-step: potential + Raman coupling
        ψ_new_up = np.zeros_like(ψ_up_x, dtype=complex)
        ψ_new_down = np.zeros_like(ψ_down_x, dtype=complex)
        
        ## Handle each position individually
        for i in range(len(sys.x)):
            n_tot = np.abs(ψ_up_x[i])**2 + np.abs(ψ_down_x[i])**2
            a = sys.U_up[i]   + g * n_tot
            b = sys.U_down[i] + g * n_tot
            Δ = a - b
            h = np.sqrt(Δ**2 + Ω**2)

            # Prevent division by zero
            h_safe = np.where(h == 0, 1e-12, h)

            c = np.exp(-dt * (a + b) / (2 * ℏ))
            cosh = np.cosh(dt * h / (2 * ℏ))
            sinh = np.sinh(dt * h / (2 * ℏ)) / h_safe

            # Construct the matrix action analytically
            ψ_new_up[i]   = c*(cosh*ψ_up_x[i]   - sinh*(Δ*ψ_up_x[i] + Ω*ψ_down_x[i]))
            ψ_new_down[i] = c*(cosh*ψ_down_x[i] - sinh*(Ω*ψ_up_x[i] - Δ*ψ_down_x[i]))
        
        ψ_up_x   = ψ_new_up
        ψ_down_x = ψ_new_down
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(ψ_up_x)**2 + np.abs(ψ_down_x)**2) * sys.dx)
        ψ_up_x   /= norm
        ψ_down_x /= norm
        ψ_x = np.stack([ψ_up_x, ψ_down_x])
    
    return ψ_x

def _phase_separation_metric(sys):
    ψ_up, ψ_down = sys.ψ_x
    n_up = np.abs(ψ_up)**2
    n_down = np.abs(ψ_down)**2

    numerator = np.sum(n_up * n_down) * sys.dx
    denom = np.sqrt(np.sum(n_up**2) * sys.dx * np.sum(n_down**2) * sys.dx)

    s = 1 - (numerator / denom)
    return s

class System:

    def __init__(self):
        
        # Construct static operators
        ops     = _build_bases()
        self.x  = ops[0]
        self.dx = ops[1]
        self.p  = ops[2]
        ops         = _build_kinetic_and_potential(self)
        self.U_0    = ops[0]
        self.U_up   = ops[1]
        self.U_down = ops[2]
        self.K      = ops[3]

    def plot_density(self):
        
        # Construct and optimize ground state
        self.ψ_x = _build_ansatz(self)
        self.ψ_x = _optimize_ansatz(self)
        
        # Plot the density
        plt.plot(self.x, np.abs(self.ψ_x[0])**2, label=r'$|\psi_\uparrow(x)|^2$')
        plt.plot(self.x, np.abs(self.ψ_x[1])**2, label=r'$|\psi_\downarrow(x)|^2$')
        plt.plot(self.x, self.U_up   / np.max(self.U_0) * np.max(np.abs(self.ψ_x[0])**2), '--', label=r'$U_\uparrow(x)$')
        plt.plot(self.x, self.U_down / np.max(self.U_0) * np.max(np.abs(self.ψ_x[0])**2), '--', label=r'$U_\downarrow(x)$')

        plt.title('Spinor BEC Ground State (No SOC yet)')
        plt.xlabel(r'$x/\sqrt{ℏ/mω}$')
        plt.ylabel(r'$|\psi(x)|^2 \cdot \sqrt{ℏ/mω}$')

        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

    def plot_fig_4(self):
        
        s_vals = []
        Ω_list = np.linspace(0, 2, 5)
        for Ω in tqdm(Ω_list):
            
            # Update Ω and rebuild potentials
            U_up   = 0.5 * m * ω**2 * self.x**2 + δ / 2
            U_down = 0.5 * m * ω**2 * self.x**2 - δ / 2
            
            self.ψ_x = _build_ansatz(self)
            self.ψ_x = _optimize_ansatz(self, Ω)

            s = _phase_separation_metric(self)
            s_vals.append(s)

        s_vals = np.array(s_vals)
        
        plt.plot(Ω_list, s_vals, '-o', label='Phase separation metric $s$')
        plt.axvline(0.19, color='gray', linestyle='--', label='Critical $Ω_c$ from Lin et al.')
        plt.xlabel(r'$Ω / E_L$')
        plt.ylabel(r'$s$')
        plt.title('Miscible to Immiscible Transition')
        plt.grid()
        plt.legend()
        plt.show()

#####################################################################################
# Generate and plot data
sys = System()
sys.plot_fig_4()

#####################################################################################