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
# Physical parameters
## Fundamental constants
π   = np.pi              # pi
ℏ   = 1.05 * 10**(-34)   # Planck's constant (in units of kg∙m^2/s)
a_0 = 5.292 * 10**(-11)  # Bohr radius (in units of m)
m   = 1.443 * 10**(-25)  # mass in kg; rubidium-87 (in units of kg)

## Experimental settings
λ = 804.1 * 10**(-9)     # wavelength in m; rubidium-87 D2 (in units of m)
ω = 2*π* 50              # harmonic trap frequency (in units of 1/s)

a_11 = 100.86 * a_0      # ↑↑ scattering length (in units of m)
a_12 = 100.36 * a_0      # ↑↓ scattering length (in units of m)
a_22 = 100.36 * a_0      # ↓↓ scattering length (in units of m)

#####################################################################################
# Dimensionless parameters
x_0 = np.sqrt(ℏ / (m*ω)) # trap size (in units of m)
k_s = 2*π/(λ*np.sqrt(2)) # recoil momentum (in units of m/s)
x_s = 1/k_s              # length scale (in units of m)
E_s = (ℏ*k_s)**2 / (2*m) # energy scale (in units of J)
t_s = ℏ/E_s              # time scale (in units of s)

g_nn = lambda a: 2*ℏ**2 * a / (m*x_0**2) # scattering coefficient (in units of J∙m)

#####################################################################################
### Dynamic parameters
Ω_0 = 0.5 * E_s     # Raman coupling strength (in units of J)
δ = 0               # detuning (in units of J)

n_steps = 512       # number of x values in the domain
x_max   = 10        # width of spatial grid (in units of x_s)
dt      = 0.01      # time step (in units of t_s)

#####################################################################################
# Functions
def _build_bases():
    """ Builds position and momentum values as 1D arrays.
        These are not operators. """
    
    x  = np.linspace(-x_max, x_max, n_steps)
    dx = x[1]-x[0]
    p  = 2*π * np.fft.fftfreq(n_steps, d=dx)
    return [x, dx, p]

def _build_kinetic_and_potential(sys):
    """ Defines potential and calculates operators as 1D arrays.
        These are not operators. """
    
    # Harmonic trap
    U_0 = ((m*ω)/(ℏ*k_s**2))**2 * sys.x**2
    
    # Kinetic energy
    K = sys.p**2
    
    return [U_0, K]

def _build_ansatz(sys):
    """ Defines and constructs a guess for the ground state as a 1D array.
        These are in the position basis. """
    
    # Gaussian ansatz
    ψ_up_x   = np.exp(-sys.x**2)
    ψ_down_x = np.exp(-sys.x**2)
    
    # Normalize
    ψ_up_x   /= np.sqrt(np.sum(np.abs(ψ_up_x)**2) * sys.dx)
    ψ_down_x /= np.sqrt(np.sum(np.abs(ψ_down_x)**2) * sys.dx)
    
    return np.stack([ψ_up_x, ψ_down_x])

def _optimize_ansatz(sys, Ω=Ω_0):
    """ Optimizes the ansatz with a split-step Fourier method and imaginary-time
        evolution via second-order Strang splitting.
        
        The spin-mixing potential needs to loop through each position, because numpy
        doesn't know how to handle an exponential of a 2D array with 1D arrays
        as elements. It can technically be worked with an exponential matrix, but
        the hyperbolic expansion is more efficient for a 2×2 matrix. """
    
    ψ_x = sys.ψ_x
    
    # Refine over n_steps
    for _ in tqdm(range(n_steps), desc=f"{'refining ansatz':<35}"):
        ψ_up_x, ψ_down_x = ψ_x
        
        # Half step in momentum basis
        exp_K    = np.exp(-sys.K * (dt/2))
        ψ_up_k   = np.fft.fft(ψ_up_x) * exp_K
        ψ_down_k = np.fft.fft(ψ_down_x) * exp_K
        
        ## Apply Raman dressing while in momentum basis
        ψ_new_up   = np.zeros_like(ψ_up_k,   dtype=complex)
        ψ_new_down = np.zeros_like(ψ_down_k, dtype=complex)
        for i in range(len(sys.p)):
            
            # Set matrix elements
            σ_11 = 2*sys.p[i] + δ/(2*E_s)
            σ_22 = -(2*sys.p[i] + δ/(2*E_s))
            σ_12 = Ω/(2*E_s)
            
            C_1 = (σ_11 + σ_22) / 2
            C_2 = (σ_11 - σ_22) / 2
            C_3 = np.sqrt(C_2**2 + σ_12**2)
            
            if np.abs(C_3) < 1e-12: C_3 = 1e-12
            arg = C_3 * dt
            if np.abs(arg) > 30: arg = np.sign(arg) * 30
            
            # Construct and apply
            exp  = np.exp(-C_1 * dt)
            cosh = np.cosh(arg)
            sinh = np.sinh(arg) / C_3
            C_4  = ψ_up_k[i]
            C_5  = ψ_down_k[i]
            C_6  = C_2  * ψ_up_k[i] + σ_12 * ψ_down_k[i]
            C_7  = σ_12 * ψ_up_k[i] - C_2  * ψ_down_k[i]
            
            ψ_new_up[i]   = exp * (cosh * C_4 - sinh * C_6)
            ψ_new_down[i] = exp * (cosh * C_5 - sinh * C_7)
        
        ψ_up_k   = ψ_new_up
        ψ_down_k = ψ_new_down
        
        # Full step in position basis; includes interaction term
        g_11, g_12, g_22 = g_nn(a_11), g_nn(a_12), g_nn(a_22)
        
        U_up     = (k_s/E_s) * (g_11 * np.abs(ψ_up_x)**2 + g_12 * np.abs(ψ_down_x)**2)
        U_down   = (k_s/E_s) * (g_12 * np.abs(ψ_up_x)**2 + g_22 * np.abs(ψ_down_x)**2)
        ψ_up_x   = np.fft.ifft(ψ_up_k)   * np.exp(-(sys.U_0 + U_up) * dt)
        ψ_down_x = np.fft.ifft(ψ_down_k) * np.exp(-(sys.U_0 + U_down) * dt)
        
        # Half step in momentum basis
        exp_K    = np.exp(-sys.K * (dt/2))
        ψ_up_k   = np.fft.fft(ψ_up_x)   * exp_K
        ψ_down_k = np.fft.fft(ψ_down_x) * exp_K
        
        # Normalize in position basis
        ψ_up_x   = np.fft.ifft(ψ_up_k)
        ψ_down_x = np.fft.ifft(ψ_down_k)
        norm     = np.sqrt(np.sum(np.abs(ψ_up_x)**2 + np.abs(ψ_down_x)**2) * sys.dx)
        ψ_up_x   /= norm
        ψ_down_x /= norm
        ψ_x      = np.stack([ψ_up_x, ψ_down_x])
    
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
        ops      = _build_bases()
        self.x   = ops[0]
        self.dx  = ops[1]
        self.p   = ops[2]
        
        ops      = _build_kinetic_and_potential(self)
        self.U_0 = ops[0]
        self.K   = ops[1]

    def plot_position(self):
        
        # Construct and optimize ground state
        self.ψ_x = _build_ansatz(self)
        self.ψ_x = _optimize_ansatz(self)
        
        # Plot the density
        plt.plot(self.x, np.abs(self.ψ_x[0])**2, label=r'$|\psi_\uparrow(x)|^2$')
        plt.plot(self.x, np.abs(self.ψ_x[1])**2, label=r'$|\psi_\downarrow(x)|^2$')
        plt.plot(self.x, self.U_0 / np.max(self.U_0) * np.max(np.abs(self.ψ_x[0])**2), '--')

        plt.title('BEC Ground State')
        plt.xlabel(r'Position $x / k_r$')
        plt.ylabel(r'Density $|\psi(x)|^2 / k_r$')

        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

    def plot_momentum(self):
        
        # Construct and optimize ground state
        self.ψ_x = _build_ansatz(self)
        self.ψ_x = _optimize_ansatz(self)
        
        self.ψ_k   = np.fft.fft(self.ψ_x)
        
        # Plot the density
        plt.plot(self.p, np.abs(self.ψ_k[0])**2, label=r'$|\psi_\uparrow(x)|^2$')
        plt.plot(self.p, np.abs(self.ψ_k[1])**2, label=r'$|\psi_\downarrow(x)|^2$')
        plt.plot(self.p, self.U_0 / np.max(self.U_0) * np.max(np.abs(self.ψ_x[0])**2), '--')
        
        plt.xlim(-20, 20)

        plt.title('BEC Ground State')
        plt.xlabel(r'Momentum $p / (ℏ k_r)$')
        plt.ylabel(r'Density $|\psi(k)|^2 \cdot ℏ k_r$')

        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

    def plot_fig_4(self):
        
        s_vals = []
        Ω_list = np.linspace(0, 1, 20)
        for Ω in tqdm(Ω_list):
            
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
sys.plot_position()

#####################################################################################