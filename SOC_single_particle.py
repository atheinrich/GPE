#####################################################################################
# SOC BEC single-particle dynamics without interaction
#
# This is used to illustrate the effect of spin-orbit coupling on a harmonic
# oscillator. Further additions might include the interaction term and projections
# of the GP result for the (presumably) Thomas-Fermi ground state(s).
#####################################################################################

#####################################################################################
# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy as sp

#####################################################################################
# Parameters
ħ = 1.0
m = 1.0
ω = 10.0
γ = 1.0
Ω = 1.0
δ = 0.0

N = 500
x_max = 5.0
n_states = 4
π = np.pi

#####################################################################################
# Bases
## Diagonal position
x   = np.linspace(-x_max, x_max, N)
x_x = np.diag(x)

## Diagonal momentum
dx  = x[1] - x[0]
p   = np.fft.fftfreq(N, d=dx) * 2*π*ħ
p   = np.fft.fftshift(p)
P_p = np.diag(p)

## Diagonal spin-1/2 projection
σ_x = np.array([[0, 1], [1, 0]])
σ_z = np.array([[1, 0], [0, -1]])
I_2 = np.eye(2)

# Ladder operators in number operator basis
dim = x_x.shape[0] * σ_x.shape[0]
a = np.zeros((dim, dim))
for i in range(1, dim):
        a[i-1, i] = np.sqrt(i)
a_dag = a.T.conj()

#####################################################################################
# Hamiltonian
## Kinetic
K_p = P_p@P_p/(2*m)
K_p = np.kron(K_p, I_2)

## Harmonic trap
U_x = (1/2)*m*ω**2 * x_x**2
F   = np.fft.fft(np.eye(N)) / np.sqrt(N)
U_p = F @ U_x @ F.T.conj()
U_p = np.kron(U_p, I_2)

## Raman SOC
SOC_p      = np.kron(γ * P_p, σ_z)
Zeeman_p   = np.kron(np.eye(N), (Ω/2) * σ_x)
Detuning_p = np.kron(np.eye(N), (δ/2) * σ_z)

## Hamiltonian
H = K_p + U_p + SOC_p + Zeeman_p + Detuning_p

#####################################################################################
# Solve
# Diagonalize
evals, evecs = sp.linalg.eigh(H)
idx   = np.argsort(evals)
evals = evals[idx]
evecs = evecs[:, idx]

#####################################################################################
# Utility
def rtc(array):
    """ Shorthand to convert a row vector to a column vector (or vice-versa). 
    
        Example
        -------
        column_vector = rtc(sys.states[1][0][:,0]) """
    
    if len(array.shape) == 1:   vector = array.reshape(array.shape[0], 1)
    elif len(array.shape) == 2: vector = array.reshape(1, array.shape[0])
    return vector

#####################################################################################
# Other
def plot_eigenstates():

    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(n_states, 2, figure=fig, wspace=0.3)
    fig.suptitle("Lowest Eigenstates of the Single-particle Hamiltonian", fontsize=14)

    for n in range(n_states):
        ax_x = fig.add_subplot(gs[n, 0])
        ax_p = fig.add_subplot(gs[n, 1])

        ψ_p = evecs[:, n].reshape(N, 2)
        ψ_x = np.fft.ifft(np.fft.ifftshift(ψ_p, axes=0), axis=0) * np.sqrt(N)
        ρ_x = np.abs(ψ_x)**2
        ρ_p = np.abs(ψ_p)**2

        ax_x.plot(x, ρ_x[:, 0], 'b')
        ax_x.plot(x, ρ_x[:, 1], 'r')
        ax_p.plot(p, ρ_p[:, 0], 'b', label='spin ↑')
        ax_p.plot(p, ρ_p[:, 1], 'r', label='spin ↓')
        
        ax_x.text(0.98, 0.95, f"n={n}", transform=ax_x.transAxes, ha='right', va='top', fontsize=10)
        if n != n_states - 1:
            ax_x.set_xticklabels([])
        else:                 
            ax_x.set_xlabel(r"$x$")
            ax_x.set_ylabel(r"$|⟨ψ|x⟩|^2$")
        
        if n != n_states - 1:
            ax_p.set_xticklabels([])
        else:
            ax_p.set_xlabel(r"$p$")
            ax_p.set_ylabel(r"$|⟨ψ|p⟩|^2$")
            ax_p.legend(loc='upper right', fontsize=8)
        
        ax_x.set_xlim(-2, 2)
        ax_p.set_xlim(-20, 20)
        ax_x.grid(True)
        ax_p.grid(True)

    plt.show()

def Thomas_Fermi_broken():
    
    # Construct and normalize TF wavefunction in position space
    n_TF = (2000 - (1/2)*m*ω**2 * x**2) / 1
    ψ_pos = np.sqrt(n_TF)
    
    ## Add a spin component
    ψ_spin = np.zeros(σ_x.shape[0])
    ψ_spin[0] = 1
    ψ = np.kron(ψ_pos, ψ_spin)
    
    ## Normalize
    ψ = rtc(ψ / np.linalg.norm(ψ))
    print(ψ)
    
    # Project onto momentum basis
    coeffs = evecs.T.conj() @ ψ
    print(coeffs)
    weights = np.abs(coeffs)**2
    
    # First four even modes: n = 0, 2, 4, 6
    even_modes = [0, 2, 4, 6]
    recons = {}
    for k in range(1, 5):
        modes = even_modes[:k]
        recons[k] = sum(coeffs[n] * evecs[:,n] for n in modes)

    # Plot
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # Row 1: TF profile
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax0.plot(x, ψ)
    ax0.set_title("Thomas–Fermi wavefunction")
    ax0.set_xlim(-xmax, xmax)

    # Row 1: coefficients
    ax1 = fig.add_subplot(gs[0, 2:4])
    ax1.bar(range(20), weights[:20])
    ax1.set_title("HO mode weights")

    # Row 2: reconstructions (first k even modes)
    for i, k in enumerate([1, 2, 3, 4]):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(x, recons[k])
        ax.plot(x, ψ, '--', color='gray')
        ax.set_title(f"First {k} even mode(s)")
        ax.set_xlim(-xmax, xmax)

    plt.show()

def Thomas_Fermi_old():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy.special import hermite, factorial
    from scipy.integrate import simps

    # Parameters
    m = 1.0
    omega = 1.0
    hbar = 1.0
    g = 1.0
    mu = 10.0
    a_ho = np.sqrt(hbar / (m * omega))

    # Spatial grid
    x_TF_radius = np.sqrt(2 * mu / (m * omega**2))
    xmax = x_TF_radius * 1.2
    x = np.linspace(-xmax, xmax, 3000)

    # Thomas–Fermi wavefunction
    V = 0.5 * m * omega**2 * x**2
    n_TF = np.maximum((mu - V) / g, 0.0)
    psi_TF = np.sqrt(n_TF)
    psi_TF /= np.sqrt(simps(np.abs(psi_TF)**2, x))

    # Harmonic oscillator eigenfunction
    def ho_wavefunction(n, x):
        xi = x / a_ho
        Hn_xi = hermite(n)(xi)
        prefactor = 1.0 / (np.pi**0.25 * np.sqrt(a_ho) * np.sqrt(2.0**n * factorial(n)))
        return prefactor * Hn_xi * np.exp(-xi**2 / 2.0)

    # Basis and projections
    N_basis = 20
    phis = [ho_wavefunction(n, x) for n in range(N_basis)]
    coeffs = np.array([simps(phis[n] * psi_TF, x) for n in range(N_basis)])
    weights = np.abs(coeffs)**2

    # First four even modes: n = 0, 2, 4, 6
    even_modes = [0, 2, 4, 6]
    recons = {}
    for k in range(1, 5):
        modes = even_modes[:k]
        recons[k] = sum(coeffs[n] * phis[n] for n in modes)

    # Plot
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # Row 1: TF profile
    ax0 = fig.add_subplot(gs[0, 0:2])
    ax0.plot(x, psi_TF)
    ax0.set_title("Thomas–Fermi wavefunction")
    ax0.set_xlim(-xmax, xmax)

    # Row 1: coefficients
    ax1 = fig.add_subplot(gs[0, 2:4])
    ax1.bar(range(20), weights[:20])
    ax1.set_title("HO mode weights")

    # Row 2: reconstructions (first k even modes)
    for i, k in enumerate([1, 2, 3, 4]):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(x, recons[k])
        ax.plot(x, psi_TF, '--', color='gray')
        ax.set_title(f"First {k} even mode(s)")
        ax.set_xlim(-xmax, xmax)

    plt.show()

#####################################################################################
# Plot
plot_eigenstates()

#####################################################################################