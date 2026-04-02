import numpy as np
import scipy.linalg as la
import scipy.optimize as opt

from ManyBodyED import ManyBodyED

class DMFTSolver:
    """
    DMFT solver for half-filled Hubbard model on Bethe lattice
    Uses 6-site chain ED solver with time-domain fitting
    Convention: μ = 0 (particle-hole symmetric around ω = 0)
    """

    def __init__(self, U, t=1.0, T=0.0, t_max=20.0, n_time=100, eta=0.01):
        """
        Parameters:
        -----------
        U : float
            Hubbard interaction strength
        t : float
            Hopping parameter (sets energy scale, default 1.0)
        T : float
            Temperature (default 0.0 for ground state)
        t_max : float
            Maximum time for real-time Green's function
        n_time : int
            Number of time points
        eta : float
            Broadening parameter for numerical stability
        """
        self.U = U
        self.t = t
        self.T = T
        self.mu = 0.5*U  # Convention 2: μ = 0 for particle-hole symmetry around zero
        self.eta = eta

        # Time grid
        self.t_grid = np.linspace(0.000001, t_max, n_time)  #a little time is added at the beginning
        self.dt = self.t_grid[1] - self.t_grid[0]
        self.n_time = n_time

        # Chain parameters (will be fitted)
        self.chain_params = None
        self.use_reduced_params = True  # Use symmetry constraints

        # ED solver
        self.ed_solver = ManyBodyED(n_sites=6)

    def bethe_dos_green(self, omega, eta=None):
        """
        Local Green's function for Bethe lattice with semi-circular DOS
        G_loc(z) = (z - sqrt(z^2 - 4t^2)) / (2t^2)
        """
        if eta is None:
            eta = self.eta
        z = omega + 1j * eta
        sqrt_term = np.sqrt(z**2 - (2*self.t)**2)
        # Choose branch with Im(G) < 0
        if np.any(sqrt_term.imag > 0):
            sqrt_term = np.where(sqrt_term.imag > 0, -sqrt_term, sqrt_term)
        return (z - sqrt_term) / (2 * self.t**2)

    def initialize_hybridization(self, method='bethe'):
        """
        Initialize target hybridization function Delta^R(t)

        Parameters:
        -----------
        method : str
            'bethe' - from non-interacting Bethe lattice
            'exponential' - simple Lorentzian approximation
        """
        if method == 'bethe':
            # Compute in frequency domain
            n_omega = 2048  # Use power of 2 for FFT
            omega_max = 10.0
            omega = np.linspace(-omega_max, omega_max, n_omega)

            # Non-interacting local Green's function
            G_loc = self.bethe_dos_green(omega)

            # Delta(omega) = t^2 * G_loc(omega)
            Delta_omega = self.t**2 * G_loc

            # Inverse FFT to time domain
            d_omega = omega[1] - omega[0]
            Delta_t_full = np.fft.ifft(np.fft.ifftshift(Delta_omega)) * n_omega * d_omega / (2 * np.pi)

            # Extract time grid points we need
            t_full = np.fft.fftfreq(n_omega, d_omega/(2*np.pi))
            t_full = np.fft.fftshift(t_full)

            # Interpolate to our time grid
            Delta_t = np.interp(self.t_grid, t_full[t_full >= 0],
                               Delta_t_full[t_full >= 0])

            # Apply causality and damping
            Delta_t = Delta_t * np.exp(-self.eta * self.t_grid)

        elif method == 'exponential':
            # Simple exponential hybridization
            Delta_t = -1j * self.t**2 * np.exp(-self.t_grid)

        return Delta_t

    def expand_reduced_params(self, p_red):
        """
        Expand reduced parameters using half-filling symmetry
        p_red = [t0, t1, t2, eps1, eps2]
        Full: [t0, t1, t2, t2, t1, eps1, eps2, 0, -eps2, -eps1]
        """
        t0, t1, t2, eps1, eps2 = p_red
        return np.array([t0, t1, t2, t2, t1, eps1, eps2, 0.0, -eps2, -eps1])

    def build_bath_hamiltonian(self, params):
        """
        Build 5x5 bath Hamiltonian (sites 1-5)
        Tridiagonal with epsilon_k on diagonal, t_k on off-diagonals
        """
        if self.use_reduced_params and len(params) == 5:
            params = self.expand_reduced_params(params)

        t0, t1, t2, t3, t4 = params[0:5]
        eps1, eps2, eps3, eps4, eps5 = params[5:10]

        H_bath = np.zeros((5, 5))

        # Diagonal: on-site energies
        H_bath[0, 0] = eps1
        H_bath[1, 1] = eps2
        H_bath[2, 2] = eps3
        H_bath[3, 3] = eps4
        H_bath[4, 4] = eps5

        # Off-diagonal: hoppings
        H_bath[0, 1] = H_bath[1, 0] = t1
        H_bath[1, 2] = H_bath[2, 1] = t2
        H_bath[2, 3] = H_bath[3, 2] = t3
        H_bath[3, 4] = H_bath[4, 3] = t4

        return H_bath, t0

    def compute_fitted_hybridization(self, params):
        """
        Compute Delta_fit^R(t) = t0^2 * g_11^R(t)
        where g_11 is the (1,1) element of the bath Green's function
        """
        H_bath, t0 = self.build_bath_hamiltonian(params)

        # Diagonalize bath
        eigenvalues, eigenvectors = la.eigh(H_bath)

        # g_11^R(t) = -i * theta(t) * sum_m |u_1m|^2 * exp(-i * lambda_m * t)
        # Site 1 is index 0 in the 5x5 bath matrix
        u_1m = eigenvectors[0, :]  # First row

        g_11_t = np.zeros(len(self.t_grid), dtype=complex)
        for i, t in enumerate(self.t_grid):
            if t > 0:
                g_11_t[i] = -1j * np.sum(np.abs(u_1m)**2 * np.exp(-1j * eigenvalues * t))

        Delta_fit = t0**2 * g_11_t
        return Delta_fit

    def fit_chain_parameters(self, Delta_target, initial_guess=None, weights=None):
        """
        Fit chain parameters to match Delta_target(t)

        Parameters:
        -----------
        Delta_target : array
            Target hybridization function in time domain
        initial_guess : array
            Initial parameter guess
        weights : array
            Weights for different time points
        """
        if initial_guess is None:
            if self.use_reduced_params:
                initial_guess = np.array([1.0, 0.5, 0.5, 0.0, 0.0])
            else:
                initial_guess = np.array([1.0, 0.5, 0.5, 0.5, 0.5,
                                        0.0, 0.0, 0.0, 0.0, 0.0])

        if weights is None:
            # Weight short times more heavily
            #weights = np.exp(-0.1 * self.t_grid)
            weights = np.ones(len(self.t_grid))


        def residuals(params):
            """Compute residuals for least-squares fitting"""
            Delta_fit = self.compute_fitted_hybridization(params)
            diff = Delta_target - Delta_fit

            # Separate real and imaginary parts, apply weights
            res_real = weights * diff.real
            res_imag = weights * diff.imag

            ####WARNING: Half filling only####
            return res_imag
            #return np.concatenate([res_real, res_imag])

        # Set bounds
        if self.use_reduced_params:
            bounds = ([0, 0, 0, 0.0, 0.0],
                     [10.0, 10.0, 10.0, 10.0, 10.0])
        else:
            bounds = ([0.1]*5 + [-5.0]*5,
                     [5.0]*10)

        # Optimize
        result = opt.least_squares(residuals, initial_guess, bounds=bounds,
                                  ftol=1e-10, max_nfev=5000, verbose=0)
        if self.use_reduced_params:
            fitted_params = self.expand_reduced_params(result.x)
        else:
            fitted_params = result.x

        return fitted_params, result

    def solve_impurity_ed(self, params):
        """
        Solve impurity problem using exact diagonalization
        Returns G_imp^R(t) computed in real time
        """
        G_imp = self.ed_solver.compute_green_function_time(
            params, self.U, self.mu, self.t_grid, T=self.T, eta=self.eta
        )

        return G_imp

    def run_dmft_iteration(self, Delta_target, max_iter=50, tol=1e-4, mixing=0.5):
        """
        Run DMFT self-consistency loop

        Parameters:
        -----------
        Delta_target : array
            Initial hybridization function
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        mixing : float
            Mixing parameter for stability (0 = no update, 1 = full update)
        """
        history = {
            'Delta': [],
            'G_imp': [],
            'error': [],
            'params': []
        }

        Delta_old = Delta_target.copy()

        for iteration in range(max_iter):
            print(f"\nIteration {iteration + 1}/{max_iter}")

            # Step 1: Fit chain parameters to current Delta
            print("  Fitting chain parameters...")
            if iteration == 0:
                initial_guess = None
            else:
                initial_guess = self.chain_params[:5] if self.use_reduced_params else self.chain_params

            self.chain_params, fit_result = self.fit_chain_parameters(
                Delta_old, initial_guess=initial_guess
            )

            print(f"  Fit cost: {fit_result.cost:.6e}")
            if self.use_reduced_params:
                print(f"  Params: t0={self.chain_params[0]:.4f}, t1={self.chain_params[1]:.4f}, "
                      f"t2={self.chain_params[2]:.4f}, eps1={self.chain_params[5]:.4f}, eps2={self.chain_params[6]:.4f}")

            # Step 2: Solve impurity problem
            print("  Solving impurity problem with ED...")
            G_imp = self.solve_impurity_ed(self.chain_params)

            # Step 3: Update hybridization
            Delta_new = self.t**2 * G_imp

            # Step 4: Check convergence
            error = np.sqrt(np.mean(np.abs(Delta_new - Delta_old)**2))
            print(f"  Convergence error: {error:.6e}")

            # Store history
            history['Delta'].append(Delta_new.copy())
            history['G_imp'].append(G_imp.copy())
            history['error'].append(error)
            history['params'].append(self.chain_params.copy())

            # Check convergence
            if error < tol:
                print(f"\n{'='*60}")
                print(f"CONVERGED after {iteration + 1} iterations!")
                print(f"{'='*60}")
                break

            # Mix old and new
            Delta_old = mixing * Delta_new + (1 - mixing) * Delta_old

        else:
            print(f"\nReached maximum iterations ({max_iter})")

        return history

    def run(self, max_iter=50, tol=1e-4, mixing=0.5, init_method='bethe'):
        """
        Run complete DMFT calculation
        """
        print("="*60)
        print(f"DMFT Solver for Hubbard Model on Bethe Lattice")
        print(f"U/t = {self.U/self.t:.2f}, Half-filling (μ = 0)")
        print(f"Particle-hole symmetric: LHB at ω≈-U/2, UHB at ω≈+U/2")
        print(f"6-site chain ED solver with real-time Green's function")
        print("="*60)

        # Initialize
        print("\nInitializing hybridization function...")
        Delta_init = self.initialize_hybridization(method=init_method)
        print(f"Initialization method: {init_method}")

        # Run DMFT loop
        history = self.run_dmft_iteration(Delta_init, max_iter, tol, mixing)

        return history
