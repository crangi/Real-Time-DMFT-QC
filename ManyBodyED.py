import numpy as np
import scipy.linalg as la
from itertools import combinations

class FockSpace:
    """
    Fock space construction for multi-site fermionic system with spin
    """
    def __init__(self, n_sites):
        """
        Parameters:
        -----------
        n_sites : int
            Number of sites
        """
        self.n_sites = n_sites
        self.n_orbitals = 2 * n_sites  # Each site has spin up and down

    def state_to_index(self, state):
        """
        Convert occupation number state to integer index
        state: list/array of length n_orbitals with 0s and 1s
        """
        return sum(bit << i for i, bit in enumerate(state))

    def index_to_state(self, index):
        """
        Convert integer index to occupation number state
        """
        return [(index >> i) & 1 for i in range(self.n_orbitals)]

    def get_sector_basis(self, n_up, n_down):
        """
        Get basis states for fixed particle number sector

        Parameters:
        -----------
        n_up : int
            Number of spin-up electrons
        n_down : int
            Number of spin-down electrons

        Returns:
        --------
        basis_states : list
            List of state indices in this sector
        """
        basis_states = []

        # Sites 0, 1, 2, ..., n_sites-1 for spin up
        # Sites n_sites, n_sites+1, ..., 2*n_sites-1 for spin down

        # All combinations of n_up sites occupied (spin up)
        for up_occupied in combinations(range(self.n_sites), n_up):
            # All combinations of n_down sites occupied (spin down)
            for down_occupied in combinations(range(self.n_sites), n_down):
                state = [0] * self.n_orbitals

                # Set spin-up occupations
                for site in up_occupied:
                    state[site] = 1

                # Set spin-down occupations
                for site in down_occupied:
                    state[self.n_sites + site] = 1

                basis_states.append(self.state_to_index(state))

        return sorted(basis_states)

    def apply_c(self, state_index, orbital, check_only=False):
        """
        Apply annihilation operator c_orbital to state

        Returns:
        --------
        new_state_index : int or None
            New state index, or None if result is zero
        sign : int
            Fermionic sign (-1 or +1)
        """
        state = self.index_to_state(state_index)

        # Check if orbital is occupied
        if state[orbital] == 0:
            return None, 0

        if check_only:
            return state_index, 1

        # Count number of occupied orbitals before this one (for sign)
        sign = (-1) ** sum(state[:orbital])

        # Annihilate electron
        new_state = state.copy()
        new_state[orbital] = 0

        return self.state_to_index(new_state), sign

    def apply_c_dagger(self, state_index, orbital, check_only=False):
        """
        Apply creation operator c†_orbital to state

        Returns:
        --------
        new_state_index : int or None
            New state index, or None if result is zero
        sign : int
            Fermionic sign (-1 or +1)
        """
        state = self.index_to_state(state_index)

        # Check if orbital is unoccupied
        if state[orbital] == 1:
            return None, 0

        if check_only:
            return state_index, 1

        # Count number of occupied orbitals before this one (for sign)
        sign = (-1) ** sum(state[:orbital])

        # Create electron
        new_state = state.copy()
        new_state[orbital] = 1

        return self.state_to_index(new_state), sign
    
class ManyBodyED:
    """
    Exact Diagonalization solver for 6-site Anderson impurity model
    """

    def __init__(self, n_sites=6):
        """
        Parameters:
        -----------
        n_sites : int
            Number of sites (default 6: 1 impurity + 5 bath)
        """
        self.n_sites = n_sites
        self.fock = FockSpace(n_sites)

    def build_hamiltonian_sector(self, params, U, mu, n_up, n_down):
        """
        Build many-body Hamiltonian for a specific particle number sector

        NOTE: Chemical potential μ applies ONLY to the impurity site (site 0)
        """
        t0, t1, t2, t3, t4 = params[0:5]
        eps1, eps2, eps3, eps4, eps5 = params[5:10]

        # On-site energies
        # Impurity site 0: eps0 = 0 (by convention, absorbed into μ)
        # Bath sites 1-5: eps1, eps2, eps3, eps4, eps5
        eps = [0.0, eps1, eps2, eps3, eps4, eps5]

        # Hoppings
        hoppings = [(0, 1, t0), (1, 2, t1), (2, 3, t2), (3, 4, t3), (4, 5, t4)]

        # Get basis states
        basis_states = self.fock.get_sector_basis(n_up, n_down)
        n_basis = len(basis_states)
        state_to_basis = {state: i for i, state in enumerate(basis_states)}

        # Initialize Hamiltonian
        H = np.zeros((n_basis, n_basis), dtype=complex)

        # Build Hamiltonian
        for i, state in enumerate(basis_states):

            # 1. On-site energies
            for site in range(self.n_sites):
                orbital_up = site
                orbital_down = self.n_sites + site

                state_arr = self.fock.index_to_state(state)
                n_up_site = state_arr[orbital_up]
                n_down_site = state_arr[orbital_down]

                # On-site energy (for ALL sites)
                H[i, i] += eps[site] * (n_up_site + n_down_site)

            # 2. Chemical potential ONLY on impurity site (site 0)
            state_arr = self.fock.index_to_state(state)
            n_up_imp = state_arr[0]  # Impurity spin-up
            n_down_imp = state_arr[self.n_sites]  # Impurity spin-down

            H[i, i] -= mu * (n_up_imp + n_down_imp)  # ← ONLY on impurity!

            # 3. Hubbard interaction on impurity site (site 0)
            H[i, i] += U * n_up_imp * n_down_imp

            # 4. Hopping terms (same as before)
            for site_i, site_j, t_val in hoppings:
                # Spin up hopping
                orbital_i_up = site_i
                orbital_j_up = site_j

                # c†_j c_i
                new_state, sign = self.fock.apply_c(state, orbital_i_up)
                if new_state is not None:
                    new_state, sign2 = self.fock.apply_c_dagger(new_state, orbital_j_up)
                    if new_state is not None and new_state in state_to_basis:
                        j_basis = state_to_basis[new_state]
                        H[j_basis, i] += -t_val * sign * sign2

                # c†_i c_j
                new_state, sign = self.fock.apply_c(state, orbital_j_up)
                if new_state is not None:
                    new_state, sign2 = self.fock.apply_c_dagger(new_state, orbital_i_up)
                    if new_state is not None and new_state in state_to_basis:
                        j_basis = state_to_basis[new_state]
                        H[j_basis, i] += -t_val * sign * sign2

                # Spin down hopping
                orbital_i_down = self.n_sites + site_i
                orbital_j_down = self.n_sites + site_j

                # c†_j c_i
                new_state, sign = self.fock.apply_c(state, orbital_i_down)
                if new_state is not None:
                    new_state, sign2 = self.fock.apply_c_dagger(new_state, orbital_j_down)
                    if new_state is not None and new_state in state_to_basis:
                        j_basis = state_to_basis[new_state]
                        H[j_basis, i] += -t_val * sign * sign2

                # c†_i c_j
                new_state, sign = self.fock.apply_c(state, orbital_j_down)
                if new_state is not None:
                    new_state, sign2 = self.fock.apply_c_dagger(new_state, orbital_i_down)
                    if new_state is not None and new_state in state_to_basis:
                        j_basis = state_to_basis[new_state]
                        H[j_basis, i] += -t_val * sign * sign2

        return H, basis_states

    def build_hamiltonian(self, params, U, mu):
        """
        Build many-body Hamiltonian in particle number conserving sector
        Uses half-filling by default: n_up = n_down = n_sites // 2

        Parameters:
        -----------
        params : array
            Chain parameters [t0, t1, t2, t3, t4, eps1, eps2, eps3, eps4, eps5]
        U : float
            Hubbard interaction (only on impurity site 0)
        mu : float
            Chemical potential

        Returns:
        --------
        H : array
            Hamiltonian matrix in the half-filling sector
        basis_states : list
            Basis state indices
        """
        # For half-filling: n_up = n_sites // 2
        # For half-filling: n_down = n_sites // 2

        n_up = self.n_sites // 2
        n_down = self.n_sites // 2

        return self.build_hamiltonian_sector(params, U, mu, n_up, n_down)

    def compute_ground_state(self, H):
        """
        Compute ground state by diagonalizing Hamiltonian

        Returns:
        --------
        E0 : float
            Ground state energy
        psi0 : array
            Ground state wavefunction
        eigenvalues : array
            All eigenvalues
        eigenvectors : array
            All eigenvectors
        """
        eigenvalues, eigenvectors = la.eigh(H)
        E0 = eigenvalues[0]
        psi0 = eigenvectors[:, 0]

        return E0, psi0, eigenvalues, eigenvectors

    def compute_green_function_time(self, params, U, mu, t_grid, T=0.0, eta=0.01):
        """
        Compute retarded Green's function in real time
        G_imp^R(t) = -i * theta(t) * <{d(t), d†(0)}>

        Parameters:
        -----------
        params : array
            Chain parameters
        U : float
            Hubbard interaction
        mu : float
            Chemical potential
        t_grid : array
            Time points
        T : float
            Temperature (0 for ground state)
        eta : float
            Broadening parameter

        Returns:
        --------
        G_imp : array
            Impurity Green's function at impurity site
        """
        # Build Hamiltonian in N-particle sector (half-filling)
        n_up = self.n_sites // 2
        n_down = self.n_sites // 2

        H_N, basis_N = self.build_hamiltonian(params, U, mu)
        state_to_basis_N = {state: i for i, state in enumerate(basis_N)}

        # Get ground state in N-particle sector
        E0, psi0, evals_N, evecs_N = self.compute_ground_state(H_N)

        # Build Hamiltonians in N-1 and N+1 particle sectors (for spin-up removal/addition)
        H_Nm1_up, basis_Nm1_up = self.build_hamiltonian_sector(params, U, mu, n_up-1, n_down)
        H_Np1_up, basis_Np1_up = self.build_hamiltonian_sector(params, U, mu, n_up+1, n_down)

        # Build Hamiltonians in N-1 and N+1 particle sectors (for spin-down removal/addition)
        H_Nm1_down, basis_Nm1_down = self.build_hamiltonian_sector(params, U, mu, n_up, n_down-1)
        H_Np1_down, basis_Np1_down = self.build_hamiltonian_sector(params, U, mu, n_up, n_down+1)

        state_to_basis_Nm1_up = {state: i for i, state in enumerate(basis_Nm1_up)}
        state_to_basis_Np1_up = {state: i for i, state in enumerate(basis_Np1_up)}
        state_to_basis_Nm1_down = {state: i for i, state in enumerate(basis_Nm1_down)}
        state_to_basis_Np1_down = {state: i for i, state in enumerate(basis_Np1_down)}

        # Diagonalize these sectors
        evals_Nm1_up, evecs_Nm1_up = la.eigh(H_Nm1_up)
        evals_Np1_up, evecs_Np1_up = la.eigh(H_Np1_up)
        evals_Nm1_down, evecs_Nm1_down = la.eigh(H_Nm1_down)
        evals_Np1_down, evecs_Np1_down = la.eigh(H_Np1_down)

        # Compute Green's function for BOTH spins
        G_imp_up = self.compute_green_spin(
            psi0, basis_N, state_to_basis_N, E0,
            0,  # orbital_imp_up = 0
            basis_Nm1_up, state_to_basis_Nm1_up, evals_Nm1_up, evecs_Nm1_up,
            basis_Np1_up, state_to_basis_Np1_up, evals_Np1_up, evecs_Np1_up,
            t_grid, eta
        )

        G_imp_down = self.compute_green_spin(
            psi0, basis_N, state_to_basis_N, E0,
            self.n_sites,  # orbital_imp_down = n_sites
            basis_Nm1_down, state_to_basis_Nm1_down, evals_Nm1_down, evecs_Nm1_down,
            basis_Np1_down, state_to_basis_Np1_down, evals_Np1_down, evecs_Np1_down,
            t_grid, eta
        )

        # Average over spins (at half-filling, should be equal by symmetry)
        G_imp = 0.5 * (G_imp_up + G_imp_down)

        return G_imp

    def compute_green_spin(self, psi0, basis_N, state_to_basis_N, E0,
                           orbital_imp,
                           basis_Nm1, state_to_basis_Nm1, evals_Nm1, evecs_Nm1,
                           basis_Np1, state_to_basis_Np1, evals_Np1, evecs_Np1,
                           t_grid, eta):
        """
        Compute Green's function for a specific spin using Lehmann representation
        """
        # Apply d (annihilation) to |psi0> (removes electron, goes to N-1 sector)
        d_psi0 = np.zeros(len(basis_Nm1), dtype=complex)
        for i, state_N in enumerate(basis_N):
            new_state, sign = self.fock.apply_c(state_N, orbital_imp)
            if new_state is not None and new_state in state_to_basis_Nm1:
                j = state_to_basis_Nm1[new_state]
                d_psi0[j] = sign * psi0[i]

        # Apply d† (creation) to |psi0> (adds electron, goes to N+1 sector)
        d_dag_psi0 = np.zeros(len(basis_Np1), dtype=complex)
        for i, state_N in enumerate(basis_N):
            new_state, sign = self.fock.apply_c_dagger(state_N, orbital_imp)
            if new_state is not None and new_state in state_to_basis_Np1:
                j = state_to_basis_Np1[new_state]
                d_dag_psi0[j] = sign * psi0[i]

        # Project onto eigenstates
        weights_hole = evecs_Nm1.T @ d_psi0  # <n, N-1| d |psi0, N>
        weights_particle = evecs_Np1.T @ d_dag_psi0  # <n, N+1| d† |psi0, N>

        # Compute Green's function using Lehmann representation
        G_imp = np.zeros(len(t_grid), dtype=complex)

        for i, t in enumerate(t_grid):
            if t > 0:
                # Particle part: <psi0| d(t) d†(0) |psi0>
                particle = np.sum(np.abs(weights_particle)**2 *
                                 np.exp(-1j * (evals_Np1 - E0) * t))

                # Hole part: <psi0| d†(0) d(t) |psi0>
                hole = np.sum(np.abs(weights_hole)**2 *
                             np.exp(1j * (evals_Nm1 - E0) * t))

                G_imp[i] = -1j * (particle + hole)

        # Apply broadening for numerical stability
        G_imp *= np.exp(-eta * t_grid)

        return G_imp
