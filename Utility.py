import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator

from RTDMFT import DMFTSolver

def plot_results(history, save_path="dmft_results.png"):
    """
    Plot DMFT results
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 15))

    # Plot 1: Convergence
    ax = axes[0]
    ax.semilogy(range(1, len(history['error'])+1), history['error'], 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration', fontsize=15)
    ax.set_ylabel('Error', fontsize=15)
    ax.set_title('DMFT Convergence', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Parameter evolution
    ax = axes[1]
    params_history = np.array(history['params'])

    # Plot first 5 parameters (chain parameters)
    labels = [r'$t_0$', r'$t_1$', r'$t_2$', r'$\varepsilon_1$', r'$\varepsilon_2$']
    indices = [0, 1, 2, 5, 6]  # t0, t1, t2, eps1, eps2

    for idx, label in zip(indices, labels):
        ax.plot(range(1, len(params_history)+1), params_history[:, idx],
                'o-', label=label, linewidth=2, markersize=5)

    ax.set_xlabel('Iteration', fontsize=15)
    ax.set_ylabel('Parameter value', fontsize=15)
    ax.set_title('Chain Parameter Evolution', fontsize=20, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print("\n" + "="*60)
    print(f"Results saved to {save_path}")
    print("="*60)
    plt.show()

def plot_high_resolution_results(solver, omega, A_lat, A_imp, Sigma_omega, save_path="high_resolution_spectral_function.png"):
    """
    Plot results from high-resolution calculation
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 15))

    omega_max_plot = 8.0
    mask = (omega >= -omega_max_plot) & (omega <= omega_max_plot)

    # Plot 1: Lattice spectral function
    ax = axes[0]
    ax.plot(omega[mask], A_lat[mask], 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
#    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=2, label='μ = 0')
#    if solver.U > 0:
#        ax.axvline(x=-solver.U/2, color='orange', linestyle=':', alpha=0.7, linewidth=2,
#                   label=f'Expected LHB ≈ {-solver.U/2:.1f}')
#        ax.axvline(x=solver.U/2, color='red', linestyle=':', alpha=0.7, linewidth=2,
#                   label=f'Expected UHB ≈ {solver.U/2:.1f}')
    ax.set_xlabel('Frequency ω', fontsize=15)
    ax.set_ylabel('$A_{lat}(ω)$', fontsize=15)
    ax.set_title(f'Lattice Spectral Function (U/t = {solver.U/solver.t:.2f})',
                 fontsize=20, fontweight='bold')
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)

    # Plot 2: Self-energy
    ax = axes[1]
    ax.plot(omega[mask], Sigma_omega[mask].real, 'b-', linewidth=2, label='Re Σ(ω)')
    ax.plot(omega[mask], Sigma_omega[mask].imag, 'r-', linewidth=2, label='Im Σ(ω)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency ω', fontsize=15)
    ax.set_ylabel('Σ(ω)', fontsize=15)
    ax.set_title('Self-Energy', fontsize=20, fontweight='bold')
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)

    # Check high-frequency
#    mask_high = np.abs(omega) > 5
#    Sigma_high = np.mean(np.abs(Sigma_omega[mask_high]))
#    color = 'lightgreen' if Sigma_high < 0.3 else ('yellow' if Sigma_high < 1 else 'salmon')
#    ax.text(0.5, 0.95, f'|Σ|(|?w|>5) ≈ {Sigma_high:.4f}',
#           transform=ax.transAxes, fontsize=9, ha='center', va='top',
#           bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Diagnostics
    print("\n" + "="*60)
    print("HIGH-RESOLUTION RESULTS")
    print("="*60)

    dw = omega[1] - omega[0]
    weight_lat = np.trapz(A_lat, omega)
    weight_imp = np.trapz(A_imp, omega)

    print(f"\n1. Spectral weights:")
    print(f"   ∫ A_lat dω = {weight_lat:.6f} (should be ~1)")
    print(f"   ∫ A_imp dω = {weight_imp:.6f}")

    mu_idx = np.argmin(np.abs(omega))
    print(f"\n2. At Fermi level (ω=0):")
    print(f"   A_lat(0) = {A_lat[mu_idx]:.6f}")
    print(f"   A_imp(0) = {A_imp[mu_idx]:.6f}")
    print(f"   Σ(0) = {Sigma_omega[mu_idx].real:.6f} + {Sigma_omega[mu_idx].imag:.6f}i")

    print(f"\n3. High-frequency behavior (|?w| > 5):")
    mask_high = np.abs(omega) > 5
    Sigma_high_re = np.mean(np.abs(Sigma_omega[mask_high].real))
    Sigma_high_im = np.mean(np.abs(Sigma_omega[mask_high].imag))
    print(f"   <|Re Σ|> = {Sigma_high_re:.6f} (should → 0)")
    print(f"   <|Im Σ|> = {Sigma_high_im:.6f} (should → 0)")

    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(A_lat[mask], height=np.max(A_lat[mask])*0.1, distance=20)
    if len(peaks) > 0:
        print(f"\n4. Spectral peaks:")
        for peak_idx in peaks[:5]:
            actual_idx = np.where(mask)[0][peak_idx]
            print(f"   ω = {omega[actual_idx]:+.4f}, A = {A_lat[actual_idx]:.4f}")

    print("="*60)

def interpolate_to_fine_grid(t_grid_coarse, G_coarse, t_max_fine=40.0, n_fine=20000,
                             method='pchip', save_path="interpolation_results.png"):
    """
    Interpolate time-domain Green's function to a finer grid

    Parameters:
    -----------
    t_grid_coarse : array
        Original time grid
    G_coarse : array (complex)
        Green's function on coarse grid
    t_max_fine : float
        Maximum time for fine grid
    n_fine : int
        Number of points in fine grid
    method : str
        'pchip', 'cubic', 'linear', or 'fourier'

    Returns:
    --------
    t_grid_fine : array
        Fine time grid
    G_fine : array (complex)
        Interpolated Green's function
    """
    print(f"\nInterpolating from {len(t_grid_coarse)} to {n_fine} points...")
    print(f"Method: {method}")

    # Create fine grid
    t_grid_fine = np.linspace(0, t_max_fine, n_fine)

    # Only interpolate within the original time range
    # For times beyond, extrapolate using exponential decay
    t_max_coarse = t_grid_coarse[-1]

    if method == 'pchip':
        # PCHIP is shape-preserving, good for decaying functions
        # Interpolate real and imaginary parts separately
        interp_re = PchipInterpolator(t_grid_coarse, G_coarse.real)
        interp_im = PchipInterpolator(t_grid_coarse, G_coarse.imag)

        # Interpolate within original range
        mask_interp = t_grid_fine <= t_max_coarse
        G_fine = np.zeros(n_fine, dtype=complex)
        G_fine[mask_interp] = interp_re(t_grid_fine[mask_interp]) + 1j * interp_im(t_grid_fine[mask_interp])

        # Extrapolate beyond using exponential decay
        if np.any(~mask_interp):
            # Fit exponential decay to last few points
            n_fit = min(50, len(t_grid_coarse) // 4)
            t_fit = t_grid_coarse[-n_fit:]
            G_fit = G_coarse[-n_fit:]

            # Fit: |G(t)| ~ exp(-gamma * t)
            log_abs_G = np.log(np.abs(G_fit) + 1e-15)
            fit = np.polyfit(t_fit, log_abs_G, 1)
            gamma = -fit[0]

            # Extrapolate
            t_extrap = t_grid_fine[~mask_interp]
            decay_factor = np.exp(-gamma * (t_extrap - t_max_coarse))
            G_fine[~mask_interp] = G_fine[mask_interp][-1] * decay_factor

    elif method == 'cubic':
        # Cubic spline interpolation
        interp_re = CubicSpline(t_grid_coarse, G_coarse.real, bc_type='natural')
        interp_im = CubicSpline(t_grid_coarse, G_coarse.imag, bc_type='natural')

        mask_interp = t_grid_fine <= t_max_coarse
        G_fine = np.zeros(n_fine, dtype=complex)
        G_fine[mask_interp] = interp_re(t_grid_fine[mask_interp]) + 1j * interp_im(t_grid_fine[mask_interp])

        # Extrapolate
        if np.any(~mask_interp):
            n_fit = min(50, len(t_grid_coarse) // 4)
            t_fit = t_grid_coarse[-n_fit:]
            G_fit = G_coarse[-n_fit:]
            log_abs_G = np.log(np.abs(G_fit) + 1e-15)
            fit = np.polyfit(t_fit, log_abs_G, 1)
            gamma = -fit[0]
            t_extrap = t_grid_fine[~mask_interp]
            decay_factor = np.exp(-gamma * (t_extrap - t_max_coarse))
            G_fine[~mask_interp] = G_fine[mask_interp][-1] * decay_factor

    elif method == 'linear':
        # Simple linear interpolation
        interp_re = interp1d(t_grid_coarse, G_coarse.real, kind='linear',
                            fill_value='extrapolate')
        interp_im = interp1d(t_grid_coarse, G_coarse.imag, kind='linear',
                            fill_value='extrapolate')
        G_fine = interp_re(t_grid_fine) + 1j * interp_im(t_grid_fine)

    elif method == 'fourier':
        # Fourier-based interpolation (assumes bandlimited signal)
        # This is more sophisticated but can be very accurate

        # Pad to power of 2 for efficiency
        n_pad = 2**int(np.ceil(np.log2(len(G_coarse))))
        G_padded = np.zeros(n_pad, dtype=complex)
        G_padded[:len(G_coarse)] = G_coarse

        # FFT
        G_freq = np.fft.fft(G_padded)

        # Zero-pad in frequency domain (increases time resolution)
        n_pad_fine = int(n_pad * n_fine / len(t_grid_coarse))
        G_freq_fine = np.zeros(n_pad_fine, dtype=complex)

        # Copy positive frequencies
        G_freq_fine[:n_pad//2] = G_freq[:n_pad//2]
        # Copy negative frequencies
        G_freq_fine[-n_pad//2:] = G_freq[-n_pad//2:]

        # Inverse FFT
        G_fine_full = np.fft.ifft(G_freq_fine) * (n_pad_fine / n_pad)

        # Extract the desired range
        t_fine_max_idx = int(n_fine * t_max_fine / t_max_coarse)
        G_fine = G_fine_full[:n_fine]
        t_grid_fine = np.linspace(0, t_max_fine, n_fine)

    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    print(f"Interpolation complete!")
    print(f"  Original dt = {t_grid_coarse[1] - t_grid_coarse[0]:.6f}")
    print(f"  Fine dt = {t_grid_fine[1] - t_grid_fine[0]:.6f}")
    print(f"  Improvement: {(t_grid_coarse[1] - t_grid_coarse[0])/(t_grid_fine[1] - t_grid_fine[0]):.1f}x")

    plot_comparison = True

    # PLOT COMPARISON
    if plot_comparison:
        fig, axes = plt.subplots(2, 1, figsize=(10, 15))

        # Determine plotting range (show first half of time for clarity)
        t_plot_max = min(t_max_coarse / 2, 15.0)
        mask_coarse = t_grid_coarse <= t_plot_max
        mask_fine = t_grid_fine <= t_plot_max

        # Plot 2: Imaginary part
        ax = axes[0]
        ax.plot(t_grid_fine[mask_fine], G_fine[mask_fine].imag, 'b-',
                linewidth=1.5, alpha=0.7, label=f'Interpolated ({method})')
        ax.plot(t_grid_coarse[mask_coarse], G_coarse[mask_coarse].imag, 'ro',
                markersize=4, alpha=0.8, label='Original data')
        ax.set_xlabel('Time t', fontsize=15)
        ax.set_ylabel('Im[G(t)]', fontsize=15)
        ax.set_title('Imaginary Part', fontsize=15, fontweight='bold')
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.3)

        # Plot 3: Magnitude (log scale)
        ax = axes[1]
        ax.semilogy(t_grid_fine[mask_fine], np.abs(G_fine[mask_fine]), 'b-',
                    linewidth=1.5, alpha=0.7, label=f'Interpolated ({method})')
        ax.semilogy(t_grid_coarse[mask_coarse], np.abs(G_coarse[mask_coarse]), 'ro',
                    markersize=4, alpha=0.8, label='Original data')
        ax.set_xlabel('Time t', fontsize=15)
        ax.set_ylabel('|G(t)|', fontsize=15)
        ax.set_title('Magnitude (Log Scale)', fontsize=20, fontweight='bold')
        ax.legend(fontsize=15)
        ax.grid(True, alpha=0.3)


        plt.suptitle(f'Interpolation Comparison: {method.upper()} Method',
                    fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_path}', dpi=150)
        plt.show()


    return t_grid_fine, G_fine

def compute_spectral_function_with_interpolation(solver, history,
                                                 t_max_fine=20.0, n_fine=20000,
                                                 interp_method='cubic',
                                                 omega_max=10.0, n_omega=4096, save_path="interpolation_results.png"):
    """
    Compute spectral functions using interpolation for finer time grid

    This is much faster than recomputing G_imp(t) on a fine grid!
    """
    print("\n" + "="*60)
    print("SPECTRAL FUNCTION WITH INTERPOLATION")
    print("="*60)

    # Get coarse-grid data
    t_grid_coarse = solver.t_grid
    G_imp_coarse = history['G_imp'][-1]
    Delta_coarse = history['Delta'][-1]

    #####WARNING, this is strictly for half-filling only####
#    G_imp_coarse = np.imag(G_imp_coarse) * complex(0,1)
#    Delta_coarse = np.real(Delta_coarse)

    # Force G_imp^R(t) to be purely imaginary
    G_imp_coarse = 1j * G_imp_coarse.imag

    # Force Δ^R(t) to be purely real
#    Delta_coarse = Delta_coarse.real + 0j
    Delta_coarse = 1j * Delta_coarse.imag

    print(f"\nOriginal grid: {len(t_grid_coarse)} points, t_max = {t_grid_coarse[-1]:.1f}")
    print(f"Target grid:   {n_fine} points, t_max = {t_max_fine:.1f}")

    # Interpolate G_imp
    print("\nInterpolating G_imp(t)...")
    t_grid_fine, G_imp_fine = interpolate_to_fine_grid(
        t_grid_coarse, G_imp_coarse, t_max_fine, n_fine, method=interp_method, save_path=save_path
    )

    # # Interpolate Δ
    # print("\nInterpolating Δ(t)...")
    # _, Delta_fine = interpolate_to_fine_grid(
    #     t_grid_coarse, Delta_coarse, t_max_fine, n_fine, method=interp_method
    # )

    # Now compute spectral functions with fine grid
    print("\nComputing spectral functions...")

    dt_fine = t_grid_fine[1] - t_grid_fine[0]
    omega_grid = np.linspace(-omega_max, omega_max, n_omega)

    # Fourier transforms
    print("  Fourier transforming...")
    G_imp_omega = np.zeros(n_omega, dtype=complex)
    Delta_omega = np.zeros(n_omega, dtype=complex)

    for i, w in enumerate(omega_grid):
        z = w + 1j * solver.eta

        integrand_G = G_imp_fine * np.exp(1j * z * t_grid_fine)
        G_imp_omega[i] = np.trapz(integrand_G, t_grid_fine)

        # integrand_Delta = Delta_fine * np.exp(1j * z * t_grid_fine)
        # Delta_omega[i] = np.trapz(integrand_Delta, t_grid_fine)

    # Self-energy
    print("  Computing self-energy...")
    Sigma_omega = np.zeros(n_omega, dtype=complex)

    for i, w in enumerate(omega_grid):
        z = w + 1j * solver.eta
        if np.abs(G_imp_omega[i]) > 1e-10:
            Sigma_omega[i] = (z - Delta_omega[i]) - 1.0 / G_imp_omega[i]
        else:
            Sigma_omega[i] = 0.0


    # Lattice Green's function
    print("  Computing lattice Green's function...")
    G_lat_omega = np.zeros(n_omega, dtype=complex)

    for i, w in enumerate(omega_grid):
        z = w + 1j * solver.eta
        z_eff = z - Sigma_omega[i]

        discriminant = z_eff**2 - (2*solver.t)**2
        sqrt_disc = np.sqrt(discriminant)

        G_plus = (z_eff - sqrt_disc) / (2 * solver.t**2)
        G_minus = (z_eff + sqrt_disc) / (2 * solver.t**2)

        if G_plus.imag < 0:
            G_lat_omega[i] = G_plus
        elif G_minus.imag < 0:
            G_lat_omega[i] = G_minus
        else:
            G_lat_omega[i] = G_plus

    # Spectral functions
    A_lat = -1.0 / np.pi * G_lat_omega.imag
    A_imp = -1.0 / np.pi * G_imp_omega.imag

    print("\nDone!")
    print("="*60)

    return omega_grid, A_lat, A_imp, G_lat_omega, Sigma_omega, t_grid_fine, G_imp_fine #, Delta_fine

def compare_interpolation_methods(solver, history):
    """
    Compare different interpolation methods
    """
    methods = ['linear', 'pchip', 'cubic']
    results = {}

    print("\n" + "="*60)
    print("COMPARING INTERPOLATION METHODS")
    print("="*60)

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")
        print(f"{'='*60}")

        omega, A_lat, A_imp, G_lat_omega, Sigma_omega, t_fine, G_fine, Delta_fine = \
            compute_spectral_function_with_interpolation(
                solver, history,
                t_max_fine=40.0,
                n_fine=4000,
                interp_method=method, # Recommended: shape-preserving
                omega_max=10.0,
                n_omega=4096
            )

        results[method] = {
            'omega': omega,
            'A_lat': A_lat,
            'A_imp': A_imp,
            'Sigma': Sigma_omega
        }

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    omega_plot = results['pchip']['omega']
    mask = (omega_plot >= -10) & (omega_plot <= 10)

    # Plot 1: A_lat for all methods
    ax = axes[0, 0]
    for method in methods:
        ax.plot(omega_plot[mask], results[method]['A_lat'][mask],
               linewidth=2, label=method, alpha=0.8)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    if solver.U > 0:
        ax.axvline(x=-solver.U/2, color='orange', linestyle=':', alpha=0.5)
        ax.axvline(x=solver.U/2, color='red', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequency ω')
    ax.set_ylabel('$A_{lat}(ω)$')
    ax.set_title('Lattice Spectral Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Σ(ω) real part
    ax = axes[0, 1]
    for method in methods:
        ax.plot(omega_plot[mask], results[method]['Sigma'][mask].real,
               linewidth=2, label=method, alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Frequency ω')
    ax.set_ylabel('Re Σ(ω)')
    ax.set_title('Self-Energy (Real Part)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Σ(ω) imaginary part
    ax = axes[1, 0]
    for method in methods:
        ax.plot(omega_plot[mask], results[method]['Sigma'][mask].imag,
               linewidth=2, label=method, alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Frequency ω')
    ax.set_ylabel('Im Σ(ω)')
    ax.set_title('Self-Energy (Imaginary Part)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: High-frequency Σ
    ax = axes[1, 1]
    mask_high = np.abs(omega_plot) > 5
    for method in methods:
        Sigma_high_mag = np.abs(results[method]['Sigma'][mask_high])
        avg = np.mean(Sigma_high_mag)
        ax.axhline(y=avg, linewidth=2, label=f'{method}: {avg:.4f}', linestyle='--')
    ax.set_xlabel('Method')
    ax.set_ylabel('$<|\u03A3|>$ for $|?w|>5$')
    ax.set_title('High-Frequency Self-Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, None])

    plt.tight_layout()
    plt.savefig('interpolation_comparison.png', dpi=150)
    plt.show()

    # Print comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    for method in methods:
        mask_high = np.abs(omega_plot) > 5
        Sigma_high = np.mean(np.abs(results[method]['Sigma'][mask_high]))
        weight = np.trapz(results[method]['A_lat'], omega_plot)

        print(f"\n{method.upper()}:")
        print(f"  |\u03A3|(|w|>5) = {Sigma_high:.6f}")
        print(f"  \u222B A_lat d\u03C9 = {weight:.6f}")

    print("="*60)

    return results


if __name__ == "__main__":

    U = 8.0
    t = 1.0

    print("Creating DMFT solver...")
    solver = DMFTSolver(U=U, t=t, t_max=20.0, n_time=100, eta=0.2)

    # Run DMFT
    history = solver.run(max_iter=200, tol=1e-4, mixing=0.2)

    results = compare_interpolation_methods(solver, history)
