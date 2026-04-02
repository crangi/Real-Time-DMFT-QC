import os

from RTDMFT import DMFTSolver
from Utility import plot_results, compute_spectral_function_with_interpolation, plot_high_resolution_results

def main():

    U = 8.0
    t = 1.0

    print("Creating DMFT solver...")
    solver = DMFTSolver(U=U, t=t, t_max=20.0, n_time=100, eta=0.2)

    # Run DMFT
    history = solver.run(max_iter=200, tol=1e-4, mixing=0.2)

    # target folder to save results
    folder_path = "./results"
    os.makedirs(folder_path, exist_ok=True)

    # Plot DMFT convergence
    plot_results(history, save_path = os.path.join(folder_path, "dmft_results.png"))

    omega, A_lat, A_imp, G_lat_omega, Sigma, t_fine, G_fine = \
        compute_spectral_function_with_interpolation(
        solver, history,
        t_max_fine=20.0,
        n_fine=20000,
        interp_method='cubic', 
        omega_max=8.0,
        n_omega=1024,
        save_path=os.path.join(folder_path, "interpolation_results.png")
    )

# Plot high-resolution spectral functions and self-energy
    plot_high_resolution_results(solver, omega, A_lat, A_imp, Sigma, 
    save_path=os.path.join(folder_path, "high_resolution_spectral_function.png"))

    return solver, history, omega, A_lat, Sigma


if __name__ == "__main__":
    solver, history, omega, A_lat, Sigma = main()