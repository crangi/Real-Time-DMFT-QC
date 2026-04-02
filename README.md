
# Real-Time DMFT Iteration Scheme for the Hubbard Model

This repository contains the numerical implementation of the real-time iteration scheme for **Dynamical Mean-Field Theory (DMFT)** as presented in:

> **"Real-time Dynamical Mean-Field Theory iteration scheme"**
> *Chakradhar Rangi, et al.*
> **Paper Link:** [arXiv:2601.19896](https://arxiv.org/abs/2601.19896)

## Overview

This toolkit provides a framework to solve the **half-filled Hubbard model** on a **Bethe lattice** using an Exact Diagonalization (ED) impurity solver. Unlike traditional DMFT which often operates in Matsubara frequency, this implementation performs the self-consistency loop directly in the **time domain**.

### Scientific Goals

The primary goal of this code is to investigate the Metal-Insulator Transition (MIT) and the evolution of spectral functions $A(\omega)$ for varying interaction strengths $U$. By utilizing a real-time Green's function approach, the solver can capture dynamic properties without the need for numerical analytic continuation.

## Computational Method

1.  **Impurity Solver:** Uses a 6-site chain (1 impurity + 5 bath sites) Anderson impurity model solved via Exact Diagonalization.
2.  **Lehmann Representation:** The impurity Green’s function $G_{imp}(t)$ is computed in the time domain using the Lehmann representation.
3.  **Time-Domain Fitting:** The hybridization function $\Delta(t)$ is fitted to the bath parameters through a least-squares optimization on the real-time grid.
4.  **High-Resolution Spectral Functions:** To obtain smooth spectral functions from a finite time-grid, the code employs advanced interpolation (PCHIP or Cubic Spline) before performing Fourier transforms.

## Repository Structure

  * `main.py`: The primary entry point for running simulations. It handles command-line arguments, executes the DMFT loop, and triggers the plotting routines.
  * `RTDMFT.py`: Contains the `DMFTSolver` class, which manages the self-consistency loop, hybridization initialization, and parameter fitting.
  * `ManyBodyED.py`: Implements the many-body physics, including Fock space construction, Hamiltonian building for specific particle sectors, and the ED solver.
  * `Utility.py`: Provides helper functions for data visualization, high-resolution interpolation, and spectral weight diagnostics.

## Getting Started

### Prerequisites

The code requires Python 3.8+ and the following scientific libraries:

  * `numpy`
  * `scipy`
  * `matplotlib`

### Running a Simulation

You can run the solver from the terminal by specifying the hopping amplitude ($t$) and the interaction strength ($U$).

```bash
# Example: Run DMFT for U = 8.0 and t = 1.0
python main.py 1.0 8.0 --t_max 20.0 --n_time 200
```

**Key Arguments:**

  * `t`: Hopping amplitude (sets the energy scale).
  * `U`: Coulomb interaction strength.
  * `--t_max`: Maximum simulation time (default: 20.0).
  * `--n_time`: Number of time points (default: 100).
  * `--eta`: Broadening parameter for numerical stability.

## Results and Visualization

All output files and visualizations are automatically saved to the `./results` folder. This includes:

  * `dmft_results.png`: Tracks the convergence of the error and the evolution of chain parameters ($t_i, \epsilon_i$) across iterations.
  * `interpolation_results.png`: A comparison of the raw time-domain data versus the interpolated fine grid.
  * `high_resolution_spectral_function.png`: The final lattice spectral function $A_{lat}(\omega)$ and self-energy $\Sigma(\omega)$.

## License

This project is licensed under the **MIT License**—see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## Citation

If you use this code in your research, please cite the accompanying paper:

```bibtex
@article{rangi2026real,
  title={Real-time Dynamical Mean-Field Theory iteration scheme},
  author={Rangi, Chakradhar and Tam, Ka-Ming and Moreno, Juana},
  journal={arXiv preprint arXiv:2601.19896},
  year={2026}
}
```
