# Real-Time DMFT: A Framework for Near-Term Quantum Simulation
This repository provides the numerical implementation of a real-time iteration scheme for Dynamical Mean-Field Theory (DMFT), serving as a framework for near-term quantum simulation of the Hubbard model.

Developed as supporting code for:
> **"Real-Time Iteration Scheme for Dynamical Mean-Field Theory: A Framework for
Near-Term Quantum Simulation"**
> *Chakradhar Rangi, et al.*
> **Paper Link:** [arXiv:2601.19896](https://arxiv.org/abs/2601.19896)

## Overview

This toolkit provides a framework to solve the **half-filled Hubbard model** on a **Bethe lattice** using an Exact Diagonalization (ED) impurity solver. Unlike traditional DMFT which often operates in Matsubara frequency, this implementation performs the self-consistency loop directly in the **time domain**.

Quantum Advantage Context: Real-time Green's functions are a "natural" output for quantum computers, as time-evolution operators ($e^{−iHt}$) can be directly implemented via Trotterization or variational circuits. This repository serves as the classical backbone for such hybrid quantum-classical workflows.

### Scientific Goals

- Real-Time Dynamics: Capture the Metal-Insulator Transition (MIT) and spectral evolution directly through real-time Green's functions.

- NISQ-Ready Mapping: Provide a template for mapping the 6-site impurity problem (1 impurity + 5 bath sites) onto quantum registers.

- Benchmarking: Establish high-precision classical results for spectral functions $A(ω)$ to validate future near-term quantum simulations.


## Computational Method

1.  **Impurity Solver:** A 6-site Anderson impurity model solved via Exact Diagonalization (ED). In a quantum workflow, this ED step is the primary candidate for replacement by a quantum sub-routine.
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
