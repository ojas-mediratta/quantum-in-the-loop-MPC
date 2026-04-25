# Quantum-in-the-loop MPC

Project files for my ECE 8803: Quantum Devices and Hardware final project.

This project explores a small quantum-in-the-loop control example where a discrete model predictive control (MPC) problem is converted into a quadratic unconstrained binary optimization (QUBO) problem and solved using QAOA in Qiskit.

## Overview

The demo uses a simple 1D point robot with position and velocity states. At each timestep, the robot chooses from a discrete set of acceleration commands:

```math
a_k \in \{-1, 0, 1\}
```

The project compares:

- brute-force discrete MPC
- exact QUBO solving
- QAOA-based approximate solving

The main goal is to demonstrate the full pipeline from a robotics MPC problem to a quantum optimization formulation.

## Files

- `mpc_qaoa_demo.py` — main Python demo
- `requirements.txt` / `environment.yml` — environment setup files, if included
- `figures/` — generated plots and visual results
- `report/` — final report files, if included

## Method

The implementation follows these steps:

1. Define a finite-horizon MPC problem for a 1D robot.
2. Encode each discrete control action using one-hot binary variables.
3. Add one-hot constraints as QUBO penalty terms.
4. Convert the MPC objective into a binary quadratic objective.
5. Solve the problem using classical baselines and QAOA.
6. Decode the resulting bitstring back into robot control actions.

## Running

Install the required Python packages, then run:

```bash
python mpc_qaoa_demo.py
```

The script prints optimization results and generates plots comparing the classical and QAOA-based controllers.

## Notes

This project is intended as an educational proof of concept. It is not meant to demonstrate near-term quantum advantage. For the small test problem, classical methods are faster and more practical, but the example shows how discrete MPC can be mapped into a QUBO form suitable for QAOA.
