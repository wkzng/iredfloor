# Irreducible Loss Floors in Gradient-Based Optimization and Energy Footprint

This repository provides the official code and reproducible experiments accompanying the paper **"Irreducible Loss Floors in Gradient-Based Optimization and Energy Footprint"**. It implements the numerical estimation of lower bounds on the training loss under idealized convergence assumptions, and illustrates the framework's behavior across various learning problems.


## Experiment 1:
Goal: Verify the validity of the SGD ODE
Result: `experiments/experiment_GD_wideMLP_2000.csv`
<img src="experiments/experiment_GD_wideMLP_2000.png" alt="non-causal 15033000" width="260">

Observation: The solution of the ODE computed with the measured rate of decay match with the exact losses observed on the train set.


## Citing

````
@misc{wkzng2025iredfloor,
  title={Irreducible Loss Floors in Gradient-Based Optimization and Energy Footprint},
  author  = {Anonymous Author},
  year    = {2025},
  eprint  = {arXiv:2506.xxxxx},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note    = {Under review at NeurIPS 2025}
}
````