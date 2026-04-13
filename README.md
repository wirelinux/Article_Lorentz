# Exploration of Poincaré Map in the Lorenz Attractor

[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.xxxxxxx-blue.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)

This repository contains the complete Python source code for the article:

> Exploration of Poincaré Map in the Lorenz Attractor 
> *Submitted for publication.*

The code implements an integrated computational framework for the analysis of the Lorenz system, including:

Code 1 -  Bifurcation diagram via dynamic Poincaré section (z = ρ – 1)

Code 2 -  Spectrum of Lyapunov exponents using Householder QR decomposition

Code 3 & Code 5 - 3D visualization of the attractor and its sections 

Code 4 & Code 6 - 2D Poincaré maps

Code 7 - One-dimensional return map (z maxima)

All implementations are optimized with parallelization (`multiprocessing`), just‑in‑time compilation (`numba`), adaptive parameter sampling, and high‑precision interpolation for crossing detection.
