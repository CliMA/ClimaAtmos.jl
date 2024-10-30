# Large Eddy Simulations

## Model description

`ClimaAtmos.jl` supports the `Large-eddy Simulation` (LES) configuration in doubly periodic channels through its `box` configuration, coupled with parameterizations of subgrid-scale turbulence. By default, this consists of the eddy-viscosity Smagorinsky-Lilly model.

## Equations

Given an implicit filter width ``\Delta``, computed in its simplest approximation as the geometric mean of the grid spacing in the horizontal and vertical directions, the SL model suggests that the eddy-viscosity is computed as 
```math
  \kappa_{SL} = (C_{s} \Delta)^2 \Big( sqrt(2 S_{ij}S_{ij}) \Big) f_{b}
```
where ``S_{ij} = \frac{1}{2}(\nabla u + (\nabla u)^{T}`` is the rate-of-strain tensor for velocity field ``u``, ``C_{s}`` is an empirical model parameter, and ``f_{b} `` represents the eddy-viscosity adjustment due to flow stratification, computed as 
```math
  f_{b} = 1, for N^{2} <= 0
  f_{b} = (max(0, 1 - \frac{N^2}{Pr_{t} S^{2}}))^{1/2}, for N^{2} > 0
```
``N^2`` is the moist buoyancy frequency, and ``Pr_{t}`` the turbulent Prandtl number, which takes value \frac{1}{3} under neutral stratification. Eddy diffusivities for scalar variables are then computed as ``D_{SL} = \frac{\kappa_{SL}}{Pr_{t}}``. The subgrid-scale stress tensor is given by 
```
  \tau_{ij} = -2 \kappa_{SL} S_{ij}
```
```
  d_{\chi} = - D_{SL} \nabla \chi
```
The subgrid-scale fluxes can further be decomposed into their horizontal and vertical contributions (which allows us to split up the implementation for the horizontal spectral operations and the vertical finite difference operations.)

## Diagnostics

