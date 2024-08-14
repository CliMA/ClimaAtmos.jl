# CalibrateAtmos.jl Notes 

## Notes from meeting
Let's redefine our observation map to get sensible column integrals for calibration. 

For column vertical water (hus) we'll do a reweighting: $ HUS = \int_0^\infty \rho q_v dz = \sum_i \rho_i q_{v_i} (\Delta z)_i$ where $q_v$ is `hus` at gridpoints. For this we'll use the trapezoid rule for the densities. For column relative humidity we'll again weight $q_v$ (note using `hus` again not `hur`) and so we get the following: $HUR = \frac{\int \rho q_v dz}{\int \rho q_v^* dz}$ where $q_v^*$ is the saturation specific humidity which is a function of temperature and pressure and can be computed using `Thermodynamics.jl`. 

Clouds are more simple. Here we'll just do a density reweighting: $CLW = \int \rho q_l dz$ and $CL = \int_0^{4km} cl dz$

We're going to replace temperature (`thetaa`) with dry static energy, recalling that $MSE = \underbrace{c_p\cdot T + gz}_\text{DSE} + L q_v$, ignoring $Lq_v$ since this is accounted for in `HUR` and `HUS` integrated values. We'll compute DSE for each of the column values and then again take the column integral, weighting based on the spacing: e.g. $\int \rho DSE dz$


## MWE 


## Comments
 - Would be great to `@assert` dimension mismatch of prior and ensemble before iteration to prevent erroring after first round of model runs.
 - Speed of different setups: M2 mac can run at 2.3 SYPD, HPC slurm can run .79 SYPD, Distributed Julia on HPC is .23 SYPD
 - Hard to dig through functions with the same name especially if the error is buried, for example, `CAL.update_ensemble()` vs `EKP.update_ensemble!()`. It's nice to have the CAL wrapper but makes it harder to debug.
 - `convergence_plot` in `postporcessing.jl` returns strange plots ... I think it's not generalized to more than one parameter.
