# CalibrateAtmos.jl Notes 

## MWE 


## Comments
 - Would be great to `@assert` dimension mismatch of prior and ensemble before iteration to prevent erroring after first round of model runs.
 - Speed of different setups: M2 mac can run at 2.3 SYPD, HPC slurm can run .79 SYPD, Distributed Julia on HPC is .23 SYPD
 - Hard to dig through functions with the same name especially if the error is buried, for example, `CAL.update_ensemble()` vs `EKP.update_ensemble!()`. It's nice to have the CAL wrapper but makes it harder to debug.
 - `convergence_plot` in `postporcessing.jl` returns strange plots ... I think it's not generalized to more than one parameter.
