# Variance adjustments (experiment notes).
#
# Core physics is in ClimaAtmos:
# - `ClimaAtmos.subcell_geometric_variance_increment` — pure (Δz, (∂q/∂z)², (∂T/∂z)²) helper
# - `sgs_quadrature_moments_from_gradients` in `src/parameterized_tendencies/microphysics/sgs_quadrature.jl`
#   (uses `src/utils/variance_statistics.jl` for pure geometry kernels)
#
# With base `sgs_distribution` (not `*_gridscale_corrected`), quadrature uses cached **(T′T′, q′q′)** and scalar
# `correlation_Tq(params)`. When true, variances get `(1/12)Δz²(∂·/∂z)²` terms and an effective **ρ_Tq** field
# combines turbulent covariance with a geometric cross term (see README).
#
# Experiment drivers: `experiment_common.jl`, `model_interface.jl`, `observation_map.jl`, README.
