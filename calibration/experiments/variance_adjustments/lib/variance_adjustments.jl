# Variance adjustments (experiment notes).
#
# Core physics is in ClimaAtmos:
# - `ClimaAtmos.subcell_geometric_variance_increment(Δz, s_q_dn, s_q_up, s_T_dn, s_T_up)` —
#   pure two-slope helper returning the Δq, ΔT variance increments from the
#   four face-anchored half-slopes.
# - `ClimaAtmos.subcell_geometric_covariance_Tq` / `ClimaAtmos.subcell_layer_mean_excursion` —
#   same file (`src/utils/variance_statistics.jl`), same two-slope signature.
# - `ClimaAtmos.integrate_over_sgs_linear_profile` in
#   `src/parameterized_tendencies/microphysics/subgrid_layer_profile_quadrature.jl` —
#   the end-to-end layer-mean quadrature driver (column-tensor and
#   `SubgridProfileRosenblatt`: composite split inner marginal (½ DN / ½ UP half-cell
#   `uniform⊛Gaussian` laws); default per-leg inverse via `ConvolutionQuantilesHalley`).
#
# With a base `sgs_distribution` (not `(gaussian|lognormal)_vertical_profile*`), quadrature uses cached
# **(T′T′, q′q′)** and scalar `correlation_Tq(params)`. With a vertical-profile
# distribution, variances receive the two-slope `(1/12) d² Δz² + (1/192) D² Δz²`
# increments and an effective **ρ_Tq** field combines turbulent covariance with a
# geometric cross term (implemented in `src/utils/variance_statistics.jl`).
#
# Experiment drivers: `experiment_common.jl`, `model_interface.jl`, `observation_map.jl`, README.
