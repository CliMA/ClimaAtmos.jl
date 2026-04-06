# Variance adjustments (experiment notes).
#
# Core physics is in ClimaAtmos:
# - `ClimaAtmos.subcell_geometric_variance_increment` — pure (Δz, (∂q/∂z)², (∂T/∂z)²) helper
# - `materialize_sgs_quadrature_moments!` / `sgs_quadrature_Tq_moments` in `src/cache/microphysics_cache.jl`
#
# With `sgs_quadrature_subcell_geometric_variance` false, quadrature uses cached **(T′T′, q′q′)** and scalar
# `correlation_Tq(params)`. When true, variances get `(1/12)Δz²(∂·/∂z)²` terms and an effective **ρ_Tq** field
# combines turbulent covariance with a geometric cross term (see README).
