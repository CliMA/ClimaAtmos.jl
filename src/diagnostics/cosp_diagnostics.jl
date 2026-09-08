# This file is included in Diagnostics.jl

compute_cloudsat_tcc(_, cache, _) = _compute_cloudsat_tcc(
    cache,
    cache.atmos.cosp,
    cache.atmos.microphysics_model,
)

_compute_cloudsat_tcc(cache, ::COSPModel, ::NonEquilibriumMicrophysics1M) =
    cache.precomputed.cloudsat_tcc

_compute_cloudsat_tcc(_, cosp, microphysics_model) = error_diagnostic_variable(
    "cloudsat_tcc requires COSPModel and NonEquilibriumMicrophysics1M; " *
    "got $(typeof(cosp)) and $(typeof(microphysics_model))",
)

add_diagnostic_variable!(
    short_name = "cloudsat_tcc",
    long_name = "CloudSat Total Cloud Cover",
    units = "%",
    comments = "Percentage of subcolumns containing reflectivity in the " *
               "inclusive -30 to 10 dBZ detection range.",
    compute = compute_cloudsat_tcc,
)

compute_cloudsat_tcc2(_, cache, _) = _compute_cloudsat_tcc2(
    cache,
    cache.atmos.cosp,
    cache.atmos.microphysics_model,
)

_compute_cloudsat_tcc2(cache, ::COSPModel, ::NonEquilibriumMicrophysics1M) =
    cache.precomputed.cloudsat_tcc2

_compute_cloudsat_tcc2(_, cosp, microphysics_model) = error_diagnostic_variable(
    "cloudsat_tcc2 requires COSPModel and NonEquilibriumMicrophysics1M; " *
    "got $(typeof(cosp)) and $(typeof(microphysics_model))",
)

add_diagnostic_variable!(
    short_name = "cloudsat_tcc2",
    long_name = "CloudSat Total Cloud Cover Excluding the Lowest 1 km",
    units = "%",
    comments = "Percentage of subcolumns containing reflectivity in the " *
               "inclusive -30 to 10 dBZ detection range above 1 km from the " *
               "local surface.",
    compute = compute_cloudsat_tcc2,
)
