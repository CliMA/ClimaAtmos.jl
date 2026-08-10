"""
Assembling the `EnsembleKalmanProcess`.

Kept separate from the driver so the EKP object can be built, inspected, and its observation
lengths checked without starting a calibration.
"""

using EnsembleKalmanProcesses: EnsembleKalmanProcesses as EKP
using Random: Random

"""
    build_ekp(interface, prior; ensemble_size, T_stops, on_terminate, rng, kwargs...)

The `EnsembleKalmanProcess` for this interface: the prior's initial ensemble against the
observation series for all cases, with a `DataMisfitController` scheduler.

`Inversion` carries EKP's `SECNice` localization and `NesterovAccelerator` by default. Both matter here:
with ~10 members against ~11,000 observations the ensemble covariance has rank ≤ 9. `TransformInversion`
would make the linear algebra far cheaper but works entirely in the ensemble subspace, so it supports
neither — and the dense solve is not the bottleneck against `N_ens × 11` column runs per iteration.

`terminate_at` starts at the *first* entry of `T_stops`; the driver advances it through the
remaining stops (see [`ratchet_terminate_at`](@ref)). `on_terminate = "continue"` leaves stopping to
the driver.
"""
function build_ekp(
    interface::SocratesInterface,
    prior;
    ensemble_size::Int,
    T_stops = nothing,
    on_terminate::AbstractString = "continue",
    rng = Random.MersenneTwister(1234),
    verbose::Bool = true,
    kwargs...,
)
    ensemble_size >= 2 ||
        error("ensemble_size must be at least 2, got $ensemble_size")
    series = observation_series(interface)
    terminate_at = isnothing(T_stops) ? 1.0 : Float64(first(T_stops))
    ekp = EKP.EnsembleKalmanProcess(
        EKP.construct_initial_ensemble(rng, prior, ensemble_size),
        series,
        EKP.Inversion();
        scheduler = EKP.DataMisfitController(; terminate_at, on_terminate),
        rng,
        verbose,
        kwargs...,
    )
    check_observation_lengths(interface, ekp)
    return ekp
end

"""
    check_observation_lengths(interface, ekp)

Verify that the stacked observation length equals the sum of the per-variable metadata lengths, and
report the per-case block sizes.

Observations on one vertical grid against model output on another produce an all-`NaN` `G`. Checking at
build time catches that before any forward model runs, rather than after a whole iteration.
"""
function check_observation_lengths(interface::SocratesInterface, ekp)
    series = EKP.get_observation_series(ekp)
    obs_len = length(EKP.get_obs(ekp))
    metadata = ClimaCalibrate.get_metadata_for_nth_iteration(series, 1)
    meta_len = sum(ClimaAnalysis.flattened_length, metadata)
    obs_len == meta_len || error(
        "Observation length ($obs_len) does not equal the summed per-variable metadata length \
         ($meta_len). `GEnsembleBuilder` fills by metadata range, so these must agree.",
    )
    per_case = [
        length(first(EKP.get_samples(o))) for o in EKP.get_observations(series)
    ]
    @info "Observations built" n_cases = length(per_case) total_length = obs_len per_case_length =
        per_case n_variables = length(interface.vars)
    return nothing
end

"""
    observation_block_report(interface)

Per-case, per-variable observation dimensions and normalizers, for inspection before a run.

Returns a vector of `(; case, length, blocks, pool_vars)` where `blocks` maps each variable to its
row range inside that case's stacked vector.
"""
function observation_block_report(interface::SocratesInterface)
    return map(interface.cases) do case
        ref = normalized_reference_series(
            case;
            transform = interface.transform,
            source = interface.reference_source,
            vars = interface.vars,
        )
        (;
            case = SS.case_name(case),
            length = size(ref.series, 1),
            n_times = size(ref.series, 2),
            blocks = ref.ranges,
            pool_vars = ref.pool_vars,
        )
    end
end
