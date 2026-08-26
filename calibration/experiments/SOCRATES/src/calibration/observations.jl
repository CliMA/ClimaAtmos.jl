"""
Atlas LES observations as `EKP.Observation`s.

One `Observation` per case, holding the normalized, window-averaged reference profile stack and its
time covariance. The layout is deliberate:

  - **One samples block and one covariance block per case.** `EKP.get_obs_noise_cov` is block-diagonal
    across whatever blocks an `Observation` is given, so splitting a case into per-variable blocks would
    discard the cross-variable covariance. Every variable is row-stacked into one
    `(Σ N_z + N_scalar, N_t)` matrix and the covariance taken over it, held as an `EKP.SVDplusD` rather
    than a dense matrix since with a few dozen snapshots that covariance is singular by construction.
  - **A vector of per-variable `ClimaAnalysis.Metadata` alongside that single block.** No trade-off
    is involved: `GEnsembleBuilder` derives its fill ranges from the metadata's flattened lengths
    rather than from `Observation.indices`, so the full within-case covariance and per-variable
    validated filling coexist.
  - **Block-diagonal across cases**, imposed by EKP and physically right: separate flights are
    separate LES runs of separate days with no shared sample dimension, so there is no cross-case
    covariance to estimate.

`pool_var` rides in each variable's attributes, so the normalization travels *inside* the
observation and cannot desynchronize from it.
"""

using EnsembleKalmanProcesses: EnsembleKalmanProcesses as EKP
using LinearAlgebra: LinearAlgebra
using Statistics: Statistics

"""
    normalized_reference_series(case; transform, source, vars, window, z_top, z_grid)

The reference for `case` over the scoring window, resampled onto `z_grid`, restricted to
`z ≤ z_top`, and normalized: a `(n_rows, n_times)` matrix stacking every variable row-wise, plus
the per-variable `pool_var` and row ranges.

The stacked matrix is what both `y` (its time mean) and `Γ` (its time covariance) are formed from,
so they live in the same normalized space by construction.
"""
function normalized_reference_series(
    case::SS.SocratesCase;
    transform::SS.ScoreTransform = SS.ScoreTransform(),
    source::Symbol = :processed,
    vars = SS.REFERENCE_VARS,
    window = SS.SM.score_window(case),
    bounds = SS.z_bounds(case; source),
    # The levels must be the ones the model will actually write, so the grid is built in the float
    # type the model runs in and then widened; a Float32 column's levels are not their Float64 values.
    float_type::Type{<:AbstractFloat} = Float64,
    grid = SS.SM.socrates_grid(float_type, case),
    z_grid = collect(Float64, SS.SM.socrates_z(grid)),
)
    reference = SS.les_outputvars(case; source, vars)
    rows = Vector{Matrix{Float64}}()
    pool_vars = Dict{String, Float64}()
    ranges = Vector{Pair{String, UnitRange{Int}}}()
    windowed = Dict{String, ClimaAnalysis.OutputVar}()
    offset = 0
    for name in vars
        var = _reference_on_grid(reference[name], z_grid, window, bounds)
        windowed[name] = var
        series = _series_matrix(var)
        # The scale comes from the time-mean profile, i.e. the same vector `y` is built from, so a
        # variable's normalized reference has nonzero magnitude one by construction.
        mean_profile = vec(Statistics.mean(series; dims = 2))
        pv = SS.pool_var(transform, name, mean_profile)
        pool_vars[name] = pv
        push!(rows, series ./ sqrt(pv))
        push!(ranges, name => (offset + 1):(offset + size(series, 1)))
        offset += size(series, 1)
    end
    return (; series = reduce(vcat, rows), pool_vars, ranges, windowed, vars)
end

# Resample onto exactly the model levels inside the scored region — the same set the model side is
# restricted to — so the two vectors match by construction. This is the step whose absence made every
# G column NaN.
function _reference_on_grid(var, z_grid, window, bounds)
    out = var
    if ClimaAnalysis.has_altitude(out)
        out = SS.reference_on_levels(out, SS.scored_levels(z_grid, bounds))
    end
    return ClimaAnalysis.window(
        out,
        ClimaAnalysis.time_name(out);
        left = first(window),
        right = last(window),
    )
end

# (rows, times): a profile contributes n_levels rows, a path contributes one.
_series_matrix(var) =
    ClimaAnalysis.has_altitude(var) ? Array{Float64}(var.data) :
    reshape(Array{Float64}(vec(var.data)), 1, :)

"""
    case_observation(case; transform, source, vars, ...)

The `EKP.Observation` for `case`.

`y` is the time mean of the normalized stacked series.

`Γ` is an `EKP.SVDplusD`: the sample time covariance held as a truncated SVD at its true rank, plus a
diagonal from [`SocratesScoring.uncertainty_diagonal`](@ref). The split is a statement about what is
known. A scoring window holds a few dozen LES snapshots, so the sample covariance has rank ≤ 23 against
684–1232 rows; the SVD part carries every direction the snapshots measure, cross-variable and
cross-level terms included, and the diagonal supplies the uncertainty on the directions they do not.

The full-length `y` is kept rather than projected onto the measured directions, so a model putting
condensate where the LES has none is still penalized and `GEnsembleBuilder` can still validate variable
by variable. Only the covariance is represented at low rank.

NaNs, meaning absent LES data, are dropped from `y` and contribute nothing to the covariance.
"""
function case_observation(
    case::SS.SocratesCase;
    transform::SS.ScoreTransform = SS.ScoreTransform(),
    source::Symbol = :processed,
    vars = SS.REFERENCE_VARS,
    rank = nothing,
    kwargs...,
)
    ref = normalized_reference_series(case; transform, source, vars, kwargs...)
    series = ref.series
    y = _nanmean_rows(series)
    diagonal = zeros(Float64, size(series, 1))
    for (name, range) in ref.ranges
        diagonal[range] .= SS.uncertainty_diagonal(transform, name, view(series, range, :))
    end
    bad = findall(<=(0.0), diagonal)
    isempty(bad) || error(
        "Observation variance for $(SS.case_name(case)) is not positive at $(length(bad)) of \
         $(length(diagonal)) entries, so Γ cannot be inverted. Every variable needs a nonzero \
         `uncertainty_floor` and a nonzero scale to apply it to.",
    )
    Γ = EKP.SVDplusD(
        _low_rank_time_covariance(series, rank),
        LinearAlgebra.Diagonal(diagonal),
    )
    metadata = [
        _flat_metadata(ref.windowed[name], ref.pool_vars[name], scored_name(case, name))
        for name in vars
    ]
    return EKP.Observation(
        Dict(
            "samples" => y,
            "covariances" => Γ,
            "names" => SS.case_name(case),
            "metadata" => metadata,
        ),
    )
end

"""
    scored_name(case, var)

The short name a variable carries inside an observation and in the `G` map: `"<case>_<var>"`.

`GEnsembleBuilder` groups metadata by short name and checks a model variable against *every* entry
sharing it. A bare `"clw"` appears once per case, so it would check RF01_Obs's 280 levels against
RF09_Obs's 292 — and its `DimValuesChecker` compares dimension values with `isapprox`, which throws a
`DimensionMismatch` on unequal lengths rather than reporting a mismatch. Naming the case makes each
model variable match exactly one metadata entry.
"""
scored_name(case::SS.SocratesCase, var::AbstractString) =
    string(SS.case_name(case), "_", var)

# The ClimaAnalysis metadata GEnsembleBuilder fills against, with the normalization carried along.
# The units are "1" because `y` is normalized: the model side is labelled the same way, and
# `GEnsembleBuilder` matches a model variable to its metadata only if the two agree exactly.
function _flat_metadata(var, pool_var, short_name)
    tagged = ClimaAnalysis.OutputVar(
        merge(
            var.attributes,
            Dict{String, Any}(
                "pool_var" => pool_var,
                "units" => "1",
                "short_name" => short_name,
            ),
        ),
        var.dims,
        var.dim_attributes,
        var.data,
    )
    mean_var = ClimaAnalysis.average_time(tagged)
    return ClimaAnalysis.flatten(mean_var).metadata
end

_nanmean_rows(m) = [SS.nanmean(view(m, i, :)) for i in axes(m, 1)]

"""
The sample time covariance of the normalized stacked series, as a truncated SVD.

Columns are the times in the scoring window, so this is the full within-case covariance — every
cross-variable and cross-level term the snapshots measure — represented at its own rank rather than as
a dense matrix that is singular by construction. `rank = nothing` lets EKP infer it from the data.
"""
function _low_rank_time_covariance(series, rank)
    samples = replace(series, NaN => 0.0)
    return isnothing(rank) ? EKP.tsvd_cov_from_samples(samples; quiet = true) :
           EKP.tsvd_cov_from_samples(samples, rank; quiet = true)
end

"""
    observation_vector(cases; kwargs...)

One `EKP.Observation` per case, in `cases` order.
"""
function observation_vector(
    cases;
    float_type::Type{<:AbstractFloat} = Float64,
    grids = [SS.SM.socrates_grid(float_type, case) for case in cases],
    kwargs...,
)
    length(grids) == length(cases) || error(
        "observation_vector needs one grid per case, in `cases` order: got $(length(grids)) \
         grids for $(length(cases)) cases.",
    )
    return [
        case_observation(case; float_type, grid, kwargs...) for
        (case, grid) in zip(cases, grids)
    ]
end
