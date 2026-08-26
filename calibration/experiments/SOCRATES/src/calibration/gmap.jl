"""
The observation map: model output → `G_ensemble`.

Filling is done by ClimaCalibrate's `GEnsembleBuilder`, which places each model `OutputVar` into the
column using the `ClimaAnalysis.Metadata` stored in the observation, checking short name, dimension
names, dimension units, variable units, **and dimension values** on the way.

That last check is the point: observations and model output built on different vertical grids give a
silently all-`NaN` `G`. Here a grid, unit, or naming mismatch is a named error instead.

`GEnsembleBuilder` lives in an extension that activates only when both `ClimaAnalysis` and
`NaNStatistics` are loaded, and it is reached as `ClimaCalibrate.EnsembleBuilder.*` — the
`ensemble_builder.jl` submodule is included by `ClimaCalibrate` but not re-exported.
"""

using ClimaCalibrate: ClimaCalibrate
using EnsembleKalmanProcesses: EnsembleKalmanProcesses as EKP
using NaNStatistics: NaNStatistics   # loaded so ClimaCalibrateClimaAnalysisExt activates

"""
    model_scored_var(output_dir, name, case, pool_var; window, bounds, reduction, period)

One model diagnostic, reduced to exactly the quantity the observation holds: restricted to the scored
levels, windowed in time, time-averaged, and divided by `sqrt(pool_var)`.

`pool_var` comes from the observation's own metadata, so the model and the reference are divided by
the same number by construction — there is no side-car normalization file to fall out of step.

The result is relabelled `"1"`, because dividing by `sqrt(pool_var)` makes it dimensionless and
`GEnsembleBuilder` matches a variable to its metadata only if their units agree exactly.
"""
function model_scored_var(
    output_dir::AbstractString,
    name::AbstractString,
    case::SS.SocratesCase,
    pool_var::Real;
    window = SS.SM.score_window(case),
    bounds,
    reduction::AbstractString = "average",
    period::AbstractString = "10m",
)
    var = SS.run_outputvars(output_dir, (name,); reduction, period)[name]
    levels = SS.model_levels(var, bounds)
    mean_var = SS.windowed_time_mean(SS.restrict_to_levels(var, levels), window)
    # A path variable is a 0-dimensional array once time is averaged away, and broadcasting unwraps
    # that to a bare number — which `OutputVar` rejects. Scale into a container of the same shape so
    # profiles and paths both stay arrays. The dimension container is retyped for the same reason: when
    # empty its element type infers as `Any` rather than an array type.
    data = similar(mean_var.data, Float64)
    data .= mean_var.data ./ sqrt(pool_var)
    dims = ClimaAnalysis.Var.OrderedDict{String, Vector{Float64}}(
        dim => collect(Float64, values) for (dim, values) in mean_var.dims
    )
    return ClimaAnalysis.OutputVar(
        merge(
            mean_var.attributes,
            Dict{String, Any}(
                "units" => "1",
                "short_name" => scored_name(case, name),
            ),
        ),
        dims,
        mean_var.dim_attributes,
        data,
    )
end

"""
    pool_var_from_metadata(metadata)

The `pool_var` stashed in an observation's per-variable metadata by
[`case_observation`](@ref), keyed by short name.
"""
function pool_var_from_metadata(metadata)
    out = Dict{String, Float64}()
    for m in metadata
        name = get(m.attributes, "short_name", nothing)
        pv = get(m.attributes, "pool_var", nothing)
        isnothing(name) && error(
            "An observation metadata entry has no `short_name`; it cannot be matched to a model \
             variable.",
        )
        isnothing(pv) && error(
            "Observation metadata for `$name` carries no `pool_var`. Observations must be built by \
             `case_observation` so the normalization travels with them.",
        )
        out[name] = Float64(pv)
    end
    return out
end

"""
    build_g_ensemble(interface, iteration) -> Matrix

`G_ensemble` for `iteration`: one column per ensemble member, filled from that member's per-case
run directories.

A member whose runs are missing or unreadable gets a `NaN` column, which is how EKP's failure
handling expects a failed forward model to be reported.
"""
function build_g_ensemble(interface, iteration::Integer)
    ekp = ClimaCalibrate.load_ekp_struct(interface.output_dir, iteration)
    builder = ClimaCalibrate.EnsembleBuilder.GEnsembleBuilder(ekp)
    metadata = ClimaCalibrate.get_metadata_for_nth_iteration(
        EKP.get_observation_series(ekp),
        EKP.get_N_iterations(ekp) + 1,
    )
    pool_vars = pool_var_from_metadata(metadata)

    for member in 1:EKP.get_N_ens(ekp)
        try
            for case in interface.cases
                dir = case_output_dir(interface, iteration, member, case)
                bounds = SS.z_bounds(case; source = interface.reference_source)
                for name in interface.vars
                    var = model_scored_var(
                        dir,
                        name,
                        case,
                        pool_vars[scored_name(case, name)];
                        window = SS.SM.score_window(case),
                        bounds,
                    )
                    ClimaCalibrate.EnsembleBuilder.fill_g_ens_col!(builder, member, var)
                end
            end
        catch e
            @warn "Member $member failed the observation map; filling its column with NaN" exception =
                (e, catch_backtrace())
            ClimaCalibrate.EnsembleBuilder.fill_g_ens_col!(builder, member, NaN)
        end
    end

    ClimaCalibrate.EnsembleBuilder.is_complete(builder) || @warn(
        "G_ensemble for iteration $iteration is not fully filled; some entries stayed NaN."
    )
    g_ensemble = ClimaCalibrate.EnsembleBuilder.get_g_ensemble(builder)
    # Every member failing is not model instability, it is a defect in the map or the runs. Say so
    # here rather than handing EKP an all-NaN G, which it accepts and then fails on obscurely.
    all(isnan, g_ensemble) && error(
        "Every member of iteration $iteration produced a NaN column. That is a failure of the \
         observation map or of every run, not ensemble spread; see the warnings above for the \
         underlying exception.",
    )
    return g_ensemble
end