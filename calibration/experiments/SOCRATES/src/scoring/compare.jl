"""
Compare a run's output against the Atlas LES reference.

This is the whole scoring path with no EKP involved: read the run, read the reference, put both on the
same levels inside the scored region, time-average over the scoring window, normalize, and report the
misfit. The calibration layer reuses these steps, so a discrepancy shows up here first and in a form
that is easy to look at.

The scored level set is computed **once**, by [`scored_levels`](@ref), and both the model and the
reference are restricted to it. Selecting them independently does not work: `ClimaAnalysis.window`
defaults to `NearestValue()`, so a bound falling between two levels snaps outward and yields one level
more than a strict filter does.
"""

using ClimaAnalysis: ClimaAnalysis
using Statistics: Statistics

"""
    run_outputvars(output_dir, vars; reduction, period)

The named diagnostics from a run directory, as `ClimaAnalysis.OutputVar`s.

`output_dir` is what [`SocratesModel.run_case`](@ref) returns, or the directory containing it: an
`activelink` run writes `output_0000/` alongside an `output_active` symlink to it, and pointing `SimDir`
at the parent aggregates both, which duplicates every timestamp.
"""
function run_outputvars(
    output_dir::AbstractString,
    vars = REFERENCE_VARS;
    reduction::AbstractString = "average",
    period::AbstractString = "10m",
)
    simdir = ClimaAnalysis.SimDir(active_output_dir(output_dir))
    return Dict{String, ClimaAnalysis.OutputVar}(
        name => ClimaAnalysis.get(simdir; short_name = name, reduction, period) for
        name in vars
    )
end

"""
    active_output_dir(dir)

The directory holding a run's NetCDF: `dir/output_active` when that exists, otherwise `dir` itself.

Resolving it is not cosmetic — aggregating a run's `output_active` symlink together with the
`output_0000` it points at yields each time twice, and reading then fails on a non-monotonic time axis.
"""
active_output_dir(dir::AbstractString) =
    isdir(joinpath(dir, "output_active")) ? joinpath(dir, "output_active") : dir

"""
    model_levels(var, bounds)

The levels of `var` inside `bounds` — the scored level set, and the levels a reference is resampled
onto. Returns `nothing` for a variable with no altitude dimension.
"""
model_levels(var::ClimaAnalysis.OutputVar, bounds) =
    ClimaAnalysis.has_altitude(var) ?
    scored_levels(var.dims[ClimaAnalysis.altitude_name(var)], bounds) : nothing

"""
    restrict_to_levels(var, levels)

`var` keeping only the altitude entries in `levels`, which must be levels of `var` itself.
"""
function restrict_to_levels(var::ClimaAnalysis.OutputVar, levels)
    (isnothing(levels) || !ClimaAnalysis.has_altitude(var)) && return var
    z_name = ClimaAnalysis.altitude_name(var)
    own = collect(Float64, var.dims[z_name])
    idx = indexin(collect(Float64, levels), own)
    any(isnothing, idx) && error(
        "Levels $(collect(levels)[findall(isnothing, idx)]) m are not levels of this variable, whose \
         grid spans $(extrema(own)) m with $(length(own)) levels.",
    )
    axis = var.dim2index[z_name]
    dims = ClimaAnalysis.Var.OrderedDict(
        name => name == z_name ? own[idx] : collect(d) for (name, d) in var.dims
    )
    data = copy(selectdim(var.data, axis, idx))
    return ClimaAnalysis.OutputVar(var.attributes, dims, var.dim_attributes, data)
end

"""
    windowed_time_mean(var, window)

`var` averaged over `window = (t0, t1)` seconds. The time dimension is dropped, leaving a profile (or a
scalar) directly comparable between model and reference.

The data is always an array, including for a path variable whose only dimension was time: averaging that
away leaves a bare number, which every downstream consumer — `OutputVar` construction, `flatten`,
plotting — needs wrapped.
"""
function windowed_time_mean(var::ClimaAnalysis.OutputVar, window)
    t = var.dims[ClimaAnalysis.time_name(var)]
    any(ti -> first(window) <= ti <= last(window), t) || error(
        "No times inside the scoring window $(window) s; the record spans $(extrema(t)) s. Check \
         that the run is long enough and that its diagnostics cover the window.",
    )
    out = ClimaAnalysis.window(
        var,
        ClimaAnalysis.time_name(var);
        left = first(window),
        right = last(window),
    )
    mean_var = ClimaAnalysis.average_time(out)
    mean_var.data isa AbstractArray && return mean_var
    return ClimaAnalysis.OutputVar(
        mean_var.attributes,
        mean_var.dims,
        mean_var.dim_attributes,
        fill(mean_var.data),
    )
end

"""
    compare_to_les(output_dir, case; transform, source, vars, window, bounds)

Normalized misfit between a run and the Atlas LES, per variable.

Returns a `Dict` of variable → `(; mse, rmse, model, reference, pool_var)`, where `model` and
`reference` are the normalized, window-averaged vectors actually compared (length 1 for a path, the
number of scored levels for a profile) and `pool_var` is the squared normalization scale.

`mse` is the mean squared difference in normalized units, so it is comparable across variables — that
is the point of the normalization.
"""
function compare_to_les(
    output_dir::AbstractString,
    case::SocratesCase;
    transform::ScoreTransform = ScoreTransform(),
    source::Symbol = :processed,
    vars = REFERENCE_VARS,
    window = score_window(case),
    bounds = z_bounds(case; source),
)
    model = run_outputvars(output_dir, vars)
    reference = les_outputvars(case; source, vars)
    out = Dict{String, Any}()
    for name in vars
        levels = model_levels(model[name], bounds)
        m = windowed_time_mean(restrict_to_levels(model[name], levels), window)
        r = windowed_time_mean(reference_on_levels(reference[name], levels), window)
        mvals, rvals = vec(m.data), vec(r.data)
        length(mvals) == length(rvals) || error(
            "`$name` has $(length(mvals)) model values but $(length(rvals)) reference values after \
             resampling to the scored levels — they must match to be compared.",
        )
        pv = pool_var(transform, name, rvals)
        mn, rn = mvals ./ sqrt(pv), rvals ./ sqrt(pv)
        d = filter(isfinite, mn .- rn)
        mse = isempty(d) ? NaN : sum(abs2, d) / length(d)
        out[name] = (; mse, rmse = sqrt(mse), model = mn, reference = rn, pool_var = pv)
    end
    return out
end

"""
    print_comparison(comparison; io = stdout)

Print a `compare_to_les` result as one line per variable.
"""
function print_comparison(comparison; io = stdout)
    println(io, rpad("variable", 10), rpad("n", 5), rpad("rmse (norm)", 14),
            rpad("mean model", 14), rpad("mean ref", 14), "pool_var")
    for name in REFERENCE_VARS
        haskey(comparison, name) || continue
        c = comparison[name]
        mm = Statistics.mean(filter(isfinite, c.model))
        mr = Statistics.mean(filter(isfinite, c.reference))
        println(io, rpad(name, 10), rpad(length(c.model), 5),
                rpad(round(c.rmse; sigdigits = 4), 14),
                rpad(round(mm; sigdigits = 4), 14),
                rpad(round(mr; sigdigits = 4), 14),
                round(c.pool_var; sigdigits = 4))
    end
    return nothing
end
