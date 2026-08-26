"""
    SocratesPlots

Figures for SOCRATES runs and calibrations.

Deliberately depends on no SOCRATES module: everything is passed in as plain data, `OutputVar`s, or an
`EnsembleKalmanProcess`. Including a layer here as well as in the caller would define a *second* copy of
it, and a `SocratesCase` from one copy is a different type from the other's.

```julia
include(".../src/calibration/SocratesCalibration.jl")
include(".../plots/SocratesPlots.jl")

SocratesPlots.parameter_evolution(ekp, prior; path = "params.png")
SocratesPlots.error_evolution(ekp; path = "error.png")
```

`scripts/plot_calibration.jl` does the gluing for a finished calibration.
"""
module SocratesPlots

using CairoMakie: CairoMakie
using ClimaAnalysis: ClimaAnalysis
using EnsembleKalmanProcesses: EnsembleKalmanProcesses as EKP
using Statistics: Statistics

"""Colour for each ensemble group, in the order groups are given."""
const GROUP_COLORS = (:steelblue, :firebrick, :seagreen, :darkorange)

"""
    parameter_evolution(ekp, prior; path, logscale = true)

Every member's parameter value against iteration, one panel per parameter, with the ensemble mean as a
thick line. Values are in physical (constrained) units.
"""
function parameter_evolution(ekp, prior; path::AbstractString, logscale::Bool = true)
    names = EKP.ParameterDistributions.get_name(prior)
    phi = EKP.get_ϕ(prior, ekp)              # one (n_par x n_ens) matrix per iteration
    n_iter = length(phi)
    n_par = length(names)
    fig = CairoMakie.Figure(; size = (450 * min(n_par, 2), 320 * cld(n_par, 2)))
    for (i, name) in enumerate(names)
        row, col = fldmod1(i, 2)
        ax = CairoMakie.Axis(
            fig[row, col];
            title = name,
            xlabel = "iteration",
            ylabel = "value [s]",
            yscale = logscale ? log10 : identity,
        )
        values = [phi[k][i, :] for k in 1:n_iter]           # per iteration, all members
        for member in 1:length(first(values))
            CairoMakie.lines!(
                ax, 1:n_iter, [v[member] for v in values];
                color = (:grey, 0.5), linewidth = 1,
            )
        end
        CairoMakie.lines!(
            ax, 1:n_iter, [Statistics.mean(v) for v in values];
            color = :firebrick, linewidth = 3, label = "ensemble mean",
        )
        i == 1 && CairoMakie.axislegend(ax; position = :rt, framevisible = false)
    end
    _save(fig, path)
    return path
end

"""
    error_evolution(ekp; path)

The EKP misfit and the accumulated algorithmic time `T = ΣΔt` against iteration.

Returns `nothing` before the first ensemble update: `get_error` reads the `"loss"` entry of the EKP
error dictionary, which `update_ensemble!` is what writes.
"""
function error_evolution(ekp; path::AbstractString)
    haskey(EKP.get_error_metrics(ekp), "loss") || begin
        @warn "No misfit recorded yet (no completed ensemble update); skipping" path
        return nothing
    end
    err = EKP.get_error(ekp)
    dt = EKP.get_Δt(ekp)
    isempty(err) && (@warn "Misfit history is empty; skipping" path; return nothing)
    fig = CairoMakie.Figure(; size = (900, 350))
    ax1 = CairoMakie.Axis(
        fig[1, 1];
        title = "misfit", xlabel = "iteration", ylabel = "error", yscale = log10,
    )
    CairoMakie.scatterlines!(ax1, 1:length(err), err; color = :firebrick, linewidth = 2)
    ax2 = CairoMakie.Axis(
        fig[1, 2];
        title = "accumulated T = ΣΔt", xlabel = "iteration", ylabel = "T", yscale = log10,
    )
    CairoMakie.scatterlines!(ax2, 1:length(dt), cumsum(dt); color = :steelblue, linewidth = 2)
    _save(fig, path)
    return path
end

"""
    ensemble_profiles(z, reference, groups; path, title, xlabel, logscale = false)

Profiles against height. `reference` is drawn as a thick black line; `groups` is a vector of
`label => members`, where `members` is a vector of profile vectors, each drawn thin with the group mean
thick.

`z`, `reference` and every member must already be restricted to the scored levels and averaged over the
scoring window — this function does no windowing of its own, so what it draws is exactly what is scored.
"""
function ensemble_profiles(
    z::AbstractVector,
    reference::AbstractVector,
    groups::AbstractVector;
    path::AbstractString,
    title::AbstractString = "",
    xlabel::AbstractString = "",
    logscale::Bool = false,
)
    fig = CairoMakie.Figure(; size = (520, 620))
    ax = CairoMakie.Axis(
        fig[1, 1];
        title, xlabel, ylabel = "z [m]",
        xscale = logscale ? log10 : identity,
    )
    for (g, (label, members)) in enumerate(groups)
        color = GROUP_COLORS[mod1(g, length(GROUP_COLORS))]
        for m in members
            CairoMakie.lines!(ax, m, z; color = (color, 0.35), linewidth = 1)
        end
        isempty(members) && continue
        mean_profile = [Statistics.mean(m[i] for m in members) for i in eachindex(z)]
        CairoMakie.lines!(ax, mean_profile, z; color, linewidth = 3, label = String(label))
    end
    CairoMakie.lines!(ax, reference, z; color = :black, linewidth = 3.5, label = "Atlas LES")
    CairoMakie.axislegend(ax; position = :rt, framevisible = false)
    _save(fig, path)
    return path
end

"""
    profile_grid(z, reference_by_var, groups_by_var, vars; path, xlabels, title)

One [`ensemble_profiles`](@ref)-style panel per variable in `vars`, on a shared figure.
"""
function profile_grid(
    z,
    reference_by_var::AbstractDict,
    groups_by_var::AbstractDict,
    vars;
    path::AbstractString,
    xlabels::AbstractDict = Dict{String, String}(),
    title::AbstractString = "",
    ncols::Int = 2,
)
    n = length(vars)
    fig = CairoMakie.Figure(; size = (460 * ncols, 340 * cld(n, ncols)))
    for (i, var) in enumerate(vars)
        row, col = fldmod1(i, ncols)
        ax = CairoMakie.Axis(
            fig[row, col];
            title = var,
            xlabel = get(xlabels, var, ""),
            ylabel = "z [m]",
        )
        for (g, (label, members)) in enumerate(groups_by_var[var])
            color = GROUP_COLORS[mod1(g, length(GROUP_COLORS))]
            for m in members
                CairoMakie.lines!(ax, m, z[var]; color = (color, 0.35), linewidth = 1)
            end
            isempty(members) && continue
            mean_profile =
                [Statistics.mean(m[k] for m in members) for k in eachindex(z[var])]
            CairoMakie.lines!(ax, mean_profile, z[var]; color, linewidth = 3, label = String(label))
        end
        CairoMakie.lines!(
            ax, reference_by_var[var], z[var];
            color = :black, linewidth = 3.5, label = "Atlas LES",
        )
        i == 1 && CairoMakie.axislegend(ax; position = :rt, framevisible = false)
    end
    isempty(title) || CairoMakie.Label(fig[0, :], title; fontsize = 18)
    _save(fig, path)
    return path
end

"""
    scalar_comparison(labels, reference, groups; path, title, ylabel)

Bar-style comparison of scalar diagnostics (the water paths), reference against each group's members.
"""
function scalar_comparison(
    labels::AbstractVector,
    reference::AbstractVector,
    groups::AbstractVector;
    path::AbstractString,
    title::AbstractString = "",
    ylabel::AbstractString = "",
)
    fig = CairoMakie.Figure(; size = (620, 380))
    ax = CairoMakie.Axis(
        fig[1, 1];
        title, ylabel, yscale = log10,
        xticks = (1:length(labels), collect(String.(labels))),
    )
    for (g, (label, members)) in enumerate(groups)
        color = GROUP_COLORS[mod1(g, length(GROUP_COLORS))]
        for m in members
            CairoMakie.scatter!(
                ax, 1:length(labels), max.(m, eps());
                color = (color, 0.4), markersize = 7,
            )
        end
        isempty(members) && continue
        mean_values =
            [max(Statistics.mean(m[i] for m in members), eps()) for i in eachindex(labels)]
        CairoMakie.scatter!(
            ax, 1:length(labels), mean_values;
            color, markersize = 16, marker = :diamond, label = String(label),
        )
    end
    CairoMakie.scatter!(
        ax, 1:length(labels), max.(reference, eps());
        color = :black, markersize = 16, marker = :star5, label = "Atlas LES",
    )
    CairoMakie.axislegend(ax; position = :rt, framevisible = false)
    _save(fig, path)
    return path
end

"""
    observation_blocks(ekp)

One entry per scored variable: `(short_name, range, z, scale)`. `range` indexes into the observation
vector, `z` is empty for column integrals, and `scale` is `sqrt(pool_var)` — multiply by it to get
physical units, since observations are stored divided by it.
"""
function observation_blocks(ekp)
    md = EKP.get_metadata(first(EKP.get_observations(EKP.get_observation_series(ekp))))
    blocks = Tuple{String, UnitRange{Int}, Vector{Float64}, Float64}[]
    offset = 0
    for m in md
        dims = collect(values(m.dims))
        n = isempty(dims) ? 1 : prod(length(v) for v in dims)
        z = isempty(dims) ? Float64[] : collect(Float64, first(dims))
        scale = sqrt(parse(Float64, string(get(m.attributes, "pool_var", "1.0"))))
        push!(
            blocks,
            (get(m.attributes, "short_name", "?"), (offset + 1):(offset + n), z, scale),
        )
        offset += n
    end
    return blocks
end

"""
    prior_posterior_profiles(ekp, prior_g, posterior_g; path, n_sigma = (1, 2))

Reference against prior (red) and posterior (blue) ensembles, one panel per scored profile, with a
band at each multiple of the ensemble standard deviation in `n_sigma`.

`prior_g` and `posterior_g` are `G_ensemble` matrices — `(n_obs, n_members)`. Everything is converted
to physical units through each block's `pool_var`, so panels read in kg/kg rather than score units.
"""
function prior_posterior_profiles(
    ekp,
    prior_g::AbstractMatrix,
    posterior_g::AbstractMatrix;
    path::AbstractString,
    n_sigma = (1, 2),
    ncols::Int = 4,
    xlim_factor::Union{Nothing, Real} = nothing,
)
    y = EKP.get_obs(ekp)
    blocks = filter(b -> !isempty(b[3]), observation_blocks(ekp))
    isempty(blocks) && error("No profile variables to plot; every block is a column integral.")
    nrows = cld(length(blocks), ncols)
    fig = CairoMakie.Figure(; size = (420 * ncols, 300 * nrows))
    for (i, (name, range, z, scale)) in enumerate(blocks)
        row, col = fldmod1(i, ncols)
        ax = CairoMakie.Axis(
            fig[row, col];
            title = name,
            xlabel = "kg/kg",
            ylabel = col == 1 ? "z [m]" : "",
        )
        for (g, color, label) in
            ((prior_g, :firebrick, "prior"), (posterior_g, :steelblue, "posterior"))
            members = view(g, range, :)
            keep = [j for j in axes(members, 2) if !any(isnan, view(members, :, j))]
            isempty(keep) && continue
            μ = Statistics.mean(view(members, :, keep); dims = 2)[:] .* scale
            σ = Statistics.std(view(members, :, keep); dims = 2)[:] .* scale
            # Widest band first so the narrower ones stay visible on top of it. `band!` fills between
            # two point sequences, which is what a horizontal spread on a z-axis profile needs.
            for k in sort(collect(n_sigma); rev = true)
                CairoMakie.band!(
                    ax,
                    CairoMakie.Point2f.(μ .- k .* σ, z),
                    CairoMakie.Point2f.(μ .+ k .* σ, z);
                    color = (color, k == minimum(n_sigma) ? 0.30 : 0.15),
                )
            end
            CairoMakie.lines!(ax, μ, z; color, linewidth = 3, label)
        end
        reference = y[range] .* scale
        CairoMakie.lines!(ax, reference, z; color = :black, linewidth = 4, label = "LES")
        # An all-zero reference gives no scale to zoom to, so those panels stay on autoscale.
        if !isnothing(xlim_factor)
            span = maximum(abs, reference)
            span > 0 && CairoMakie.xlims!(ax, -0.1 * xlim_factor * span, xlim_factor * span)
        end
        i == 1 && CairoMakie.axislegend(ax; position = :rt, framevisible = false)
    end
    _save(fig, path)
    return path
end

"""
    tendency_budget(variable, budget, columns; path, styles, threshold = 0.0)

The tendency budget of one prognostic variable, with one column per flight.

`budget` is the signed term list for the variable — `(term, sign)` pairs, e.g. an entry of
`SocratesModel.MP1M_BUDGETS`. `columns` is a vector of `(name, z, series)` where `series` maps a
label to that label's `term => profile` dictionary; a label may be missing terms (the LES has no
counterpart for some) and is then simply absent from that panel.

`styles` maps a label to `(linestyle, alpha, linewidth)` so posterior, prior and reference read
differently. Terms whose signed contribution never exceeds `threshold` in any column are dropped.
"""
function tendency_budget(
    variable::AbstractString,
    budget,
    columns::AbstractVector;
    path::AbstractString,
    styles::AbstractDict = Dict(
        "LES" => (:solid, 1.0, 2.5),
        "best_final" => (:dash, 1.0, 1.8),
        "best" => (:dot, 0.9, 1.8),
    ),
    threshold::Real = 0.0,
)
    isempty(columns) && error("`tendency_budget` needs at least one flight column.")
    terms = [t for (t, _) in budget]
    active = filter(terms) do term
        any(columns) do (_, _, series)
            any(values(series)) do rates
                haskey(rates, term) && maximum(abs, rates[term]) > threshold
            end
        end
    end
    isempty(active) &&
        error("Every term of the $variable budget is below the $threshold threshold.")
    palette = CairoMakie.cgrad(:tab20, max(length(active), 2); categorical = true)
    signof(term) = only(s for (t, s) in budget if t == term)
    fig = CairoMakie.Figure(; size = (330 * length(columns) + 280, 430))
    for (col, (name, z, series)) in enumerate(columns)
        ax = CairoMakie.Axis(
            fig[1, col];
            title = name,
            xlabel = "kg/kg/s",
            ylabel = col == 1 ? "z [m]" : "",
        )
        widest = 0.0
        for (label, rates) in sort!(collect(series); by = first)
            style, alpha, width = get(styles, label, (:solid, 1.0, 1.5))
            for (i, term) in enumerate(active)
                haskey(rates, term) || continue
                signed = signof(term) .* rates[term]
                widest = max(widest, maximum(abs, signed))
                CairoMakie.lines!(
                    ax, signed, z;
                    color = (palette[i], alpha), linestyle = style, linewidth = width,
                )
            end
        end
        CairoMakie.vlines!(ax, [0.0]; color = (:black, 0.4), linestyle = :dash)
        # A panel where everything is zero has no axis range, and tick-finding then warns per panel.
        widest > 0 ? CairoMakie.xlims!(ax, -1.05 * widest, 1.05 * widest) :
        CairoMakie.xlims!(ax, -1, 1)
    end
    entries = [
        CairoMakie.LineElement(; color = palette[i], linewidth = 2) for
        i in eachindex(active)
    ]
    labels = [
        "$(signof(t) > 0 ? "+" : "−") $(replace(t, "S_" => "", "_" => " "))" for t in active
    ]
    CairoMakie.Legend(
        fig[1, length(columns) + 1], entries, labels;
        framevisible = false, labelsize = 9, patchsize = (14, 2),
    )
    CairoMakie.Label(
        fig[0, :],
        "$variable tendency budget — Atlas LES solid, best_final dashed, best dotted";
        fontsize = 15,
    )
    _save(fig, path)
    return path
end

function _save(fig, path)
    mkpath(dirname(abspath(path)))
    CairoMakie.save(path, fig)
    @info "wrote" path
    return path
end

end # module
