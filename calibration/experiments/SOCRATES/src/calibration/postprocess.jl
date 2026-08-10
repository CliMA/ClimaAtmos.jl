"""
Rerunning selected ensemble members with the full process-rate diagnostics.

A calibration writes only the scored variables, so it cannot answer "which process is doing what".
This layer picks members out of a finished calibration by their misfit and reruns them asking for
[`SocratesModel.TENDENCY_DIAGNOSTIC_VARS`](@ref) — every 1-moment microphysics rate in the grid mean,
the updraft and the environment, plus the state fields needed to read them.

The selection metric is the same one the EKP scheduler reacts to,
`Φⱼ = ½ (gⱼ - y)ᵀ Γ⁻¹ (gⱼ - y)`, so "best" here means "lowest misfit", not "closest in any one
variable".
"""

using ClimaCalibrate: ClimaCalibrate
using EnsembleKalmanProcesses: EnsembleKalmanProcesses as EKP
using JLD2: JLD2
using LinearAlgebra: LinearAlgebra

"""
    member_misfits(ekp, g_ensemble) -> Vector

`Φⱼ` for each column of `g_ensemble`, `NaN` for a member that failed (any `NaN` in its column).
"""
function member_misfits(ekp, g_ensemble::AbstractMatrix)
    y = EKP.get_obs(ekp)
    Γ = Matrix(EKP.get_obs_noise_cov(ekp))
    factorized = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Γ))
    return map(axes(g_ensemble, 2)) do j
        d = view(g_ensemble, :, j) .- y
        any(isnan, d) ? NaN : 0.5 * LinearAlgebra.dot(d, factorized \ d)
    end
end

"""
    g_ensemble_path(output_dir, iteration)

Where `run_iteration`'s observation map wrote `G_ensemble` for `iteration`.
"""
g_ensemble_path(output_dir::AbstractString, iteration::Integer) =
    joinpath(output_dir, "iteration_" * lpad(iteration, 3, '0'), "G_ensemble.jld2")

"""
    best_members(interface; last_iteration) -> (; best, best_final)

The `(iteration, member, misfit)` of the lowest-misfit member over all completed iterations, and of
the lowest-misfit member within the last one. Iterations whose `G_ensemble` is missing are skipped,
and members that failed are ignored.
"""
function best_members(
    interface::SocratesInterface;
    last_iteration::Integer = ClimaCalibrate.last_completed_iteration(
        interface.output_dir,
    ),
)
    last_iteration >= 1 ||
        error("No completed iterations in $(interface.output_dir); nothing to postprocess.")
    ekp = ClimaCalibrate.load_ekp_struct(interface.output_dir, last_iteration)
    candidates = NamedTuple{(:iteration, :member, :misfit), Tuple{Int, Int, Float64}}[]
    for iteration in 1:last_iteration
        path = g_ensemble_path(interface.output_dir, iteration)
        isfile(path) || continue
        misfits = member_misfits(ekp, JLD2.load_object(path))
        for (member, φ) in enumerate(misfits)
            isnan(φ) || push!(candidates, (; iteration, member, misfit = φ))
        end
    end
    isempty(candidates) &&
        error("Every ensemble member failed; there is nothing to rerun.")
    final = filter(c -> c.iteration == last_iteration, candidates)
    isempty(final) && error(
        "Every member of iteration $last_iteration failed, so there is no best final member.",
    )
    return (;
        best = argmin(c -> c.misfit, candidates),
        best_final = argmin(c -> c.misfit, final),
    )
end

"""
    rerun_member(interface, iteration, member; output_dir, period, kwargs...)

Rerun every case for one member with the full process-rate diagnostics, into
`output_dir/<case_name>`. Returns the directories written, in `interface.cases` order.

Uses the member's own `parameters.toml`, so this reproduces exactly what the calibration ran, and
the same grid and `run_kwargs` the calibration used.
"""
function rerun_member(
    interface::SocratesInterface,
    iteration::Integer,
    member::Integer;
    output_dir::AbstractString,
    period::AbstractString = "10mins",
    vars = SS.SM.TENDENCY_DIAGNOSTIC_VARS,
    executor = SS.SM.SerialExecutor(),
)
    params = ClimaCalibrate.parameter_path(interface.output_dir, iteration, member)
    isfile(params) || error("No parameters.toml for iteration $iteration member $member: $params")
    run_one = case -> begin
        grid = case_grid(interface, case)
        SS.SM.run_case(
            case;
            FT = interface.float_type,
            params,
            grid,
            output_dir = joinpath(output_dir, SS.case_name(case)),
            verbose = false,
            diagnostics = SS.SM.socrates_diagnostics(
                vars;
                period,
                reduction = "average",
                n_levels = length(SS.SM.socrates_z(grid)),
            ),
            interface.run_kwargs...,
        )
    end
    return SS.SM.run_tasks(run_one, interface.cases, executor)
end

"""
    postprocess_best_members(interface; output_dir, last_iteration, period)

Rerun the best and best-final members with full diagnostics.

Returns a `Dict` from label (`"best"`, `"best_final"`) to `(; iteration, member, misfit, dirs)`. The
two are rerun separately even when they are the same member, so the caller can always index both.
"""
function postprocess_best_members(
    interface::SocratesInterface;
    output_dir::AbstractString = joinpath(interface.output_dir, "postprocess"),
    last_iteration::Integer = ClimaCalibrate.last_completed_iteration(
        interface.output_dir,
    ),
    period::AbstractString = "10mins",
    executor = SS.SM.SerialExecutor(),
)
    selected = best_members(interface; last_iteration)
    results = Dict{String, Any}()
    for label in ("best", "best_final")
        pick = getproperty(selected, Symbol(label))
        @info "Rerunning $label member with process-rate diagnostics" pick.iteration pick.member pick.misfit
        dirs = rerun_member(
            interface,
            pick.iteration,
            pick.member;
            output_dir = joinpath(output_dir, label),
            period,
            executor,
        )
        results[label] = (; pick.iteration, pick.member, pick.misfit, dirs)
    end
    return results
end

"""
Atlas LES counterparts of the model process rates, as `(SAM variable, sign)` sums.

The processed reference files already carry `kg/kg/s`, so no per-day conversion
is applied here. Signs are chosen so each entry is the same *magnitude* the model rate is — the
budget's own sign is applied by [`SocratesModel.MP1M_BUDGETS`](@ref).

Rates absent from this table have no LES counterpart: the freezing terms need nucleation the 1-moment
scheme does not have, `S_accr_icl_rai`/`S_accr_melt_*` have no single SAM variable, and the file
carries no `QCDIFF`. Those panels are model-only.

The `QxSED`/`QxADV`/`QxDIFF` transport terms are signed tendencies in the same convention as the
model's: measured over the RF01_Obs score window every `QxSED` is positive at the base of the active
layer and negative at its top, with a negative column sum.

`PCC` (condensation/evaporation) and `PRE` (rain evaporation) are annotated in that source as
"seems to be broken in Atlas files", so treat their reference lines with suspicion.
"""
const LES_RATE_SOURCES = Dict{String, Vector{Tuple{String, Int}}}(
    "sed_q_lcl" => [("QCSED", +1)],
    "sed_q_icl" => [("QISED", +1)],
    "sed_q_rai" => [("QRSED", +1)],
    "sed_q_sno" => [("QSSED", +1)],
    "adv_q_lcl" => [("QCADV", +1)],
    "adv_q_icl" => [("QIADV", +1)],
    "adv_q_rai" => [("QRADV", +1)],
    "adv_q_sno" => [("QSADV", +1)],
    "dif_q_icl" => [("QIDIFF", +1)],
    "dif_q_rai" => [("QRDIFF", +1)],
    "dif_q_sno" => [("QSDIFF", +1)],
    "S_phase_change_vap_lcl" => [("PCC", +1)],
    "S_phase_change_vap_icl" => [("PRD", +1), ("EPRD", +1)],
    "S_phase_change_vap_sno" => [("PRDS", +1), ("EPRDS", +1)],
    "S_phase_change_vap_rai" => [("PRE", +1)],
    "S_acnv_lcl_rai" => [("PRC", +1)],
    "S_acnv_icl_sno" => [("PRCI", +1), ("PITOSN", +1)],
    "S_accr_lcl_rai" => [("PRA", +1)],
    "S_accr_lcl_sno_cold" => [("PSACWS", +1)],
    "S_accr_icl_sno" => [("PRAI", +1)],
    "S_accr_rai_sno_cold" => [("PRACS", +1)],
    "S_melt_sno_rai" => [("PSMLT", +1)],
)

"""
    case_les_rates(case, z; window, source = :processed) -> Dict

The LES process rates for `case`, time-averaged over `window` and resampled onto levels `z`, keyed by
the model rate name they correspond to. Rates with no counterpart are absent.
"""
function case_les_rates(
    case::SS.SocratesCase,
    z::AbstractVector;
    window = SS.SM.score_window(case),
)
    wanted = unique(v for sources in values(LES_RATE_SOURCES) for (v, _) in sources)
    raw = SS.les_raw_profiles(case, wanted)
    keep = findall(t -> first(window) <= t <= last(window), raw.time)
    isempty(keep) && error(
        "No LES times inside $(window) s for $(SS.case_name(case)); the record spans \
         $(extrema(raw.time)) s.",
    )
    # Linear in height with flat ends: the LES grid is finer than any model grid here, so this is a
    # plain resample, not the cell-extent padding `reference_on_levels` needs for scored variables.
    onto(profile) = map(z) do zi
        k = searchsortedfirst(raw.z, zi)
        k <= 1 && return profile[1]
        k > length(raw.z) && return profile[end]
        z0, z1 = raw.z[k - 1], raw.z[k]
        w = (zi - z0) / (z1 - z0)
        (1 - w) * profile[k - 1] + w * profile[k]
    end
    rates = Dict{String, Vector{Float64}}()
    for (rate, sources) in LES_RATE_SOURCES
        all(haskey(raw.data, v) for (v, _) in sources) || continue
        total = zeros(Float64, length(raw.z))
        for (v, sign) in sources
            a = raw.data[v]
            total .+= sign .* [NaNStatistics.nanmean(view(a, k, keep)) for k in axes(a, 1)]
        end
        rates[rate] = collect(Float64, onto(total))
    end
    return rates
end

"""
    case_budget_terms(dir, case; window, location = "mp1m") -> (terms, z)

Every term of a tendency budget written by a postprocessing run in `dir`, time-averaged over `window`.

Keys are budget term names: the microphysics process rates with the `location` prefix stripped, and
the transport diagnostics under their own names. Together they cover
[`SocratesModel.MP1M_BUDGETS`](@ref) and [`SocratesModel.TRANSPORT_BUDGETS`](@ref). Transport is
registered in the grid mean only, so it is read as such whatever `location` selects for the rates.

`dir` is one of the directories [`rerun_member`](@ref) returned.
"""
function case_budget_terms(
    dir::AbstractString,
    case::SS.SocratesCase;
    window = SS.SM.score_window(case),
    location::AbstractString = "mp1m",
    period::AbstractString = "10m",
)
    transport = (
        "$(prefix)_$(var)" for prefix in SS.SM.TRANSPORT_PREFIXES for
        var in SS.SM.MP1M_BUDGET_VARS
    )
    wanted = Iterators.flatten((
        ("$(location)_$(term)" => term for term in SS.SM.MP1M_SOURCE_TERMS),
        (name => name for name in transport),
    ))
    terms = Dict{String, Vector{Float64}}()
    z = Float64[]
    for (name, key) in wanted
        var = try
            only(values(SS.run_outputvars(dir, (name,); period)))
        catch
            continue
        end
        averaged = SS.windowed_time_mean(var, window)
        terms[key] = collect(Float64, vec(averaged.data))
        isempty(z) &&
            (z = collect(Float64, var.dims[ClimaAnalysis.altitude_name(var)]))
    end
    isempty(terms) &&
        error("No budget terms found in $dir; was it run with `TENDENCY_DIAGNOSTIC_VARS`?")
    return terms, z
end
