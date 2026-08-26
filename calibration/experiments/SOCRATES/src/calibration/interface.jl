"""
The ClimaCalibrate model interface for SOCRATES.

The experiment configuration is a plain Julia struct built in memory — no YAML file, no
`Dict` string-indexing. Everything the forward model needs is either on the struct or derived from
the case, and the struct is serializable so it can be shipped to a worker as-is.

`forward_model` runs a member's cases through Layer 1's runner, so the calibration adds no second
way of running a case.
"""

using ClimaCalibrate: ClimaCalibrate

"""
    SocratesInterface(; cases, output_dir, vars, transform, reference_source, float_type, run_kwargs)

Multi-case SOCRATES calibration against the Atlas LES.

# Fields

  - `cases`: the cases scored, one `EKP.Observation` each.
  - `output_dir`: root for iteration/member directories and the EKP objects.
  - `vars`: Atmos short names scored, in the order observations stack them.
  - `transform`: the score normalization, shared by the observations and the observation map.
  - `reference_source`: `:processed` or `:sscf`.
  - `float_type`: float type the forward model runs in. The calibration arithmetic stays `Float64`
    regardless — `Float32` covariances invert badly.
  - `grids`: one grid per case, in `cases` order, from [`SocratesModel.socrates_grid`](@ref).
    Defaults to each case's native LES column. The same grid feeds that case's runs and its
    observations, so the model and the reference cannot end up on different levels.
  - `run_kwargs`: everything else forwarded to [`SocratesModel.run_case`](@ref), e.g. `(; dt = 10)`.
  - `prune_output`: delete each iteration's NetCDF output once its `G_ensemble` is written. On by
    default — 11 cases times an ensemble times many iterations is a lot of disk for data nothing
    reads again. Turn it off to keep the runs for [`plot_calibration`](../../scripts/plot_calibration.jl).
Nothing here removes output. An existing `output_dir` is resumed — case-runs already marked complete
are skipped and their output scored as-is — so clearing it is the caller's `rm`.
"""
struct SocratesInterface{FT <: AbstractFloat, T, G, K} <:
       ClimaCalibrate.AbstractModelInterface
    cases::Vector{SS.SocratesCase}
    output_dir::String
    vars::Vector{String}
    transform::T
    reference_source::Symbol
    float_type::Type{FT}
    grids::Vector{G}
    run_kwargs::K
    prune_output::Bool
end

function SocratesInterface(;
    cases = SS.SM.socrates_cases(),
    output_dir::AbstractString,
    vars = collect(String, SS.REFERENCE_VARS),
    transform = SS.ScoreTransform(),
    reference_source::Symbol = :processed,
    float_type::Type{<:AbstractFloat} = Float64,
    grids = nothing,
    run_kwargs = (;),
    prune_output::Bool = true,
)
    any(k -> haskey(run_kwargs, k), (:grid, :dz_min, :faces)) && error(
        "Pass the vertical grid as `grids`, not `run_kwargs`: it also determines the levels the \
         observations are built on.",
    )
    cases = collect(SS.SocratesCase, cases)
    isempty(cases) && error("SocratesInterface needs at least one case")
    grids =
        isnothing(grids) ? [SS.SM.socrates_grid(float_type, case) for case in cases] :
        collect(grids)
    length(grids) == length(cases) || error(
        "SocratesInterface needs one grid per case, in `cases` order: got $(length(grids)) \
         grids for $(length(cases)) cases.",
    )
    foreach(SS.SM.validate, cases)
    isempty(vars) && error("SocratesInterface needs at least one scored variable")
    for v in vars
        haskey(SS.LES_VARIABLE, v) ||
            error("No LES reference mapping for scored variable `$v`")
    end
    output_dir = abspath(output_dir)
    mkpath(output_dir)
    return SocratesInterface(
        cases,
        output_dir,
        collect(String, vars),
        transform,
        reference_source,
        float_type,
        grids,
        run_kwargs,
        prune_output,
    )
end

"""
    case_output_dir(interface, iteration, member, case)

Where one `(iteration, member, case)` run writes its diagnostics:
`<output_dir>/iteration_XXX/member_YYY/<case_name>`.
"""
case_output_dir(
    interface::SocratesInterface,
    iteration::Integer,
    member::Integer,
    case::SS.SocratesCase,
) = joinpath(
    ClimaCalibrate.path_to_ensemble_member(interface.output_dir, iteration, member),
    SS.case_name(case),
)

"""
    case_grid(interface, case)

The grid `interface` runs and scores `case` on.
"""
function case_grid(interface::SocratesInterface, case::SS.SocratesCase)
    idx = findfirst(c -> SS.case_name(c) == SS.case_name(case), interface.cases)
    isnothing(idx) &&
        error("`$(SS.case_name(case))` is not one of this interface's cases.")
    return interface.grids[idx]
end

"""
    run_case_for_member(interface, iteration, member, case)

Run one case for one ensemble member with that member's sampled parameters.

This is the unit of work the flat scheduler distributes, and the only place the calibration touches
the model: it calls Layer 1's `run_case` with the member's `parameters.toml` as a parameter source.
"""
function run_case_for_member(
    interface::SocratesInterface,
    iteration::Integer,
    member::Integer,
    case::SS.SocratesCase,
)
    params = ClimaCalibrate.parameter_path(interface.output_dir, iteration, member)
    isfile(params) ||
        error("Sampled parameter file not found for member $member: $params")
    return SS.SM.run_case(
        case;
        FT = interface.float_type,
        params,
        output_dir = case_output_dir(interface, iteration, member, case),
        verbose = false,
        grid = case_grid(interface, case),
        interface.run_kwargs...,
    )
end

"""
    ClimaCalibrate.forward_model(interface, iteration, member)

Run every case for one ensemble member.

Kept intact so the stock `JuliaBackend` and `HPCBackend` work unchanged; the `WorkerBackend` path
overrides `run_iteration` instead to flatten `(member, case)` into one task pool.
"""
function ClimaCalibrate.forward_model(
    interface::SocratesInterface,
    iteration,
    member,
)
    for case in interface.cases
        run_case_for_member(interface, iteration, member, case)
    end
    return nothing
end

"""
    ClimaCalibrate.observation_map(interface, iteration)

`G_ensemble` for `iteration`, assembled and validated by `GEnsembleBuilder`.
"""
ClimaCalibrate.observation_map(interface::SocratesInterface, iteration) =
    build_g_ensemble(interface, iteration)

ClimaCalibrate.model_interface_filepath(::SocratesInterface) =
    abspath(joinpath(@__DIR__, "SocratesCalibration.jl"))

ClimaCalibrate.experiment_dir(::SocratesInterface) =
    abspath(joinpath(@__DIR__, "..", ".."))

"""
    observations(interface)

The `EKP.Observation` vector for this interface's cases, in `cases` order.
"""
observations(interface::SocratesInterface) = observation_vector(
    interface.cases;
    transform = interface.transform,
    source = interface.reference_source,
    vars = interface.vars,
    grids = interface.grids,
    float_type = interface.float_type,
)

"""
    observation_series(interface)

Every case as one `EKP.Observation`, one block per case, wrapped in a single-entry
`EKP.ObservationSeries`.

The series axis is for repeated observations of the same quantity, so its entries are assumed to
share a length (`list_update_groups_over_minibatch` strides by `length(get_obs(os)) / len_mb`).
Cases have different lengths — different columns, different `z_bounds` — so they are blocks of one
observation, not entries of the series. `combine_observations` appends the covariance blocks, so Γ
stays block-diagonal across cases and `y` is unchanged.
"""
observation_series(interface::SocratesInterface) =
    EKP.ObservationSeries(EKP.combine_observations(observations(interface)))
