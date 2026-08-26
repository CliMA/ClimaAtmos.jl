# Plot a finished calibration: parameter and misfit evolution, and prior-vs-posterior profiles.
#
# Parameter and misfit plots need only the EKP files, so they always work. Profile plots read each
# member's model output, which `prune_output = true` deletes after every iteration — build the interface
# with `prune_output = false` if you want them.

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

EXPERIMENT = joinpath(@__DIR__, "..")
OUTPUT = joinpath(EXPERIMENT, "calibrations", "run01")
FIGURES = joinpath(OUTPUT, "figures")

include(joinpath(EXPERIMENT, "src", "calibration", "SocratesCalibration.jl"))
include(joinpath(EXPERIMENT, "plots", "SocratesPlots.jl"))

# Must match the calibration that produced OUTPUT.
cases = SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase[
    SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase(
        9,
        SocratesCalibration.SocratesScoring.SocratesModel.SSCF.ObsForcing(),
    ),
]
interface = SocratesCalibration.SocratesInterface(;
    cases,
    output_dir = OUTPUT,
    prune_output = false,
)
prior = SocratesCalibration.default_prior()

last_iteration = SocratesCalibration.ClimaCalibrate.last_completed_iteration(OUTPUT)
ekp = SocratesCalibration.ClimaCalibrate.load_ekp_struct(OUTPUT, last_iteration)
@info "loaded" last_iteration n_ens = SocratesCalibration.EKP.get_N_ens(ekp) n_iterations =
    SocratesCalibration.EKP.get_N_iterations(ekp)

# --- always available: the EKP objects alone -------------------------------------------------- #

SocratesPlots.parameter_evolution(ekp, prior; path = joinpath(FIGURES, "parameters.png"))
SocratesPlots.error_evolution(ekp; path = joinpath(FIGURES, "error.png"))

# --- profiles, from each member's model output ------------------------------------------------ #

"""Every member's scored profile (or scalar) for one variable, at one iteration."""
function member_values(interface, iteration, case, var, levels)
    window = SocratesCalibration.SS.SM.score_window(case)
    values = Vector{Float64}[]
    for member in 1:SocratesCalibration.EKP.get_N_ens(ekp)
        dir = SocratesCalibration.case_output_dir(interface, iteration, member, case)
        isdir(dir) || continue
        outputs = SocratesCalibration.SS.run_outputvars(dir, (var,))
        restricted = SocratesCalibration.SS.restrict_to_levels(outputs[var], levels)
        mean_var = SocratesCalibration.SS.windowed_time_mean(restricted, window)
        push!(values, vec(Array{Float64}(mean_var.data)))
    end
    return values
end

PROFILE_VARS = ("clw", "cli", "husra", "hussn")
PATH_VARS = ("lwp", "iwp", "rwp", "swp")
ITERATIONS = (1, last_iteration)          # prior ensemble, then the latest one

for case in cases
    name = SocratesCalibration.SS.case_name(case)
    window = SocratesCalibration.SS.SM.score_window(case)
    bounds = SocratesCalibration.SS.z_bounds(case)
    levels = SocratesCalibration.SS.scored_levels(
        SocratesCalibration.SS.SM.socrates_z(
            SocratesCalibration.case_grid(interface, case),
        ),
        bounds,
    )
    reference = SocratesCalibration.SS.les_outputvars(case)

    have = [
        it for it in ITERATIONS if
        isdir(SocratesCalibration.case_output_dir(interface, it, 1, case))
    ]
    isempty(have) && error(
        "No model output under $(OUTPUT) for $name. Profile plots read each member's run; \
         `prune_output = true` deletes it after every iteration, so rebuild the interface with \
         `prune_output = false` and rerun, or re-run the calibrated parameters with `run_case`.",
    )

    z = Dict(v => levels for v in PROFILE_VARS)
    reference_by_var = Dict{String, Vector{Float64}}()
    groups_by_var = Dict{String, Vector{Pair{String, Vector{Vector{Float64}}}}}()
    for var in PROFILE_VARS
        r = SocratesCalibration.SS.windowed_time_mean(
            SocratesCalibration.SS.reference_on_levels(reference[var], levels),
            window,
        )
        reference_by_var[var] = vec(Array{Float64}(r.data))
        groups_by_var[var] =
            ["iteration $it" => member_values(interface, it, case, var, levels) for it in have]
    end
    SocratesPlots.profile_grid(
        z, reference_by_var, groups_by_var, PROFILE_VARS;
        path = joinpath(FIGURES, "$(name)_profiles.png"),
        xlabels = Dict(v => "kg/kg" for v in PROFILE_VARS),
        title = "$name — scored levels, averaged over $(round.(Int, window)) s",
    )

    reference_paths = [
        only(SocratesCalibration.SS.windowed_time_mean(reference[v], window).data) for
        v in PATH_VARS
    ]
    path_groups = map(have) do it
        # member_values gives one 1-element vector per member; regroup into one vector per member
        # across the four path variables.
        per_var = [member_values(interface, it, case, v, levels) for v in PATH_VARS]
        n_members = minimum(length, per_var)
        "iteration $it" =>
            [[only(per_var[j][m]) for j in eachindex(PATH_VARS)] for m in 1:n_members]
    end
    SocratesPlots.scalar_comparison(
        collect(PATH_VARS), reference_paths, path_groups;
        path = joinpath(FIGURES, "$(name)_paths.png"),
        title = "$name water paths", ylabel = "kg/m²",
    )
end

@info "figures written" FIGURES
