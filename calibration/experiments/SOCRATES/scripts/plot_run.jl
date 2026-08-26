# Plot one run against the Atlas LES. Loads the scoring layer only — no EKP, no ClimaCalibrate.
#
# Profiles are on the scored levels and averaged over the scoring window, so what is drawn is exactly
# what `compare_to_les` scores.

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

EXPERIMENT = joinpath(@__DIR__, "..")

include(joinpath(EXPERIMENT, "src", "scoring", "SocratesScoring.jl"))
include(joinpath(EXPERIMENT, "plots", "SocratesPlots.jl"))

RUN = get(ENV, "SOCRATES_RUN", joinpath(EXPERIMENT, "runs", "cases", "rf09"))
case = SocratesScoring.SM.socrates_case("RF09_Obs")
FIGURES = joinpath(EXPERIMENT, "runs", "figures")

PROFILE_VARS = ("clw", "cli", "husra", "hussn")
PATH_VARS = ("lwp", "iwp", "rwp", "swp")

window = SocratesScoring.SM.score_window(case)
bounds = SocratesScoring.z_bounds(case)
model = SocratesScoring.run_outputvars(RUN)
reference = SocratesScoring.les_outputvars(case)
levels = SocratesScoring.model_levels(model["clw"], bounds)

@info "plotting" RUN case = SocratesScoring.case_name(case) window bounds n_levels =
    length(levels)

z = Dict(v => levels for v in PROFILE_VARS)
reference_by_var = Dict{String, Vector{Float64}}()
groups_by_var = Dict{String, Vector{Pair{String, Vector{Vector{Float64}}}}}()
for var in PROFILE_VARS
    m = SocratesScoring.windowed_time_mean(
        SocratesScoring.restrict_to_levels(model[var], levels),
        window,
    )
    r = SocratesScoring.windowed_time_mean(
        SocratesScoring.reference_on_levels(reference[var], levels),
        window,
    )
    reference_by_var[var] = vec(Array{Float64}(r.data))
    groups_by_var[var] = ["model" => [vec(Array{Float64}(m.data))]]
end

SocratesPlots.profile_grid(
    z, reference_by_var, groups_by_var, PROFILE_VARS;
    path = joinpath(FIGURES, "$(SocratesScoring.case_name(case))_profiles.png"),
    xlabels = Dict(v => "kg/kg" for v in PROFILE_VARS),
    title = "$(SocratesScoring.case_name(case)) — scored levels, averaged over $(round.(Int, window)) s",
)

SocratesPlots.scalar_comparison(
    collect(PATH_VARS),
    [only(SocratesScoring.windowed_time_mean(reference[v], window).data) for v in PATH_VARS],
    ["model" => [[only(SocratesScoring.windowed_time_mean(model[v], window).data) for v in PATH_VARS]]];
    path = joinpath(FIGURES, "$(SocratesScoring.case_name(case))_paths.png"),
    title = "$(SocratesScoring.case_name(case)) water paths",
    ylabel = "kg/m²",
)

# The same numbers, as a table
SocratesScoring.print_comparison(SocratesScoring.compare_to_les(RUN, case))
