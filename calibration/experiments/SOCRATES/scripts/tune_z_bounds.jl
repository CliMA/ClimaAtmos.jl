# Print the LES cloud top and the derived scored region for every case, so the buffer can be chosen
# against the data rather than guessed.

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "scoring", "SocratesScoring.jl"))

BUFFER = 1000.0
ROUND_TO = 500.0

println(
    rpad("case", 12), rpad("window [s]", 20), rpad("cloud top [m]", 15),
    rpad("derived top", 13), rpad("override", 11), rpad("in use", 11), "levels scored",
)
for case in SocratesScoring.SocratesModel.socrates_cases()
    name = SocratesScoring.SocratesModel.case_name(case)
    try
        window = SocratesScoring.SocratesModel.score_window(case)
        top = SocratesScoring.les_cloud_top(case; window)
        z_max = SocratesScoring.SocratesModel.z_max_default(case)
        derived = min(z_max, ROUND_TO * ceil((top + BUFFER) / ROUND_TO))
        # Overrides apply only to the Obs-forced cases, so only show one where it is actually used.
        override =
            case.forcing_type isa SS.SSCF.ObsForcing ?
            get(SS.OBS_Z_TOP, case.flight_number, nothing) : nothing
        bounds = SocratesScoring.z_bounds(case; buffer = BUFFER, round_to = ROUND_TO)
        levels = SocratesScoring.scored_levels(SocratesScoring.SocratesModel.socrates_z(Float64, case), bounds)
        println(
            rpad(name, 12), rpad(string(round.(Int, window)), 20),
            rpad(round(top; digits = 1), 15), rpad(derived, 13),
            rpad(isnothing(override) ? "-" : override, 11),
            rpad(last(bounds), 11), length(levels),
        )
    catch e
        println(rpad(name, 12), "skipped: ", first(sprint(showerror, e), 90))
    end
end

# To derive every case rather than honour the hand-set Obs numbers:
#   SS.z_bounds(case; overrides = Base.ImmutableDict{Int, Float64}())