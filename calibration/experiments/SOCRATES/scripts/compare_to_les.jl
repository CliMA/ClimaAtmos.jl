# Score a run against the Atlas LES. No EKP, no ClimaCalibrate.

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "scoring", "SocratesScoring.jl"))

RUN = get(ENV, "SOCRATES_RUN", joinpath(@__DIR__, "..", "runs", "cases", "rf09"))
case = SocratesScoring.SocratesModel.socrates_case("RF09_Obs")

@info "scoring" RUN case = SocratesScoring.SocratesModel.case_name(case) window = SocratesScoring.SocratesModel.score_window(case) bounds =
SocratesScoring.z_bounds(case)

comparison = SocratesScoring.compare_to_les(RUN, case)
SocratesScoring.print_comparison(comparison)

# `comparison[var]` also carries the normalized vectors actually compared, so a bad variable can be
# looked at directly:
#   c = comparison["clw"]; c.model, c.reference, c.pool_var
#
# To read the reference straight from the SSCF artifact instead of the reduced files:
#   SS.compare_to_les(RUN, case; source = :sscf)