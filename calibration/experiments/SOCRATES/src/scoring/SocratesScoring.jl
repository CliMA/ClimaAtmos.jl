"""
    SocratesScoring

Compare SOCRATES runs against the Atlas LES reference.

This layer owns the reference data, the vertical bounds of the scored region, and the score
normalization — everything needed to say how far a run is from the LES, with no notion of EKP or
ClimaCalibrate. `scripts/compare_to_les.jl` uses it directly, and the calibration layer builds its
observations and `G` on top of it.

```julia
include(".../src/scoring/SocratesScoring.jl")
const SS = SocratesScoring
case = SS.SM.socrates_case("RF09_Obs")

SS.les_outputvars(case)                 # reference, as ClimaAnalysis OutputVars
SS.z_bounds(case)                       # scored region
SS.compare_to_les("runs/rf09", case)    # normalized misfit per variable
```
"""
module SocratesScoring

using ClimaAnalysis: ClimaAnalysis
using NCDatasets: NCDatasets as NC
using SOCRATESSingleColumnForcings: SOCRATESSingleColumnForcings as SSCF
using Statistics: Statistics

# The model layer supplies case identity, the scoring window and the vertical grid; scoring is
# defined against those, so it depends on them rather than restating them.
include(joinpath(@__DIR__, "..", "model", "SocratesModel.jl"))
using .SocratesModel:
    SocratesModel as SM,
    SocratesCase,
    case_name,
    score_window,
    socrates_z,
    z_max_default

# no exports, qualified uses only

include("reference.jl")
include("z_bounds.jl")
include("score.jl")
include("compare.jl")

end # module
