"""
    SocratesModel

Set up and run the SOCRATES Atlas LES single-column cases in ClimaAtmos.

This layer is self-contained: it knows about cases, vertical grids, forcing, the model, and how
to run one case or many. It has no notion of observations, EKP, or ClimaCalibrate, so running a
case — including one with a calibrated parameter set — costs nothing but ClimaAtmos and SSCF.

# Running

```julia
include(".../src/model/SocratesModel.jl")

SocratesModel.run_case(SocratesModel.socrates_case("RF09_Obs"); output_dir = "runs/rf09")

# a calibrated parameter set, a coarser grid, Float32, a shorter run
SocratesModel.run_case(SocratesModel.socrates_case("RF09_Obs");
    params = "calibrated_parameters.toml", z_elem = 30, FT = Float32, t_end = 3600)

# every case, in parallel across an existing worker pool
SocratesModel.run_cases(SocratesModel.socrates_cases(); executor = SocratesModel.WorkerPoolExecutor(pool),
             output_dir = "runs/all")
```

Build a simulation without solving it — for inspection, custom callbacks, or debugging — with
[`socrates_simulation`](@ref).
"""
module SocratesModel

using ClimaAtmos: ClimaAtmos as CA
using ClimaComms: ClimaComms
using ClimaParams: ClimaParams
using Dates: Dates
using Distributed: Distributed
using NCDatasets: NCDatasets as NC
using SOCRATESSingleColumnForcings: SOCRATESSingleColumnForcings as SSCF
using TOML: TOML


# no exports, qualified uses only, especially as this is a throwaway module for running sims

include("cases.jl")
include("grid.jl")
include("params.jl")
include("memory_tvi.jl")
include("forcing.jl")
include("setup.jl")
include("diagnostics.jl")
include("transport_diagnostics.jl")
include("model.jl")
include("run.jl")
include("climacolumn.jl")

end # module