# Run SOCRATES cases. Edit and run in a REPL, or `julia --project=. scripts/run_cases.jl`.
#
# Loads the model layer only: no EKP, no ClimaCalibrate.

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "model", "SocratesModel.jl"))

OUTPUT = joinpath(@__DIR__, "..", "runs", "cases")

# --- one case, on the Atlas LES's own vertical grid ------------------------------------------- #

case = SocratesModel.socrates_case("RF09_Obs")
dir = SocratesModel.run_case(case; output_dir = joinpath(OUTPUT, "rf09"))
@info "wrote" dir

# --- other things you might want ------------------------------------------------------------- #

# A calibrated (or any) parameter set, as a TOML path:
#   SocratesModel.run_case(case; params = "path/to/parameters.toml", output_dir = ...)
#
# Inline overrides, no file needed. Later sources win, and paths and Dicts can be mixed:
#   SocratesModel.run_case(case;
#       params = Dict("rain_autoconversion_timescale" => Dict("value" => 500.0, "type" => "float")),
#       output_dir = ...)
#
# A different vertical resolution: `grid` is the single knob, built by `socrates_grid`.
#   grid = SocratesModel.socrates_grid(Float64, case; dz_min = 200)   # merge adjacent LES cells
#   grid = SocratesModel.socrates_grid(Float64, case; faces = collect(range(0, 6000; length = 61)))
#   SocratesModel.run_case(case; grid, output_dir = ...)
#
# Any ClimaAtmos grid works too:
#   grid = SocratesModel.CA.ColumnGrid(Float64; z_elem = 60,
#                                      z_max = SocratesModel.z_max_default(case),
#                                      z_stretch = true, dz_bottom = 30)
#
# Float32:
#   SocratesModel.run_case(case; FT = Float32, output_dir = ...)
#
# A shorter run:
#   SocratesModel.run_case(case; t_end = 3600, output_dir = ...)

# --- every case ------------------------------------------------------------------------------- #
# Serial by default. For a worker pool:
#
#   using Distributed
#   addprocs(6)
#   @everywhere include(joinpath(@__DIR__, "..", "src", "model", "SocratesModel.jl"))
#   executor = SocratesModel.WorkerPoolExecutor(WorkerPool(workers()))
#   SocratesModel.run_cases(SocratesModel.socrates_cases(); executor, output_dir = OUTPUT)

# dirs = SocratesModel.run_cases(SocratesModel.socrates_cases(); output_dir = OUTPUT)

# --- build a simulation without solving it, to inspect or add callbacks ----------------------- #
# sim = SocratesModel.socrates_simulation(Float64, case; output_dir = joinpath(OUTPUT, "inspect"))