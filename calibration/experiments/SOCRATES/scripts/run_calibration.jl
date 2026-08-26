# Calibrate the SOCRATES microphysics timescales against the Atlas LES.
#
# Work is distributed as a flat pool of (member, case) runs, so the worker count is not capped by the
# ensemble size: 10 members over 11 cases is 110 independent tasks.

using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Distributed: Distributed
using ClimaCalibrate: ClimaCalibrate

EXPERIMENT = joinpath(@__DIR__, "..")
OUTPUT = joinpath(EXPERIMENT, "calibrations", "run01")

ENSEMBLE_SIZE = 16
N_ITERATIONS = 20
T_STOPS = [1.0, 10.0, 100.0, 1000.0]

Δt = 10.0
CFL_limit = 0.5 # it's supposed to be implicit but apparently that's not working out for us
w_up_max = 5.0
dz_min = w_up_max * Δt / CFL_limit
Δt /= 2 # we had some crashes, see if this helps...


# Work is ENSEMBLE_SIZE * n_cases tasks, so more workers than ENSEMBLE_SIZE are useful. The binding
# constraint is memory: a worker holds ~7 GiB on a coarse grid and ~9 GiB on the native 320-level one,
# against a per-user cgroup cap (`systemctl show user-$UID.slice -p MemoryMax`) shared with everything
# else you are running. The first task on a worker costs ~570 s of compilation and every later one ~10 s,
# so the pool is worth keeping alive across iterations rather than resizing it.
N_WORKERS = 41

Pkg.precompile()

# Start the workers and load the model code on all of them, including here.
#
# `addprocs` and `@everywhere` both block, so once they return every worker has the code — there is no
# window in which a task can be sent to a worker that cannot deserialize it. `ClimaCalibrate.add_workers`
# / `@worker_setup` are the asynchronous equivalents, meant for cluster backends where workers join over
# the life of the job; for a local pool started up front they only add a race.
#
# Only the shortfall is added, so re-including this file in a live session does not keep growing the
# pool. The `@everywhere` include must be the only one: it also runs here, and including the module
# separately beforehand would define it twice, so objects built from the first definition would not
# match methods looked up in the second.
let missing_workers = N_WORKERS - (Distributed.nprocs() - 1)
    @info "worker pool" requested = N_WORKERS existing = Distributed.nprocs() - 1 adding =
        max(missing_workers, 0)
    missing_workers > 0 &&
        Distributed.addprocs(missing_workers; exeflags = "--project=$(EXPERIMENT)")
end
Distributed.@everywhere begin
    include(joinpath($EXPERIMENT, "src", "calibration", "SocratesCalibration.jl"))
end

# cases = SocratesCalibration.SS.SM.socrates_cases()
cases = SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase[
    SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase( 1, SocratesCalibration.SocratesScoring.SocratesModel.SSCF.ObsForcing()),
    SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase( 9, SocratesCalibration.SocratesScoring.SocratesModel.SSCF.ObsForcing()),
    SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase(10, SocratesCalibration.SocratesScoring.SocratesModel.SSCF.ObsForcing()),
    SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase(12, SocratesCalibration.SocratesScoring.SocratesModel.SSCF.ObsForcing()),
    SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase(13, SocratesCalibration.SocratesScoring.SocratesModel.SSCF.ObsForcing()),
#  SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase( 1, SocratesCalibration.SocratesScoring.SocratesModel.SSCF.ERA5Forcing()),
#  SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase( 9, SocratesCalibration.SocratesScoring.SocratesModel.SSCF.ERA5Forcing()),
#  SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase(10, SocratesCalibration.SocratesScoring.SocratesModel.SSCF.ERA5Forcing()),
#  SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase(11, SocratesCalibration.SocratesScoring.SocratesModel.SSCF.ERA5Forcing()),
#  SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase(12, SocratesCalibration.SocratesScoring.SocratesModel.SSCF.ERA5Forcing()),
#  SocratesCalibration.SocratesScoring.SocratesModel.SocratesCase(13, SocratesCalibration.SocratesScoring.SocratesModel.SSCF.ERA5Forcing()),
]


grids = [
    SocratesCalibration.SocratesScoring.SocratesModel.socrates_grid(
        Float64,
        case;
        dz_min = dz_min,
    ) for case in cases
]

interface = SocratesCalibration.SocratesInterface(;
    cases,
    output_dir = OUTPUT,
    grids,
    run_kwargs = (; dt = Δt),
)

# After any `rm` above, so the workers' log files are not unlinked from under them.
Distributed.@everywhere SocratesCalibration.use_worker_log(joinpath($OUTPUT, "logs"))

prior = SocratesCalibration.default_prior()
ekp = SocratesCalibration.build_ekp(interface, prior; ensemble_size = ENSEMBLE_SIZE, T_stops = T_STOPS)

backend = ClimaCalibrate.WorkerBackend(;
    worker_pool = Distributed.WorkerPool(Distributed.workers()),
)

# ekp = SocratesCalibration.calibrate(
#     backend,
#     ekp,
#     interface;
#     prior,
#     n_iterations = N_ITERATIONS,
#     T_stops = T_STOPS,
#     overwrite = false,   # false resumes, skipping case-runs already marked complete
# )

@info "done" iterations = SocratesCalibration.EKP.get_N_iterations(ekp) T = SocratesCalibration.accumulated_T(ekp) parameters =
    SocratesCalibration.EKP.get_ϕ_mean_final(prior, ekp)

# The calibrated parameters are written per iteration as
# <OUTPUT>/iteration_XXX/member_YYY/parameters.toml, and any of them can be run directly:
#   SocratesCalibration.SS.SM.run_case(cases[1]; params = ".../parameters.toml", output_dir = "runs/calibrated")



# All three figures come from the EKP objects and `G_ensemble.jld2`, which are never pruned, so they
# work regardless of `prune_output`.
include(joinpath(EXPERIMENT, "plots", "SocratesPlots.jl"))
FIGURES = joinpath(OUTPUT, "figures")
SocratesPlots.parameter_evolution(ekp, prior; path = joinpath(FIGURES, "parameters.png"))
SocratesPlots.error_evolution(ekp; path = joinpath(FIGURES, "error.png"))

let last_iteration = ClimaCalibrate.last_completed_iteration(OUTPUT)
    g(it) = SocratesCalibration.JLD2.load_object(
        joinpath(OUTPUT, "iteration_" * lpad(it, 3, '0'), "G_ensemble.jld2"),
    )
    for (name, factor) in (("profiles.png", nothing), ("profiles_zoom.png", 2.0))
        SocratesPlots.prior_posterior_profiles(
            ekp,
            g(1),
            g(last_iteration);
            path = joinpath(FIGURES, name),
            xlim_factor = factor,
        )
    end
end

# Rerun the lowest-misfit member overall and the lowest-misfit member of the final iteration, asking
# for every 1-moment process rate in the grid mean, the updraft and the environment, plus the
# sedimentation, advection and diffusion terms of the same budgets, so the profiles above can be
# attributed to processes. This is 2 members x n_cases full-length runs, so it reuses the calibration's
# worker pool rather than running serially.
POSTPROCESS = SocratesCalibration.postprocess_best_members(
    interface;
    executor = SocratesCalibration.SS.SM.WorkerPoolExecutor(
        Distributed.WorkerPool(Distributed.workers()),
    ),
)

# One figure per prognostic variable, every flight a column, each panel carrying that variable's
# microphysics rates and its transport terms for the LES and both reruns.
BUDGET_COLUMNS = let columns = []
    for (i, case) in enumerate(cases)
        series = Dict{String, Dict{String, Vector{Float64}}}()
        z = Float64[]
        for (label, result) in POSTPROCESS
            dir = result.dirs[i]
            isnothing(dir) && continue
            terms, zc = SocratesCalibration.case_budget_terms(dir, case)
            series[label] = terms
            isempty(z) && (z = zc)
        end
        isempty(z) && continue
        series["LES"] = SocratesCalibration.case_les_rates(case, z)
        push!(columns, (SocratesCalibration.SS.case_name(case), z, series))
    end
    columns
end

for variable in SocratesCalibration.SS.SM.MP1M_BUDGET_VARS
    SocratesPlots.tendency_budget(
        variable,
        vcat(
            SocratesCalibration.SS.SM.MP1M_BUDGETS[variable],
            SocratesCalibration.SS.SM.TRANSPORT_BUDGETS[variable],
        ),
        BUDGET_COLUMNS;
        path = joinpath(FIGURES, "budget_$(variable).png"),
    )
end

# Full model profiles over the whole column (not just the scored region) need each member's NetCDF,
# which `prune_output = true` deletes. Build the interface with `prune_output = false`, then:
#   julia --project=. scripts/plot_calibration.jl