using Pkg: Pkg
using Revise: Revise

# Driver to run BOMEX NEQ 1M A/B (WF ON/OFF) in a temporary experiment env
# and produce NetCDF + a comparison PNG, without touching the core Project.toml.
#
# Usage 1 (REPL):
#   include("scripts/experiments/run_bomex_wf_ab_run.jl"); run_bomex_wf_ab_run("/abs/outdir")
# or just: run_bomex_wf_ab_run_now()
#
# Usage 2 (CLI) [STRONLGY NOT RECOMMENDED.. THIS IS NOT THE IDEAL WAY TO USE JULIA AND BEARS SEVERE REPEATED PRECOMPILATION COSTS]:
#   julia -q scripts/experiments/run_bomex_wf_ab_run.jl /abs/path/to/outdir
# If no argument is provided, a temporary directory is created and used.

function _setup_experiment_env()
    p = mktempdir()
    Pkg.activate(p; io = devnull)
    # Develop local ClimaAtmos checkout
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    Pkg.develop(Pkg.PackageSpec(path = repo_root); io = devnull)
    # Add plotting and NetCDF only to this temp env
    Pkg.add(["CairoMakie"]; io = devnull)
    return p
end
function run_bomex_wf_ab_run(
    base_out::AbstractString;
    t_end_secs::FT = 3600. * 12,
    dt::FT = 100.,
    turbconv::Symbol = :diagnostic_edmfx,
) where FT

    mkpath(base_out)
    _setup_experiment_env() # Create and activate temp env
    # Load/update definitions
    Revise.includet(joinpath(@__DIR__, "run_bomex_wf_ab.jl"))
    # Call latest method safely
    png_path = Base.invokelatest(
        run_bomex_wf_ab,
        base_out;
        t_end_secs = t_end_secs,
        dt = dt,
        turbconv = turbconv,
    )

    println("Saved plot: ", png_path)
    return png_path
end

function run_bomex_wf_ab_run_now(;
    t_end_secs::FT = 3600. * 12,
    dt::FT = 100.,
    turbconv::Symbol = :diagnostic_edmfx,
    ) where FT
    out = mktempdir()
    return run_bomex_wf_ab_run(out; t_end_secs=t_end_secs, dt=dt, turbconv=turbconv)
end

# Replot without rerunning, using the same temp env bootstrap
function replot_bomex_wf_ab_run(base_out::AbstractString)
    _setup_experiment_env()
    Revise.includet(joinpath(@__DIR__, "run_bomex_wf_ab.jl"))
    return Base.invokelatest(eval, :(replot_bomex_wf_ab($base_out)))
end
