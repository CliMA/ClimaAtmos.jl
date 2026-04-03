"""
Runs a set of Larcform1 sensitivity tests from a baseline AtmosConfig and
generates the standard CI plots for each case.

Launch with:

julia --project=.buildkite runscripts/larcform1_sensitivity_driver.jl
"""

# When Julia 1.10+ is used interactively, stacktraces contain reduced type information to make them shorter.
# On the other hand, the full type information is printed when julia is not run interactively.
# Given that ClimaCore objects are heavily parametrized, non-abbreviated stacktraces are hard to read,
# so we force abbreviated stacktraces even in non-interactive runs.
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Random
Random.seed!(1234)

include(joinpath(pkgdir(CA), "post_processing", "ci_plots.jl"))

const BASE_CONFIG_FILE = "config/model_configs/larcform1_1M_prognostic_edmfx.yml"
const OUTPUT_ROOT = joinpath("output", "larcform1_sensitivity")
const REFERENCE_JOB_ID = "larcform1_1M"

const VARIANTS = [
    (
        name = "larcform1_prog_base",
        description = "Baseline prognostic EDMFx run",
        overrides = Dict{String, Any}(),
    ),
    (
        name = "larcform1_prog_no_impdiff",
        description = "Disable implicit diffusion",
        overrides = Dict{String, Any}(
            "implicit_diffusion" => false,
        ),
    ),
    (
        name = "larcform1_prog_dt30s",
        description = "Reduce timestep to 30 seconds",
        overrides = Dict{String, Any}(
            "dt" => "30secs",
        ),
    ),
    (
        name = "larcform1_prog_imp_sgs_adv",
        description = "Enable implicit SGS advection",
        overrides = Dict{String, Any}(
            "implicit_sgs_advection" => true,
        ),
    ),
    (
        name = "larcform1_prog_fine_surface_grid",
        description = "Increase lower-level vertical resolution",
        overrides = Dict{String, Any}(
            "dz_bottom" => 10.0,
            "z_elem" => 110,
        ),
    ),
]

function make_variant(base::CA.AtmosConfig, variant_name::AbstractString, overrides)
    args = copy(base.parsed_args)
    merge!(args, overrides)
    args["output_dir"] = joinpath(OUTPUT_ROOT, variant_name)
    args["reference_job_id"] = REFERENCE_JOB_ID
    return CA.AtmosConfig(args; job_id = variant_name, comms_ctx = base.comms_ctx)
end

function plot_case(reference_job_id::AbstractString, simulation, comms_ctx)
    if ClimaComms.iamroot(comms_ctx)
        @info "Plotting case" job_id = simulation.job_id output_dir = simulation.output_dir
        make_plots(Val(Symbol(reference_job_id)), simulation.output_dir)
    end
end

function plot_comparison(reference_job_id::AbstractString, output_dirs, comms_ctx)
    if ClimaComms.iamroot(comms_ctx) && length(output_dirs) > 1
        @info "Plotting comparison" reference_job_id output_dirs
        make_plots(Val(Symbol(reference_job_id)), output_dirs)
    end
end

function run_variant(base::CA.AtmosConfig, variant)
    config = make_variant(base, variant.name, variant.overrides)
    @info "Starting sensitivity case" name = variant.name description = variant.description overrides = variant.overrides

    simulation = CA.AtmosSimulation(config)
    sol_res = CA.solve_atmos!(simulation)
    (; sol) = sol_res

    CA.error_if_crashed(sol_res.ret_code)
    CA.verify_callbacks(sol.t)
    CA.write_diagnostics_as_txt(simulation)

    ref_job_id = get(config.parsed_args, "reference_job_id", nothing)
    reference_job_id = isnothing(ref_job_id) ? simulation.job_id : ref_job_id
    plot_case(reference_job_id, simulation, config.comms_ctx)

    return (; simulation, config, reference_job_id)
end

mkpath(OUTPUT_ROOT)

context = CA.get_comms_context(Dict("device" => "auto"))
base = CA.AtmosConfig(BASE_CONFIG_FILE; comms_ctx = context)

results = []
for variant in VARIANTS
    push!(results, run_variant(base, variant))
end

comparison_reference_job_id = first(results).reference_job_id
comparison_output_dirs = [result.simulation.output_dir for result in results]
plot_comparison(comparison_reference_job_id, comparison_output_dirs, context)

@info "Sensitivity driver complete" output_dirs = comparison_output_dirs