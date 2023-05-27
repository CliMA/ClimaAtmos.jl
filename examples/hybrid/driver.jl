import ClimaAtmos.Parameters as CAP
import ClimaAtmos as CA
import Thermodynamics as TD
import ClimaComms
using OrdinaryDiffEq
using PrettyTables
using DiffEqCallbacks
using JLD2
using NCDatasets
using ClimaCore
using ClimaTimeSteppers
import Random
import ClimaCore
using Statistics: mean
import ClimaAtmos.InitialConditions as ICs
Random.seed!(1234)

config = CA.AtmosConfig()
integrator = CA.get_integrator(config)
sol_res = CA.solve_atmos!(integrator)

(; simulation, atmos, params) = integrator.p
(; p) = integrator
import JSON
using Test
import OrderedCollections
using ClimaCoreTempestRemap
using ClimaCorePlots, Plots
using ClimaCoreMakie, CairoMakie
include(joinpath(pkgdir(CA), "post_processing", "contours_and_profiles.jl"))
include(joinpath(pkgdir(CA), "post_processing", "post_processing_funcs.jl"))
include(
    joinpath(pkgdir(CA), "post_processing", "define_tc_quicklook_profiles.jl"),
)

ref_job_id = config.parsed_args["reference_job_id"]
reference_job_id = isnothing(ref_job_id) ? simulation.job_id : ref_job_id

is_edmfx =
    atmos.turbconv_model isa CA.EDMFX ||
    atmos.turbconv_model isa CA.DiagnosticEDMFX
if is_edmfx && config.parsed_args["post_process"]
    contours_and_profiles(simulation.output_dir, reference_job_id)
    zip_and_cleanup_output(simulation.output_dir, "hdf5files.zip")
end

if config.parsed_args["debugging_tc"] && !is_edmfx
    include(
        joinpath(
            @__DIR__,
            "..",
            "..",
            "regression_tests",
            "self_reference_or_path.jl",
        ),
    )

    main_branch_root = get_main_branch_buildkite_path()
    main_branch_data_path = joinpath(main_branch_root, reference_job_id)

    day = floor(Int, simulation.t_end / (60 * 60 * 24))
    sec = floor(Int, simulation.t_end % (60 * 60 * 24))

    zip_file = "hdf5files.zip"
    mktempdir(
        simulation.output_dir;
        prefix = "temp_unzip_path_",
    ) do temp_main_branch_path
        # Unzip files to temp directory, to avoid collisions with other jobs
        unzip_file_in_path(
            main_branch_data_path,
            zip_file,
            temp_main_branch_path,
        )
        # hdf5 files from the main branch are in `temp_dir`

        plot_tc_contours(
            simulation.output_dir;
            main_branch_data_path = temp_main_branch_path,
        )
        plot_tc_profiles(
            simulation.output_dir;
            hdf5_filename = "day$day.$sec.hdf5",
            main_branch_data_path = temp_main_branch_path,
        )
    end
    if atmos.model_config isa CA.SingleColumnModel
        zip_and_cleanup_output(simulation.output_dir, zip_file)
    end
end

if sol_res.ret_code == :simulation_crashed
    error(
        "The ClimaAtmos simulation has crashed. See the stack trace for details.",
    )
end
# Simulation did not crash
(; sol, walltime) = sol_res
@assert last(sol.t) == simulation.t_end
CA.verify_callbacks(sol.t)

if CA.is_distributed(config.comms_ctx)
    CA.export_scaling_file(
        sol,
        simulation.output_dir,
        walltime,
        config.comms_ctx,
        ClimaComms.nprocs(config.comms_ctx),
    )
end

if !CA.is_distributed(config.comms_ctx) &&
   config.parsed_args["post_process"] &&
   !is_edmfx &&
   !(atmos.model_config isa CA.SphericalModel)
    ENV["GKSwstype"] = "nul" # avoid displaying plots
    if CA.is_column_without_edmf(config.parsed_args)
        custom_postprocessing(sol, simulation.output_dir, p)
    elseif CA.is_column_edmf(config.parsed_args)
        postprocessing_edmf(
            sol,
            simulation.output_dir,
            config.parsed_args["fps"],
        )
    elseif CA.is_solid_body(config.parsed_args)
        postprocessing(sol, simulation.output_dir, config.parsed_args["fps"])
    elseif atmos.model_config isa CA.BoxModel
        postprocessing_box(sol, simulation.output_dir)
    elseif atmos.model_config isa CA.PlaneModel
        postprocessing_plane(sol, simulation.output_dir, p)
    else
        error("Uncaught case")
    end
end

include(joinpath(@__DIR__, "..", "..", "regression_tests", "mse_tables.jl"))
if config.parsed_args["regression_test"]
    # Test results against main branch
    include(
        joinpath(
            @__DIR__,
            "..",
            "..",
            "regression_tests",
            "regression_tests.jl",
        ),
    )
    @testset "Test regression table entries" begin
        mse_keys = sort(collect(keys(all_best_mse[simulation.job_id])))
        pcs = collect(Fields.property_chains(sol.u[end]))
        for prop_chain in mse_keys
            @test prop_chain in pcs
        end
    end
    perform_regression_tests(
        simulation.job_id,
        sol.u[end],
        all_best_mse,
        simulation.output_dir,
    )
end



if config.parsed_args["check_conservation"]
    FT = Spaces.undertype(axes(sol.u[end].c.ρ))
    @test sum(sol.u[1].c.ρ) ≈ sum(sol.u[end].c.ρ) rtol = 25 * eps(FT)
    @test sum(sol.u[1].c.ρe_tot) +
          (p.net_energy_flux_sfc[][] - p.net_energy_flux_toa[][]) ≈
          sum(sol.u[end].c.ρe_tot) rtol = 30 * eps(FT)
end
