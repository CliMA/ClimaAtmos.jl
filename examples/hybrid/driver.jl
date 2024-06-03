# When Julia 1.10+ is used interactively, stacktraces contain reduced type information to make them shorter.
# On the other hand, the full type information is printed when julia is not run interactively.
# Given that ClimaCore objects are heavily parametrized, non-abbreviated stacktraces are hard to read,
# so we force abbreviated stacktraces even in non-interactive runs.
# (See also Base.type_limited_string_from_context())
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Random
Random.seed!(1234)

import ClimaComms

if !(@isdefined config)
    (; config_file, job_id) = CA.commandline_kwargs()
    config = CA.AtmosConfig(config_file; job_id)
end
simulation = CA.get_simulation(config)
(; integrator) = simulation
sol_res = CA.solve_atmos!(simulation)

(; atmos, params) = integrator.p
(; p) = integrator

import ClimaCore
import ClimaCore: Topologies, Quadratures, Spaces
import ClimaAtmos.InitialConditions as ICs
using Statistics: mean
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD
using SciMLBase
using PrettyTables
using JLD2
using NCDatasets
using ClimaTimeSteppers
import JSON
using Test
import Tar
import Base.Filesystem: rm
import OrderedCollections
include(joinpath(pkgdir(CA), "post_processing", "ci_plots.jl"))

ref_job_id = config.parsed_args["reference_job_id"]
reference_job_id = isnothing(ref_job_id) ? simulation.job_id : ref_job_id

if sol_res.ret_code == :simulation_crashed
    error(
        "The ClimaAtmos simulation has crashed. See the stack trace for details.",
    )
end
# Simulation did not crash
(; sol, walltime) = sol_res

# we gracefully exited, so we won't have reached t_end
if !isempty(integrator.tstops)
    @assert last(sol.t) == simulation.t_end
end
CA.verify_callbacks(sol.t)

# Scaling check
if CA.is_distributed(config.comms_ctx)
    nprocs = ClimaComms.nprocs(config.comms_ctx)
    comms_ctx = config.comms_ctx
    output_dir = simulation.output_dir
    # replace sol.u on the root processor with the global sol.u
    if ClimaComms.iamroot(comms_ctx)
        Y = sol.u[1]
        center_space = axes(Y.c)
        horz_space = Spaces.horizontal_space(center_space)
        horz_topology = Spaces.topology(horz_space)
        quadrature_style = Spaces.quadrature_style(horz_space)
        Nq = Quadratures.degrees_of_freedom(quadrature_style)
        nlocalelems = Topologies.nlocalelems(horz_topology)
        ncols_per_process = nlocalelems * Nq * Nq
        scaling_file =
            joinpath(output_dir, "scaling_data_$(nprocs)_processes.jld2")
        @info(
            "Writing scaling data",
            "walltime (seconds)" = walltime,
            scaling_file
        )
        JLD2.jldsave(scaling_file; nprocs, ncols_per_process, walltime)
    end
end

# Check if selected output has changed from the previous recorded output (bit-wise comparison)
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

@info "Callback verification, n_expected_calls: $(CA.n_expected_calls(integrator))"
@info "Callback verification, n_measured_calls: $(CA.n_measured_calls(integrator))"

# Conservation checks
if config.parsed_args["check_conservation"]
    FT = Spaces.undertype(axes(sol.u[end].c.ρ))
    @info "Checking conservation"
    (; energy_conservation, mass_conservation, water_conservation) =
        CA.check_conservation(sol)

    @info "    Net energy change / total energy: $energy_conservation"
    @info "    Net mass change / total mass: $mass_conservation"
    @info "    Net water change / total water: $water_conservation"

    sfc = p.atmos.surface_model

    if CA.has_no_source_or_sink(config.parsed_args)
        @test energy_conservation ≈ 0 atol = 50 * eps(FT)
        @test mass_conservation ≈ 0 atol = 50 * eps(FT)
        @test water_conservation ≈ 0 atol = 50 * eps(FT)
    else
        @test energy_conservation ≈ 0 atol = sqrt(eps(FT))
        @test mass_conservation ≈ 0 atol = sqrt(eps(FT))
        if sfc isa CA.PrognosticSurfaceTemperature
            @test water_conservation ≈ 0 atol = sqrt(eps(FT))
        end
    end
end

# Visualize the solution
if ClimaComms.iamroot(config.comms_ctx)
    include(
        joinpath(pkgdir(CA), "regression_tests", "self_reference_or_path.jl"),
    )
    @info "Plotting"
    path = self_reference_or_path() # __build__ path (not job path)
    if path == :self_reference
        make_plots(Val(Symbol(reference_job_id)), simulation.output_dir)
    else
        main_job_path = joinpath(path, reference_job_id)
        nc_dir = joinpath(main_job_path, "nc_files")
        if ispath(nc_dir)
            @info "nc_dir exists"
        else
            mkpath(nc_dir)
            # Try to extract nc files from tarball:
            @info "Comparing against $(readdir(nc_dir))"
        end
        if isempty(readdir(nc_dir))
            if isfile(joinpath(main_job_path, "nc_files.tar"))
                Tar.extract(joinpath(main_job_path, "nc_files.tar"), nc_dir)
            else
                @warn "No nc_files found"
            end
        else
            @info "Files already extracted"
        end

        paths = if isempty(readdir(nc_dir))
            simulation.output_dir
        else
            [simulation.output_dir, nc_dir]
        end
        make_plots(Val(Symbol(reference_job_id)), paths)
    end
    @info "Plotting done"

    if islink(simulation.output_dir)
        symlink_to_fullpath(path) = joinpath(dirname(path), readlink(path))
    else
        symlink_to_fullpath(path) = path
    end

    @info "Creating tarballs"
    # These NC files are used by our reproducibility tests,
    # and need to be found later when comparing against the
    # main branch. If "nc_files.tar" is renamed, then please
    # search for "nc_files.tar" globally and rename it in the
    # reproducibility test folder.
    Tar.create(
        f -> endswith(f, ".nc"),
        symlink_to_fullpath(simulation.output_dir),
        joinpath(simulation.output_dir, "nc_files.tar"),
    )
    Tar.create(
        f -> endswith(f, r"hdf5|h5"),
        symlink_to_fullpath(simulation.output_dir),
        joinpath(simulation.output_dir, "hdf5_files.tar"),
    )

    foreach(readdir(simulation.output_dir)) do f
        endswith(f, r"nc|hdf5|h5") && rm(joinpath(simulation.output_dir, f))
    end
    @info "Tarballs created"
end
