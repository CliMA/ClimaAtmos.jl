# When Julia 1.10+ is used interactively, stacktraces contain reduced type information to make them shorter.
# On the other hand, the full type information is printed when julia is not run interactively.
# Given that ClimaCore objects are heavily parametrized, non-abbreviated stacktraces are hard to read,
# so we force abbreviated stacktraces even in non-interactive runs.
# (See also Base.type_limited_string_from_context())
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
# PrecompileCI is a local package that forces commonly used methods to be precompiled,
# allowing them to be reused between Julia sessions.
# To load in the precompiled methods, run `using PrecompileCI` before loading ClimaAtmos.
# To see what methods are precompiled, open julia: `julia --project=.buildkite/PrecompileCI`
# and run `using PrecompileTools; PrecompileTools.verbose[] = true; include(".buildkite/PrecompileCI/src/PrecompileCI.jl")`
haskey(ENV, "CI") && (using PrecompileCI)

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Random
Random.seed!(1234)

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
import ClimaCore: Topologies, Quadratures, Spaces, Fields
import ClimaComms
using SciMLBase
using PrettyTables
using JLD2
using NCDatasets
using ClimaTimeSteppers
using Test
import Tar
import Base.Filesystem: rm
import Statistics: mean
import LinearAlgebra: norm_sqr, diag, UniformScaling
include(joinpath(pkgdir(CA), "post_processing", "ci_plots.jl"))

ref_job_id = config.parsed_args["reference_job_id"]
reference_job_id = isnothing(ref_job_id) ? simulation.job_id : ref_job_id

if (
    config.parsed_args["debug_jacobian"] &&
    !config.parsed_args["use_dense_jacobian"]
)
    @info "Debugging Jacobian in first column of final state"
    include(joinpath(@__DIR__, "..", "post_processing", "jacobian_summary.jl"))
    print_jacobian_summary(integrator)
end

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
if config.parsed_args["reproducibility_test"]
    # Test results against main branch
    include(
        joinpath(
            @__DIR__,
            "..",
            "reproducibility_tests",
            "reproducibility_tools.jl",
        ),
    )
    export_reproducibility_results(
        sol.u[end],
        config.comms_ctx;
        job_id = simulation.job_id,
        computed_dir = simulation.output_dir,
    )
end

@info "Callback verification, n_expected_calls: $(CA.n_expected_calls(integrator))"
@info "Callback verification, n_measured_calls: $(CA.n_measured_calls(integrator))"

# Write diagnostics that are in DictWriter to text files
CA.write_diagnostics_as_txt(simulation)

if config.parsed_args["check_steady_state"]
    Y_end = integrator.sol.u[end]
    t_end = integrator.sol.t[end]
    (; steady_state_velocity, params) = integrator.p
    (; zd_rayleigh) = params
    FT = eltype(Y_end)

    @info "Comparing velocity fields to predicted steady state at t = $t_end"
    ᶜu_normsqr = norm_sqr.(steady_state_velocity.ᶜu)
    ᶠu_normsqr = norm_sqr.(steady_state_velocity.ᶠu)
    ᶜuₕ_err_normsqr = norm_sqr.(Y_end.c.uₕ .- CA.C12.(steady_state_velocity.ᶜu))
    ᶠu₃_err_normsqr = norm_sqr.(Y_end.f.u₃ .- CA.C3.(steady_state_velocity.ᶠu))

    # Average all errors below the sponge layer.
    ᶜsponge_mask = FT.(Fields.coordinate_field(Y_end.c).z .< zd_rayleigh)
    ᶠsponge_mask = FT.(Fields.coordinate_field(Y_end.f).z .< zd_rayleigh)
    ᶜu_rms = sqrt(sum(ᶜu_normsqr .* ᶜsponge_mask) / sum(ᶜsponge_mask))
    ᶠu_rms = sqrt(sum(ᶠu_normsqr .* ᶠsponge_mask) / sum(ᶠsponge_mask))
    ᶜuₕ_rmse = sqrt(sum(ᶜuₕ_err_normsqr .* ᶜsponge_mask) / sum(ᶜsponge_mask))
    ᶠu₃_rmse = sqrt(sum(ᶠu₃_err_normsqr .* ᶠsponge_mask) / sum(ᶠsponge_mask))
    ᶜuₕ_rel_err = ᶜuₕ_rmse / ᶜu_rms
    ᶠu₃_rel_err = ᶠu₃_rmse / ᶠu_rms

    # Average the errors on several levels close to the surface.
    n_levels = 3
    level_uₕ_rel_errs = map(1:n_levels) do level
        level_u_rms = sqrt(mean(Fields.level(ᶜu_normsqr, level)))
        level_uₕ_rmse = sqrt(mean(Fields.level(ᶜuₕ_err_normsqr, level)))
        level_uₕ_rmse / level_u_rms
    end
    level_u₃_rel_errs = map((1:n_levels) .- Fields.half) do level
        level_u_rms = sqrt(mean(Fields.level(ᶠu_normsqr, level)))
        level_u₃_rmse = sqrt(mean(Fields.level(ᶠu₃_err_normsqr, level)))
        level_u₃_rmse / level_u_rms
    end

    @info "    Absolute RMSE of uₕ below sponge layer: $ᶜuₕ_rmse"
    @info "    Absolute RMSE of u₃ below sponge layer: $ᶠu₃_rmse"
    @info "    Relative RMSE of uₕ below sponge layer: $ᶜuₕ_rel_err"
    @info "    Relative RMSE of u₃ below sponge layer: $ᶠu₃_rel_err"
    @info "    Relative RMSE of uₕ on $n_levels levels closest to the surface:"
    @info "        $level_uₕ_rel_errs"
    @info "    Relative RMSE of u₃ on $n_levels levels closest to the surface:"
    @info "        $level_u₃_rel_errs"

    if t_end > 24 * 60 * 60
        # TODO: Float32 simulations currently show significant divergence of uₕ.
        @test ᶜuₕ_rel_err < (FT == Float32 ? 0.05 : 0.005)
        @test ᶠu₃_rel_err < 0.0005
    end
end

# Conservation checks
if config.parsed_args["check_conservation"]
    FT = Spaces.undertype(axes(sol.u[end].c.ρ))
    @info "Checking conservation"
    (; energy_conservation, mass_conservation, water_conservation) =
        CA.check_conservation(sol)

    @info "    Net energy change / total energy: $energy_conservation"
    @info "    Net mass change / total mass: $mass_conservation"
    @info "    Net water change / total water: $water_conservation"

    @test energy_conservation ≈ 0 atol = 100 * eps(FT)
    @test mass_conservation ≈ 0 atol = 250 * eps(FT)
    @test water_conservation ≈ 0 atol = 1000 * eps(FT)
end

# Visualize the solution
if ClimaComms.iamroot(config.comms_ctx)
    include(
        joinpath(
            pkgdir(CA),
            "reproducibility_tests",
            "reproducibility_utils.jl",
        ),
    )
    @info "Plotting"
    paths = latest_comparable_dirs() # __build__ path (not job path)
    if isempty(paths)
        make_plots(Val(Symbol(reference_job_id)), simulation.output_dir)
    else
        main_job_path = joinpath(first(paths), reference_job_id)
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
