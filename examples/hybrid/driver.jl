import ClimaAtmos as CA
import Random
Random.seed!(1234)

if !(@isdefined config)
    config = CA.AtmosConfig()
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
import ClimaComms
using SciMLBase
using PrettyTables
import DiffEqCallbacks as DECB
using JLD2
using NCDatasets
using ClimaTimeSteppers
import JSON
using Test
import Tar
import Base.Filesystem: rm
import OrderedCollections
using ClimaCoreTempestRemap
using ClimaCorePlots, Plots
using ClimaCoreMakie, CairoMakie
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
@assert last(sol.t) == simulation.t_end
CA.verify_callbacks(sol.t)

if ClimaComms.iamroot(config.comms_ctx)
    @info "Plotting"
    make_plots(Val(Symbol(reference_job_id)), simulation.output_dir)
    @info "Plotting done"

    @info "Creating tarballs"
    Tar.create(
        f -> endswith(f, ".nc"),
        simulation.output_dir,
        joinpath(simulation.output_dir, "nc_files.tar"),
    )
    Tar.create(
        f -> endswith(f, r"hdf5|h5"),
        simulation.output_dir,
        joinpath(simulation.output_dir, "hdf5_files.tar"),
    )

    foreach(readdir(simulation.output_dir)) do f
        endswith(f, r"nc|hdf5|h5") && rm(joinpath(simulation.output_dir, f))
    end
    @info "Tarballs created"
end

if CA.is_distributed(config.comms_ctx)
    nprocs = ClimaComms.nprocs(config.comms_ctx)
    comms_ctx = config.comms_ctx
    output_dir = simulation.output_dir
    # replace sol.u on the root processor with the global sol.u
    if ClimaComms.iamroot(comms_ctx)
        Y = sol.u[1]
        center_space = axes(Y.c)
        horz_space = Spaces.horizontal_space(center_space)
        horz_topology = horz_space.topology
        Nq = Quadratures.degrees_of_freedom(horz_space.quadrature_style)
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

if config.parsed_args["check_conservation"]
    FT = Spaces.undertype(axes(sol.u[end].c.ρ))

    # energy
    energy_total = sum(sol.u[end].c.ρe_tot)
    energy_atmos_change = sum(sol.u[end].c.ρe_tot) - sum(sol.u[1].c.ρe_tot)
    sfc = p.atmos.surface_model
    if sfc isa CA.PrognosticSurfaceTemperature
        sfc_cρh = sfc.ρ_ocean * sfc.cp_ocean * sfc.depth_ocean
        energy_total +=
            CA.horizontal_integral_at_boundary(sol.u[end].sfc.T .* sfc_cρh)
        energy_surface_change =
            CA.horizontal_integral_at_boundary(
                sol.u[end].sfc.T .- sol.u[1].sfc.T,
            ) * sfc_cρh
    else
        energy_surface_change = -p.net_energy_flux_sfc[][]
    end
    energy_radiation_input = -p.net_energy_flux_toa[][]
    @test (energy_atmos_change + energy_surface_change) / energy_total ≈
          energy_radiation_input / energy_total atol = 5 * sqrt(eps(FT))

    if p.atmos.moisture_model isa CA.DryModel
        # density
        @test sum(sol.u[1].c.ρ) ≈ sum(sol.u[end].c.ρ) rtol = 50 * eps(FT)
    else
        if sfc isa CA.PrognosticSurfaceTemperature
            # water
            water_total = sum(sol.u[end].c.ρq_tot)
            water_atmos_change =
                sum(sol.u[end].c.ρq_tot) - sum(sol.u[1].c.ρq_tot)
            water_surface_change = CA.horizontal_integral_at_boundary(
                sol.u[end].sfc.water .- sol.u[1].sfc.water,
            )
            @test (water_atmos_change + water_surface_change) / water_total ≈ 0 atol =
                100 * sqrt(eps(FT))
        end
    end
end

if config.parsed_args["check_precipitation"]
    # run some simple tests based on the output
    FT = Spaces.undertype(axes(sol.u[end].c.ρ))
    Yₜ = similar(sol.u[end])

    Yₜ_ρ = similar(Yₜ.c.ρq_rai)
    Yₜ_ρqₚ = similar(Yₜ.c.ρq_rai)
    Yₜ_ρqₜ = similar(Yₜ.c.ρq_rai)

    CA.remaining_tendency!(Yₜ, sol.u[end], sol.prob.p, sol.t[end])

    @. Yₜ_ρqₚ = -Yₜ.c.ρq_rai - Yₜ.c.ρq_sno
    @. Yₜ_ρqₜ = Yₜ.c.ρq_tot
    @. Yₜ_ρ = Yₜ.c.ρ

    ClimaCore.Fields.bycolumn(axes(sol.u[end].c.ρ)) do colidx

        # no nans
        @assert !any(isnan, Yₜ.c.ρ[colidx])
        @assert !any(isnan, Yₜ.c.ρq_tot[colidx])
        @assert !any(isnan, Yₜ.c.ρe_tot[colidx])
        @assert !any(isnan, Yₜ.c.ρq_rai[colidx])
        @assert !any(isnan, Yₜ.c.ρq_sno[colidx])
        @assert !any(isnan, sol.prob.p.precomputed.ᶜwᵣ[colidx])
        @assert !any(isnan, sol.prob.p.precomputed.ᶜwₛ[colidx])

        # treminal velocity is positive
        @test minimum(sol.prob.p.precomputed.ᶜwᵣ[colidx]) >= FT(0)
        @test minimum(sol.prob.p.precomputed.ᶜwₛ[colidx]) >= FT(0)

        # checking for water budget conservation
        # in the presence of precipitation sinks
        # (This test only works without surface flux of q_tot)
        @test all(
            ClimaCore.isapprox(
                Yₜ_ρqₜ[colidx],
                Yₜ_ρqₚ[colidx],
                rtol = 1e2 * eps(FT),
            ),
        )

        # mass budget consistency
        @test all(
            ClimaCore.isapprox(Yₜ_ρ[colidx], Yₜ_ρqₜ[colidx], rtol = eps(FT)),
        )

        # cloud fraction diagnostics
        @assert !any(isnan, sol.prob.p.precomputed.ᶜcloud_fraction[colidx])
        @test minimum(sol.prob.p.precomputed.ᶜcloud_fraction[colidx]) >= FT(0)
        @test maximum(sol.prob.p.precomputed.ᶜcloud_fraction[colidx]) <= FT(1)
    end
end
