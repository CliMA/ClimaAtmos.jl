if !haskey(ENV, "BUILDKITE")
    import Pkg
    Pkg.develop(Pkg.PackageSpec(; path = dirname(dirname(@__DIR__))))
end
using Test

using OrdinaryDiffEq: SSPRK33, CallbackSet
using ClimaCorePlots, Plots
using UnPack


using CLIMAParameters
using ClimaAtmos.Utils.InitialConditions: init_2d_moist_bubble
using ClimaAtmos.Domains
using ClimaAtmos.BoundaryConditions
using ClimaAtmos.Models
using ClimaAtmos.Models.Nonhydrostatic2DModels
using ClimaAtmos.Callbacks
using ClimaAtmos.Simulations


# Set up parameters
struct Bubble2DParameters <: CLIMAParameters.AbstractEarthParameterSet end

function run_2d_moist_bubble(
    ::Type{FT};
    stepper = SSPRK33(),
    nelements = (20, 80),
    npolynomial = 4,
    dt = 0.02,
    callbacks = (),
    test_mode = :regression,
) where {FT}
    params = Bubble2DParameters()

    domain = HybridPlane(
        FT,
        xlim = (0.0, 10e3),
        zlim = (0.0, 10e3),
        nelements = nelements,
        npolynomial = npolynomial,
    )

    model = Nonhydrostatic2DModel(
        domain = domain,
        thermodynamics = TotalEnergy(),
        moisture = EquilibriumMoisture(),
        precipitation = PrecipitationRemoval(),
        boundary_conditions = nothing,
        parameters = params,
    )

    # execute differently depending on testing mode
    if test_mode == :regression
        # TODO!: run with input callbacks = ...
        cb_cfl = CFLAdaptive(model, 0.01, 1.0, false)
        cb = CallbackSet(generate_callback(cb_cfl))
        simulation = Simulation(
            model,
            stepper,
            dt = dt,
            tspan = (0.0, 1.0),
            callbacks = cb,
        )
        @test simulation isa Simulation

        # test error handling
        @test_throws ArgumentError set!(simulation, quack = 0.0)
        @test_throws ArgumentError set!(simulation, ρ = "quack")

        # test sim
        @unpack ρ, ρuh, ρw, ρe_tot, ρq_tot = init_2d_moist_bubble(FT, params, thermovar = :ρe_tot)
        set!(simulation, :base, ρ = ρ, ρuh = ρuh, ρw = ρw)
        set!(simulation, :thermodynamics, ρe_tot = ρe_tot)
        set!(simulation, :moisture, ρq_tot = ρq_tot)

        # initial state
        u = simulation.integrator.u
        ∫ρ_0 = sum(u.base.ρ)
        ∫ρe_tot_0 = sum(u.thermodynamics.ρe_tot)
        ∫ρq_tot_0 = sum(u.moisture.ρq_tot)

        # 1 step
        step!(simulation)
        u_end = simulation.integrator.u
        ∫ρ_e = sum(u.base.ρ)
        ∫ρe_tot_e = sum(u.thermodynamics.ρe_tot)
        ∫ρq_tot_e = sum(u.moisture.ρq_tot)

        # perform regression check
        current_min = 21928.206023189607
        current_max = 28343.530694118865

        @test minimum(parent(u_end.thermodynamics.ρe_tot)) ≈ current_min atol = 0.01
        @test maximum(parent(u_end.thermodynamics.ρe_tot)) ≈ current_max atol = 0.01

        # conservation check
        Δρ = (∫ρ_e - ∫ρ_0) ./ ∫ρ_0 * 100
        Δρe_tot = (∫ρe_tot_e - ∫ρe_tot_0) ./ ∫ρe_tot_0 * 100
        Δρq_tot = (∫ρq_tot_e - ∫ρq_tot_0) ./ ∫ρq_tot_0 * 100
        
        if FT == Float32
            @test abs(Δρ) < 3e-5
            @test abs(Δρe_tot) < 1e-5
            @test abs(Δρq_tot) < 1e-3
        else
            @test abs(Δρ) < 1e-12
            @test abs(Δρe_tot) < 1e-5
            @test abs(Δρq_tot) < 1e-3
        end

    elseif test_mode == :validation
        simulation = Simulation(
            model,
            stepper,
            dt = dt,
            tspan = (0.0, 500.0),
        )
        @unpack ρ, ρuh, ρw, ρe_tot, ρq_tot = init_2d_moist_bubble(FT, params, thermovar = :ρe_tot)
        set!(simulation, :base, ρ = ρ, ρuh = ρuh, ρw = ρw)
        set!(simulation, :thermodynamics, ρe_tot = ρe_tot)
        set!(simulation, :moisture, ρq_tot = ρq_tot)
        run!(simulation)
        u_end = simulation.integrator.u

        # post-processing
        ENV["GKSwstype"] = "nul"
        Plots.GRBackend()

        # make output directory
        path = joinpath(@__DIR__, first(split(basename(@__FILE__), ".jl")))
        mkpath(path)

        foi = Plots.plot(
            u_end.moisture.ρq_tot ./ u_end.base.ρ,
        )
        Plots.png(foi, joinpath(path, "moist_rising_bubble_2d_FT_$FT"))

        @test true # check is visual
    else
        throw(ArgumentError("$test_mode incompatible with test case."))
    end

    nothing
end

@testset "2D dry rising bubble" begin
    for FT in (Float32, Float64)
        run_2d_moist_bubble(FT)
    end
end