if !haskey(ENV, "BUILDKITE")
    import Pkg
    Pkg.develop(Pkg.PackageSpec(; path = dirname(dirname(@__DIR__))))
end
using Test

using OrdinaryDiffEq: SSPRK33, CallbackSet
using ClimaCorePlots, Plots
using UnPack

using CLIMAParameters
using ClimaAtmos.Utils.InitialConditions: init_2d_rising_bubble
using ClimaAtmos.Domains
using ClimaAtmos.BoundaryConditions
using ClimaAtmos.Models
using ClimaAtmos.Models.Nonhydrostatic2DModels
using ClimaAtmos.Callbacks
using ClimaAtmos.Simulations

# Set up parameters
struct Bubble2DParameters <: CLIMAParameters.AbstractEarthParameterSet end

function run_2d_rising_bubble(
    ::Type{FT};
    stepper = SSPRK33(),
    nelements = (10, 50),
    npolynomial = 4,
    dt = 0.02,
    callbacks = (),
    test_mode = :regression,
    thermo_mode = :ρθ,
    moisture_mode = :dry,
) where {FT}
    params = Bubble2DParameters()

    domain = HybridPlane(
        FT,
        xlim = (-5e2, 5e2),
        zlim = (0.0, 1e3),
        nelements = nelements,
        npolynomial = npolynomial,
    )

    boundary_conditions = (;
        base = (;
            ρ = (top = NoFluxCondition(), bottom = NoFluxCondition()),
            ρuh = (top = NoFluxCondition(), bottom = NoFluxCondition()),
            ρw = (top = NoFluxCondition(), bottom = NoFluxCondition()),
        ),
        thermodynamics = (;
            ρθ = (top = NoFluxCondition(), bottom = NoFluxCondition()),
        ),
    )

    model = Nonhydrostatic2DModel(
        domain = domain,
        boundary_conditions = boundary_conditions,
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
        @unpack ρ, ρuh, ρw, ρθ = init_2d_rising_bubble(FT, params)
        set!(simulation, :base, ρ = ρ, ρuh = ρuh, ρw = ρw)
        set!(simulation, :thermodynamics, ρθ = ρθ)
        step!(simulation)
        u = simulation.integrator.u

        # perform regression check
        current_min = 299.9999997747195
        current_max = 300.49999999996226
        @test minimum(parent(u.thermodynamics.ρθ ./ u.base.ρ)) ≈ current_min atol =
            1e-3
        @test maximum(parent(u.thermodynamics.ρθ ./ u.base.ρ)) ≈ current_max atol =
            1e-3
    elseif test_mode == :validation
        # for now plot θ for the ending step;
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 500.0))
        @unpack ρ, ρuh, ρw, ρθ = init_2d_rising_bubble(FT, params)
        set!(simulation, :base, ρ = ρ, ρuh = ρuh, ρw = ρw)
        set!(simulation, :thermodynamics, ρθ = ρθ)
        run!(simulation)
        u_end = simulation.integrator.u

        # post-processing
        ENV["GKSwstype"] = "nul"
        Plots.GRBackend()

        # make output directory
        path = joinpath(@__DIR__, first(split(basename(@__FILE__), ".jl")))
        mkpath(path)

        foi = Plots.plot(
            u_end.thermodynamics.ρθ ./ u_end.base.ρ,
            clim = (300.0, 300.8),
        )
        Plots.png(foi, joinpath(path, "dry_rising_bubble_2d_FT_$FT"))

        @test true # check is visual
    else
        throw(ArgumentError("$test_mode incompatible with test case."))
    end

    # TODO!: Implement the rest plots and analyses
    # 1. sort out saveat kwarg for Simulation
    # 2. create animation for a rising bubble; timeseries of total energy

    nothing
end

@testset "2D dry rising bubble" begin
    for FT in (Float32, Float64)
        run_2d_rising_bubble(FT)
    end
end
