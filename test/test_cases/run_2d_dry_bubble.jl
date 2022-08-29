using Test

using OrdinaryDiffEq: SSPRK33, CallbackSet
using ClimaCorePlots, Plots
using UnPack
using DiffEqCallbacks

using ClimaAtmos.Experimental.Utils.InitialConditions: init_2d_dry_bubble
using ClimaAtmos.Experimental.Domains
using ClimaAtmos.Experimental.BoundaryConditions
using ClimaAtmos.Experimental.Models
using ClimaAtmos.Experimental.Models.Nonhydrostatic2DModels
using ClimaAtmos.Experimental.Callbacks
using ClimaAtmos.Experimental.Simulations

# Set up parameters
import ClimaAtmos
include(joinpath(pkgdir(ClimaAtmos), "parameters", "create_parameters.jl"))


"""
    PNGOutput{M, I} <: AbstractCallback

Specifies a `DiffEqCallbacks.PeriodicCallback` that
plots some of the state variables from the integrator
into a `.png` file.
"""
struct PNGOutput{M <: AbstractModel, I <: Number} <: AbstractCallback
    model::M
    filedir::String
    filename::String
    interval::I
end

function (F::PNGOutput)(integrator)
    state = integrator.u

    # Create directory
    mkpath(F.filedir)

    ENV["GKSwstype"] = "nul"
    Plots.GRBackend()

    foi = Plots.plot(state.base.ρ)
    Plots.png(
        foi,
        joinpath(F.filedir, F.filename * "_rho" * "_$(integrator.t)" * ".png"),
    )

    foi = Plots.plot(state.base.ρw)
    Plots.png(
        foi,
        joinpath(
            F.filedir,
            F.filename * "_rho_w" * "_$(integrator.t)" * ".png",
        ),
    )

    foi = Plots.plot(state.base.ρuh)
    Plots.png(
        foi,
        joinpath(
            F.filedir,
            F.filename * "_rho_uh" * "_$(integrator.t)" * ".png",
        ),
    )

    return nothing
end

function run_2d_dry_bubble(
    ::Type{FT};
    stepper = SSPRK33(),
    nelements = (10, 40),
    npolynomial = 4,
    dt = 0.03,
    callbacks = (),
    test_mode = :regression,
    thermo_mode = :ρθ,
    moisture_mode = :dry,
) where {FT}
    params = create_climaatmos_parameter_set(FT)

    domain = HybridPlane(
        FT,
        xlim = (-5e2, 5e2),
        zlim = (0.0, 1e3),
        nelements = nelements,
        npolynomial = npolynomial,
    )

    model = Nonhydrostatic2DModel(
        domain = domain,
        boundary_conditions = nothing,
        precipitation = NoPrecipitation(),
        parameters = params,
        hyperdiffusivity = FT(100),
        cache = CacheBase(),
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
        @unpack ρ, ρuh, ρw, ρθ = init_2d_dry_bubble(FT, params)
        set!(simulation, :base, ρ = ρ, ρuh = ρuh, ρw = ρw)
        set!(simulation, :thermodynamics, ρθ = ρθ)
        step!(simulation)
        u = simulation.integrator.u

        # perform regression check
        current_min = 299.9999997747195
        current_max = 300.49692207965666
        @test minimum(parent(u.thermodynamics.ρθ ./ u.base.ρ)) ≈ current_min atol =
            1e-3
        @test maximum(parent(u.thermodynamics.ρθ ./ u.base.ρ)) ≈ current_max atol =
            1e-3
    elseif test_mode == :validation

        path = joinpath(@__DIR__, first(split(basename(@__FILE__), ".jl")))
        mkpath(path)

        cb_png = PNGOutput(model, path, "plots_dry_bubble", 100)
        cb_set = CallbackSet(
            DiffEqCallbacks.PeriodicCallback(
                cb_png,
                cb_png.interval;
                initial_affect = true,
            ),
        )

        simulation = Simulation(
            model,
            stepper,
            dt = dt,
            tspan = (0.0, 500.0),
            callbacks = cb_set,
        )
        @unpack ρ, ρuh, ρw, ρθ = init_2d_dry_bubble(FT, params)
        set!(simulation, :base, ρ = ρ, ρuh = ρuh, ρw = ρw)
        set!(simulation, :thermodynamics, ρθ = ρθ)
        run!(simulation)

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
        run_2d_dry_bubble(FT)
    end
end
# run_2d_dry_bubble(Float32, test_mode = :validation)
