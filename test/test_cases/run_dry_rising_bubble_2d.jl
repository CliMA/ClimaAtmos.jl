include("initial_conditions/dry_rising_bubble_2d.jl")

# Set up parameters
using CLIMAParameters
struct Bubble2DParameters <: CLIMAParameters.AbstractEarthParameterSet end

function run_dry_rising_bubble_2d(
    ::Type{FT};
    stepper = SSPRK33(),
    nelements = (10, 50),
    npolynomial = 4,
    dt = 0.02,
    callbacks = (),
    mode = :regression,
) where {FT}
    params = Bubble2DParameters()

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
        parameters = params,
    )

    # execute differently depending on testing mode
    if mode == :integration
        # TODO!: run with input callbacks = ...
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))
        @test simulation isa Simulation

        # test set function
        @unpack ρ, ρuh, ρw, ρθ = init_dry_rising_bubble_2d(FT, params)
        set!(simulation, :base, ρ = ρ, ρuh = ρuh, ρw = ρw)
        set!(simulation, :thermodynamics, ρθ = ρθ)

        # test error handling
        @test_throws ArgumentError set!(simulation, quack = ρ)
        @test_throws ArgumentError set!(simulation, ρ = "quack")

        # test successful integration
        @test step!(simulation) isa Nothing # either error or integration runs
    elseif mode == :regression
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))
        @unpack ρ, ρuh, ρw, ρθ = init_dry_rising_bubble_2d(FT, params)
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
    elseif mode == :validation
        # for now plot θ for the ending step;
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 500.0))
        @unpack ρ, ρuh, ρw, ρθ = init_dry_rising_bubble_2d(FT, params)
        set!(simulation, :base, ρ = ρ, ρuh = ρuh, ρw = ρw)
        set!(simulation, :thermodynamics, ρθ = ρθ)
        run!(simulation)
        u_end = simulation.integrator.u

        # post-processing
        ENV["GKSwstype"] = "nul"
        Plots.GRBackend()

        # make output directory
        path = joinpath(@__DIR__, "output_validation")
        mkpath(path)

        foi = Plots.plot(
            u_end.thermodynamics.ρθ ./ u_end.base.ρ,
            clim = (300.0, 300.8),
        )
        Plots.png(foi, joinpath(path, "dry_rising_bubble_2d_FT_$FT"))

        @test true # check is visual
    else
        throw(ArgumentError("$mode incompatible with test case."))
    end

    # TODO!: Implement the rest plots and analyses
    # 1. sort out saveat kwarg for Simulation
    # 2. create animation for a rising bubble; timeseries of total energy

    nothing
end
