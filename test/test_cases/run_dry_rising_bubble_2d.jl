include("initial_conditions/dry_rising_bubble_2d.jl")

function run_dry_rising_bubble_2d(
    FT;
    stepper = SSPRK33(),
    nelements = (10, 50),
    npolynomial = 4,
    dt = 0.02,
    callbacks = (),
    mode = :regression,
)
    if FT <: Float32
        @info "Dry rising bubble 2D test does not run for $FT."
        return nothing
    end

    params = map(
        FT,
        (
            x_c = 0.0,
            z_c = 350.0,
            r_c = 250.0,
            θ_b = 300.0,
            θ_c = 0.5,
            p_0 = 1e5,
            cp_d = 1004.0,
            cv_d = 717.5,
            R_d = 287.0,
            g = 9.80616,
        ),
    )

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
    if mode == :unit
        # TODO!: run with input callbacks = ...
        simulation = Simulation(model, stepper, Δt = dt, tspan = (0.0, 1.0))

        # test show function
        show(simulation)
        println()
        @test simulation isa Simulation

        # test set function
        @unpack ρ, ρuh, ρw, ρθ = init_dry_rising_bubble_2d(FT, params)
        set!(simulation, ρ = ρ, ρuh = ρuh, ρw = ρw, ρθ = ρθ)

        # test error handling
        @test_throws ArgumentError set!(simulation, quack = ρ)
        @test_throws ArgumentError set!(simulation, ρ = "quack")

        # test successful integration
        @test time_step!(simulation) isa Nothing # either error or integration runs
    elseif mode == :regression
        simulation = Simulation(model, stepper, Δt = dt, tspan = (0.0, 1.0))
        @unpack ρ, ρuh, ρw, ρθ = init_dry_rising_bubble_2d(FT, params)
        set!(simulation, :nhm, ρ = ρ, ρuh = ρuh, ρw = ρw, ρθ = ρθ)
        time_step!(simulation)
        u = simulation.integrator.u.nhm

        # perform regression check
        current_min = 299.9999997747195
        current_max = 300.49999999996226
        @test minimum(parent(u.ρθ ./ u.ρ)) ≈ current_min atol = 1e-3
        @test maximum(parent(u.ρθ ./ u.ρ)) ≈ current_max atol = 1e-3
    elseif mode == :validation
        # for now plot θ for the ending step;
        simulation = Simulation(model, stepper, Δt = dt, tspan = (0.0, 500.0))
        @unpack ρ, ρuh, ρw, ρθ = init_dry_rising_bubble_2d(FT, params)
        set!(simulation, :nhm, ρ = ρ, ρuh = ρuh, ρw = ρw, ρθ = ρθ)
        run!(simulation)
        u_end = simulation.integrator.u.nhm

        # post-processing
        ENV["GKSwstype"] = "nul"
        Plots.GRBackend()

        # make output directory
        path = joinpath(@__DIR__, "output_validation")
        mkpath(path)

        foi = Plots.plot(u_end.ρθ ./ u_end.ρ, clim = (300.0, 300.8))
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
