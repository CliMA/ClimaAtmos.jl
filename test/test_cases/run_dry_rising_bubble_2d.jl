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
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))

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
        @test step!(simulation) isa Nothing # either error or integration runs
    elseif mode == :regression
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))
        @unpack ρ, ρuh, ρw, ρθ = init_dry_rising_bubble_2d(FT, params)
        set!(simulation, :nhm, ρ = ρ, ρuh = ρuh, ρw = ρw, ρθ = ρθ)
        step!(simulation)
        u = simulation.integrator.u.nhm

        # TODO!: After validation -> set up regression test
        # perform regression check
        # current_min = -0.0
        # current_max = 0.0
        # @test minimum(parent(u.ρw)) ≈ current_min atol = 1e-3
        # @test maximum(parent(u.ρw)) ≈ current_max atol = 1e-3
        @test true
    elseif mode == :validation
        # TODO!: Implement all the plots and analyses
        # 1. Video
        # 2. Slices at different x and z
        # 3. Time series of total energy -> should not increase
        @test true # check is visual
    else
        throw(ArgumentError("$mode incompatible with test case."))
    end

    nothing
end
