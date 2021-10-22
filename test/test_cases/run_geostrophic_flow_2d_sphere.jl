include("initial_conditions/geostrophic_flow_2d_sphere.jl")

function run_bickley_jet_2d_plane(
    FT;
    stepper = SSPRK33(),
    nelements = (16, 16),
    npolynomial = 3,
    dt = 0.04,
    callbacks = (),
    mode = :regression,
)
    if FT <: Float32
        @info "Bickley jet 2D plane test does not run for $FT."
        return nothing
    end

    params = map(FT, (
        g = 9.8,  # gravitational constant
        D₄ = 1e-4,  # hyperdiffusion constant
        ϵ = 0.1,  # perturbation size for initial condition
        l = 0.5,  # Gaussian width
        k = 0.5,  # sinusoidal wavenumber
        h₀ = 1.0,  # reference density
    ))

    domain = PeriodicPlane(
        FT,
        xlim = (-2π, 2π),
        ylim = (-2π, 2π),
        nelements = nelements,
        npolynomial = npolynomial,
    )

    model = ShallowWaterModel(domain = domain, parameters = params)

    # execute differently depending on testing mode
    if mode == :unit
        # TODO!: run with input callbacks = ...
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))

        # test show function
        show(simulation)
        println()
        @test simulation isa Simulation
        @test simulation.restart isa NoRestart

        # test set function
        @unpack h, u, c = init_bickley_jet_2d_plane(params)
        set!(simulation, h = h, u = u, c = c)

        # test error handling
        @test_throws ArgumentError set!(simulation, quack = c)
        @test_throws ArgumentError set!(simulation, h = "quack")

        # test successful integration
        @test step!(simulation) isa Nothing # either error or integration runs
    elseif mode == :regression
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))
        @unpack h, u, c = init_bickley_jet_2d_plane(params)

        @test simulation.restart isa NoRestart
        # here we set the initial condition with an array for testing
        space = axes(simulation.integrator.u.swm.c) # get tracer field
        local_geometry = Fields.local_geometry_field(space)
        c_field = c.(local_geometry)

        set!(simulation, :swm, h = h, u = u, c = c_field)
        step!(simulation)
        u = simulation.integrator.u.swm

        # perform regression check
        current_min = -0.08674718288150758
        current_max = 0.41810635564122384
        @test minimum(parent(u.u)) ≈ current_min atol = 1e-3
        @test maximum(parent(u.u)) ≈ current_max atol = 1e-3
    elseif mode == :validation
        # TODO!: run with callbacks = ...
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 80.0))
        @test simulation.restart isa NoRestart
        @unpack h, u, c = init_bickley_jet_2d_plane(params)
        set!(simulation, :swm, h = h, u = u, c = c)
        run!(simulation)
        u_end = simulation.integrator.u.swm

        # post-processing
        ENV["GKSwstype"] = "nul"
        Plots.GRBackend()

        # make output directory
        path = joinpath(@__DIR__, "output_validation")
        mkpath(path)

        # plot final state
        foi = Plots.plot(u_end.c, clim = (-1, 1))
        Plots.png(foi, joinpath(path, "bickley_jet_2d_plane_FT_$FT"))

        @test true # check is visual
    else
        throw(ArgumentError("$mode incompatible with test case."))
    end

    nothing
end
