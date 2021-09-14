using ClimaCore: Fields, Geometry, Operators, Spaces
using ClimaAtmos.Domains: make_function_space
using ClimaAtmos.Domains: Plane, PeriodicPlane, Column, HybridPlane # remove
using IntervalSets #remove
using OrdinaryDiffEq: SSPRK33 # remove

function initial_conditions(hv_center_space)
    h_init(x_init, z_init) = begin
        coords = Fields.coordinate_field(hv_center_space)
        h = map(coords) do coord
            exp(-((coord.x + x_init)^2 + (coord.z + z_init)^2) / (2 * 0.2^2))
        end

        return h
    end
    U = Fields.FieldVector(h = h_init(0.5, 0.5))
    
    return U
end

function rhs!(dudt, u, _, t)
    h = u.h
    dh = dudt.h

    # vertical advection no inflow at bottom 
    # and outflow at top
    Ic2f = Operators.InterpolateC2F(top = Operators.Extrapolate())
    divf2c = Operators.DivergenceF2C(
        bottom = Operators.SetValue(Geometry.Cartesian13Vector(0.0, 0.0)),
    )
    # only upward advection
    @. dh = -divf2c(Ic2f(h) * Geometry.Cartesian13Vector(0.0, cᵥ))

    # only horizontal advection
    hdiv = Operators.Divergence()
    @. dh += -hdiv(h * Geometry.Cartesian1Vector(cₕ))
    Spaces.weighted_dss!(dh)

    return dudt
end

function run_diagonal_advection_hybridplane(
    FT;
    stepper = SSPRK33(),
    nelements = (10, 64),
    npolynomial = 7,
    dt = 0.001,
    callbacks = (),
    mode = :regression,
)
    if FT <: Float32
        @info "Diagonal advection on hybrid plane test does not run for $FT."
        return nothing
    end

    params = map(FT, (
        cₕ = 0.0,
        cᵥ = 1.0,
    ))

    domain = HybridPlane(
        FT,
        xlim = (-π, π),
        zlim = (0, 4π),
        nelements = nelements,
        npolynomial = npolynomial,
    )
    hv_center_space, hv_face_space = make_function_space(domain)

    U = initial_conditions(hv_center_space)
    t_end = 1.0
    prob = ODEProblem(rhs!, U, (0.0, t_end))
    sol = solve(prob, SSPRK33(), dt = Δt)

    print("test")

    #model = DiagonalAdvectionModel(domain = domain, parameters = params)

    # execute differently depending on testing mode
    # if mode == :unit
    #     # TODO!: run with input callbacks = ...
    #     simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))
    #     @unpack h, u, c = init_bickley_jet_2d_plane(params)

    #     set!(simulation, h = h, u = u, c = c)
    #     step!(simulation)

    #     @test true # either error or integration runs
    # elseif mode == :regression
    #     simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))
    #     @unpack h, u, c = init_bickley_jet_2d_plane(params)

    #     # here we set the initial condition with an array for testing
    #     space = axes(simulation.integrator.u.swm.c)
    #     local_geometry = Fields.local_geometry_field(space)
    #     c_array = c.(local_geometry)

    #     set!(simulation, :swm, h = h, u = u, c = c_array)
    #     step!(simulation)
    #     u = simulation.integrator.u.swm

    #     # perform regression check
    #     current_min = -0.08674718288150758
    #     current_max = 0.41810635564122384
    #     @test minimum(parent(u.u)) ≈ current_min atol = 1e-3
    #     @test maximum(parent(u.u)) ≈ current_max atol = 1e-3
    # elseif mode == :validation
    #     # TODO!: run with callbacks = ...
    #     simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 80.0))
    #     @unpack h, u, c = init_bickley_jet_2d_plane(params)
    #     set!(simulation, :swm, h = h, u = u, c = c)
    #     run!(simulation)
    #     u_end = simulation.integrator.u.swm

    #     # post-processing
    #     ENV["GKSwstype"] = "nul"
    #     Plots.GRBackend()

    #     # make output directory
    #     path = joinpath(@__DIR__, "output_validation")
    #     mkpath(path)

    #     # plot final state
    #     foi = Plots.plot(u_end.c, clim = (-1, 1))
    #     Plots.png(foi, joinpath(path, "bickley_jet_2d_plane_FT_$FT"))

    #     @test true # check is visual
    # else
    #     throw(ArgumentError("$mode incompatible with test case."))
    # end

    nothing
end
