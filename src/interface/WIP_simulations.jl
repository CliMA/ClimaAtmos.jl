abstract type AbstractSimulation end

struct Simulation{A,B,C,D,E} <: AbstractSimulation
    backend::A
    model::B
    timestepper::C
    callbacks::D
    ode_problem::E
end

function Simulation(
    backend::AbstractBackend;
    model,
    timestepper::AbstractTimestepper,
    callbacks,
)
    ode_problem = create_ode_problem(
        backend, 
        model,
        timestepper, 
    )
  
    return Simulation(
        backend,
        model,
        timestepper, 
        callbacks, 
        ode_problem,
    )
end

function evolve(simulation::Simulation)
    return solve(
        simulation.ode_problem,
        simulation.timestepper.method,
        dt = simulation.timestepper.dt,
        saveat = simulation.timestepper.saveat,
        progress = simulation.timestepper.progress, 
        progress_message = simulation.timestepper.progress_message,
    )
end

function create_ode_problem(backend::ClimaCoreBackend, model, timestepper)
    function_space = create_function_space(backend, model.domain)
    rhs! = create_rhs(backend, model.equation_set, function_space)
    y0 = model.initial_conditions.(
        Fields.coordinate_field(function_space), 
        Ref(model.parameters)
    )

    return ODEProblem(
        rhs!, 
        y0, 
        timestepper.tspan,
    )
end

function create_rhs(backend::ClimaCoreBackend, equation_set, function_space)
    function rhs!(dydt, y, _, t)
        @unpack D₄, g = model.parameters

        sdiv = Operators.Divergence()
        wdiv = Operators.WeakDivergence()
        grad = Operators.Gradient()
        wgrad = Operators.WeakGradient()
        curl = Operators.Curl()
        wcurl = Operators.WeakCurl()

        # compute hyperviscosity first
        @. dydt.u =
            wgrad(sdiv(y.u)) -
            Cartesian12Vector(wcurl(Geometry.Covariant3Vector(curl(y.u))))
        @. dydt.ρθ = wdiv(grad(y.ρθ))

        Spaces.weighted_dss!(dydt)

        @. dydt.u =
            -D₄ * (
                wgrad(sdiv(dydt.u)) -
                Cartesian12Vector(wcurl(Geometry.Covariant3Vector(curl(dydt.u))))
            )
        @. dydt.ρθ = -D₄ * wdiv(grad(dydt.ρθ))

        # add in pieces
        J = Fields.Field(function_space.local_geometry.J, function_space)
        @. begin
            dydt.ρ = -wdiv(y.ρ * y.u)
            dydt.u +=
                -grad(g * y.ρ + norm(y.u)^2 / 2) +
                Cartesian12Vector(J * (y.u × curl(y.u)))
            dydt.ρθ += -wdiv(y.ρθ * y.u)
        end
        Spaces.weighted_dss!(dydt)
        return dydt
    end

    return rhs!
end

function create_function_space(::ClimaCoreBackend, domain::Rectangle)
    rectangle = Domains.RectangleDomain(
        domain.xlim,
        domain.ylim,
        x1periodic = domain.periodic[1],
        x2periodic = domain.periodic[2],
    )
    mesh = Meshes.EquispacedRectangleMesh(
        rectangle, 
        domain.nelements[1], 
        domain.nelements[2]
    )
    grid_topology = Topologies.GridTopology(mesh)
    quad = Spaces.Quadratures.GLL{domain.npolynomial}()
    function_space = Spaces.SpectralElementSpace2D(grid_topology, quad)

    return function_space
end