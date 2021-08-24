using ClimaAtmos.Domains: PeriodicPlane
using ClimaAtmos.Models: ShallowWaterModel
using ClimaAtmos.Simulations: Simulation, step!, run!, set!
using OrdinaryDiffEq: SSPRK33, ODEProblem
using DiffEqBase: DEIntegrator, init
using ClimaCore.Geometry: Cartesian12Vector

function instantiate_simulation(FT)
    domain = PeriodicPlane(
        FT,
        xlim = (-2π, 2π),
        ylim = (-2π, 2π),
        nelements = (16, 16),
        npolynomial = 4,
    )
    model = ShallowWaterModel(
        domain = domain,
        boundary_conditions = nothing,
        initial_conditions = nothing,
        parameters = nothing,
    )
    rhs!(dY, Y, p, t) = Y
    problem = ODEProblem(rhs!, [0.01], (0.0, 1.0))
    integrator = init(problem, SSPRK33(), dt = 0.01)
    sim = Simulation(model, integrator)

    check1 = sim.model == model
    check2 = sim.integrator == integrator

    return check1 && check2
end

function construct_simulation(FT)
    parameters = map(
        FT,
        (
            ϵ = 0.1,  # perturbation size for initial condition
            l = 0.5,  # Gaussian width
            k = 0.5,  # sinusoidal wavenumber
            ρ₀ = 1.0,  # reference density
            c = 2.0,
            g = 9.8,  # gravitational constant
            D₄ = 1e-4, # hyperdiffusion coefficient
        ),
    )

    function initialize_state(x, y, parameters)
        @unpack ρ₀, l, k, ϵ = parameters

        # density
        ρ = ρ₀

        # velocity
        U₁ = cosh(y)^(-2)
        gaussian = exp(-(y + l / 10)^2 / 2 * l^2)
        u₁′ = gaussian * (y + l / 10) / l^2 * cos(k * x) * cos(k * y)
        u₁′ += k * gaussian * cos(k * x) * sin(k * y)
        u₂′ = -k * gaussian * sin(k * x) * cos(k * y)
        u = Cartesian12Vector(U₁ + ϵ * u₁′, ϵ * u₂′)

        # passive tracer
        θ = sin(k * y)

        return (ρ = ρ, u = u, ρθ = ρ * θ)
    end
    domain = PeriodicPlane(
        FT,
        xlim = (-2π, 2π),
        ylim = (-2π, 2π),
        nelements = (16, 16),
        npolynomial = 4,
    )
    model = ShallowWaterModel(
        domain = domain,
        boundary_conditions = nothing,
        initial_conditions = initialize_state,
        parameters = parameters,
    )
    simulation = Simulation(model, SSPRK33(), dt = 0.04, tspan = (0.0, 1.0))
    check1 = simulation.model == model
    check2 = simulation.integrator isa DEIntegrator

    return check1 && check2
end

@testset "Simulations" begin
    @info "Testing ClimaAtmos.Simulations..."

    @testset "Simulations" begin
        for FT in float_types
            @test instantiate_simulation(FT)
            @test construct_simulation(FT)
        end

        # test
    end
end
