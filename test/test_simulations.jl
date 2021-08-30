using ClimaAtmos.Domains: PeriodicPlane
using ClimaAtmos.Models: ShallowWaterModel
using ClimaAtmos.Simulations: Simulation, step!, run!, set!
using OrdinaryDiffEq: SSPRK33, ODEProblem
using DiffEqBase: DEIntegrator, init

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

@testset "Simulations" begin
    @info "Testing ClimaAtmos.Simulations..."

    @testset "Simulations" begin
        for FT in float_types
            @test instantiate_simulation(FT)
        end

        # test
    end
end
