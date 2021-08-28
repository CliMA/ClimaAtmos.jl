using ClimaAtmos.Domains: Plane
using ClimaAtmos.Models: ShallowWaterModel, prognostic_state_names, diagnostic_state_names

function instantiate_shallow_water_model(FT)
    parameters = (
        ϵ  = 0.1,  # perturbation size for initial condition
        l  = 0.5,  # Gaussian width
        k  = 0.5,  # sinusoidal wavenumber
        ρ₀ = 1.0,  # reference density
        c  = 2.0,  
        g  = 9.8,  # gravitational constant
        D₄ = 1e-4, # hyperdiffusion coefficient
    )
    domain = Plane(
        FT, 
        xlim = (0.0, 1.0), 
        ylim = (0.0, 2.0), 
        nelements = (3, 2),
        npolynomial = 16,
        periodic = (false, false),
    )
    swm = ShallowWaterModel(
        domain = domain,
        parameters = parameters,
    )
    check1 = swm.domain == domain
    check2 = swm.parameters == parameters

    return check1 && check2
end

@testset "ShallowWaterModels" begin
    @info "Testing ClimaAtmos.ShallowWaterModels..."

    @testset "ShallowWaterModels" begin
        for FT in float_types
            @test instantiate_shallow_water_model(FT)
        
            # Setup model instance
            domain = Plane(
                FT, 
                xlim = (0.0, 1.0), 
                ylim = (0.0, 2.0), 
                nelements = (3, 2),
                npolynomial = 16,
                periodic = (false, false),
            )
            model = ShallowWaterModel(
                domain = domain,
                parameters = nothing,
            )

            # Test prognostic_state_names
            @test prognostic_state_names(model) == (:h, :u, :c)

            # Test diagnostic_state_names
            @test diagnostic_state_names(model) === nothing
        end
    end
end
