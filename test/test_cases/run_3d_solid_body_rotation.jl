using Test

using OrdinaryDiffEq: SSPRK33
using Plots
using UnPack

using CLIMAParameters
using ClimaAtmos.Utils.InitialConditions: init_3d_solid_body_rotation
using ClimaAtmos.Domains
using ClimaAtmos.BoundaryConditions
using ClimaAtmos.Models.Nonhydrostatic3DModels
using ClimaAtmos.Simulations

# Set up parameters
struct SolidBodyRotationParameters <: CLIMAParameters.AbstractEarthParameterSet end

function run_3d_solid_body_rotation(
    ::Type{FT};
    stepper = SSPRK33(),
    nelements = (4, 15),
    npolynomial = 5,
    dt = 5.0,
    callbacks = (),
    test_mode = :regression,
) where {FT}
    params = SolidBodyRotationParameters()

    domain = SphericalShell(
        FT,
        radius = CLIMAParameters.Planet.planet_radius(params),
        height = FT(30.0e3),
        nelements = nelements,
        npolynomial = npolynomial,
    )

    model = Nonhydrostatic3DModel(
        domain = domain,
        boundary_conditions = nothing,
        parameters = params,
        flux_corr = false,
        hyperdiffusivity = FT(0),
    )

    # execute differently depending on testing mode
    if test_mode == :regression
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, dt))
        @unpack ρ, uh, w, ρe_tot = init_3d_solid_body_rotation(FT, params)
        set!(simulation, :base, ρ = ρ, uh = uh, w = w)
        set!(simulation, :thermodynamics, ρe_tot = ρe_tot)
        step!(simulation)
        u = simulation.integrator.u.base
        uh_phy = Geometry.transform.(Ref(Geometry.UVAxis()), u.uh)
        w_phy = Geometry.transform.(Ref(Geometry.WAxis()), u.w)

        # perform regression check
        if FT == Float64
            current_uh_max = 3.834751379171411e-15
        elseif FT == Float32
            current_uh_max = 1.6886055f-6
        end
        current_w_max = 0.21114581947634953

        @test (abs.(uh_phy |> parent) |> maximum) ≈ current_uh_max rtol = 1e-3
        @test (abs.(w_phy |> parent) |> maximum) ≈ current_w_max rtol = 1e-3
    elseif test_mode == :validation
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 3600))
        @unpack ρ, uh, w, ρe_tot = init_3d_solid_body_rotation(FT, params)
        set!(simulation, :base, ρ = ρ, uh = uh, w = w)
        set!(simulation, :thermodynamics, ρe_tot = ρe_tot)
        run!(simulation)
        u_end = simulation.integrator.u.base

        uh_phy = Geometry.transform.(Ref(Geometry.UVAxis()), u_end.uh)
        w_phy = Geometry.transform.(Ref(Geometry.WAxis()), u_end.w)

        @test maximum(abs.(uh_phy.components.data.:1)) ≤ 1e-11
        @test maximum(abs.(uh_phy.components.data.:2)) ≤ 1e-11
        @test maximum(abs.(w_phy |> parent)) ≤ 1.0
    else
        throw(ArgumentError("$test_mode incompatible with test case."))
    end

    nothing
end
