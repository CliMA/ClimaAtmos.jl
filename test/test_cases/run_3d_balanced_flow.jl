if !haskey(ENV, "BUILDKITE")
    import Pkg
    Pkg.develop(Pkg.PackageSpec(; path = dirname(dirname(@__DIR__))))
end
using Test

using OrdinaryDiffEq: SSPRK33
using Plots
using UnPack

using ClimaCore: Geometry
using CLIMAParameters
using ClimaAtmos.Utils.InitialConditions: init_3d_baroclinic_wave
using ClimaAtmos.Domains
using ClimaAtmos.BoundaryConditions
using ClimaAtmos.Models
using ClimaAtmos.Models.Nonhydrostatic3DModels
using ClimaAtmos.Simulations

# Set up parameters
struct BalancedFlowParameters <: CLIMAParameters.AbstractEarthParameterSet end

function run_3d_balanced_flow(
    ::Type{FT};
    stepper = SSPRK33(),
    nelements = (4, 10),
    npolynomial = 4,
    dt = 5.0,
    callbacks = (),
    test_mode = :regression,
) where {FT}
    params = BalancedFlowParameters()

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
        @unpack ρ, uh, w, ρe_tot =
            init_3d_baroclinic_wave(FT, params, isbalanced = true)
        set!(simulation, :base, ρ = ρ, uh = uh, w = w)
        set!(simulation, :thermodynamics, ρe_tot = ρe_tot)
        step!(simulation)

        ub = simulation.integrator.u.base
        ut = simulation.integrator.u.thermodynamics
        uh_phy = Geometry.transform.(Ref(Geometry.UVAxis()), ub.uh)
        w_phy = Geometry.transform.(Ref(Geometry.WAxis()), ub.w)

        # perform regression check
        current_u_max = 27.8838476993762
        current_w_max = 1.1618705020355047
        current_ρ_max = 1.2104825974679885
        current_ρe_max = 34865.88430268488

        @test (uh_phy.components.data.:1 |> maximum) ≈ current_u_max rtol = 1e-2
        @test (abs.(w_phy |> parent) |> maximum) ≈ current_w_max rtol = 1e-2
        @test (ub.ρ |> maximum) ≈ current_ρ_max rtol = 1e-2
        @test (ut.ρe_tot |> maximum) ≈ current_ρe_max rtol = 1e-2
    elseif test_mode == :validation
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 3600))
        @unpack ρ, uh, w, ρe_tot =
            init_3d_baroclinic_wave(FT, params, isbalanced = true)
        set!(simulation, :base, ρ = ρ, uh = uh, w = w)
        set!(simulation, :thermodynamics, ρe_tot = ρe_tot)
        u_init = simulation.integrator.u

        run!(simulation)
        u_end = simulation.integrator.u

        @test u_end.base.ρ ≈ u_init.base.ρ rtol = 5e-2
        @test u_end.thermodynamics.ρe_tot ≈ u_init.thermodynamics.ρe_tot rtol =
            5e-2
        @test u_end.base.uh ≈ u_init.base.uh rtol = 5e-2
    else
        throw(ArgumentError("$test_mode incompatible with test case."))
    end

    nothing
end

@testset "3D balanced flow" begin
    for FT in (Float32, Float64)
        run_3d_balanced_flow(FT)
    end
end
