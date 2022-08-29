using Test

using OrdinaryDiffEq: SSPRK33
using ClimaCorePlots, Plots
using UnPack

import ClimaAtmos.Parameters as CAP
using ClimaAtmos.Experimental.Utils.InitialConditions: init_3d_baroclinic_wave
using ClimaAtmos.Experimental.Domains
using ClimaAtmos.Experimental.BoundaryConditions
using ClimaAtmos.Experimental.Models: ConstantViscosity
using ClimaAtmos.Experimental.Models.Nonhydrostatic3DModels
using ClimaAtmos.Experimental.Simulations

# Set up parameters
import ClimaAtmos
include(joinpath(pkgdir(ClimaAtmos), "parameters", "create_parameters.jl"))

function run_3d_baroclinic_wave(
    ::Type{FT};
    stepper = SSPRK33(),
    nelements = (6, 10),
    npolynomial = 3,
    case = :default,
    dt = 0.02,
    callbacks = (),
    test_mode = :regression,
) where {FT}
    params = create_climaatmos_parameter_set(FT)

    domain = SphericalShell(
        FT,
        radius = CAP.planet_radius(params),
        height = FT(30.0e3),
        nelements = nelements,
        npolynomial = npolynomial,
    )

    boundary_conditions = (;
        ρe_tot = (top = NoFlux(), bottom = NoFlux()),
        uh = (top = NoVectorFlux(), bottom = NoVectorFlux()),
    )

    model = Nonhydrostatic3DModel(
        domain = domain,
        boundary_conditions = boundary_conditions,
        parameters = params,
        hyperdiffusivity = FT(1e16),
        vertical_diffusion = ConstantViscosity(ν = FT(5)), #default: 5 [m^2/s]
    )

    # execute differently depending on testing mode
    if test_mode == :regression
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 100.0))
        @test simulation isa Simulation

        # test set function
        @unpack ρ, uh, w, ρe_tot = init_3d_baroclinic_wave(FT, params)
        set!(simulation, :base, ρ = ρ, uh = uh, w = w)
        set!(simulation, :thermodynamics, ρe_tot = ρe_tot)

        # test successful integration
        @test step!(simulation) isa Nothing # either error or integration runs

    # TODO!: Implement meaningful(!) regression test

    elseif test_mode == :validation
        # TODO!: Implement the rest plots and analyses
        # 1. sort out saveat kwarg for Simulation
        # 2. create animation for a rising bubble; timeseries of total energy
    else
        throw(ArgumentError("$test_mode incompatible with test case."))
    end

    simulation
end

@testset "3D baroclinic wave" begin
    for FT in (Float32, Float64)
        run_3d_baroclinic_wave(FT)
    end
end
