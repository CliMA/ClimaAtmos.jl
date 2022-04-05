using OrdinaryDiffEq: SSPRK33
using ClimaCorePlots, Plots
using UnPack

using CLIMAParameters
using ClimaAtmos.Utils.InitialConditions: init_3d_baroclinic_wave
using ClimaAtmos.Domains
using ClimaAtmos.BoundaryConditions
using ClimaAtmos.Models: ConstantViscosity
using ClimaAtmos.Models.Nonhydrostatic3DModels
using ClimaAtmos.Simulations

# Set up parameters
const FT = Float64
const day = FT(86400)
Base.@kwdef struct HeldSuarezParameters <: CLIMAParameters.AbstractEarthParameterSet 
    k_a = FT(1 / (40 * day))
    k_f = FT(1 / day)
    k_s = FT(1 / (4 * day))
    ΔT_y = FT(60)
    Δθ_z = FT(10)
    T_equator = FT(315)
    T_min = FT(200)
    σ_b = FT(7 / 10)
end

stepper = SSPRK33()
nelements = (4, 10)
npolynomial = 4
dt = FT(5)

params = HeldSuarezParameters()
# function run_3d_baroclinic_wave(
#     ::Type{FT};
#     stepper = SSPRK33(),
#     nelements = (6, 10),
#     npolynomial = 3,
#     case = :default,
#     dt = 0.02,
#     callbacks = (),
#     test_mode = :regression,
# ) where {FT}
    

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
        hyperdiffusivity = FT(1e16),
        vertical_diffusion = ConstantViscosity(ν = FT(1)),
    )

    # execute differently depending on testing mode
    if test_mode == :regression
        simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 1.0))
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

    nothing
end

@testset "3D baroclinic wave" begin
    for FT in (Float32, Float64)
        run_3d_baroclinic_wave(FT)
    end
end
