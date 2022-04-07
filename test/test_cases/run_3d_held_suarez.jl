using Test

using OrdinaryDiffEq: SSPRK33
using ClimaCorePlots, Plots
using UnPack

using CLIMAParameters

using ClimaAtmos.Utils.Forcings
using ClimaAtmos.Utils.InitialConditions: init_3d_baroclinic_wave
using ClimaAtmos.Domains
using ClimaAtmos.BoundaryConditions
using ClimaAtmos.Models: ConstantViscosity
using ClimaAtmos.Models.Nonhydrostatic3DModels
using ClimaAtmos.Simulations

# Set up parameters
const FT = Float32
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
    vertical_diffusion = ConstantViscosity(ν = FT(0)),
    sources = (;forcing = HeldSuarezForcing(params))
)

simulation = Simulation(model, stepper, dt = dt, tspan = (0.0, 2dt))
@test simulation isa Simulation

# test set function
@unpack ρ, uh, w, ρe_tot = init_3d_baroclinic_wave(FT, params)
set!(simulation, :base, ρ = ρ, uh = uh, w = w)
set!(simulation, :thermodynamics, ρe_tot = ρe_tot)

step!(simulation) 


