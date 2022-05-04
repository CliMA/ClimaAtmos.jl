using PrettyTables

struct EarthParameterSet <: AbstractEarthParameterSet end

Δx = FT(1) # Note: This value shouldn't matter, since we only have 1 column.

params = EarthParameterSet()
horizontal_mesh =
    periodic_rectangle_mesh(; x_max = Δx, y_max = Δx, x_elem = 1, y_elem = 1)
quad = Spaces.Quadratures.GL{1}()
z_max = FT(90e3)
z_elem = 90
z_stretch = Meshes.GeneralizedExponentialStretching(FT(100), FT(10000))
t_end = FT(60 * 60 * 24 * 365.25)
dt = FT(60 * 60 * 3)
dt_save_to_sol = 10 * dt
ode_algorithm = OrdinaryDiffEq.Rosenbrock23

additional_tendency!(Yₜ, Y, p, t) = rrtmgp_model_tendency!(Yₜ, Y, p, t)
additional_callbacks = (PeriodicCallback(
    rrtmgp_model_callback!,
    dt; # this will usually be bigger than dt, but for this example it can be dt
    initial_affect = true, # run callback at t = 0
    save_positions = (false, false), # do not save Y before and after callback
),)

# TODO: dispatch into this method
function center_initial_condition(
    local_geometry,
    params,
    ᶜ𝔼_name,
    moisture_mode,
)
    z = local_geometry.coordinates.z
    FT = eltype(z)

    R_d = FT(Planet.R_d(params))
    MSLP = FT(Planet.MSLP(params))
    grav = FT(Planet.grav(params))

    T = FT(300)
    p = MSLP * exp(-z * grav / (R_d * T))
    ρ = p / (R_d * T)
    ts = TD.PhaseDry_ρp(params, ρ, p)

    if ᶜ𝔼_name === Val(:ρθ)
        𝔼_kwarg = (; ρθ = ρ * TD.liquid_ice_pottemp(params, ts))
    elseif ᶜ𝔼_name === Val(:ρe)
        𝔼_kwarg = (; ρe = ρ * (TD.internal_energy(params, ts) + grav * z))
    elseif ᶜ𝔼_name === Val(:ρe_int)
        𝔼_kwarg = (; ρe_int = ρ * TD.internal_energy(params, ts))
    end
    return (; ρ, 𝔼_kwarg..., uₕ = Geometry.Covariant12Vector(FT(0), FT(0)))
end
