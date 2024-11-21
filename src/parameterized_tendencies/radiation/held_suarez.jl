#####
##### Held-Suarez
#####

import Thermodynamics as TD
import Thermodynamics.Parameters as TDP
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

#####
##### No forcing
#####

forcing_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

#####
##### Held-Suarez forcing
#####

function held_suarez_ΔT_y_T_equator(params, moisture_model::DryModel)
    FT = eltype(params)
    ΔT_y = FT(CAP.ΔT_y_dry(params))
    T_equator = FT(CAP.T_equator_dry(params))
    return ΔT_y, T_equator
end

function held_suarez_ΔT_y_T_equator(
    params,
    moisture_model::T,
) where {T <: Union{EquilMoistModel, NonEquilMoistModel}}
    FT = eltype(params)
    ΔT_y = FT(CAP.ΔT_y_wet(params))
    T_equator = FT(CAP.T_equator_wet(params))
    return ΔT_y, T_equator
end

struct HeldSuarezForcingParams{FT}
    ΔT_y::FT
    day::FT
    σ_b::FT
    R_d::FT
    T_min::FT
    T_equator::FT
    Δθ_z::FT
    p_ref_theta::FT
    κ_d::FT
    grav::FT
    MSLP::FT
end
Base.Broadcast.broadcastable(x::HeldSuarezForcingParams) = tuple(x)

function compute_ΔρT(
    thermo_params::TDP.ThermodynamicsParameters,
    ts_surf::TD.ThermodynamicState,
    ρ::FT,
    p::FT,
    lat::FT,
    z_surface::FT,
    s::HeldSuarezForcingParams,
) where {FT}
    σ = compute_σ(thermo_params, z_surface, p, ts_surf, s)
    k_a = 1 / (40 * s.day)
    k_s = 1 / (4 * s.day)

    φ = deg2rad(lat)
    return (k_a + (k_s - k_a) * height_factor(σ, s.σ_b) * abs2(abs2(cos(φ)))) *
           ρ *
           ( # ᶜT - ᶜT_equil
               p / (ρ * s.R_d) - max(
                   s.T_min,
                   (
                       s.T_equator - s.ΔT_y * abs2(sin(φ)) -
                       s.Δθ_z * log(p / s.p_ref_theta) * abs2(cos(φ))
                   ) * fast_pow(p / s.p_ref_theta, s.κ_d),
               )
           )
end

function compute_σ(
    thermo_params::TDP.ThermodynamicsParameters,
    z_surface::FT,
    p::FT,
    ts_surf::TD.ThermodynamicState,
    s::HeldSuarezForcingParams,
) where {FT}
    p / (
        s.MSLP * exp(
            -s.grav * z_surface / s.R_d /
            TD.air_temperature(thermo_params, ts_surf),
        )
    )
end

height_factor(σ::FT, σ_b::FT) where {FT} = max(0, (σ - σ_b) / (1 - σ_b))
height_factor(
    thermo_params::TDP.ThermodynamicsParameters,
    z_surface::FT,
    p::FT,
    ts_surf::TD.ThermodynamicState,
    s::HeldSuarezForcingParams,
) where {FT} =
    height_factor(compute_σ(thermo_params, z_surface, p, ts_surf, s), s.σ_b)

function forcing_tendency!(Yₜ, Y, p, t, ::HeldSuarezForcing)
    (; params) = p
    (; ᶜp, sfc_conditions) = p.precomputed

    # TODO: Don't need to enforce FT here, it should be done at param creation.
    FT = Spaces.undertype(axes(Y.c))
    R_d = FT(CAP.R_d(params))
    κ_d = FT(CAP.kappa_d(params))
    cv_d = FT(CAP.cv_d(params))
    day = FT(CAP.day(params))
    MSLP = FT(CAP.MSLP(params))
    p_ref_theta = FT(CAP.p_ref_theta(params))
    grav = FT(CAP.grav(params))
    Δθ_z = FT(CAP.Δθ_z(params))
    T_min = FT(CAP.T_min_hs(params))
    thermo_params = CAP.thermodynamics_params(params)
    σ_b = CAP.σ_b(params)
    k_f = 1 / day

    z_surface = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)

    ΔT_y, T_equator = held_suarez_ΔT_y_T_equator(params, p.atmos.moisture_model)

    hs_params = HeldSuarezForcingParams{FT}(
        ΔT_y,
        day,
        σ_b,
        R_d,
        T_min,
        T_equator,
        Δθ_z,
        p_ref_theta,
        κ_d,
        grav,
        MSLP,
    )

    lat = Fields.coordinate_field(Y.c).lat
    @. Yₜ.c.uₕ -=
        (
            k_f * height_factor(
                thermo_params,
                z_surface,
                ᶜp,
                sfc_conditions.ts,
                hs_params,
            )
        ) * Y.c.uₕ
    @. Yₜ.c.ρe_tot -=
        compute_ΔρT(
            thermo_params,
            sfc_conditions.ts,
            Y.c.ρ,
            ᶜp,
            lat,
            z_surface,
            hs_params,
        ) * cv_d
    return nothing
end
