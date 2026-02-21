#####
##### Held-Suarez
#####

import Thermodynamics as TD
import Thermodynamics.Parameters as TDP
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

#####
##### Held-Suarez forcing
#####

function held_suarez_ΔT_y_T_equator(params, microphysics_model::DryModel)
    FT = eltype(params)
    ΔT_y = FT(CAP.ΔT_y_dry(params))
    T_equator = FT(CAP.T_equator_dry(params))
    return ΔT_y, T_equator
end

function held_suarez_ΔT_y_T_equator(
    params,
    microphysics_model::T,
) where {T <: MoistMicrophysics}
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
    T_sfc::FT,
    ρ::FT,
    p::FT,
    lat::FT,
    z_surface::FT,
    s::HeldSuarezForcingParams,
) where {FT}
    σ = compute_σ(z_surface, p, T_sfc, s)
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
    z_surface::FT,
    p::FT,
    T_sfc::FT,
    s::HeldSuarezForcingParams,
) where {FT}
    p / (s.MSLP * exp(-s.grav * z_surface / s.R_d / T_sfc))
end

height_factor(σ::FT, σ_b::FT) where {FT} = max(0, (σ - σ_b) / (1 - σ_b))
height_factor(z_surface::FT, p::FT, T_sfc::FT, s::HeldSuarezForcingParams) where {FT} =
    height_factor(compute_σ(z_surface, p, T_sfc, s), s.σ_b)

function held_suarez_forcing_tendency_ρe_tot(
    ᶜρ,
    ᶜuₕ,
    ᶜp,
    params,
    T_sfc,
    microphysics_model,
    forcing,
)
    forcing isa Nothing && return NullBroadcasted()
    ᶜspace = axes(ᶜρ)
    (; ᶜz, ᶠz) = z_coordinate_fields(ᶜspace)
    lat = Fields.coordinate_field(ᶜspace).lat

    # TODO: Don't need to enforce FT here, it should be done at param creation.
    FT = Spaces.undertype(ᶜspace)
    R_d = FT(CAP.R_d(params))
    κ_d = FT(CAP.kappa_d(params))
    cv_d = FT(CAP.cv_d(params))
    day = FT(CAP.day(params))
    MSLP = FT(CAP.MSLP(params))
    p_ref_theta = FT(CAP.p_ref_theta(params))
    grav = FT(CAP.grav(params))
    Δθ_z = FT(CAP.Δθ_z(params))
    T_min = FT(CAP.T_min_hs(params))
    σ_b = CAP.σ_b(params)
    k_f = 1 / day

    z_surface = Fields.level(ᶠz, Fields.half)

    ΔT_y, T_equator = held_suarez_ΔT_y_T_equator(params, microphysics_model)

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

    return @. lazy(
        -compute_ΔρT(T_sfc, ᶜρ, ᶜp, lat, z_surface, hs_params) * cv_d,
    )
end

function held_suarez_forcing_tendency_uₕ(
    ᶜuₕ,
    ᶜp,
    params,
    T_sfc,
    microphysics_model,
    forcing,
)
    forcing isa Nothing && return NullBroadcasted()
    ᶜspace = axes(ᶜp)
    (; ᶜz, ᶠz) = z_coordinate_fields(axes(ᶜp))
    # TODO: Don't need to enforce FT here, it should be done at param creation.
    FT = Spaces.undertype(ᶜspace)
    R_d = FT(CAP.R_d(params))
    κ_d = FT(CAP.kappa_d(params))
    cv_d = FT(CAP.cv_d(params))
    day = FT(CAP.day(params))
    MSLP = FT(CAP.MSLP(params))
    p_ref_theta = FT(CAP.p_ref_theta(params))
    grav = FT(CAP.grav(params))
    Δθ_z = FT(CAP.Δθ_z(params))
    T_min = FT(CAP.T_min_hs(params))
    σ_b = CAP.σ_b(params)
    k_f = 1 / day

    z_surface = Fields.level(ᶠz, Fields.half)

    ΔT_y, T_equator = held_suarez_ΔT_y_T_equator(params, microphysics_model)

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

    return @. lazy(-(k_f * height_factor(z_surface, ᶜp, T_sfc, hs_params)) * ᶜuₕ)
end
