#####
##### Held-Suarez
#####

import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

forcing_cache(Y, atmos::AtmosModel) = forcing_cache(Y, atmos.forcing_type)

#####
##### No forcing
#####

forcing_cache(Y, forcing_type::Nothing) = (;)
forcing_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) = nothing

#####
##### Held-Suarez forcing
#####

function forcing_cache(Y, forcing_type::HeldSuarezForcing)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        ᶜσ = similar(Y.c, FT),
        ᶜheight_factor = similar(Y.c, FT),
        ᶜΔρT = similar(Y.c, FT),
        ᶜφ = deg2rad.(Fields.coordinate_field(Y.c).lat),
    )
end

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

function forcing_tendency!(Yₜ, Y, p, t, colidx, ::HeldSuarezForcing)
    (; params) = p
    (; ᶜp, sfc_conditions) = p.precomputed
    (; ᶜσ, ᶜheight_factor, ᶜΔρT, ᶜφ) = p.forcing

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
    k_a = 1 / (40 * day)
    k_s = 1 / (4 * day)
    k_f = 1 / day

    z_surface =
        Fields.level(Fields.coordinate_field(Y.f).z[colidx], Fields.half)

    ΔT_y, T_equator = held_suarez_ΔT_y_T_equator(params, p.atmos.moisture_model)

    @. ᶜσ[colidx] =
        ᶜp[colidx] / (
            MSLP * exp(
                -grav * z_surface / R_d /
                TD.air_temperature(thermo_params, sfc_conditions.ts[colidx]),
            )
        )

    @. ᶜheight_factor[colidx] = max(0, (ᶜσ[colidx] - σ_b) / (1 - σ_b))
    @. ᶜΔρT[colidx] =
        (
            k_a +
            (k_s - k_a) * ᶜheight_factor[colidx] * abs2(abs2(cos(ᶜφ[colidx])))
        ) *
        Y.c.ρ[colidx] *
        ( # ᶜT - ᶜT_equil
            ᶜp[colidx] / (Y.c.ρ[colidx] * R_d) - max(
                T_min,
                (
                    T_equator - ΔT_y * abs2(sin(ᶜφ[colidx])) -
                    Δθ_z *
                    log(ᶜp[colidx] / p_ref_theta) *
                    abs2(cos(ᶜφ[colidx]))
                ) * fast_pow(ᶜp[colidx] / p_ref_theta, κ_d),
            )
        )

    @. Yₜ.c.uₕ[colidx] -= (k_f * ᶜheight_factor[colidx]) * Y.c.uₕ[colidx]
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ[colidx] -=
            ᶜΔρT[colidx] * fast_pow((p_ref_theta / ᶜp[colidx]), κ_d)
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot[colidx] -= ᶜΔρT[colidx] * cv_d
    end
    return nothing
end
