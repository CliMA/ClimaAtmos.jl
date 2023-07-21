#####
##### Held-Suarez
#####

import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

#####
##### No forcing
#####

forcing_cache(Y, forcing_type::Nothing) = (; forcing_type)
forcing_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) = nothing

#####
##### Held-Suarez forcing
#####

function forcing_cache(Y, forcing_type::HeldSuarezForcing)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        forcing_type,
        ᶜσ = similar(Y.c, FT),
        ᶜheight_factor = similar(Y.c, FT),
        ᶜΔρT = similar(Y.c, FT),
        ᶜφ = deg2rad.(Fields.coordinate_field(Y.c).lat),
    )
end

function forcing_tendency!(Yₜ, Y, p, t, colidx, ::HeldSuarezForcing)
    (; sfc_conditions, ᶜp, ᶜσ, ᶜheight_factor, ᶜΔρT, ᶜφ, params) = p

    # TODO: Don't need to enforce FT here, it should be done at param creation.
    FT = Spaces.undertype(axes(Y.c))
    R_d = FT(CAP.R_d(params))
    κ_d = FT(CAP.kappa_d(params))
    cv_d = FT(CAP.cv_d(params))
    day = FT(CAP.day(params))
    MSLP = FT(CAP.MSLP(params))
    grav = FT(CAP.grav(params))
    ΔT_y_dry = FT(CAP.ΔT_y_dry(params))
    ΔT_y_wet = FT(CAP.ΔT_y_wet(params))
    thermo_params = CAP.thermodynamics_params(params)

    z_surface =
        Fields.level(Fields.coordinate_field(Y.f).z[colidx], Fields.half)

    σ_b = FT(7 / 10)
    k_a = 1 / (40 * day)
    k_s = 1 / (4 * day)
    k_f = 1 / day
    if :ρq_tot in propertynames(Y.c)
        ΔT_y = ΔT_y_wet
        T_equator = FT(294)
    else
        ΔT_y = ΔT_y_dry
        T_equator = FT(315)
    end
    Δθ_z = FT(10)
    T_min = FT(200)

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
                    Δθ_z * log(ᶜp[colidx] / MSLP) * abs2(cos(ᶜφ[colidx]))
                ) * fast_pow(ᶜp[colidx] / MSLP, κ_d),
            )
        )

    @. Yₜ.c.uₕ[colidx] -= (k_f * ᶜheight_factor[colidx]) * Y.c.uₕ[colidx]
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ[colidx] -= ᶜΔρT[colidx] * fast_pow((MSLP / ᶜp[colidx]), κ_d)
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot[colidx] -= ᶜΔρT[colidx] * cv_d
    end
    return nothing
end
