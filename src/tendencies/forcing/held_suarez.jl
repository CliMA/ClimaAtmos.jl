#####
##### Held-Suarez forcing
#####

import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

function held_suarez_cache(Y)
    FT = Spaces.undertype(axes(Y.c))
    return (;
        ᶜσ = similar(Y.c, FT),
        ᶜheight_factor = similar(Y.c, FT),
        ᶜΔρT = similar(Y.c, FT),
        ᶜφ = deg2rad.(Fields.coordinate_field(Y.c).lat),
    )
end

function held_suarez_tendency!(Yₜ, Y, p, t, colidx)
    (; T_sfc, z_sfc, ᶜp, ᶜσ, ᶜheight_factor, ᶜΔρT, ᶜφ, params) = p # assume ᶜp has been updated

    FT = Spaces.undertype(axes(Y.c))
    R_d = FT(CAP.R_d(params))
    κ_d = FT(CAP.kappa_d(params))
    cv_d = FT(CAP.cv_d(params))
    day = FT(CAP.day(params))
    MSLP = FT(CAP.MSLP(params))
    grav = FT(CAP.grav(params))

    z_bottom = Spaces.level(Fields.coordinate_field(Y.c).z[colidx], 1)
    z_surface = Fields.Field(Fields.field_values(z_sfc[colidx]), axes(z_bottom))

    σ_b = FT(7 / 10)
    k_a = 1 / (40 * day)
    k_s = 1 / (4 * day)
    k_f = 1 / day
    if :ρq_tot in propertynames(Y.c)
        ΔT_y = FT(65)
        T_equator = FT(294)
    else
        ΔT_y = FT(60)
        T_equator = FT(315)
    end
    Δθ_z = FT(10)
    T_min = FT(200)

    @. ᶜσ[colidx] =
        ᶜp[colidx] / (MSLP * exp(-grav * z_surface / R_d / T_sfc[colidx]))

    @. ᶜheight_factor[colidx] = max(0, (ᶜσ[colidx] - σ_b) / (1 - σ_b))
    @. ᶜΔρT[colidx] =
        (k_a + (k_s - k_a) * ᶜheight_factor[colidx] * (cos(ᶜφ[colidx])^2)^2) *
        Y.c.ρ[colidx] *
        ( # ᶜT - ᶜT_equil
            ᶜp[colidx] / (Y.c.ρ[colidx] * R_d) - max(
                T_min,
                (
                    T_equator - ΔT_y * sin(ᶜφ[colidx])^2 -
                    Δθ_z * log(ᶜp[colidx] / MSLP) * cos(ᶜφ[colidx])^2
                ) * fast_pow(ᶜσ[colidx], κ_d),
            )
        )

    @. Yₜ.c.uₕ[colidx] -= (k_f * ᶜheight_factor[colidx]) * Y.c.uₕ[colidx]
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ[colidx] -= ᶜΔρT[colidx] * fast_pow((MSLP / ᶜp[colidx]), κ_d)
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot[colidx] -= ᶜΔρT[colidx] * cv_d
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int[colidx] -= ᶜΔρT[colidx] * cv_d
    end
    return nothing
end
