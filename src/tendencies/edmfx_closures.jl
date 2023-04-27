#####
##### EDMF entrainment detrainment
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

"""
    Return physical vertical velocity - a projection of full velocity vector
    onto the vertical axis.
"""
get_physical_w(u, local_geometry) = Geometry.WVector(u, local_geometry)[1]

"""
    Return buoyancy on cell centers.
"""
function ᶜbuoyancy(params, ᶜρ_ref::FT, ᶜρ::FT) where {FT}
    # TODO - replace by grad(Φ) when we move to deep atmosphere
    g = CAP.grav(params)
    return g * (ᶜρ_ref - ᶜρ) / ᶜρ
end

"""
   Return entrainment and detrainment rates [1/s].

   We want the rates to be based on linear combination of pi-groups.
   But for now:
    - entr = const * w / z
    - detr = const * w

   Inputs (everything defined on cell centers):
   - params set with model parameters
   - entr_detr_flag - bool flag for if we want/don't want to compute
                      entrainment and detrainment
   - ᶜz, ᶜp, ᶜρ, - grid mean height, pressure and density
   - buoy_flux_surface - buoyancy flux at the surface
   - ᶜaʲ, ᶜwʲ, ᶜRHʲ, ᶜbuoyʲ - updraft area, physical vertical velocity,
                                   relative humidity and buoyancy
   - ᶜw⁰, ᶜRH⁰, ᶜbuoy⁰ - environment physical vertical velocity,
                              relative humidity and buoyancy
)
"""
function pi_groups_entr_detr(
    params,
    entr_detr_flag::Bool,
    ᶜz::FT,
    ᶜp::FT,
    ᶜρ::FT,
    buoy_flux_surface::FT,
    ᶜaʲ::FT,
    ᶜwʲ::FT,
    ᶜRHʲ::FT,
    ᶜbuoyʲ::FT,
    ᶜw⁰::FT,
    ᶜRH⁰::FT,
    ᶜbuoy⁰::FT,
) where {FT}

    if ᶜaʲ <= FT(0) || !entr_detr_flag
        return (; entr = FT(0), detr = FT(0))
    else
        g = CAP.grav(params)

        turbconv_params = CAP.turbconv_params(params)
        ᶜaʲ_max = TCP.max_area(turbconv_params)
        max_area_limiter = 0.1 * exp(-10 * (ᶜaʲ_max - ᶜaʲ))

        # pressure scale height (height where pressure drops by 1/e)
        ref_H = ᶜp / (ᶜρ * g)
        # convective velocity
        w_star = TC.get_wstar(buoy_flux_surface)

        # non-dimensional pi-groups
        # TODO - using Π₁ blows things up
        Π₁ = ᶜz * (ᶜbuoyʲ - ᶜbuoy⁰) / ((ᶜwʲ - ᶜw⁰)^2 + w_star^2 + eps(FT))
        Π₃ = sqrt(ᶜaʲ)
        Π₄ = ᶜRHʲ - ᶜRH⁰
        Π₆ = ᶜz / ref_H
        entr = max(
            0,
            ᶜwʲ / ᶜz * (
                -4.013288 - 0.000968 * Π₁ + 0.356974 * Π₃ - 0.403124 * Π₄ +
                1.503261 * Π₆
            ),
        )
        detr = max(
            0,
            ᶜwʲ * (
                3.535208 + 0.598496 * Π₁ + 1.583348 * Π₃ + 0.046275 * Π₄ -
                0.344836 * Π₆ + max_area_limiter
            ),
        )

        # TODO - Temporary: Switch to Π groups after simple tests are done
        # (kinematic, bubble, Bomex)
        # and/or we can calibrate things in ClimaAtmos
        entr = max(0, 1e-1 * ᶜwʲ / ᶜz)
        detr = max(0, 1e-3 * ᶜwʲ)

        return (; entr, detr)
    end
end

edmfx_entr_detr_cache(Y, turbconv_model) = (;)
function edmfx_entr_detr_cache(Y, turbconv_model::EDMFX)
    n = n_mass_flux_subdomains(turbconv_model)
    FT = eltype(Y)
    return (;
        ᶜentr_detrʲs = similar(
            Y.c,
            NTuple{n, NamedTuple{(:entr, :detr), NTuple{2, FT}}},
        )
    )
end

edmfx_entr_detr_tendency!(Yₜ, Y, p, t, colidx, turbconv_model) = nothing
function edmfx_entr_detr_tendency!(Yₜ, Y, p, t, colidx, turbconv_model::EDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    ᶜlg = Fields.local_geometry_field(Y.c)

    (; params, ᶜp, ᶜρ_ref, buoy_flux_surface) = p
    (; ᶜρʲs, ᶜtsʲs, ᶜuʲs, ᶜspecificʲs, ᶜentr_detrʲs) = p
    (; ᶜρ⁰, ᶜts⁰, ᶜu⁰, ᶜspecific⁰) = p

    thermo_params = CAP.thermodynamics_params(params)

    ᶜz = Fields.coordinate_field(Y.c).z

    for j in 1:n
        @. ᶜentr_detrʲs.:($$j)[colidx] = pi_groups_entr_detr(
            params,
            p.atmos.edmfx_entr_detr,
            ᶜz[colidx],
            ᶜp[colidx],
            Y.c.ρ[colidx],
            buoy_flux_surface[colidx],
            Y.c.sgsʲs.:($$j).ρa[colidx] / ᶜρʲs.:($$j)[colidx],
            get_physical_w(ᶜuʲs.:($$j)[colidx], ᶜlg[colidx]),
            TD.relative_humidity(thermo_params, ᶜtsʲs.:($$j)[colidx]),
            ᶜbuoyancy(params, ᶜρ_ref[colidx], ᶜρʲs.:($$j)[colidx]),
            get_physical_w(ᶜu⁰[colidx], ᶜlg[colidx]),
            TD.relative_humidity(thermo_params, ᶜts⁰[colidx]),
            ᶜbuoyancy(params, ᶜρ_ref[colidx], ᶜρ⁰[colidx]),
        )

        @. Yₜ.c.sgsʲs.:($$j).ρa[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] * (
                ᶜentr_detrʲs.:($$j).entr[colidx] -
                ᶜentr_detrʲs.:($$j).detr[colidx]
            )

        @. Yₜ.c.sgsʲs.:($$j).ρae_tot[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] * (
                ᶜentr_detrʲs.:($$j).entr[colidx] * TD.total_specific_enthalpy(
                    thermo_params,
                    ᶜts⁰[colidx],
                    ᶜspecific⁰.e_tot[colidx],
                ) -
                ᶜentr_detrʲs.:($$j).detr[colidx] * TD.total_specific_enthalpy(
                    thermo_params,
                    ᶜtsʲs.:($$j)[colidx],
                    ᶜspecificʲs.:($$j).e_tot[colidx],
                )
            )

        @. Yₜ.c.sgsʲs.:($$j).ρaq_tot[colidx] +=
            Y.c.sgsʲs.:($$j).ρa[colidx] * (
                ᶜspecific⁰.q_tot[colidx] * ᶜentr_detrʲs.:($$j).entr[colidx] -
                ᶜspecificʲs.:($$j).q_tot[colidx] *
                ᶜentr_detrʲs.:($$j).detr[colidx]
            )

        @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] +=
            ᶠinterp(ᶜentr_detrʲs.:($$j).entr[colidx] * C3(ᶜu⁰[colidx])) -
            ᶠinterp(ᶜentr_detrʲs.:($$j).entr[colidx]) *
            Y.f.sgsʲs.:($$j).u₃[colidx]
    end
    return nothing
end
