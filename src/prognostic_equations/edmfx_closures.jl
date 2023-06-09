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
    # TODO - replace by ᶜgradᵥᶠΦ when we move to deep atmosphere
    g = CAP.grav(params)
    return (ᶜρ_ref - ᶜρ) / ᶜρ * g
end
"""
    Return buoyancy on cell faces.
"""
function ᶠbuoyancy(ᶠρ_ref, ᶠρ, ᶠgradᵥ_ᶜΦ)
    return (ᶠρ_ref - ᶠρ) / ᶠρ * ᶠgradᵥ_ᶜΦ
end

"""
   Return the nonhydrostatic pressure drag for updrafts [m/s2 * m]

   Inputs (everything defined on cell faces):
   - params - set with model parameters
   - nh_presssure_flag - bool flag for if we want/don't want to compute
                         pressure drag
   - ᶠlg - local geometry (needed to compute the norm inside a local function)
   - ᶠbuoyʲ - updraft buoyancy
   - ᶠu₃ʲ, ᶠu₃⁰ - Covariant3Vector velocity for updraft and environment
   - updraft top height
"""
function ᶠupdraft_nh_pressure(
    params,
    nh_pressure_flag,
    ᶠlg,
    ᶠbuoyʲ,
    ᶠu₃ʲ,
    ᶠu₃⁰,
    updraft_top,
)
    FT = eltype(updraft_top)

    if !nh_pressure_flag
        return C3(FT(0))
    else
        turbconv_params = CAP.turbconv_params(params)
        # factor multiplier for pressure buoyancy terms (effective buoyancy is (1-α_b))
        α_b = TCP.pressure_normalmode_buoy_coeff1(turbconv_params)
        # factor multiplier for pressure drag
        α_d = TCP.pressure_normalmode_drag_coeff(turbconv_params)

        # Independence of aspect ratio hardcoded: α₂_asp_ratio² = FT(0)

        H_up_min = TCP.min_updraft_top(turbconv_params)
        plume_scale_height = max(updraft_top, H_up_min)

        # We also used to have advection term here: α_a * w_up * div_w_up
        return α_b * ᶠbuoyʲ -
               α_d * (ᶠu₃ʲ - ᶠu₃⁰) * CC.Geometry._norm(ᶠu₃ʲ - ᶠu₃⁰, ᶠlg) /
               plume_scale_height
    end
end

edmfx_nh_pressure_cache(Y, turbconv_model) = (;)

edmfx_nh_pressure_tendency!(Yₜ, Y, p, t, colidx, turbconv_model) = nothing
function edmfx_nh_pressure_tendency!(Yₜ, Y, p, t, colidx, turbconv_model::EDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    (; params, ᶜρʲs, ᶜρ_ref, ᶠgradᵥ_ᶜΦ, ᶜuʲs, ᶜu⁰, ᶠu₃⁰) = p
    FT = eltype(Y)
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠlg = Fields.local_geometry_field(Y.f)

    turbconv_params = CAP.turbconv_params(params)
    a_min = TCP.min_area(turbconv_params)

    for j in 1:n

        # look for updraft top
        updraft_top = FT(0)
        for level in 1:Spaces.nlevels(axes(ᶜz))
            if Spaces.level(Y.c.sgsʲs.:($j).ρa[colidx], level)[] /
               Spaces.level(ᶜρʲs.:($j)[colidx], level)[] > a_min
                updraft_top = Spaces.level(ᶜz[colidx], level)[]
            end
        end

        @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] -= ᶠupdraft_nh_pressure(
            params,
            p.atmos.edmfx_nh_pressure,
            ᶠlg[colidx],
            ᶠbuoyancy(
                ᶠinterp(ᶜρ_ref[colidx]),
                ᶠinterp(ᶜρʲs.:($$j)[colidx]),
                ᶠgradᵥ_ᶜΦ[colidx],
            ),
            Y.f.sgsʲs.:($$j).u₃[colidx],
            ᶠu₃⁰[colidx],
            updraft_top,
        )
    end
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
        w_star = get_wstar(buoy_flux_surface)

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
        entr = max(0, 1 * ᶜwʲ / ᶜz)
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

    (; params, ᶜp, ᶜρ_ref, sfc_conditions) = p
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
            sfc_conditions[colidx].buoyancy_flux,
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
