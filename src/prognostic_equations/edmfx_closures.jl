#####
##### EDMF closures (nonhydrostatic pressure drag and mixing length)
#####

import StaticArrays as SA
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

"""
    Return draft area given ρa and ρ
"""
function draft_area(ρa::FT, ρ::FT) where {FT}
    return ρa / ρ
end

"""
    Return buoyancy on cell centers.
"""
function ᶜphysical_buoyancy(params, ᶜρ_ref::FT, ᶜρ::FT) where {FT}
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
    Return surface flux of TKE, a C3 vector used by ClimaAtmos operator boundary conditions
"""
function surface_flux_tke(
    turbconv_params,
    ρ_int,
    u_int,
    ustar,
    interior_local_geometry,
    surface_local_geometry,
)
    c_d = CAP.tke_diss_coeff(turbconv_params)
    c_m = CAP.tke_ed_coeff(turbconv_params)
    k_star² = CAP.tke_surf_scale(turbconv_params)
    speed = Geometry._norm(
        CA.CT12(u_int, interior_local_geometry),
        interior_local_geometry,
    )
    c3_unit = C3(unit_basis_vector_data(C3, surface_local_geometry))
    return ρ_int * (1 - c_d * c_m * k_star²^2) * ustar^2 * speed * c3_unit
end

"""
   Return the nonhydrostatic pressure drag for updrafts [m/s2 * m]

   Inputs (everything defined on cell faces):
   - params - set with model parameters
   - ᶠlg - local geometry (needed to compute the norm inside a local function)
   - ᶠbuoyʲ - covariant3 or contravariant3 updraft buoyancy
   - ᶠu3ʲ, ᶠu3⁰ - covariant3 or contravariant3 velocity for updraft and environment.
                  covariant3 velocity is used in prognostic edmf, and contravariant3
                  velocity is used in diagnostic edmf.
   - updraft top height
"""
function ᶠupdraft_nh_pressure(params, ᶠlg, ᶠbuoyʲ, ᶠu3ʲ, ᶠu3⁰, plume_height)
    turbconv_params = CAP.turbconv_params(params)
    # factor multiplier for pressure buoyancy terms (effective buoyancy is (1-α_b))
    α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
    # factor multiplier for pressure drag
    α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)

    # Independence of aspect ratio hardcoded: α₂_asp_ratio² = FT(0)

    H_up_min = CAP.min_updraft_top(turbconv_params)

    # We also used to have advection term here: α_a * w_up * div_w_up
    return α_b * ᶠbuoyʲ +
           α_d * (ᶠu3ʲ - ᶠu3⁰) * CC.Geometry._norm(ᶠu3ʲ - ᶠu3⁰, ᶠlg) /
           max(plume_height, H_up_min)
end

edmfx_nh_pressure_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing
function edmfx_nh_pressure_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)

    n = n_mass_flux_subdomains(turbconv_model)
    (; params) = p
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜρʲs, ᶠnh_pressure₃ʲs, ᶠu₃⁰) = p.precomputed
    ᶠlg = Fields.local_geometry_field(Y.f)

    scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)

    for j in 1:n
        if p.atmos.edmfx_model.nh_pressure isa Val{true}
            @. ᶠnh_pressure₃ʲs.:($$j) = ᶠupdraft_nh_pressure(
                params,
                ᶠlg,
                ᶠbuoyancy(ᶠinterp(Y.c.ρ), ᶠinterp(ᶜρʲs.:($$j)), ᶠgradᵥ_ᶜΦ),
                Y.f.sgsʲs.:($$j).u₃,
                ᶠu₃⁰,
                scale_height,
            )
            @. Yₜ.f.sgsʲs.:($$j).u₃ -= ᶠnh_pressure₃ʲs.:($$j)
        else
            @. ᶠnh_pressure₃ʲs.:($$j) = C3(0)
        end
    end
end

# lambert_2_over_e(::Type{FT}) where {FT} = FT(LambertW.lambertw(FT(2) / FT(MathConstants.e)))
lambert_2_over_e(::Type{FT}) where {FT} = FT(0.46305551336554884) # since we can evaluate

function lamb_smooth_minimum(
    l::SA.SVector,
    lower_bound::FT,
    upper_bound::FT,
) where {FT}
    x_min = minimum(l)
    λ_0 = max(x_min * lower_bound / lambert_2_over_e(FT), upper_bound)

    num = sum(l_i -> l_i * exp(-(l_i - x_min) / λ_0), l)
    den = sum(l_i -> exp(-(l_i - x_min) / λ_0), l)
    smin = num / den
    return smin
end

"""
    mixing_length(params, ustar, ᶜz, sfc_tke, ᶜlinear_buoygrad, ᶜtke, obukhov_length, ᶜstrain_rate_norm, ᶜPr, ᶜtke_exch)

where:
- `params`: set with model parameters
- `ustar`: friction velocity
- `ᶜz`: height
- `tke_sfc`: env kinetic energy at first cell center
- `ᶜlinear_buoygrad`: buoyancy gradient
- `ᶜtke`: env turbulent kinetic energy
- `obukhov_length`: surface Monin Obukhov length
- `ᶜstrain_rate_norm`: Frobenius norm of strain rate tensor
- `ᶜPr`: Prandtl number
- `ᶜtke_exch`: subdomain exchange term

Returns mixing length as a smooth minimum between
wall-constrained length scale,
production-dissipation balanced length scale,
effective static stability length scale, and
Smagorinsky length scale.
"""
function mixing_length(
    params,
    ustar,
    ᶜz::FT,
    z_sfc,
    ᶜdz,
    sfc_tke,
    ᶜlinear_buoygrad,
    ᶜtke,
    obukhov_length,
    ᶜstrain_rate_norm,
    ᶜPr,
    ᶜtke_exch,
) where {FT}

    turbconv_params = CAP.turbconv_params(params)
    c_m = CAP.tke_ed_coeff(turbconv_params)
    c_d = CAP.tke_diss_coeff(turbconv_params)
    smin_ub = CAP.smin_ub(turbconv_params)
    smin_rm = CAP.smin_rm(turbconv_params)
    c_b = CAP.static_stab_coeff(turbconv_params)
    vkc = CAP.von_karman_const(params)

    # compute the maximum mixing length at height z
    l_z = ᶜz - z_sfc

    # compute the l_W - the wall constraint mixing length
    # which imposes an upper limit on the size of eddies near the surface
    # kz scale (surface layer)
    if obukhov_length < 0.0 #unstable
        l_W =
            vkc * (ᶜz - z_sfc) /
            max(sqrt(sfc_tke / ustar / ustar) * c_m, eps(FT)) *
            min((1 - 100 * (ᶜz - z_sfc) / obukhov_length)^FT(0.2), 1 / vkc)
    else # neutral or stable
        l_W =
            vkc * (ᶜz - z_sfc) /
            max(sqrt(sfc_tke / ustar / ustar) * c_m, eps(FT))
    end

    # compute l_TKE - the production-dissipation balanced length scale
    a_pd = c_m * (2 * ᶜstrain_rate_norm - ᶜlinear_buoygrad / ᶜPr) * sqrt(ᶜtke)
    # Dissipation term
    c_neg = c_d * ᶜtke * sqrt(ᶜtke)
    if abs(a_pd) > eps(FT) && 4 * a_pd * c_neg > -(ᶜtke_exch * ᶜtke_exch)
        l_TKE = max(
            -(ᶜtke_exch / 2 / a_pd) +
            sqrt(ᶜtke_exch * ᶜtke_exch + 4 * a_pd * c_neg) / 2 / a_pd,
            0,
        )
    elseif abs(a_pd) < eps(FT) && abs(ᶜtke_exch) > eps(FT)
        l_TKE = c_neg / ᶜtke_exch
    else
        l_TKE = FT(0)
    end

    # compute l_N - the effective static stability length scale.
    N_eff = sqrt(max(ᶜlinear_buoygrad, 0))
    if N_eff > 0.0
        l_N = min(sqrt(max(c_b * ᶜtke, 0)) / N_eff, l_z)
    else
        l_N = l_z
    end

    # compute l_smag - smagorinsky length scale
    l_smag = smagorinsky_lilly_length(
        CAP.c_smag(params),
        N_eff,
        ᶜdz,
        ᶜPr,
        ᶜstrain_rate_norm,
    )

    # add limiters
    l = SA.SVector(
        l_N > l_z ? l_z : l_N,
        l_TKE > l_z ? l_z : l_TKE,
        l_W > l_z ? l_z : l_W,
    )
    # get soft minimum
    l_smin = lamb_smooth_minimum(l, smin_ub, smin_rm)
    l_limited = max(l_smag, min(l_smin, l_z))

    return MixingLength(l_limited, l_W, l_TKE, l_N)
end

"""
    turbulent_prandtl_number(params, obukhov_length, ᶜRi_grad)

where:
- `params`: set with model parameters
- `obukhov_length`: surface Monin Obukhov length
- `ᶜRi_grad`: gradient Richardson number

Returns the turbulent Prandtl number give the obukhov length sign and
the gradient Richardson number, which is calculated from the linearized
buoyancy gradient and shear production.
"""
function turbulent_prandtl_number(
    params,
    obukhov_length::FT,
    ᶜlinear_buoygrad::FT,
    ᶜstrain_rate_norm::FT,
) where {FT}
    turbconv_params = CAP.turbconv_params(params)
    Ri_c = CAP.Ri_crit(turbconv_params)
    ω_pr = CAP.Prandtl_number_scale(turbconv_params)
    Pr_n = CAP.Prandtl_number_0(turbconv_params)
    ᶜRi_grad = min(ᶜlinear_buoygrad / max(2 * ᶜstrain_rate_norm, eps(FT)), Ri_c)
    if obukhov_length > 0 && ᶜRi_grad > 0 #stable
        # CSB (Dan Li, 2019, eq. 75), where ω_pr = ω_1 + 1 = 53.0 / 13.0
        prandtl_nvec =
            Pr_n * (
                2 * ᶜRi_grad / (
                    1 + ω_pr * ᶜRi_grad -
                    sqrt((1 + ω_pr * ᶜRi_grad)^2 - 4 * ᶜRi_grad)
                )
            )
    else
        prandtl_nvec = Pr_n
    end
    return prandtl_nvec
end

edmfx_filter_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

"""
   Apply EDMF filters:
   - Relax u_3 to zero when it is negative
   - Relax ρa to zero when it is negative
"""
function edmfx_filter_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    (; dt) = p

    if p.atmos.edmfx_model.filter isa Val{true}
        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                C3(min(Y.f.sgsʲs.:($$j).u₃.components.data.:1, 0)) / dt
            @. Yₜ.c.sgsʲs.:($$j).ρa -= min(Y.c.sgsʲs.:($$j).ρa, 0) / dt
        end
    end
end
