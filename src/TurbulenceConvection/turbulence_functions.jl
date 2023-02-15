# convective velocity scale
function get_wstar(bflux::FT) where {FT}
    # average depth of the mixed layer (prescribed for now)
    zi = FT(1000)
    return cbrt(max(bflux * zi, 0))
end

function buoyancy_c(thermo_params, ρ::FT, ρ_i::FT) where {FT}
    g::FT = TD.Parameters.grav(thermo_params)
    return g * (ρ - ρ_i) / ρ
end

# MO scaling of near surface tke and scalar variance
function get_surface_tke(
    mixing_length_params,
    ustar::FT,
    zLL::FT,
    oblength::FT,
) where {FT}
    κ_star² = mixing_length_params.κ_star²
    if oblength < 0
        return (
            (κ_star² + cbrt(zLL / oblength * zLL / oblength)) * ustar * ustar
        )
    else
        return κ_star² * ustar * ustar
    end
end
function get_surface_variance(
    flux1::FT,
    flux2::FT,
    ustar::FT,
    zLL::FT,
    oblength::FT,
) where {FT}
    c_star1 = -flux1 / ustar
    c_star2 = -flux2 / ustar
    if oblength < 0
        return 4 *
               c_star1 *
               c_star2 *
               (1 - FT(8.3) * zLL / oblength)^(-FT(2 / 3))
    else
        return 4 * c_star1 * c_star2
    end
end

function gradient_Richardson_number(
    mixing_length_params,
    ∂b∂z::FT,
    Shear²::FT,
    ϵ::FT,
) where {FT}
    Ri_c::FT = mixing_length_params.Ri_c
    return min(∂b∂z / max(Shear², ϵ), Ri_c)
end

# Turbulent Prandtl number given the obukhov length sign and the gradient Richardson number
function turbulent_Prandtl_number(
    mixing_length_params,
    obukhov_length::FT,
    ∇Ri::FT,
) where {FT}
    ω_pr = mixing_length_params.ω_pr
    Pr_n = mixing_length_params.Pr_n
    if obukhov_length > 0 && ∇Ri > 0 #stable
        # CSB (Dan Li, 2019, eq. 75), where ω_pr = ω_1 + 1 = 53.0/13.0
        prandtl_nvec =
            Pr_n *
            (2 * ∇Ri / (1 + ω_pr * ∇Ri - sqrt((1 + ω_pr * ∇Ri)^2 - 4 * ∇Ri)))
    else
        prandtl_nvec = Pr_n
    end
    return prandtl_nvec
end
