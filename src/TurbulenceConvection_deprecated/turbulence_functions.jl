function buoyancy_c(thermo_params, ρ::FT, ρ_i::FT) where {FT}
    g::FT = TD.Parameters.grav(thermo_params)
    return g * (ρ - ρ_i) / ρ
end

# MO scaling of near surface tke and scalar variance
function get_surface_tke(param_set, ustar::FT, zLL::FT, oblength::FT) where {FT}
    κ_star² = TCP.κ_star²(param_set)
    if oblength < 0
        return (
            (κ_star² + cbrt(zLL / oblength * zLL / oblength)) * ustar * ustar
        )
    else
        return κ_star² * ustar * ustar
    end
end
function get_surface_variance(
    flux,
    ustar::FT,
    zLL::FT,
    oblength::FT,
    local_geometry_data,
) where {FT}
    c_star = -flux / ustar
    if oblength < 0
        return 4 *
               _norm_sqr(c_star, local_geometry_data) *
               (1 - FT(8.3) * zLL / oblength)^(-FT(2 / 3))
    else
        return 4 * _norm_sqr(c_star, local_geometry_data)
    end
end

function gradient_Richardson_number(
    param_set,
    ∂b∂z::FT,
    Shear²::FT,
    ϵ::FT,
) where {FT}
    Ri_c = TCP.Ri_crit(param_set)
    return min(∂b∂z / max(Shear², ϵ), Ri_c)
end

# Turbulent Prandtl number given the obukhov length sign and the gradient Richardson number
function turbulent_Prandtl_number(
    param_set,
    obukhov_length::FT,
    ∇Ri::FT,
) where {FT}
    ω_pr = TCP.Prandtl_number_scale(param_set)
    Pr_n = TCP.Prandtl_number_0(param_set)
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
