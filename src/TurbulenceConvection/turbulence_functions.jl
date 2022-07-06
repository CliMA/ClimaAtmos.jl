# convective velocity scale
get_wstar(bflux, zi) = cbrt(max(bflux * zi, 0))

function buoyancy_c(param_set::APS, ρ::FT, ρ_i::FT) where {FT}
    g::FT = TCP.grav(param_set)
    return g * (ρ - ρ_i) / ρ
end

# BL height
function get_inversion(grid::Grid, state::State, param_set::APS, Ri_bulk_crit)
    FT = float_type(state)
    g::FT = TCP.grav(param_set)
    kc_surf = kc_surface(grid)
    θ_virt = center_aux_grid_mean(state).θ_virt
    u = physical_grid_mean_u(state)
    v = physical_grid_mean_v(state)
    Ri_bulk = center_aux_grid_mean(state).Ri
    θ_virt_b = θ_virt[kc_surf]
    z_c = grid.zc
    ∇c = CCO.DivergenceF2C()
    wvec = CC.Geometry.WVector

    # test if we need to look at the free convective limit
    if (u[kc_surf]^2 + v[kc_surf]^2) <= 0.01
        ∇θ_virt = center_aux_turbconv(state).ϕ_temporary
        k_star = findlast_center(k -> θ_virt[k] > θ_virt_b, grid)
        LB = CCO.LeftBiasedC2F(; bottom = CCO.SetValue(θ_virt[kc_surf]))
        @. ∇θ_virt = ∇c(wvec(LB(θ_virt)))
        h = (θ_virt_b - θ_virt[k_star - 1]) / ∇θ_virt[k_star] + z_c[k_star - 1].z
    else
        ∇Ri_bulk = center_aux_turbconv(state).ϕ_temporary
        Ri_bulk_fn(k) = g * (θ_virt[k] - θ_virt_b) * z_c[k].z / θ_virt_b / (u[k] * u[k] + v[k] * v[k])

        @inbounds for k in real_center_indices(grid)
            Ri_bulk[k] = Ri_bulk_fn(k)
        end
        k_star = findlast_center(k -> Ri_bulk_fn(k) > Ri_bulk_crit, grid)
        LB = CCO.LeftBiasedC2F(; bottom = CCO.SetValue(Ri_bulk[kc_surf]))
        @. ∇Ri_bulk = ∇c(wvec(LB(Ri_bulk)))
        h = (Ri_bulk_crit - Ri_bulk[k_star - 1]) / ∇Ri_bulk[k_star] + z_c[k_star - 1].z
    end

    return h
end
# Teixiera convective tau
function get_mixing_tau(zi::FT, wstar::FT) where {FT}
    # return 0.5 * zi / wstar
    #return zi / (max(wstar, FT(1e-5)))
    return zi / (wstar + FT(0.001))
end

# MO scaling of near surface tke and scalar variance
function get_surface_tke(mixing_length_params, ustar::FT, zLL::FT, oblength::FT) where {FT}
    κ_star² = mixing_length_params.κ_star²
    if oblength < 0
        return ((κ_star² + cbrt(zLL / oblength * zLL / oblength)) * ustar * ustar)
    else
        return κ_star² * ustar * ustar
    end
end
function get_surface_variance(flux1::FT, flux2::FT, ustar::FT, zLL::FT, oblength::FT) where {FT}
    c_star1 = -flux1 / ustar
    c_star2 = -flux2 / ustar
    if oblength < 0
        return 4 * c_star1 * c_star2 * (1 - FT(8.3) * zLL / oblength)^(-FT(2 / 3))
    else
        return 4 * c_star1 * c_star2
    end
end

function gradient_Richardson_number(mixing_length_params, ∂b∂z::FT, Shear²::FT, ϵ::FT) where {FT}
    Ri_c::FT = mixing_length_params.Ri_c
    return min(∂b∂z / max(Shear², ϵ), Ri_c)
end

# Turbulent Prandtl number given the obukhov length sign and the gradient Richardson number
function turbulent_Prandtl_number(mixing_length_params, obukhov_length::FT, ∇Ri::FT) where {FT}
    ω_pr = mixing_length_params.ω_pr
    Pr_n = mixing_length_params.Pr_n
    if obukhov_length > 0 && ∇Ri > 0 #stable
        # CSB (Dan Li, 2019, eq. 75), where ω_pr = ω_1 + 1 = 53.0/13.0
        prandtl_nvec = Pr_n * (2 * ∇Ri / (1 + ω_pr * ∇Ri - sqrt((1 + ω_pr * ∇Ri)^2 - 4 * ∇Ri)))
    else
        prandtl_nvec = Pr_n
    end
    return prandtl_nvec
end
