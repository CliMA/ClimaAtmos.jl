"""
function mixing_length(param_set, ustar_surf, zc, tke_surf, ∂b∂z, tke,
                       obukhov_length, Shear², Pr, b_exch)
where:
- `param_set`: set with model parameters
- `ustar_surf`: friction velocity
- `zc`: height
- `tke_surf`: env kinetic energy at first cell center
- `∂b∂z`: buoyancy gradient
- `tke`: env turbulent kinetic energy
- `obukhov_length`: surface Monin Obukhov length
- `Shear²`: shear term
- `Pr`: Prandtl number
- `b_exch`: subdomain echange term

Returns mixing length as a smooth minimum between
wall-constrained length scale,
production-dissipation balanced length scale and
effective static stability length scale.
"""
function mixing_length(
    param_set,
    ustar_surf::FT,
    zc::FT,
    tke_surf::FT,
    ∂b∂z::FT,
    tke::FT,
    obukhov_length::FT,
    Shear²::FT,
    Pr::FT,
    b_exch::FT,
) where {FT}

    c_m = TCP.tke_ed_coeff(param_set)
    c_d = TCP.tke_diss_coeff(param_set)
    smin_ub = TCP.smin_ub(param_set)
    smin_rm = TCP.smin_rm(param_set)
    l_max = TCP.l_max(param_set)
    c_b = TCP.static_stab_coeff(param_set)
    vkc = TCP.von_karman_const(param_set)

    # compute the l_W - the wall constraint mixing length
    # which imposes an upper limit on the size of eddies near the surface
    # kz scale (surface layer)
    if obukhov_length < 0.0 #unstable
        l_W =
            vkc * zc / (sqrt(tke_surf / ustar_surf / ustar_surf) * c_m) *
            min((1 - 100 * zc / obukhov_length)^FT(0.2), 1 / vkc)
    else # neutral or stable
        l_W = vkc * zc / (sqrt(tke_surf / ustar_surf / ustar_surf) * c_m)
    end

    # compute l_TKE - the production-dissipation balanced length scale
    a_pd = c_m * (Shear² - ∂b∂z / Pr) * sqrt(tke)
    # Dissipation term
    c_neg = c_d * tke * sqrt(tke)
    if abs(a_pd) > eps(FT) && 4 * a_pd * c_neg > -b_exch * b_exch
        l_TKE = max(
            -b_exch / 2 / a_pd +
            sqrt(b_exch * b_exch + 4 * a_pd * c_neg) / 2 / a_pd,
            0,
        )
    elseif abs(a_pd) < eps(FT) && abs(b_exch) > eps(FT)
        l_TKE = c_neg / b_exch
    else
        l_TKE = FT(0)
    end

    # compute l_N - the effective static stability length scale.
    N_eff = sqrt(max(∂b∂z, 0))
    if N_eff > 0.0
        l_N = min(sqrt(max(c_b * tke, 0)) / N_eff, l_max)
    else
        l_N = l_max
    end

    # add limiters
    l = SA.SVector(
        (l_N < eps(FT) || l_N > l_max) ? l_max : l_N,
        (l_TKE < eps(FT) || l_TKE > l_max) ? l_max : l_TKE,
        (l_W < eps(FT) || l_W > l_max) ? l_max : l_W,
    )
    # get soft minimum
    return lamb_smooth_minimum(l, smin_ub, smin_rm)
end
