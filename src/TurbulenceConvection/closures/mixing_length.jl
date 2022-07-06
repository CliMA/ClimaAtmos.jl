function mixing_length(mix_len_params, param_set, ml_model::MinDisspLen{FT}) where {FT}
    c_m = mix_len_params.c_m
    c_d = mix_len_params.c_d
    smin_ub = mix_len_params.smin_ub
    smin_rm = mix_len_params.smin_rm
    l_max = mix_len_params.l_max
    c_b = mix_len_params.c_b
    g::FT = TCP.grav(param_set)
    molmass_ratio::FT = TCP.molmass_ratio(param_set)
    vkc::FT = TCP.von_karman_const(param_set)
    ustar = ml_model.ustar
    z = ml_model.z
    tke_surf = ml_model.tke_surf
    ∂b∂z = ml_model.∇b.∂b∂z
    tke = ml_model.tke

    # kz scale (surface layer)
    if ml_model.obukhov_length < 0.0 #unstable
        l_W =
            vkc * z / (sqrt(tke_surf / ustar / ustar) * c_m) *
            min((1 - 100 * z / ml_model.obukhov_length)^FT(0.2), 1 / vkc)
    else # neutral or stable
        l_W = vkc * z / (sqrt(tke_surf / ustar / ustar) * c_m)
    end

    # compute l_TKE - the production-dissipation balanced length scale
    a_pd = c_m * (ml_model.Shear² - ∂b∂z / ml_model.Pr) * sqrt(tke)
    # Dissipation term
    c_neg = c_d * tke * sqrt(tke)
    # Subdomain exchange term
    b_exch = ml_model.b_exch

    if abs(a_pd) > eps(FT) && 4 * a_pd * c_neg > -b_exch * b_exch
        l_TKE = max(-b_exch / 2 / a_pd + sqrt(b_exch * b_exch + 4 * a_pd * c_neg) / 2 / a_pd, 0)
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
    min_len, min_len_ind = findmin(l)
    mix_len = lamb_smooth_minimum(l, smin_ub, smin_rm)
    ml_ratio = mix_len / min_len
    return MixLen(min_len_ind, mix_len, ml_ratio)
end
