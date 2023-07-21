import StaticArrays as SA

function get_cloud_fraction(thermo_params, env_thermo_quad, ᶜp, ᶜts)
    q_tot = @. TD.total_specific_humidity(thermo_params, ᶜts)
    FT = eltype(thermo_params)
    θ_liq_ice = @. TD.liquid_ice_pottemp(thermo_params, ᶜts)
    qt′qt′ = @. (FT(0.05) * q_tot)^2
    θl′θl′ = @. FT(5)
    θl′qt′ = @. FT(0)
    return @. compute_cloud_fraction(
        env_thermo_quad,
        thermo_params,
        ᶜp,
        q_tot,
        θ_liq_ice,
        qt′qt′,
        θl′θl′,
        θl′qt′,
    )
end

function compute_cloud_fraction(
    env_thermo_quad,
    thermo_params,
    ᶜp,
    q_tot,
    θ_liq_ice,
    qt′qt′,
    θl′θl′,
    θl′qt′,
)
    vars = (;
        p_c = ᶜp,
        qt_mean = q_tot,
        θl_mean = θ_liq_ice,
        qt′qt′,
        θl′θl′,
        θl′qt′,
    )
    return quad_loop(env_thermo_quad, vars, thermo_params).cf
end

function quad_loop(env_thermo_quad::SGSQuadrature, vars, thermo_params)

    env_len = 8
    i_ql, i_qi, i_T, i_cf, i_qt_sat, i_qt_unsat, i_T_sat, i_T_unsat = 1:env_len

    quadrature_type = env_thermo_quad.quadrature_type
    quad_order = quadrature_order(env_thermo_quad)
    χ = env_thermo_quad.a
    weights = env_thermo_quad.w

    # qt - total water specific humidity
    # θl - liquid ice potential temperature
    # _mean and ′ - subdomain mean and (co)variances
    (; qt′qt′, qt_mean, θl′θl′, θl_mean, θl′qt′, p_c) = vars

    FT = eltype(qt_mean)

    inner_env = SA.MVector{env_len, FT}(undef)
    outer_env = SA.MVector{env_len, FT}(undef)

    sqpi_inv = FT(1 / sqrt(π))
    sqrt2 = FT(sqrt(2))

    # Epsilon defined per typical variable fluctuation
    eps_q = eps(FT) * max(eps(FT), qt_mean)
    eps_θ = eps(FT)

    if quadrature_type isa LogNormalQuad
        # Lognormal parameters (ν, s) from mean and variance
        ν_q = log(qt_mean^2 / max(sqrt(qt_mean^2 + qt′qt′), eps_q))
        ν_θ = log(θl_mean^2 / sqrt(θl_mean^2 + θl′θl′))
        s_q = sqrt(log(qt′qt′ / max(qt_mean, eps_q)^2 + 1))
        s_θ = sqrt(log(θl′θl′ / θl_mean^2 + 1))

        # Enforce Cauchy-Schwarz inequality, numerically stable compute
        corr = θl′qt′ / max(sqrt(qt′qt′), eps_q)
        corr = max(min(corr / max(sqrt(θl′θl′), eps_θ), 1), -1)

        # Conditionals
        s2_θq = log(
            corr * sqrt(θl′θl′ * qt′qt′) / θl_mean / max(qt_mean, eps_q) + 1,
        )
        s_c = sqrt(max(s_θ^2 - s2_θq^2 / max(s_q, eps_q)^2, 0))

    elseif quadrature_type isa GaussianQuad
        # limit σ_q to prevent negative qt_hat
        σ_q_lim = -qt_mean / (sqrt2 * χ[1])
        σ_q = min(sqrt(qt′qt′), σ_q_lim)
        σ_θ = sqrt(θl′θl′)

        # Enforce Cauchy-Schwarz inequality, numerically stable compute
        corr = θl′qt′ / max(σ_q, eps_q)
        corr = max(min(corr / max(σ_θ, eps_θ), 1), -1)

        # Conditionals
        σ_c = sqrt(max(1 - corr * corr, 0)) * σ_θ
    end

    # zero outer quadrature points
    @inbounds for idx in 1:env_len
        outer_env[idx] = 0
    end

    @inbounds for m_q in 1:quad_order
        if quadrature_type isa LogNormalQuad
            qt_hat = exp(ν_q + sqrt2 * s_q * χ[m_q])
            ν_c = ν_θ + s2_θq / max(s_q, eps_q)^2 * (log(qt_hat) - ν_q)
        elseif quadrature_type isa GaussianQuad
            qt_hat = qt_mean + sqrt2 * σ_q * χ[m_q]
            μ_c = θl_mean + sqrt2 * corr * σ_θ * χ[m_q]
        end

        # zero inner quadrature points
        inner_env .= 0

        for m_h in 1:quad_order
            if quadrature_type isa LogNormalQuad
                h_hat = exp(ν_c + sqrt2 * s_c * χ[m_h])
            elseif quadrature_type isa GaussianQuad
                h_hat = (μ_c) + sqrt2 * σ_c * χ[m_h]
            end

            # condensation
            ts = thermo_state(thermo_params; p = p_c, θ = h_hat, q_tot = qt_hat)
            q_liq_en = TD.liquid_specific_humidity(thermo_params, ts)
            q_ice_en = TD.ice_specific_humidity(thermo_params, ts)
            T = TD.air_temperature(thermo_params, ts)
            # autoconversion and accretion

            # environmental variables
            inner_env[i_ql] += q_liq_en * weights[m_h] * sqpi_inv
            inner_env[i_qi] += q_ice_en * weights[m_h] * sqpi_inv
            inner_env[i_T] += T * weights[m_h] * sqpi_inv
            # cloudy/dry categories for buoyancy in TKE
            if TD.has_condensate(q_liq_en + q_ice_en)
                inner_env[i_cf] += weights[m_h] * sqpi_inv
                inner_env[i_qt_sat] += qt_hat * weights[m_h] * sqpi_inv
                inner_env[i_T_sat] += T * weights[m_h] * sqpi_inv
            else
                inner_env[i_qt_unsat] += qt_hat * weights[m_h] * sqpi_inv
                inner_env[i_T_unsat] += T * weights[m_h] * sqpi_inv
            end
        end

        for idx in 1:env_len
            outer_env[idx] += inner_env[idx] * weights[m_q] * sqpi_inv
        end
    end
    outer_env_nt = (;
        ql = outer_env[i_ql],
        qi = outer_env[i_qi],
        T = outer_env[i_T],
        cf = outer_env[i_cf],
        qt_sat = outer_env[i_qt_sat],
        qt_unsat = outer_env[i_qt_unsat],
        T_sat = outer_env[i_T_sat],
        T_unsat = outer_env[i_T_unsat],
    )
    return outer_env_nt
end
