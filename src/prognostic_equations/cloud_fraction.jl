import StaticArrays as SA

function get_cloud_fraction(thermo_params, env_thermo_quad, ᶜp, ᶜts)
    q_tot = TD.total_specific_humidity(thermo_params, ᶜts)
    FT = eltype(thermo_params)
    θ_liq_ice = TD.liquid_ice_pottemp(thermo_params, ᶜts)
    qt′qt′ = (FT(0.05) * q_tot)^2
    θl′θl′ = FT(5)
    θl′qt′ = FT(0)
    return compute_cloud_fraction(
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
    ᶜp::FT,
    q_tot,
    θ_liq_ice,
    qt′qt′,
    θl′θl′,
    θl′qt′,
) where {FT}
    vars = (;
        p_c = ᶜp,
        qt_mean = q_tot,
        θl_mean = θ_liq_ice,
        qt′qt′,
        θl′θl′,
        θl′qt′,
    )
    return quad_loop(
        env_thermo_quad,
        env_thermo_quad.quadrature_type,
        vars,
        thermo_params,
    )::FT
end

function quad_loop(
    env_thermo_quad::SGSQuadrature,
    quadrature_type::GaussianQuad,
    vars,
    thermo_params,
)

    # qt - total water specific humidity
    # θl - liquid ice potential temperature
    # _mean and ′ - subdomain mean and (co)variances
    (; qt′qt′, qt_mean, θl′θl′, θl_mean, θl′qt′, p_c) = vars

    FT = eltype(qt_mean)

    sqrt2 = FT(sqrt(2))

    # Epsilon defined per typical variable fluctuation
    eps_q = FT(eps(FT)) * max(FT(eps(FT)), qt_mean)
    eps_θ = FT(eps(FT))

    # limit σ_q to prevent negative q_tot_hat
    σ_q_lim = -qt_mean / (sqrt2 * env_thermo_quad.a[1]) # TODO: is this correct?
    σ_q::FT = min(sqrt(qt′qt′), σ_q_lim)
    σ_θ::FT = sqrt(θl′θl′)

    # Enforce Cauchy-Schwarz inequality, numerically stable compute
    _corr::FT = (θl′qt′ / max(σ_q, eps_q))
    corr::FT = max(min(_corr / max(σ_θ, eps_θ), 1), -1)

    # Conditionals
    σ_c = sqrt(max(1 - corr * corr, 0)) * σ_θ

    function get_x_hat(χ::Tuple{<:Real, <:Real})
        μ_c = θl_mean + sqrt2 * corr * σ_θ * χ[1]
        θ_hat = μ_c + sqrt2 * σ_c * χ[2]
        q_tot_hat = qt_mean + sqrt2 * σ_q * χ[1]
        return (θ_hat, q_tot_hat)
    end

    # cloudy/dry categories for buoyancy in TKE
    f_q_tot_sat(x_hat::Tuple{<:Real, <:Real}, hc) =
        hc ? x_hat[2] : eltype(x_hat)(0)

    get_ts(x_hat::Tuple{<:Real, <:Real}) =
        thermo_state(thermo_params; p = p_c, θ = x_hat[1], q_tot = x_hat[2])
    f_cf(x_hat::Tuple{<:Real, <:Real}, hc) =
        hc ? eltype(x_hat)(1) : eltype(x_hat)(0)
    function f(x_hat::Tuple{<:Real, <:Real})
        ts = get_ts(x_hat)
        hc = TD.has_condensate(thermo_params, ts)
        return (; cf = f_cf(x_hat, hc), q_tot_sat = f_q_tot_sat(x_hat, hc))
    end

    return quad(f, get_x_hat, env_thermo_quad).cf
end

import ClimaCore.RecursiveApply: rzero, ⊞, ⊠
function quad(f, get_x_hat::F, quad_type) where {F <: Function}
    χ = quad_type.a
    weights = quad_type.w
    quad_order = quadrature_order(quad_type)
    FT = eltype(χ)
    # zero outer quadrature points
    T = typeof(f(get_x_hat((χ[1], χ[1]))))
    outer_env = rzero(T)
    @inbounds for m_q in 1:quad_order
        # zero inner quadrature points
        inner_env = rzero(T)
        for m_h in 1:quad_order
            x_hat = get_x_hat((χ[m_q], χ[m_h]))
            inner_env = inner_env ⊞ f(x_hat) ⊠ weights[m_h] ⊠ FT(1 / sqrt(π))
        end
        outer_env = outer_env ⊞ inner_env ⊠ weights[m_q] ⊠ FT(1 / sqrt(π))
    end
    return outer_env
end
