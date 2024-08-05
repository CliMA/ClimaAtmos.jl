#=
References:
https://www.cosmo-model.org/content/model/documentation/newsLetters/newsLetter09/cnl9-04.pdf
http://dx.doi.org/10.1175/1520-0493(2003)131%3C1229:NCOMTI%3E2.0.CO;2
https://atmos.uw.edu/academics/classes/2010Q1/536/1503AP_lee_waves.pdf
=#
import StaticArrays: @SVector

background_N(::Type{FT}) where {FT} = FT(0.01)
background_T_sfc(::Type{FT}) where {FT} = FT(288)
background_u(::Type{FT}) where {FT} = FT(10)

function background_p_and_T(params, ζ)
    FT = eltype(params)
    g = CAP.grav(params)
    R_d = CAP.R_d(params)
    cp_d = CAP.cp_d(params)
    p_sfc = CAP.MSLP(params)
    N = background_N(FT)
    T_sfc = background_T_sfc(FT)

    g == 0 && return (p_sfc, T_sfc)

    β = N^2 / g
    a = g / (cp_d * T_sfc * β)
    p = p_sfc * (1 - a + a * exp(-β * ζ))^(cp_d / R_d)
    T = T_sfc * ((1 - a) * exp(β * ζ) + a)

    # Replace the constant-N profile with an isothermal profile above T = T_iso
    # to avoid unreasonably small or large values of T.
    T_min = FT(100)
    T_max = FT(500)
    T_iso = a > 1 ? T_min : T_max
    ζ_iso = log((T_iso / T_sfc - a) / (1 - a)) / β
    p_iso = p_sfc * ((1 - a) / (1 - a * T_sfc / T_iso))^(cp_d / R_d)
    if ζ > ζ_iso
        p = p_iso * exp(-g / (R_d * T_iso) * (ζ - ζ_iso))
        T = T_iso
    end

    return (p, T)
end

# Replace ζ with z in the background thermodynamic state so that the initial
# condition is hydrostatically balanced.
initial_thermo_state(params, coord) = TD.PhaseDry_pT(
    CAP.thermodynamics_params(params),
    background_p_and_T(params, coord.z)...,
)

function approximate_FΔuvw(params, k_x, k_y, ζ, Fh)
    FT = eltype(params)
    g = CAP.grav(params)
    R_d = CAP.R_d(params)
    cp_d = CAP.cp_d(params)
    p_sfc = CAP.MSLP(params)
    N = background_N(FT)
    T_sfc = background_T_sfc(FT)
    u = background_u(FT)
    (; z_top) = CAP.topography_params(params)

    if g == 0
        Fh == 0 ||
            error("the analytic solution for topography without gravity has \
                   not been implemented yet") # TODO: compute limits for g → 0
        return @SVector(FT[0, 0, 0])
    end

    β = N^2 / g
    a = g / (cp_d * T_sfc * β)
    γ = cp_d / (cp_d - R_d)
    α = g / (γ * R_d * T_sfc * β)
    r = k_y / k_x
    k_h² = k_x^2 + k_y^2
    FΔζ_xh = -im * k_x * Fh # ≈ F(-∂h/∂x / (1 - h/z_top)) to first order in h
    FΔζ_yh = -im * k_y * Fh # ≈ F(-∂h/∂y / (1 - h/z_top)) to first order in h
    FΔζ_z = Fh / z_top # ≈ F(h/z_top / (1 - h/z_top)) to first order in h

    (p, T) = background_p_and_T(params, ζ)

    ρ_sfc = p_sfc / (R_d * T_sfc)
    m_sfc = u^2 / (γ * R_d * T_sfc)
    μ_sfc = 1 - k_x^2 / k_h² * m_sfc
    ν_sfc = k_x^2 / k_h² * m_sfc / μ_sfc
    d_sfc = ρ_sfc / μ_sfc
    FΔζ_x_sfc = FΔζ_xh
    FΔζ_y_sfc = FΔζ_yh

    ρ = p / (R_d * T)
    m = u^2 / (γ * R_d * T)
    μ = 1 - k_x^2 / k_h² * m
    ν = k_x^2 / k_h² * m / μ
    d = ρ / μ
    FΔζ_x = FΔζ_xh * (1 - ζ / z_top)
    FΔζ_y = FΔζ_yh * (1 - ζ / z_top)

    k_ζ_sfc² =
        k_h² * (-μ_sfc + (N / (k_x * u))^2 * (1 + ν_sfc * (1 - a))) -
        β^2 / 4 * (
            (1 + α)^2 +
            2 / μ_sfc * α * (1 - a) +
            ν_sfc * (1 + 3 / μ_sfc) * (1 - a)^2
        )
    Δf_sfc =
        g / u * (
            im * k_h² / k_x * μ_sfc * FΔζ_z +
            ((1 - m_sfc) * FΔζ_xh + r * FΔζ_yh) / z_top +
            β * r * (-r * FΔζ_x_sfc + FΔζ_y_sfc) * (1 + ν_sfc * (1 - a))
        )

    k_ζ² =
        k_h² * (-μ + (N / (k_x * u))^2 * (1 + ν * (1 - a))) -
        β^2 / 4 *
        ((1 + α)^2 + 2 / μ * α * (1 - a) + ν * (1 + 3 / μ) * (1 - a)^2)
    Δf =
        g / u * (
            im * k_h² / k_x * μ * FΔζ_z +
            ((1 - m) * FΔζ_xh + r * FΔζ_yh) / z_top +
            β * r * (-r * FΔζ_x + FΔζ_y) * (1 + ν * (1 - a))
        )

    plus_or_minus = k_ζ² > 0 ? sign(k_x * u) : 1

    # Approximate FΔw and ∂FΔw_∂ζ to zeroth order in β.
    FΔw =
        Δf / k_ζ² +
        (sqrt(d_sfc / d) * (im * k_x * u * Fh - Δf_sfc / k_ζ_sfc²)) *
        exp(plus_or_minus * im * sqrt(Complex(k_ζ²)) * ζ)
    ∂FΔw_∂ζ = plus_or_minus * im * sqrt(Complex(k_ζ²)) * (FΔw - Δf / k_ζ²)

    FΔp =
        -(im * k_x / k_h² * g * d) *
        ((1 - m) * FΔζ_x + r * FΔζ_y - m / u * FΔw + u / g * ∂FΔw_∂ζ)
    FΔu = -(im / k_x * g * ρ * FΔζ_x + FΔp) / (ρ * u)
    FΔv = -(im / k_x * g * ρ * FΔζ_y + r * FΔp) / (ρ * u)

    return @SVector([FΔu, FΔv, FΔw])
end

# Integrate f(x) from x1 to x2 with n_calls function calls. Note that QuadGK is
# not GPU-compatible, so it should not be used here.
function definite_integral(f::F, x1, x2, n_calls) where {F}
    dx = (x2 - x1) / n_calls
    return sum(index -> f(x1 + index * dx), 2:n_calls; init = f(x1 + dx)) * dx
end

function predicted_velocity_mountain_2d(
    params,
    coord,
    x_center,
    k_x_max,
    mountain_topography::F1,
    centered_mountain_topography_transform::F2,
) where {F1, F2}
    FT = eltype(params)
    u = background_u(FT)
    (; z_top) = CAP.topography_params(params)
    (; x, z) = coord

    h = mountain_topography(params, coord)
    ζ = (z - h) / (1 - h / z_top)

    # Compute the inverse Fourier transform of FΔuvw. Since the integral is
    # symmetric, we can just evaluate it over positive values of k_x and
    # multiply the result by 2. The integral should go to k_x = Inf, but we can
    # assume that it is negligibly small when k_x > k_x_max.
    (Δu, Δv, Δw) = 2 * definite_integral(FT(0), k_x_max, 50000) do k_x
        # Use the transform of the centered topography, h(x - x_center),
        # instead of the actual topography, h(x), to avoid propagating the
        # quantity exp(-im * k_x * x_center) through approximate_FΔuvw, as
        # this can introduce numerical errors.
        Fh = centered_mountain_topography_transform(k_x)
        FΔuvw = approximate_FΔuvw(params, k_x, 0, ζ, Fh)
        real.(FΔuvw * exp(im * k_x * (x - x_center)))
    end

    return UVW(u + Δu, Δv, Δw)
end

################################################################################

agnesi_topography_params(::Type{FT}) where {FT} =
    (; a = FT(5e3), x_center = FT(50e3))
schar_topography_params(::Type{FT}) where {FT} =
    (; λ = FT(4e3), a = FT(5e3), x_center = FT(50e3))
cosine_topography_params(::Type{FT}) where {FT} = (; λ = FT(100e3))

################################################################################

function topography_agnesi(params, coord)
    FT = eltype(params)
    (; h_max) = CAP.topography_params(params)
    (; a, x_center) = agnesi_topography_params(FT)
    (; x) = coord

    return h_max / (1 + ((x - x_center) / a)^2)
end

function predicted_velocity_agnesi(params, coord)
    FT = eltype(params)
    (; h_max) = CAP.topography_params(params)
    (; a, x_center) = agnesi_topography_params(FT)

    centered_topography_transform_agnesi(k_x) =
        h_max * a / 2 * exp(-a * abs(k_x))

    n_efolding_intervals = -log(eps(FT))
    k_x_max = n_efolding_intervals / a
    # centered_topography_transform_agnesi(k_x) < eps(FT) when k_x > k_x_max

    return predicted_velocity_mountain_2d(
        params,
        coord,
        x_center,
        k_x_max,
        topography_agnesi,
        centered_topography_transform_agnesi,
    )
end

################################################################################

function topography_schar(params, coord)
    FT = eltype(params)
    (; h_max) = CAP.topography_params(params)
    (; λ, a, x_center) = schar_topography_params(FT)
    (; x) = coord

    return h_max * exp(-(x - x_center)^2 / a^2) * cospi((x - x_center) / λ)^2
end

function predicted_velocity_schar(params, coord)
    FT = eltype(params)
    (; h_max) = CAP.topography_params(params)
    (; λ, a, x_center) = schar_topography_params(FT)

    k_peak = 2 * FT(π) / λ
    centered_topography_transform_schar(k_x::FT) where {FT} =
        h_max * a / (8 * sqrt(FT(π))) * (
            exp(-a^2 / 4 * (k_x + k_peak)^2) +
            2 * exp(-a^2 / 4 * k_x^2) +
            exp(-a^2 / 4 * (k_x - k_peak)^2)
        ) # FT needs to be redefined for type stability

    n_efolding_intervals = -log(eps(FT))
    k_x_max = k_peak + 2 * sqrt(n_efolding_intervals) / a
    # centered_topography_transform_schar(k_x) < eps(FT) when k_x > k_x_max

    return predicted_velocity_mountain_2d(
        params,
        coord,
        x_center,
        k_x_max,
        topography_schar,
        centered_topography_transform_schar,
    )
end

################################################################################

function topography_cosine_2d(params, coord)
    FT = eltype(params)
    (; h_max) = CAP.topography_params(params)
    (; λ) = cosine_topography_params(FT)
    (; x) = coord

    return h_max * cospi(2 * x / λ)
end

function predicted_velocity_cosine_2d(params, coord)
    FT = eltype(params)
    u = background_u(FT)
    (; h_max, z_top) = CAP.topography_params(params)
    (; λ) = cosine_topography_params(FT)
    (; x, z) = coord

    h = topography_cosine_2d(params, coord)
    ζ = (z - h) / (1 - h / z_top)

    # Instead of using Fh = h_max / 2 * (δ(k_x + k_peak) + δ(k_x - k_peak)) and
    # integrating FΔuvw * exp(im * k_x * x) over all values of k_x, we can drop
    # the delta functions and just evaluate the integrand at k_x = ±k_peak.
    k_peak = 2 * FT(π) / λ
    Fh_δ_coef = h_max / 2
    FΔuvw_δ_coef = approximate_FΔuvw(params, k_peak, 0, ζ, Fh_δ_coef)
    (Δu, Δv, Δw) = 2 * real.(FΔuvw_δ_coef * exp(im * k_peak * x))

    return UVW(u + Δu, Δv, Δw)
end

################################################################################

function topography_cosine_3d(params, coord)
    FT = eltype(params)
    (; h_max) = CAP.topography_params(params)
    (; λ) = cosine_topography_params(FT)
    (; x, y) = coord

    return h_max * cospi(2 * x / λ) * cospi(2 * y / λ)
end

function predicted_velocity_cosine_3d(params, coord)
    FT = eltype(params)
    u = background_u(FT)
    (; h_max, z_top) = CAP.topography_params(params)
    (; λ) = cosine_topography_params(FT)
    (; x, y, z) = coord

    h = topography_cosine_3d(params, coord)
    ζ = (z - h) / (1 - h / z_top)

    # Drop all delta functions and evaluate the integrand at k_x, k_y = ±k_peak.
    k_peak = 2 * FT(π) / λ
    Fh_δ_coef = h_max / 4
    FΔuvw_δ_coef = approximate_FΔuvw(params, k_peak, k_peak, ζ, Fh_δ_coef)
    (Δu, Δv, Δw) = 4 * real.(FΔuvw_δ_coef * exp(im * k_peak * (x + y)))

    return UVW(u + Δu, Δv, Δw)
end
