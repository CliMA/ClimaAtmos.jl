import StaticArrays: @SVector

background_u(::Type{FT}) where {FT} = FT(10)
background_N(::Type{FT}) where {FT} = FT(0.01) # This needs to be a small value.
background_T_sfc(::Type{FT}) where {FT} = FT(288)
background_T_min(::Type{FT}) where {FT} = FT(100)
background_T_max(::Type{FT}) where {FT} = FT(500)

function background_p_and_T(params, η)
    FT = eltype(params)
    g = CAP.grav(params)
    R_d = CAP.R_d(params)
    cp_d = CAP.cp_d(params)
    p_sfc = CAP.MSLP(params)
    N = background_N(FT)
    T_sfc = background_T_sfc(FT)
    T_min = background_T_min(FT)
    T_max = background_T_max(FT)

    g == 0 && return (p_sfc, T_sfc)

    β = N^2 / g
    a = g / (cp_d * T_sfc * β)
    p = p_sfc * (1 - a + a * exp(-β * η))^(cp_d / R_d)
    T = T_sfc * ((1 - a) * exp(β * η) + a)

    # Replace the constant-N profile with an isothermal profile above T = T_iso
    # to avoid unreasonably small or large values of T.
    T_iso = a > 1 ? T_min : T_max
    η_iso = log((T_iso / T_sfc - a) / (1 - a)) / β
    p_iso = p_sfc * ((1 - a) / (1 - a * T_sfc / T_iso))^(cp_d / R_d)
    if η > η_iso
        p = p_iso * exp(-g / (R_d * T_iso) * (η - η_iso))
        T = T_iso
    end

    return (p, T)
end

# Replace η with z in the initial state so that it is hydrostatically balanced.
function constant_buoyancy_frequency_initial_state(params, coord)
    FT = eltype(params)
    thermo_state = TD.PhaseDry_pT(
        CAP.thermodynamics_params(params),
        background_p_and_T(params, coord.z)...,
    )
    velocity = Geometry.UVector(background_u(FT))
    return (; thermo_state, velocity)
end

"""
    ∂FΔU_∂Fh_approximation(params, η, k_x, k_y, z_top)

Approximates the derivative `∂FΔU/∂Fh`, where `FΔU` is the Fourier transform of
the 3D velocity perturbation `ΔU`, and `Fh` is the Fourier transform of the
topography elevation `h`. This approximation is first-order in the maximum
topography elevation `h_max` and zeroth-order in the static stability paramter
`β = N^2 / g`. We assume that the topographic grid warping is a `LinearAdaption`
and the perturbation's background state is a `ConstantBuoyancyFrequencyProfile`.

TODO: Generalize this to work with any `HypsographyAdaption` from ClimaCore.

Arguments:
  - `params`: `ClimaAtmosParameters` used to define the background state
  - `η`: nondimensional terrain-following coordinate (0 at surface and 1 at top)
  - `k_x`: wavenumber along `x`-direction
  - `k_y`: wavenumber along `y`-direction
  - `z_top`: elevation at the top of the model domain

References:
  - https://www.cosmo-model.org/content/model/documentation/newsLetters/newsLetter09/cnl9-04.pdf
  - http://dx.doi.org/10.1175/1520-0493(2003)131%3C1229:NCOMTI%3E2.0.CO;2
  - https://atmos.uw.edu/academics/classes/2010Q1/536/1503AP_lee_waves.pdf
"""
function ∂FΔU_∂Fh_approximation(params, η, k_x, k_y, z_top)
    FT = eltype(params)
    g = CAP.grav(params)
    R_d = CAP.R_d(params)
    cp_d = CAP.cp_d(params)
    p_sfc = CAP.MSLP(params)
    N = background_N(FT)
    T_sfc = background_T_sfc(FT)
    u = background_u(FT)

    g == 0 && return @SVector(FT[0, 0, 0])
    # TODO: This is only correct when h_max is 0. We need to find the limit of
    # ∂FΔU/∂Fh as g approaches 0.

    β = N^2 / g
    a = g / (cp_d * T_sfc * β)
    γ = cp_d / (cp_d - R_d)
    α = g / (γ * R_d * T_sfc * β)
    r = k_y / k_x
    k_h² = k_x^2 + k_y^2
    ∂FΔη_xh_∂Fh = -im * k_x # ∂F(-∂h/∂x / (1 - h/z_top))/∂Fh to 1st order in h
    ∂FΔη_yh_∂Fh = -im * k_y # ∂F(-∂h/∂y / (1 - h/z_top))/∂Fh to 1st order in h
    ∂FΔη_z_∂Fh = 1 / z_top # ∂F(h/z_top / (1 - h/z_top))/∂Fh to 1st order in h

    (p, T) = background_p_and_T(params, η)

    ρ_sfc = p_sfc / (R_d * T_sfc)
    m_sfc = u^2 / (γ * R_d * T_sfc)
    μ_sfc = 1 - k_x^2 / k_h² * m_sfc
    ν_sfc = k_x^2 / k_h² * m_sfc / μ_sfc
    d_sfc = ρ_sfc / μ_sfc
    ∂FΔη_x_sfc_∂Fh = ∂FΔη_xh_∂Fh
    ∂FΔη_y_sfc_∂Fh = ∂FΔη_yh_∂Fh

    ρ = p / (R_d * T)
    m = u^2 / (γ * R_d * T)
    μ = 1 - k_x^2 / k_h² * m
    ν = k_x^2 / k_h² * m / μ
    d = ρ / μ
    ∂FΔη_x_∂Fh = ∂FΔη_xh_∂Fh * (1 - η / z_top)
    ∂FΔη_y_∂Fh = ∂FΔη_yh_∂Fh * (1 - η / z_top)

    k_η_sfc² =
        k_h² * (-μ_sfc + (N / (k_x * u))^2 * (1 + ν_sfc * (1 - a))) -
        β^2 / 4 * (
            (1 + α)^2 +
            2 / μ_sfc * α * (1 - a) +
            ν_sfc * (1 + 3 / μ_sfc) * (1 - a)^2
        )
    ∂Δf_sfc_∂Fh =
        g / u * (
            im * k_h² / k_x * μ_sfc * ∂FΔη_z_∂Fh +
            ((1 - m_sfc) * ∂FΔη_xh_∂Fh + r * ∂FΔη_yh_∂Fh) * ∂FΔη_z_∂Fh +
            β * r * (-r * ∂FΔη_x_sfc_∂Fh + ∂FΔη_y_sfc_∂Fh) * (1 + ν_sfc * (1 - a))
        )

    k_η² =
        k_h² * (-μ + (N / (k_x * u))^2 * (1 + ν * (1 - a))) -
        β^2 / 4 *
        ((1 + α)^2 + 2 / μ * α * (1 - a) + ν * (1 + 3 / μ) * (1 - a)^2)
    ∂Δf_∂Fh =
        g / u * (
            im * k_h² / k_x * μ * ∂FΔη_z_∂Fh +
            ((1 - m) * ∂FΔη_xh_∂Fh + r * ∂FΔη_yh_∂Fh) * ∂FΔη_z_∂Fh +
            β * r * (-r * ∂FΔη_x_∂Fh + ∂FΔη_y_∂Fh) * (1 + ν * (1 - a))
        )

    plus_or_minus = k_η² > 0 ? sign(k_x * u) : 1

    # Approximate the derivatives of FΔw and ∂FΔw_∂η to zeroth order in β.
    ∂FΔw_∂Fh =
        ∂Δf_∂Fh / k_η² +
        (sqrt(d_sfc / d) * (im * k_x * u - ∂Δf_sfc_∂Fh / k_η_sfc²)) *
        exp(plus_or_minus * im * sqrt(Complex(k_η²)) * η)
    ∂²FΔw_∂Fh∂η = plus_or_minus * im * sqrt(Complex(k_η²)) * (∂FΔw_∂Fh - ∂Δf_∂Fh / k_η²)

    ∂FΔp_∂Fh =
        -(im * k_x / k_h² * g * d) * (
            (1 - m) * ∂FΔη_x_∂Fh + r * ∂FΔη_y_∂Fh - m / u * ∂FΔw_∂Fh +
            u / g * ∂²FΔw_∂Fh∂η
        )
    ∂FΔu_∂Fh = -(im / k_x * g * ρ * ∂FΔη_x_∂Fh + ∂FΔp_∂Fh) / (ρ * u)
    ∂FΔv_∂Fh = -(im / k_x * g * ρ * ∂FΔη_y_∂Fh + r * ∂FΔp_∂Fh) / (ρ * u)

    return @SVector([∂FΔu_∂Fh, ∂FΔv_∂Fh, ∂FΔw_∂Fh])
end

##
## Steady-state solutions for periodic topography
##

function steady_state_velocity_no_warp(params, coord, z_top)
    FT = eltype(params)
    u = background_u(FT)
    return UVW(u, FT(0), FT(0))
end

function steady_state_velocity_cosine_2d(params, coord, z_top)
    FT = eltype(params)
    (; λ) = cosine_params(FT)
    (; x, z) = coord
    return steady_state_velocity_cosine(params, x, FT(0), z, λ, FT(Inf), z_top)
end

function steady_state_velocity_cosine_3d(params, coord, z_top)
    FT = eltype(params)
    (; λ) = cosine_params(FT)
    (; x, y, z) = coord
    return steady_state_velocity_cosine(params, x, y, z, λ, λ, z_top)
end

function steady_state_velocity_cosine(params, x, y, z, λ_x, λ_y, z_top)
    FT = eltype(params)
    (; h_max) = cosine_params(FT)
    u = background_u(FT)
    h = topography_cosine(x, y, λ_x, λ_y, h_max)
    η = (z - h) / (1 - h / z_top)
    k_x = 2 * FT(π) / λ_x
    k_y = 2 * FT(π) / λ_y

    # Compute ΔU as the inverse Fourier transform of the approximation of FΔU.
    # Instead of integrating FΔU * exp(im * (k_x * x + k_y * y)) over all values
    # of k_x and k_y, with the Fourier transformed elevation Fh(k_x′, k_y′) =
    # h_max (δ(k_x′ + k_x) + δ(k_x′ - k_x)) (δ(k_y′ + k_y) + δ(k_y′ - k_y)) / 4,
    # we can drop the delta functions and evaluate the integrand at (±k_x, ±k_y)
    # using Fh = h_max / 4. Since the integrand is symmetric around the origin,
    # we can also just evaluate it at (k_x, k_y) and multiply the result by 4.
    FΔU = ∂FΔU_∂Fh_approximation(params, η, k_x, k_y, z_top) * h_max
    (Δu, Δv, Δw) = real.(FΔU * exp(im * (k_x * x + k_y * y)))
    return UVW(u + Δu, Δv, Δw)
end

##
## Steady-state solutions for mountain topography
##

# Integrate f(x) from x1 to x2 by sampling its values n_samples times. We do not
# use QuadGK for this because it is not compatible with GPUs.
function definite_integral(f::F, x1, x2, n_samples) where {F}
    dx = (x2 - x1) / n_samples
    return sum(index -> f(x1 + index * dx), 2:n_samples; init = f(x1 + dx)) * dx
end

function steady_state_velocity_mountain_2d(
    mountain_elevation::F1,
    centered_mountain_elevation_fourier_transform::F2,
    params,
    coord,
    z_top,
    x_center,
    k_x_max,
) where {F1, F2}
    FT = eltype(params)
    u = background_u(FT)
    h = mountain_elevation(coord)
    (; x, z) = coord
    η = (z - h) / (1 - h / z_top)
    k_y = FT(0)

    # Compute ΔU as the inverse Fourier transform of the approximation of FΔU.
    # Since the integrand FΔU * exp(im * k_x * (x - x_center)) is symmetric
    # around the origin, we can just evaluate it over positive values of k_x and
    # multiply the result by 2. The integral should go to k_x = Inf, but we can
    # assume that FΔU is negligible when k_x > k_x_max because Fh < eps(FT).
    (Δu, Δv, Δw) =
        2 * definite_integral(FT(0), k_x_max, 1_000_000) do k_x
            # Use the transform of the centered elevation h(x - x_center) rather
            # than the actual elevation h(x) to avoid propagating the quantity
            # exp(-im * k_x * x_center) through the approximation of FΔU. When
            # x_center >> 1, this quantity introduces large numerical errors.
            Fh = centered_mountain_elevation_fourier_transform(k_x)
            FΔU = ∂FΔU_∂Fh_approximation(params, η, k_x, k_y, z_top) * Fh
            real.(FΔU * exp(im * k_x * (x - x_center)))
        end
    return UVW(u + Δu, Δv, Δw)
end

function steady_state_velocity_agnesi(params, coord, z_top)
    FT = eltype(params)
    (; h_max, x_center, a) = agnesi_params(FT)
    topography_agnesi_Fh(k_x) = h_max * a / 2 * exp(-a * abs(k_x))
    n_efolding_intervals = -log(eps(FT))
    k_x_max = n_efolding_intervals / a
    return steady_state_velocity_mountain_2d(
        topography_agnesi, topography_agnesi_Fh,
        params,
        coord, z_top, x_center, k_x_max,
    )
end

function steady_state_velocity_schar(params, coord, z_top)
    FT = eltype(params)
    (; h_max, x_center, λ, a) = schar_params(FT)
    k_peak = 2 * FT(π) / λ
    Fh_coef = h_max * a / (8 * sqrt(FT(π)))
    topography_schar_Fh(k_x) =
        Fh_coef * (
            exp(-a^2 / 4 * (k_x + k_peak)^2) +
            2 * exp(-a^2 / 4 * k_x^2) +
            exp(-a^2 / 4 * (k_x - k_peak)^2)
        )
    n_efolding_intervals = -log(eps(FT))
    k_x_max = k_peak + 2 * sqrt(n_efolding_intervals) / a
    return steady_state_velocity_mountain_2d(
        topography_schar, topography_schar_Fh,
        params,
        coord, z_top, x_center, k_x_max,
    )
end
