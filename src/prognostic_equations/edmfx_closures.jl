#####
##### EDMF closures (nonhydrostatic pressure drag and mixing length)
#####

import StaticArrays as SA
import Thermodynamics.Parameters as TDP
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields

"""
    Return draft area given ρa and ρ
"""
function draft_area(ρa, ρ)
    return ρa / ρ
end

"""
    Return buoyancy on cell centers.
"""
function ᶜphysical_buoyancy(thermo_params, ᶜρ_ref, ᶜρ)
    # TODO - replace by ᶜgradᵥᶠΦ when we move to deep atmosphere
    g = TDP.grav(thermo_params)
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
   Return the virtual mass term of the pressure closure for updrafts [m/s2 * m]

   Inputs (everything defined on cell faces):
   - params - set with model parameters
   - ᶠbuoyʲ - covariant3 or contravariant3 updraft buoyancy
"""
function ᶠupdraft_nh_pressure_buoyancy(params, ᶠbuoyʲ)
    turbconv_params = CAP.turbconv_params(params)
    # factor multiplier for pressure buoyancy terms (effective buoyancy is (1-α_b))
    α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
    return α_b * ᶠbuoyʲ
end

"""
   Return the drag term of the pressure closure for updrafts [m/s2 * m]

   Inputs (everything defined on cell faces):
   - params - set with model parameters
   - ᶠlg - local geometry (needed to compute the norm inside a local function)
   - ᶠu3ʲ, ᶠu3⁰ - covariant3 or contravariant3 velocity for updraft and environment.
                  covariant3 velocity is used in prognostic edmf, and contravariant3
                  velocity is used in diagnostic edmf.
   - scale height - an approximation for updraft top height
"""
function ᶠupdraft_nh_pressure_drag(params, ᶠlg, ᶠu3ʲ, ᶠu3⁰, scale_height)
    turbconv_params = CAP.turbconv_params(params)
    # factor multiplier for pressure drag
    α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)
    H_up_min = CAP.min_updraft_top(turbconv_params)

    # Independence of aspect ratio hardcoded: α₂_asp_ratio² = FT(0)
    # We also used to have advection term here: α_a * w_up * div_w_up
    return α_d * (ᶠu3ʲ - ᶠu3⁰) * CC.Geometry._norm(ᶠu3ʲ - ᶠu3⁰, ᶠlg) /
           max(scale_height, H_up_min)
end

edmfx_nh_pressure_drag_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing
function edmfx_nh_pressure_drag_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)
    (; ᶠnh_pressure₃_dragʲs) = p.precomputed
    n = n_mass_flux_subdomains(turbconv_model)
    for j in 1:n
        @. Yₜ.f.sgsʲs.:($$j).u₃ -= ᶠnh_pressure₃_dragʲs.:($$j)
    end
end

"""
    lamb_smooth_minimum(l::SA.SVector{N, FT}, smoothness_param::FT, λ_floor::FT) where {N, FT}

Calculates a smooth minimum of the elements in the StaticVector `l`.

This function provides a differentiable approximation to the `minimum` function,
yielding a value slightly larger than the true minimum, weighted towards the
smallest elements. The degree of smoothness is controlled by an internally
calculated parameter `λ₀`, which depends on the input parameters
`smoothness_param` and `λ_floor`. A larger `λ₀` results in a smoother
(less sharp) minimum approximation.

This implementation is based on an exponentially weighted average, with `λ₀`
determined involving the minimum element `x_min` and a factor related to the
Lambert W function evaluated at 2/e.

Arguments:
 - `l`: An `SVector{N, FT}` of N numbers for which to find the smooth minimum.
 - `smoothness_param`: A parameter (`FT`) influencing the scaling of the smoothness
                      parameter `λ₀`. A larger value generally leads to a larger `λ₀`
                      and a smoother minimum.
 - `λ_floor`: The minimum value (`FT`) allowed for the smoothness parameter `λ₀`.
                  Ensures a minimum level of smoothing and prevents `λ₀` from
                  becoming zero or negative. Must be positive.
Returns:
 - The smooth minimum value (`FT`).

Algorithm:
 1. Find the hard minimum `x_min = minimum(l)`.
 2. Calculate the smoothness scale:
    `λ₀ = max(x_min * smoothness_param / W(2/e), λ_floor)`,
    where `W(2/e)` is the Lambert W function evaluated at 2/e.
 3. Ensure `λ₀` is positive (`>= eps(FT)`).
 4. Compute the exponentially weighted average:
    `smin = Σᵢ(lᵢ * exp(-(lᵢ - x_min) / λ₀)) / Σᵢ(exp(-(lᵢ - x_min) / λ₀))`
"""
function lamb_smooth_minimum(l, smoothness_param, λ_floor)
    FT = typeof(smoothness_param)

    # Precomputed constant value of LambertW(2/e) for efficiency.
    # LambertW.lambertw(FT(2) / FT(MathConstants.e)) ≈ 0.46305551336554884
    lambert_2_over_e = FT(0.46305551336554884)

    # Ensure the floor for the smoothness parameter is positive
    @assert λ_floor > 0 "λ_floor must be positive"

    # 1. Find the minimum value in the vector
    x_min = minimum(l)

    # 2. Calculate the smoothing parameter λ_0.
    # It scales with the minimum value and smoothness_param, bounded below by λ_floor.
    # Using a precomputed value for lambertw(2/e) for type stability and efficiency.
    lambda_scaling_term = x_min * smoothness_param / lambert_2_over_e
    λ_0 = max(lambda_scaling_term, λ_floor)

    # 3. Ensure λ_0 is numerically positive (should be guaranteed by λ_floor > 0)
    λ_0_safe = max(λ_0, eps(FT))

    # Calculate the numerator and denominator for the weighted average.
    # The exponent is -(l_i - x_min)/λ_0_safe, which is <= 0.
    numerator = sum(l_i -> l_i * exp(-(l_i - x_min) / λ_0_safe), l)
    denominator = sum(l_i -> exp(-(l_i - x_min) / λ_0_safe), l)

    # 4. Calculate the smooth minimum.
    # The denominator is guaranteed to be >= 1 because the term with l_i = x_min
    # contributes exp(0) = 1. Add a safeguard for (unlikely) underflow issues.
    return numerator / max(eps(FT), denominator)
end

function blend_scales(
    method::SmoothMinimumBlending,
    l::SA.SVector,
    turbconv_params,
)
    FT = eltype(l)
    smin_ub = CAP.smin_ub(turbconv_params)
    smin_rm = CAP.smin_rm(turbconv_params)
    l_final = lamb_smooth_minimum(l, smin_ub, smin_rm)
    return max(l_final, FT(0))
end

function blend_scales(
    method::HardMinimumBlending,
    l::SA.SVector,
    turbconv_params,
)
    FT = eltype(l)
    return max(minimum(l), FT(0))
end

"""
    mixing_length(params, ustar, ᶜz, z_sfc, ᶜdz, 
                   sfc_tke, ᶜlinear_buoygrad, ᶜtke, obukhov_length, 
                   ᶜstrain_rate_norm, ᶜPr, ᶜtke_exch, scale_blending_method)

where:
- `params`: Parameter set (e.g., CLIMAParameters.AbstractParameterSet).
- `ustar`: Friction velocity [m/s].
- `ᶜz`: Cell center height [m].
- `z_sfc`: Surface elevation [m].
- `ᶜdz`: Cell vertical thickness [m].
- `sfc_tke`: TKE near the surface (e.g., first cell center) [m^2/s^2].
- `ᶜlinear_buoygrad`: N^2, Brunt-Väisälä frequency squared [1/s^2].
- `ᶜtke`: Turbulent kinetic energy at cell center [m^2/s^2].
- `obukhov_length`: Surface Monin-Obukhov length [m].
- `ᶜstrain_rate_norm`: Frobenius norm of strain rate tensor [1/s].
- `ᶜPr`: Turbulent Prandtl number [-].
- `ᶜtke_exch`: TKE exchange term [m^2/s^3].
- `scale_blending_method`: The method to use for blending physical scales.

Calculates the turbulent mixing length, limited by physical constraints and grid resolution.
"""
function mixing_length(
    params,
    ustar,
    ᶜz,
    z_sfc,
    ᶜdz,
    sfc_tke,
    ᶜlinear_buoygrad,
    ᶜtke,
    obukhov_length,
    ᶜstrain_rate_norm,
    ᶜPr,
    ᶜtke_exch,
    scale_blending_method,
)

    FT = eltype(params)
    eps_FT = eps(FT)


    turbconv_params = CAP.turbconv_params(params)
    sf_params = CAP.surface_fluxes_params(params) # Businger params

    c_m = CAP.tke_ed_coeff(turbconv_params)
    c_d = CAP.tke_diss_coeff(turbconv_params)
    smin_ub = CAP.smin_ub(turbconv_params)
    smin_rm = CAP.smin_rm(turbconv_params)
    c_b = CAP.static_stab_coeff(turbconv_params)
    vkc = CAP.von_karman_const(params)

    # MOST stability function coefficients
    most_a_m = sf_params.ufp.a_m # Businger a_m
    most_b_m = sf_params.ufp.b_m # Businger b_m
    most_g_m = CAP.coefficient_b_m_gryanik(params)  # Gryanik b_m

    # l_z: Geometric distance from the surface
    l_z = ᶜz - z_sfc
    # Ensure l_z is non-negative when ᶜz is numerically smaller than z_sfc.
    l_z = max(l_z, FT(0))

    # l_W: Wall-constrained length scale (near-surface limit, to match 
    # Monin-Obukhov Similarity Theory in the surface layer, with Businger-Dyer 
    # type stability functions)
    tke_sfc_safe = max(sfc_tke, eps_FT)
    ustar_sq_safe = max(ustar * ustar, eps_FT) # ustar^2 may vanish in certain LES setups

    # Denominator of the base length scale (always positive):
    #     c_m * √(tke_sfc / u_*²) = c_m * √(e_sfc) / u_*
    # The value increases when u_* is small and decreases when e_sfc is small.
    l_W_denom_factor = sqrt(tke_sfc_safe / ustar_sq_safe)
    l_W_denom = max(c_m * l_W_denom_factor, eps_FT)

    # Base length scale (neutral, but adjusted for TKE level)
    # l_W_base = κ * l_z / (c_m * sqrt(e_sfc) / u_star)
    # This can be Inf if l_W_denom is eps_FT and l_z is large.
    # This can be 0 if l_z is 0.
    # The expression approaches ∞ when l_W_denom ≈ eps_FT and l_z > eps_FT,
    # and approaches 0 when l_z → 0.
    l_W_base = vkc * l_z / l_W_denom

    if obukhov_length < FT(0) # Unstable case
        obukhov_len_safe = min(obukhov_length, -eps_FT) # Ensure L < 0
        zeta = l_z / obukhov_len_safe # Stability parameter zeta = z/L (<0)

        # Calculate MOST term (1 - b_m * zeta)
        # Since zeta is negative, this term is > 1
        inner_term = 1 - most_b_m * zeta

        # Numerical safety check – by theory the value is ≥ 1.
        inner_term_safe = max(inner_term, eps_FT)

        # Unstable-regime correction factor:
        #     (1 − b_m ζ)^(1/4) = φ_m⁻¹,
        # where φ_m is the Businger stability function φ_m = (1 − b_m ζ)^(-1/4).
        stability_correction = sqrt(sqrt(inner_term_safe))
        l_W = l_W_base * stability_correction

    else # Neutral or stable case
        # Ensure L > 0 for Monin-Obukhov length
        obukhov_len_safe_stable = max(obukhov_length, eps_FT)
        zeta = l_z / obukhov_len_safe_stable # zeta >= 0

        # Stable/neutral-regime correction after Gryanik (2020):
        #     φ_m = 1 + a_m ζ / (1 + g_m ζ)^(2/3),
        # a nonlinear refinement to the Businger formulation.
        phi_m_denom_term = (1 + most_g_m * zeta)
        # Guard against a negative base in the fractional power
        # (theoretically impossible for ζ ≥ 0 and g_m > 0, retained for robustness).
        phi_m_denom_cubed_sqrt = cbrt(phi_m_denom_term)
        phi_m_denom =
            max(phi_m_denom_cubed_sqrt * phi_m_denom_cubed_sqrt, eps_FT) # (val)^(2/3)

        phi_m = 1 + (most_a_m * zeta) / phi_m_denom

        # Stable-regime correction factor: 1 / φ_m.
        # phi_m should be >= 1 for stable/neutral
        stability_correction = 1 / max(phi_m, eps_FT)

        # Apply the correction factor
        l_W = l_W_base * stability_correction
    end
    l_W = max(l_W, FT(0)) # Ensure non-negative

    # --- l_TKE: TKE production-dissipation balance scale ---
    tke_pos = max(ᶜtke, FT(0)) # Ensure TKE is not negative
    sqrt_tke_pos = sqrt(tke_pos)

    # Net production of TKE from shear and buoyancy is approximated by
    #     (S² − N²/Pr_t) · √TKE · l,
    # where S² denotes the gradient involved in shear production and
    # N²/Pr_t denotes the gradient involved in buoyancy production.
    # The factor below corresponds to that production term normalised by l.
    a_pd = c_m * (2 * ᶜstrain_rate_norm - ᶜlinear_buoygrad / ᶜPr) * sqrt_tke_pos

    # Dissipation is modelled as c_d · k^{3/2} / l.
    # For the quadratic expression below, c_neg ≡ c_d · k^{3/2}.
    c_neg = c_d * tke_pos * sqrt_tke_pos

    l_TKE = FT(0)
    # Solve for l_TKE in
    #     a_pd · l_TKE − c_neg / l_TKE + ᶜtke_exch = 0
    #  ⇒  a_pd · l_TKE² + ᶜtke_exch · l_TKE − c_neg = 0
    # yielding
    #     l_TKE = (−ᶜtke_exch ± √(ᶜtke_exch² + 4 a_pd c_neg)) / (2 a_pd).
    if abs(a_pd) > eps_FT # If net of shear and buoyancy production (a_pd) is non-zero
        discriminant = ᶜtke_exch * ᶜtke_exch + 4 * a_pd * c_neg
        if discriminant >= FT(0) # Ensure real solution exists
            # Select the physically admissible (positive) root for l_TKE.
            # When a_pd > 0 (production exceeds dissipation) the root
            #     (−ᶜtke_exch + √D) / (2 a_pd)
            # is positive.  For a_pd < 0 the opposite root is required.
            l_TKE_sol1 = (-(ᶜtke_exch) + sqrt(discriminant)) / (2 * a_pd)
            # For a_pd < 0 (local destruction exceeds production) use
            #     (−ᶜtke_exch − √D) / (2 a_pd).
            if a_pd > FT(0)
                l_TKE = l_TKE_sol1
            else # a_pd < FT(0)
                l_TKE = (-(ᶜtke_exch) - sqrt(discriminant)) / (2 * a_pd)
            end
            l_TKE = max(l_TKE, FT(0)) # Ensure it's non-negative
        end
    elseif abs(ᶜtke_exch) > eps_FT # If a_pd is zero, balance is between exchange and dissipation
        # ᶜtke_exch = c_neg / l_TKE  => l_TKE = c_neg / ᶜtke_exch
        # Ensure division is safe and result is positive
        if ᶜtke_exch > eps_FT # Assuming positive exchange means TKE sink from env perspective
            l_TKE = c_neg / ᶜtke_exch # if c_neg is positive, l_TKE is positive
        elseif ᶜtke_exch < -eps_FT # Negative exchange means TKE source for env
            # -|ᶜtke_exch| = c_neg / l_TKE. If c_neg > 0, this implies l_TKE < 0, which is unphysical.
            # This case (a_pd=0, tke_exch < 0, c_neg > 0) implies TKE source and dissipation, no production.
            # Dissipation = Source. So, c_d * k_sqrt_k / l = -tke_exch. l = c_d * k_sqrt_k / (-tke_exch)
            l_TKE = c_neg / (-(ᶜtke_exch))
        end
        l_TKE = max(l_TKE, FT(0))
    end
    # If a_pd = 0 and ᶜtke_exch = 0 (or c_neg = 0), l_TKE remains zero.

    # --- l_N: Static-stability length scale (buoyancy limit), constrained by l_z ---
    N_eff_sq = max(ᶜlinear_buoygrad, FT(0)) # Use N^2 only if stable (N^2 > 0)
    l_N = l_z # Default to wall distance if not stably stratified or TKE is zero
    if N_eff_sq > eps_FT && tke_pos > eps_FT
        N_eff = sqrt(N_eff_sq)
        # l_N ~ sqrt(c_b * TKE) / N_eff
        l_N_physical = sqrt(c_b * tke_pos) / N_eff
        # Limit by distance from wall
        l_N = min(l_N_physical, l_z)
    end
    l_N = max(l_N, FT(0)) # Ensure non-negative


    # --- Combine Scales ---

    # Vector of *physical* scales (wall, TKE, stability)
    # These scales (l_W, l_TKE, l_N) are already ensured to be non-negative.
    # l_N is already limited by l_z. l_W and l_TKE are not necessarily.
    l_physical_scales = SA.SVector(l_W, l_TKE, l_N)

    l_smin =
        blend_scales(scale_blending_method, l_physical_scales, turbconv_params)

    # 1. Limit the combined physical scale by the distance from the wall.
    #    This step mitigates excessive values of l_W or l_TKE.
    l_limited_phys_wall = min(l_smin, l_z)

    # 2. Impose the grid-scale limit
    l_final = min(l_limited_phys_wall, ᶜdz)

    # Final check: guarantee that the mixing length is at least a small positive
    # value.  This prevents division-by-zero in
    #     ε_d = C_d · TKE^{3/2} / l_mix
    # when TKE > 0.  When TKE = 0, l_mix is inconsequential, but eps_FT
    # provides a conservative lower bound.
    l_final = max(l_final, eps_FT)

    return MixingLength{FT}(l_final, l_W, l_TKE, l_N)
end

"""
    turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm)

where:
- `params`: Parameters set
- `ᶜlinear_buoygrad`: N^2, Brunt-Väisälä frequency squared [1/s^2].
- `ᶜstrain_rate_norm`: Frobenius norm of strain rate tensor, |S| [1/s].

Returns the turbulent Prandtl number based on the gradient Richardson number.

The formula implemented is from Li et al. (JAS 2015, DOI: 10.1175/JAS-D-14-0335.1, their Eq. 39),
with a reformulation and correction of an algebraic error in their expression:

    Pr_t(Ri) = (X + sqrt(max(X^2 - 4*Pr_n*Ri, 0))) / 2

where X = Pr_n + ω_pr * Ri and Ri = N^2 / max(2*|S|, eps)
using parameters Pr_n = Prandtl_number_0, ω_pr = Prandtl_number_scale.
This formula applies in both stable (Ri > 0) and unstable (Ri < 0) conditions.
The returned turbulent Prandtl number is limited by Pr_max parameter.
"""
function turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm)
    FT = eltype(params)
    turbconv_params = CAP.turbconv_params(params)
    eps_FT = eps(FT)

    # Parameters from CliMAParams
    Pr_n = CAP.Prandtl_number_0(turbconv_params) # Neutral Prandtl number
    ω_pr = CAP.Prandtl_number_scale(turbconv_params) # Prandtl number scale coefficient
    Pr_max = CAP.Pr_max(turbconv_params) # Maximum Prandtl number limit

    # Calculate the raw gradient Richardson number
    # Using the definition Ri = N^2 / (2*|S|)
    ᶜshear_term_safe = max(2 * ᶜstrain_rate_norm, eps_FT)
    ᶜRi_grad = ᶜlinear_buoygrad / ᶜshear_term_safe

    # --- Apply the Pr_t(Ri) formula valid for stable and unstable conditions ---

    # Calculate the intermediate term X = Pr_n + ω_pr * Ri
    X = Pr_n + ω_pr * ᶜRi_grad

    # Calculate the discriminant term: (Pr_n + ω_pr*Ri)^2 - 4*Pr_n*Ri = X^2 - 4*Pr_n*Ri
    discriminant = X * X - 4 * Pr_n * ᶜRi_grad
    # Ensure the discriminant is non-negative before taking the square root
    discriminant_safe = max(discriminant, FT(0))

    # Calculate the Prandtl number using the positive root solution of the quadratic eq.
    # Pr_t = ( X + sqrt(discriminant_safe) ) / 2
    prandtl_nvec = (X + sqrt(discriminant_safe)) / 2

    # Optional safety: ensure Pr_t is not excessively small or negative,
    # though the formula should typically yield positive values if Pr_n > 0.
    # Also ensure that it's not larger than the Pr_max parameter.
    return min(max(prandtl_nvec, eps_FT), Pr_max)
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
                C3(min(Y.f.sgsʲs.:($$j).u₃.components.data.:1, 0)) / float(dt)
            @. Yₜ.c.sgsʲs.:($$j).ρa -= min(Y.c.sgsʲs.:($$j).ρa, 0) / float(dt)
        end
    end
end
