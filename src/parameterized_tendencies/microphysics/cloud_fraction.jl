import NVTX
import StaticArrays as SA
import ClimaCore.RecursiveApply: rzero, ⊞, ⊠

"""
    set_covariance_cache_and_cloud_fraction!(Y, p)

Update the covariance cache and cloud fraction in a way that is consistent with
the coupling between cloud fraction, buoyancy gradient, and mixing length.

The buoyancy gradient depends on the cloud fraction, while the cloud fraction
depends on the covariance cache, whose mixing length depends on the buoyancy
gradient. This circular dependency is resolved by performing two Picard
iterations on cloud fraction and then applying a guarded Aitken Δ²
acceleration,

    cₐ = c₀ - (c₁ - c₀)^2 / (c₂ - 2c₁ + c₀),

where `c₀` is the initial cloud fraction, `c₁ = f(c₀)`, and `c₂ = f(c₁)`.

The accelerated update is only applied when the first two Picard increments
change sign, since in that case the Aitken value lies between the previously
computed iterates. Otherwise, the second Picard iterate is retained.

For reproducible restart, the initial cloud fraction is first recomputed using
`GridScaleCloud()` so that the starting iterate is deterministic.

Note: Vertical gradients (`ᶜgradᵥ_q_tot`, `ᶜgradᵥ_θ_liq_ice`) are always computed
from grid-mean variables. Ideally PrognosticEDMFX would use environmental
gradients since the covariances represent sub-grid fluctuations within the
environment, but this is a current approximation.
"""
function set_covariance_cache_and_cloud_fraction!(Y, p)
    (; cloud_model, microphysics_model) = p.atmos
    (; ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice, ᶜcloud_fraction) = p.precomputed
    (; ᶜlinear_buoygrad, ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜlg = Fields.local_geometry_field(Y.c)

    # Precompute gradients
    @. ᶜgradᵥ_q_tot = ᶜgradᵥ(ᶠinterp(ᶜq_tot_safe))
    @. ᶜgradᵥ_θ_liq_ice = ᶜgradᵥ(
        ᶠinterp(
            TD.liquid_ice_pottemp(
                thermo_params,
                ᶜT,
                Y.c.ρ,
                ᶜq_tot_safe,
                ᶜq_liq_rai,
                ᶜq_ice_sno,
            ),
        ),
    )

    # The buoyancy gradient depends on cloud fraction, and cloud fraction depends
    # on the covariance cache through the mixing length. For reproducible restart,
    # first reconstruct the initial cloud fraction deterministically.
    if p.atmos.numerics.reproducible_restart isa ReproducibleRestart
        set_cloud_fraction!(Y, p, microphysics_model, GridScaleCloud())
    end

    # One Picard step: use the current cloud fraction to update buoyancy
    # gradient and covariance cache, then recompute cloud fraction.
    function picard_step!()
        @. ᶜlinear_buoygrad = buoyancy_gradients(
            BuoyGradMean(), # TODO: modify for NonEq + 1M tracers if needed
            thermo_params,
            ᶜT,
            Y.c.ρ,
            ᶜq_tot_safe,
            ᶜq_liq_rai,
            ᶜq_ice_sno,
            ᶜcloud_fraction,
            C3,
            ᶜgradᵥ_q_tot,
            ᶜgradᵥ_θ_liq_ice,
            ᶜlg,
        )

        # Cache SGS covariances (no-op for dry/0M/GridScaleCloud configs).
        # For EDMF: gradients are precomputed above.
        # For non-EDMF: gradients are computed inside set_covariance_cache!.
        set_covariance_cache!(Y, p, thermo_params)
        set_cloud_fraction!(Y, p, microphysics_model, cloud_model)
        return nothing
    end

    # Scratch storage for Picard/Aitken iterates:
    #   c0 = initial cloud fraction
    #   c1 = first Picard iterate
    #   c2 = second Picard iterate
    # ᶜtemp_scalar, ᶜtemp_scalar_2, ᶜtemp_scalar_3, ᶜtemp_scalar_5, ᶜtemp_scalar_6 might
    # change inside the functions that are called in picard_step!() and should not be used
    # here to store variables before calling picard_step!
    c0 = p.scratch.ᶜtemp_scalar_4
    c1 = p.scratch.ᶜtemp_scalar_7
    c2 = p.scratch.ᶜtemp_scalar

    # Picard iterates: c1 = f(c0), c2 = f(c1)
    @. c0 = ᶜcloud_fraction
    picard_step!()
    @. c1 = ᶜcloud_fraction
    picard_step!()
    @. c2 = ᶜcloud_fraction

    # Apply aitken Δ² acceleration for better convergence
    @. ᶜcloud_fraction = _aitken_picard_helper(c0, c1, c2)

    # Recompute buoyancy gradient and covariance cache with the final cloud fraction.
    @. ᶜlinear_buoygrad = buoyancy_gradients(
        BuoyGradMean(), # TODO: modify for NonEq + 1M tracers if needed
        thermo_params,
        ᶜT,
        Y.c.ρ,
        ᶜq_tot_safe,
        ᶜq_liq_rai,
        ᶜq_ice_sno,
        ᶜcloud_fraction,
        C3,
        ᶜgradᵥ_q_tot,
        ᶜgradᵥ_θ_liq_ice,
        ᶜlg,
    )
    set_covariance_cache!(Y, p, thermo_params)

    return nothing
end

"""
Guarded Aitken Δ² acceleration:
  c_acc = c0 - (c1 - c0)^2 / (c2 - 2c1 + c0)

Apply Aitken only when the Picard increments change sign, i.e. when the
Picard iterates oscillate around the fixed point. In that case, the
accelerated value is expected to remain between previously computed iterates.
Otherwise, retain the second Picard iterate.
"""
@inline function _aitken_picard_helper(c0, c1, c2)
    FT = typeof(c0)
    Δ1 = c1 - c0
    Δ2 = c2 - c1
    denom = c2 - 2c1 + c0
    tol = eps(FT)
    return ifelse(
        (Δ1 * Δ2 < zero(FT)) & (abs(denom) > tol),
        c0 - Δ1^2 / denom,
        c2,
    )
end

# ============================================================================
# Utility Functions
# ============================================================================


"""
    compute_∂T_∂θ!(dest, Y, p, thermo_params)

Materialize the θ→T Jacobian (∂T/∂θ_li) into `dest`.

Always uses grid-mean variables, consistent with the gradient computation
(see `set_covariance_cache!`).
"""
function compute_∂T_∂θ!(dest, Y, p, thermo_params)
    (; ᶜT) = p.precomputed
    ᶜρ = Y.c.ρ
    if p.atmos.microphysics_model isa Union{DryModel, EquilibriumMicrophysics0M}
        (; ᶜq_liq_rai, ᶜq_ice_sno, ᶜq_tot_safe) = p.precomputed
        # TODO - follow up with renaming the cached variables to liq and ice
        ᶜq_liq = ᶜq_liq_rai
        ᶜq_ice = ᶜq_ice_sno
        ᶜq_tot = ᶜq_tot_safe
    else
        # TODO - change in the next PR. Keeping this non behavior changing
        ᶜq_liq = @. lazy(specific(Y.c.ρq_lcl, Y.c.ρ)) # TODO + specific(Y.c.ρq_rai, Y.c.ρ))
        ᶜq_ice = @. lazy(specific(Y.c.ρq_icl, Y.c.ρ)) # TODO + specific(Y.c.ρq_sno, Y.c.ρ))
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    end
    ᶜθ_li = @. lazy(
        TD.liquid_ice_pottemp(thermo_params, ᶜT, ᶜρ, ᶜq_tot, ᶜq_liq, ᶜq_ice),
    )
    @. dest = ∂T_∂θ_li(
        thermo_params, ᶜT, ᶜθ_li, ᶜq_liq, ᶜq_ice, ᶜq_tot, ᶜρ,
    )
    return dest
end

"""
    set_covariance_cache!(Y, p, thermo_params)

Materializes T-based SGS covariances into cached fields for use by downstream
computations (SGS quadrature, cloud fraction). Populates `p.precomputed.(ᶜT′T′, ᶜq′q′)`.

Pipeline:
1. Compute mixing length via `compute_gm_mixing_length` or `ᶜmixing_length`
2. Materialize θ-based covariances from gradients
3. Transform θ→T using `compute_∂T_∂θ!`
"""
function set_covariance_cache!(Y, p, thermo_params)
    # Covariance fields are only allocated when microphysics needs the
    # quadrature API or QuadratureCloud/MLCloud is active.
    # No-op otherwise (e.g. EquilMoist + 0M + GridScaleCloud).
    uses_covariances =
        !isnothing(p.atmos.sgs_quadrature) ||
        p.atmos.microphysics_model isa
        Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M} ||
        p.atmos.cloud_model isa Union{QuadratureCloud, MLCloud}
    uses_covariances || return nothing

    (; ᶜT′T′, ᶜq′q′) = p.precomputed

    coeff = CAP.diagnostic_covariance_coeff(p.params)
    turbconv_model = p.atmos.turbconv_model
    (; ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice) = p.precomputed

    # NOTE: gradients must be precomputed when using compute_gm_mixing_length
    # compute_gm_mixing_length materializes into p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field =
        turbconv_model isa PrognosticEDMFX || turbconv_model isa DiagnosticEDMFX ?
        ᶜmixing_length(Y, p) :
        compute_gm_mixing_length(Y, p)

    # Compute θ-based covariances from gradients and mixing length
    cov_from_grad(C, L, ∇Φ, ∇Ψ) = 2 * C * L^2 * dot(∇Φ, ∇Ψ)

    # Materialize q′q′ into cache (same in θ and T basis)
    @. ᶜq′q′ = cov_from_grad(
        coeff,
        ᶜmixing_length_field,
        Geometry.WVector(ᶜgradᵥ_q_tot),
        Geometry.WVector(ᶜgradᵥ_q_tot),
    )
    # Materialize θ′θ′ into ᶜT′T′ temporarily
    @. ᶜT′T′ = cov_from_grad(
        coeff,
        ᶜmixing_length_field,
        Geometry.WVector(ᶜgradᵥ_θ_liq_ice),
        Geometry.WVector(ᶜgradᵥ_θ_liq_ice),
    )
    # Transform θ′θ′ → T′T′ in-place using Jacobian ∂T/∂θ
    ᶜ∂T_∂θ = p.scratch.ᶜtemp_scalar_2
    compute_∂T_∂θ!(ᶜ∂T_∂θ, Y, p, thermo_params)
    @. ᶜT′T′ = ᶜ∂T_∂θ^2 * ᶜT′T′  # θ′θ′ → T′T′
    return nothing
end


# ============================================================================
# Cloud Fraction: Sommeria-Deardorff Moment Matching
# ============================================================================

"""
    compute_cloud_fraction_sd(
        thermo_params, T, ρ, q_tot, q_liq, q_ice, T′T′, q′q′, corr_Tq,
        cf_steepness_scale, q_min, sgs_dist
    )

Compute cloud fraction using the Sommeria & Deardorff (1977) approach, but with 
moment-matching to obtain the cloud fraction given condensate specific humidities 
(rather than jointly determining condensate specific humidities and cloud fraction, 
as in the original approach).

Given grid-mean condensate (`q_liq`, `q_ice`) and the subgrid variances and
correlation of `(T, q_tot)`, the cloud fraction is determined by approximately
matching the predicted condensate to the width of the subgrid saturation deficit PDF.

# Algorithm
1. Compute phase-specific saturation specific humidities and
   Clausius-Clapeyron slopes `b = ∂q_sat/∂T = L·q_sat / (R_v·T²)`.
2. Compute the SGS variance of the saturation deficit:
   `σ_s² = σ_q² + b²·σ_T² − 2b·corr(T,q)·σ_T·σ_q`
3. Normalize grid-mean condensate by the PDF width: `Q̂ = q_cond / σ_s`.
4. Approximate the Gaussian CDF with `tanh(π/√6 · Q̂)` and match implied condensate 
   from supersaturation to obtain cloud fraction.
5. Merge liquid and ice via maximum overlap: `cf = max(cf_l, cf_i)`.
6. Enforce zero cloud fraction when no condensate exists.

With zero variance the function returns 0 when no condensate exists and
1 when condensate is present, recovering the grid-scale behaviour.

# Arguments
- `thermo_params`: Thermodynamics parameters
- `T`: Grid-mean temperature [K]
- `ρ`: Air density [kg/m³]
- `q_tot`: Grid-mean total specific humidity [kg/kg]
- `q_liq`: Grid-mean cloud liquid [kg/kg]
- `q_ice`: Grid-mean cloud ice [kg/kg]
- `T′T′`: Temperature variance [K²]
- `q′q′`: Moisture variance [(kg/kg)²]
- `corr_Tq`: Correlation coefficient corr(T', q')
- `cf_steepness_scale`: Scaling factor for the steepness of the cloud fraction transition versus condensate (default 1).
- `q_min`: Minimum specific humidity threshold [kg/kg] (from ClimaParams `specific_humidity_minimum`).
- `sgs_dist`: Assumed sub-grid scale distribution type (`GaussianSGS`, `LogNormalSGS`, or `GridMeanSGS`).

# Returns
Cloud fraction ∈ [0, 1]
"""
@inline function compute_cloud_fraction_sd(
    thermo_params,
    T,
    ρ,
    q_tot,
    q_liq,
    q_ice,
    T′T′,
    q′q′,
    corr_Tq,
    cf_steepness_scale,
    q_min,
    sgs_dist::AbstractSGSDistribution,
)
    FT = eltype(thermo_params)

    # --- 1. Thermodynamic sensitivities (Clausius-Clapeyron) ---
    qsat_l = TD.q_vap_saturation(thermo_params, T, ρ, TD.Liquid())
    qsat_i = TD.q_vap_saturation(thermo_params, T, ρ, TD.Ice())

    R_v = TD.Parameters.R_v(thermo_params)
    L_v = TD.latent_heat_vapor(thermo_params, T)
    L_s = TD.latent_heat_sublim(thermo_params, T)

    # b = ∂q_sat/∂T  (Clausius-Clapeyron slope, assuming constant pressure)
    b_l = L_v * qsat_l / (R_v * T^2)
    b_i = L_s * qsat_i / (R_v * T^2)

    # Standard deviations
    σ_q = sqrt(max(q′q′, zero(FT)))
    σ_T = sqrt(max(T′T′, zero(FT)))

    # --- 2. SGS variance of saturation deficit ---
    # σ_s² = σ_q² + b²·σ_T² − 2b·corr(T', q')·σ_T·σ_q
    sig2_l = q′q′ + b_l * b_l * T′T′ - FT(2) * b_l * corr_Tq * σ_T * σ_q
    sig2_i = q′q′ + b_i * b_i * T′T′ - FT(2) * b_i * corr_Tq * σ_T * σ_q

    # Safety floor
    sig_l = sqrt(max(sig2_l, ϵ_numerics(FT)))
    sig_i = sqrt(max(sig2_i, ϵ_numerics(FT)))

    # --- 3. Normalize condensate by PDF width ---
    Q_hat_l = q_liq / sig_l
    Q_hat_i = q_ice / sig_i

    # --- 4. Formulation-Specific Cloud Fraction Helpers ---
    cf_l =
        _cloud_fraction_helper(Q_hat_l, sig_l, q_tot, cf_steepness_scale, q_min, sgs_dist)
    cf_i =
        _cloud_fraction_helper(Q_hat_i, sig_i, q_tot, cf_steepness_scale, q_min, sgs_dist)

    # --- 5. Maximum overlap ---
    cf = max(cf_l, cf_i)

    # --- 6. No condensate → no cloud (branchless) ---
    has_cond = TD.has_condensate(thermo_params, q_liq + q_ice)
    return ifelse(has_cond, cf, zero(FT))
end

"""
    _cloud_fraction_helper(Q_hat, sig_s, q_tot, cf_steepness_scale, q_min, sgs_dist)

Compute the phase-specific cloud fraction from normalized condensate.

# Arguments
- `Q_hat`: Condensate normalized by the standard deviation of saturation deficit
- `sig_s`: Standard deviation of saturation deficit [kg/kg]
- `q_tot`: Grid-mean total specific humidity [kg/kg]
- `cf_steepness_scale`: Scaling factor for the steepness of the cloud fraction transition versus condensate
- `q_min`: Minimum specific humidity threshold [kg/kg]
- `sgs_dist`: Assumed sub-grid scale distribution type (`GaussianSGS`, `LogNormalSGS`, or `GridMeanSGS`)

# Returns
- Phase-specific cloud fraction ∈ [0, 1]
"""
@inline function _cloud_fraction_helper(
    Q_hat,
    sig_s,
    q_tot,
    cf_steepness_scale,
    q_min,
    ::GaussianSGS,
)
    FT = typeof(Q_hat)
    # Analytical CDF approximation: π/√6 matches the variance of the logistic distribution to the normal distribution
    coeff = (FT(π) / sqrt(FT(6))) * cf_steepness_scale
    return tanh(coeff * Q_hat)
end

@inline function _cloud_fraction_helper(
    Q_hat,
    sig_s,
    q_tot,
    cf_steepness_scale,
    q_min,
    ::LogNormalSGS,
)
    FT = typeof(Q_hat)
    # Coefficient of variation (protect against division by zero)
    C_v = sig_s / max(q_tot, q_min)

    # Base coefficient corresponds to Gaussian limit (Cv -> 0)
    coeff = (FT(π) / sqrt(FT(6))) * cf_steepness_scale

    # Modulate coefficient with Cv based on offline optimal fits to lognormal 
    # distribution for q_tot (RMSE < 5% for C_v in [0, 1])
    c = coeff * (FT(1) + FT(0.3) * C_v)

    return tanh(c * Q_hat)
end

@inline function _cloud_fraction_helper(
    Q_hat,
    sig_s,
    q_tot,
    cf_steepness_scale,
    q_min,
    ::GridMeanSGS,
)
    FT = typeof(Q_hat)
    return Q_hat > zero(FT) ? FT(1) : zero(FT)
end

# ============================================================================
# Cloud Fraction Dispatch Methods
# ============================================================================

"""
    set_cloud_fraction!(Y, p, microphysics_model, cloud_model)

Compute and store grid-scale cloud fraction based on sub-grid scale properties.

Dispatches on `microphysics_model` and `cloud_model`:
- `DryModel`: Cloud fraction and cloud condensate are zero.
- `GridScaleCloud`: Cloud fraction is 1 if grid-scale condensate exists, 0 otherwise.
- `QuadratureCloud`: Cloud fraction from Sommeria-Deardorff moment matching.
- `MLCloud`: Cloud fraction from neural network.

For EDMF turbulence models, updraft contributions are added to the environment values.
"""
NVTX.@annotate function set_cloud_fraction!(Y, p, ::DryModel, _)
    FT = eltype(p.params)
    p.precomputed.ᶜcloud_fraction .= FT(0)
end
NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::MoistMicrophysics,
    ::GridScaleCloud,
)
    (; ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    FT = eltype(p.params)
    @. p.precomputed.ᶜcloud_fraction =
        ifelse(TD.has_condensate(thermo_params, ᶜq_liq_rai + ᶜq_ice_sno), FT(1), FT(0))
end
NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::MoistMicrophysics,
    ::QuadratureCloud,
)
    thermo_params = CAP.thermodynamics_params(p.params)
    turbconv_model = p.atmos.turbconv_model
    microphysics_model = p.atmos.microphysics_model

    # Get environment density, temperature, and total specific humidity
    ᶜρ_env, ᶜT_mean, ᶜq_mean = _get_env_ρ_T_q(Y, p, thermo_params, turbconv_model)

    # Get condensate means (dispatches on microphysics_model)
    ᶜq_lcl, ᶜq_icl = _get_condensate_means(Y, p, turbconv_model, microphysics_model)

    # Get T-based variances from cache
    (; ᶜT′T′, ᶜq′q′) = p.precomputed

    sgs_dist =
        isnothing(p.atmos.sgs_quadrature) ? GaussianSGS() :
        p.atmos.sgs_quadrature.dist

    corr_Tq = correlation_Tq(p.params)
    cf_steepness_scale = CAP.cloud_fraction_steepness_scale(p.params)
    q_min = CAP.q_min(p.params)

    @. p.precomputed.ᶜcloud_fraction = compute_cloud_fraction_sd(
        thermo_params,
        ᶜT_mean,
        ᶜρ_env,
        ᶜq_mean,
        ᶜq_lcl,
        ᶜq_icl,
        ᶜT′T′,
        ᶜq′q′,
        corr_Tq,
        cf_steepness_scale,
        q_min,
        sgs_dist,
    )

    _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)
end

NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::MoistMicrophysics,
    qc::MLCloud,
)
    thermo_params = CAP.thermodynamics_params(p.params)
    turbconv_model = p.atmos.turbconv_model
    microphysics_model = p.atmos.microphysics_model

    # Get environment state, condensate, and covariances
    ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜθ_mean, ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′ =
        _compute_cloud_state(Y, p, thermo_params, turbconv_model, microphysics_model)

    set_ml_cloud_fraction!(
        Y,
        p,
        qc,
        thermo_params,
        turbconv_model,
        ᶜρ_env,
        ᶜT_mean,
        ᶜq_mean,
        ᶜθ_mean,
    )
    _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)
end

# ============================================================================
# Internal Helper Functions  
# ============================================================================

"""
    _get_env_ρ_T_q(Y, p, thermo_params, turbconv_model)

Get environment density, temperature, and specific humidity for cloud fraction.
Lightweight alternative to `_compute_cloud_state` when only ρ, T, and q are needed.
"""
function _get_env_ρ_T_q(Y, p, thermo_params, turbconv_model)
    (; ᶜp, ᶜT, ᶜq_tot_safe) = p.precomputed
    if turbconv_model isa PrognosticEDMFX
        (; ᶜT⁰, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed
        ᶜρ_env = @. lazy(
            TD.air_density(
                thermo_params,
                ᶜT⁰,
                ᶜp,
                ᶜq_tot_safe⁰,
                ᶜq_liq_rai⁰,
                ᶜq_ice_sno⁰,
            ),
        )
        return ᶜρ_env, ᶜT⁰, ᶜq_tot_safe⁰
    else
        return Y.c.ρ, ᶜT, ᶜq_tot_safe
    end
end

"""
    _compute_cloud_state(Y, p, thermo_params, turbconv_model, microphysics_model)

Compute environment state, condensate means, and variances for cloud fraction.

For PrognosticEDMFX, uses environment (⁰) fields; otherwise uses grid-scale fields.

# Returns
Tuple: `(ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜθ_mean, ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′)`
"""
function _compute_cloud_state(Y, p, thermo_params, turbconv_model, microphysics_model)
    (; ᶜp, ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed

    if turbconv_model isa PrognosticEDMFX
        (; ᶜT⁰, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed
        ᶜρ_env = @. lazy(
            TD.air_density(thermo_params, ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰),
        )
        ᶜT_mean = ᶜT⁰
        ᶜq_mean = ᶜq_tot_safe⁰
        ᶜθ_mean = @. lazy(
            TD.liquid_ice_pottemp(
                thermo_params,
                ᶜT⁰,
                ᶜρ_env,
                ᶜq_tot_safe⁰,
                ᶜq_liq_rai⁰,
                ᶜq_ice_sno⁰,
            ),
        )
    else
        ᶜρ_env = Y.c.ρ
        ᶜT_mean = ᶜT
        ᶜq_mean = ᶜq_tot_safe
        ᶜθ_mean = @. lazy(
            TD.liquid_ice_pottemp(
                thermo_params,
                ᶜT,
                Y.c.ρ,
                ᶜq_tot_safe,
                ᶜq_liq_rai,
                ᶜq_ice_sno,
            ),
        )
    end

    # Get condensate means
    ᶜq_lcl, ᶜq_icl = _get_condensate_means(Y, p, turbconv_model, microphysics_model)

    # Get T-based variances from cache
    (; ᶜT′T′, ᶜq′q′) = p.precomputed

    return ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜθ_mean, ᶜq_lcl, ᶜq_icl, ᶜT′T′, ᶜq′q′
end

"""
    _get_condensate_means(Y, p, turbconv_model, microphysics_model)

Dispatch condensate mean retrieval based on microphysics model.
"""
_get_condensate_means(Y, p, turbconv_model, ::EquilibriumMicrophysics0M) =
    _get_condensate_means_equil(p, turbconv_model)
_get_condensate_means(Y, p, turbconv_model, ::NonEquilibriumMicrophysics) =
    _get_condensate_means_nonequil(Y, p, turbconv_model)

"""
    _get_condensate_means_equil(p, turbconv_model)

Retrieve grid-mean cloud condensate for EquilibriumMicrophysics0M.

For PrognosticEDMFX, uses environment condensate fields (ᶜq_liq_rai⁰, ᶜq_ice_sno⁰).
Otherwise (including DiagnosticEDMFX), uses grid-scale precomputed condensate.

# Returns
Tuple: `(ᶜq_lcl_mean, ᶜq_icl_mean)` as lazy field expressions.
"""
function _get_condensate_means_equil(p, turbconv_model)
    if turbconv_model isa PrognosticEDMFX
        (; ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed
        return ᶜq_liq_rai⁰, ᶜq_ice_sno⁰
    else
        (; ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
        return ᶜq_liq_rai, ᶜq_ice_sno
    end
end

"""
    _get_condensate_means_nonequil(Y, p, turbconv_model)

Retrieve grid-mean cloud condensate for NonEquilibriumMicrophysics.

For PrognosticEDMFX, uses environment condensate fields (ᶜq_liq_rai⁰, ᶜq_ice_sno⁰).
Otherwise (including DiagnosticEDMFX), computes cloud-only condensate from prognostic variables.

# Returns
Tuple: `(ᶜq_lcl_mean, ᶜq_icl_mean)` as lazy field expressions.
"""
function _get_condensate_means_nonequil(Y, p, turbconv_model)
    if turbconv_model isa PrognosticEDMFX # TODO Shouldn't we do this for DiagnosticEDMFX too?
        (; ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed
        return ᶜq_liq_rai⁰, ᶜq_ice_sno⁰
    else
        ᶜq_lcl_mean = @. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))
        ᶜq_icl_mean = @. lazy(specific(Y.c.ρq_icl, Y.c.ρ))
        return ᶜq_lcl_mean, ᶜq_icl_mean
    end
end

"""
    _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)

Apply EDMF-specific adjustments to cloud diagnostics.

For PrognosticEDMFX:
1. Weights environment cloud diagnostics by environment area fraction
2. Adds updraft contributions weighted by their respective area fractions

For DiagnosticEDMFX:
1. Adds updraft contributions (environment area fraction assumed = 1)

Updraft cloud fraction is binary: 1 if updraft contains condensate, 0 otherwise.
"""
function _apply_edmf_cloud_weighting!(Y, p, turbconv_model, thermo_params)
    (; ᶜp) = p.precomputed

    # Weight by environment area fraction if using PrognosticEDMFX (assumed 1 otherwise)
    if turbconv_model isa PrognosticEDMFX
        ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, p.atmos.turbconv_model))
        (; ᶜT⁰, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed
        ᶜρ⁰ = @. lazy(
            TD.air_density(thermo_params, ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰),
        )
        @. p.precomputed.ᶜcloud_fraction *= draft_area(ᶜρa⁰, ᶜρ⁰)
    end

    # Add contributions from the updrafts if using EDMF
    if turbconv_model isa PrognosticEDMFX || turbconv_model isa DiagnosticEDMFX
        n = n_mass_flux_subdomains(turbconv_model)
        (; ᶜρʲs, ᶜq_liq_raiʲs, ᶜq_ice_snoʲs) = p.precomputed
        for j in 1:n
            ᶜρaʲ =
                turbconv_model isa PrognosticEDMFX ? Y.c.sgsʲs.:($j).ρa :
                p.precomputed.ᶜρaʲs.:($j)

            @. p.precomputed.ᶜcloud_fraction +=
                ifelse(
                    TD.has_condensate(
                        thermo_params,
                        ᶜq_liq_raiʲs.:($$j) + ᶜq_ice_snoʲs.:($$j),
                    ),
                    draft_area(ᶜρaʲ, ᶜρʲs.:($$j)),
                    0,
                )
        end
    end
end

# ============================================================================
# Machine Learning Cloud Fraction
# ============================================================================

"""
    set_ml_cloud_fraction!(Y, p, ml_cloud, thermo_params, turbconv_model,
                          ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜθ_mean)

Overwrite the environment cloud fraction with ML-predicted values.

The ML model uses non-dimensional π groups derived from thermodynamic state
and turbulence quantities. Only the cloud fraction is replaced by the ML
prediction; condensate is computed from the grid-mean thermodynamic state.

# Arguments
- `Y`: State vector
- `p`: Cache/parameters
- `ml_cloud`: MLCloud configuration with trained neural network
- `thermo_params`: Thermodynamics parameters
- `turbconv_model`: Turbulence-convection model type
- `ᶜρ_env`: Environment air density [kg/m³]
- `ᶜT_mean`: Mean temperature [K]
- `ᶜq_mean`: Mean total specific humidity [kg/kg]
- `ᶜθ_mean`: Mean liquid-ice potential temperature [K]
"""
function set_ml_cloud_fraction!(
    Y,
    p,
    ml_cloud::MLCloud,
    thermo_params,
    turbconv_model,
    ᶜρ_env,
    ᶜT_mean,
    ᶜq_mean,
    ᶜθ_mean,
)
    # compute_gm_mixing_length materializes into p.scratch.ᶜtemp_scalar
    ᶜmixing_length_lazy =
        turbconv_model isa PrognosticEDMFX || turbconv_model isa DiagnosticEDMFX ?
        ᶜmixing_length(Y, p) :
        compute_gm_mixing_length(Y, p)

    # Materialize mixing length into scratch field to break the lazy broadcast
    # chain. For PrognosticEDMFX, ᶜmixing_length returns a lazy broadcast
    # carrying mixing_length_lopez_gomez_2020 with its parameter structs,
    # which would exceed the 4 KiB GPU kernel parameter limit if nested.
    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_6
    ᶜmixing_length_field .= ᶜmixing_length_lazy

    # Vertical gradients of q_tot and θ_liq_ice
    ᶜ∇q = p.scratch.ᶜtemp_scalar_2
    ᶜ∇q .=
        projected_vector_data.(
            C3,
            p.precomputed.ᶜgradᵥ_q_tot,
            Fields.level(Fields.local_geometry_field(Y.c)),
        )
    ᶜ∇θ = p.scratch.ᶜtemp_scalar_3
    ᶜ∇θ .=
        projected_vector_data.(
            C3,
            p.precomputed.ᶜgradᵥ_θ_liq_ice,
            Fields.level(Fields.local_geometry_field(Y.c)),
        )

    p.precomputed.ᶜcloud_fraction .=
        compute_ml_cloud_fraction.(
            Ref(ml_cloud.model),
            ᶜmixing_length_field,
            ᶜ∇q,
            ᶜ∇θ,
            ᶜρ_env,
            ᶜT_mean,
            ᶜq_mean,
            ᶜθ_mean,
            thermo_params,
        )
end

"""
    compute_ml_cloud_fraction(nn_model, mixing_length, ∇q, ∇θ, ρ, T, q_tot, θli, thermo_params)

Compute ML-predicted cloud fraction at a single grid point using non-dimensional π groups.

The ML model was trained on four non-dimensional features:
- π₁: Saturation deficit `(q_sat - q_tot) / q_sat`
- π₂: Normalized distance to saturation `Δθ / θ_sat`
- π₃: Moisture gradient term `((dq_sat/dθ × ∇θ - ∇q) × L) / q_sat`
- π₄: Temperature gradient term `(∇θ × L) / θ_sat`

# Arguments
- `nn_model`: Trained neural network model
- `mixing_length`: Turbulent mixing length [m]
- `∇q`: Vertical gradient of total specific humidity [kg/kg/m]
- `∇θ`: Vertical gradient of liquid-ice potential temperature [K/m]
- `ρ`: Air density [kg/m³]
- `T`: Temperature [K]
- `q_tot`: Total specific humidity [kg/kg]
- `θli`: Liquid-ice potential temperature [K]
- `thermo_params`: Thermodynamics parameters

# Returns
- Cloud fraction ∈ [0, 1]
"""
function compute_ml_cloud_fraction(
    nn_model,
    mixing_length,
    ∇q,
    ∇θ,
    ρ,
    T,
    q_tot,
    θli,
    thermo_params,
)
    FT = eltype(thermo_params)

    # Finite difference step size [K] for computing ∂q_sat/∂θ
    Δθ_fd_step = FT(0.1)

    # Compute saturation using functional API
    q_sat = TD.q_vap_saturation(thermo_params, T, ρ)

    # Distance to saturation in θ-space (needed for π groups)
    Δθli, θli_sat, dqsatdθli =
        saturation_distance(q_tot, q_sat, T, ρ, θli, thermo_params, Δθ_fd_step)

    # Form non-dimensional π groups
    π_1 = (q_sat - q_tot) / q_sat
    π_2 = Δθli / θli_sat
    π_3 = ((dqsatdθli * ∇θ - ∇q) * mixing_length) / q_sat
    π_4 = (∇θ * mixing_length) / θli_sat

    return apply_cf_nn(nn_model, π_1, π_2, π_3, π_4)
end

"""
    saturation_distance(q_tot, q_sat, T, ρ, θli, thermo_params, Δθ_fd)

Compute the distance to saturation in θ-space using finite differences.

This function estimates how far the current state is from saturation
by computing a Newton step in θ_liq_ice space. Used for ML feature engineering.

# Arguments
- `q_tot`: Total specific humidity [kg/kg]
- `q_sat`: Saturation specific humidity [kg/kg]
- `T`: Temperature [K]
- `ρ`: Air density [kg/m³]
- `θli`: Liquid-ice potential temperature [K]
- `thermo_params`: Thermodynamics parameters
- `Δθ_fd`: Finite difference step size for computing ∂q_sat/∂θ [K]

# Returns
- `Δθli`: Distance to saturation in θ-space [K]
- `θli_sat`: θ value at saturation [K]
- `dq_sat_dθli`: Sensitivity of saturation humidity to θ [kg/kg/K]
"""
function saturation_distance(q_tot, q_sat, T, ρ, θli, thermo_params, Δθ_fd)
    FT = typeof(T)

    # Estimate perturbed temperature from perturbed θ
    # Using chain rule: ΔT ≈ (∂T/∂θ) × Δθ ≈ (T/θ) × Δθ (Exner factor approximation)
    ∂T_∂θ = T / max(θli, eps(FT))
    T_perturbed = T + ∂T_∂θ * Δθ_fd

    # Compute perturbed saturation using functional API
    q_sat_perturbed = TD.q_vap_saturation(thermo_params, T_perturbed, ρ)

    # Finite-difference derivative ∂q_sat / ∂θli
    dq_sat_dθli = (q_sat_perturbed - q_sat) / Δθ_fd

    # Newton step to saturation distance in θli-space
    # Avoids division by zero when derivative is very small
    Δθli = ifelse(
        abs(dq_sat_dθli) > eps(FT),
        (q_sat - q_tot) / dq_sat_dθli,
        FT(0),
    )
    θli_sat = θli + Δθli

    return Δθli, θli_sat, dq_sat_dθli
end

"""
    apply_cf_nn(model, π_1, π_2, π_3, π_4) -> FT

Apply the neural network model to compute cloud fraction from π groups.

# Arguments
- `model`: Trained neural network (callable with SVector input)
- `π_1`: Saturation deficit `(q_sat - q_tot) / q_sat`
- `π_2`: Normalized distance to saturation `Δθ / θ_sat`
- `π_3`: Moisture gradient term `((dq_sat/dθ × ∇θ - ∇q) × L) / q_sat`
- `π_4`: Temperature gradient term `(∇θ × L) / θ_sat`

# Returns
Cloud fraction clamped to [0, 1].
"""
function apply_cf_nn(model, π_1::FT, π_2::FT, π_3::FT, π_4::FT) where {FT}
    return clamp((model(SA.SVector(π_1, π_2, π_3, π_4))[]), FT(0.0), FT(1.0))
end
