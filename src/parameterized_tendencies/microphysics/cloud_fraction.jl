import NVTX
import StaticArrays as SA
import ClimaCore.RecursiveApply: rzero, ⊞, ⊠

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
    if base_microphysics_model(p.atmos.microphysics_model) isa Union{DryModel, EquilibriumMicrophysics0M}
        (; ᶜq_liq_rai, ᶜq_ice_sno, ᶜq_tot_safe) = p.precomputed
        ᶜq_liq = ᶜq_liq_rai
        ᶜq_ice = ᶜq_ice_sno
        ᶜq_tot = ᶜq_tot_safe
    else
        ᶜq_liq = @. lazy(specific(Y.c.ρq_liq, Y.c.ρ))
        ᶜq_ice = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))
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
computations (SGS quadrature, cloud fraction).

Called once per stage in `set_explicit_precomputed_quantities!`.
Populates `p.precomputed.(ᶜT′T′, ᶜq′q′)`.

Note: Vertical gradients (ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice) are always computed
from grid-mean variables. For EDMF configurations, these gradients are computed
in `set_explicit_precomputed_quantities!` before this function is called. For
non-EDMF, they are computed here. Ideally PrognosticEDMFX would use environmental
gradients since the covariances represent sub-grid fluctuations within the
environment, but this is a current approximation.

Pipeline:
1. Compute vertical gradients (non-EDMF only; EDMF gradients are precomputed)
2. Compute mixing length via `compute_gm_mixing_length` or `ᶜmixing_length`
3. Materialize θ-based covariances from gradients
4. Transform θ→T using `compute_∂T_∂θ!`
"""
function set_covariance_cache!(Y, p, thermo_params)
    # Covariance fields are only allocated when microphysics needs the
    # quadrature API or QuadratureCloud/MLCloud is active.
    # No-op otherwise (e.g. EquilMoist + 0M + GridScaleCloud).
    uses_covariances =
        p.atmos.microphysics_model isa
        Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M, QuadratureMicrophysics} ||
        p.atmos.cloud_model isa Union{QuadratureCloud, MLCloud}
    uses_covariances || return nothing

    (; ᶜT′T′, ᶜq′q′) = p.precomputed

    coeff = CAP.diagnostic_covariance_coeff(p.params)
    turbconv_model = p.atmos.turbconv_model
    (; ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice) = p.precomputed

    # Compute gradients for non-EDMF cases (EDMF gradients are precomputed)
    if isnothing(turbconv_model)
        needs_gradients =
            p.atmos.microphysics_model isa QuadratureMicrophysics ||
            p.atmos.cloud_model isa Union{QuadratureCloud, MLCloud}
        if needs_gradients
            (; ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
            # TODO: replace by 3d gradients
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
        end
    end
    # For EDMF: gradients are precomputed in set_explicit_precomputed_quantities!

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
        thermo_params, T, ρ, q_liq, q_ice, T′T′, q′q′, corr_Tq
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
- `q_liq`: Grid-mean cloud liquid [kg/kg]
- `q_ice`: Grid-mean cloud ice [kg/kg]
- `T′T′`: Temperature variance [K²]
- `q′q′`: Moisture variance [(kg/kg)²]
- `corr_Tq`: Correlation coefficient corr(T', q')

# Returns
Cloud fraction ∈ [0, 1]
"""
@inline function compute_cloud_fraction_sd(
    thermo_params,
    T,
    ρ,
    q_liq,
    q_ice,
    T′T′,
    q′q′,
    corr_Tq,
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

    # --- 4. Analytical CDF approximation ---
    # Coefficients from comparison with Gaussian integrals
    # π/√6 matches the variance of the logistic distribution to the normal distribution
    coeff = FT(π) / sqrt(FT(6))
    cf_l = tanh(coeff * Q_hat_l)
    cf_i = tanh(coeff * Q_hat_i)

    # --- 5. Maximum overlap ---
    cf = max(cf_l, cf_i)

    # --- 6. No condensate → no cloud (branchless) ---
    has_cond = (q_liq + q_ice) > FT(0)
    return ifelse(has_cond, cf, zero(FT))
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
set_cloud_fraction!(Y, p, qm::QuadratureMicrophysics, cloud_model) =
    set_cloud_fraction!(Y, p, qm.base_model, cloud_model)

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
    FT = eltype(p.params)
    @. p.precomputed.ᶜcloud_fraction =
        ifelse(TD.has_condensate(ᶜq_liq_rai + ᶜq_ice_sno), FT(1), FT(0))
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

    # Get environment density and temperature
    ᶜρ_env, ᶜT_mean = _get_env_ρ_T(Y, p, thermo_params, turbconv_model)

    # Get condensate means (dispatches on microphysics_model)
    ᶜq_liq, ᶜq_ice = _get_condensate_means(Y, p, turbconv_model, microphysics_model)

    # Get T-based variances from cache
    (; ᶜT′T′, ᶜq′q′) = p.precomputed

    @. p.precomputed.ᶜcloud_fraction = compute_cloud_fraction_sd(
        thermo_params,
        ᶜT_mean,
        ᶜρ_env,
        ᶜq_liq,
        ᶜq_ice,
        ᶜT′T′,
        ᶜq′q′,
        correlation_Tq(p.params),
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
    ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜθ_mean, ᶜq_liq, ᶜq_ice, ᶜT′T′, ᶜq′q′ =
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
    _get_env_ρ_T(Y, p, thermo_params, turbconv_model)

Get environment density and temperature for cloud fraction.
Lightweight alternative to `_compute_cloud_state` when only ρ and T are needed.
"""
function _get_env_ρ_T(Y, p, thermo_params, turbconv_model)
    (; ᶜp, ᶜT) = p.precomputed
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
        return ᶜρ_env, ᶜT⁰
    else
        return Y.c.ρ, ᶜT
    end
end

"""
    _compute_cloud_state(Y, p, thermo_params, turbconv_model, microphysics_model)

Compute environment state, condensate means, and variances for cloud fraction.

For PrognosticEDMFX, uses environment (⁰) fields; otherwise uses grid-scale fields.

# Returns
Tuple: `(ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜθ_mean, ᶜq_liq, ᶜq_ice, ᶜT′T′, ᶜq′q′)`
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
    ᶜq_liq, ᶜq_ice = _get_condensate_means(Y, p, turbconv_model, microphysics_model)

    # Get T-based variances from cache
    (; ᶜT′T′, ᶜq′q′) = p.precomputed

    return ᶜρ_env, ᶜT_mean, ᶜq_mean, ᶜθ_mean, ᶜq_liq, ᶜq_ice, ᶜT′T′, ᶜq′q′
end

"""
    _get_condensate_means(Y, p, turbconv_model, microphysics_model)

Dispatch condensate mean retrieval based on microphysics model.
"""
_get_condensate_means(Y, p, turbconv_model, qm::QuadratureMicrophysics) =
    _get_condensate_means(Y, p, turbconv_model, qm.base_model)
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
Tuple: `(ᶜq_liq_mean, ᶜq_ice_mean)` as lazy field expressions.
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
Tuple: `(ᶜq_liq_mean, ᶜq_ice_mean)` as lazy field expressions.
"""
function _get_condensate_means_nonequil(Y, p, turbconv_model)
    if turbconv_model isa PrognosticEDMFX # TODO Shouldn't we do this for DiagnosticEDMFX too?
        (; ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed
        return ᶜq_liq_rai⁰, ᶜq_ice_sno⁰
    else
        ᶜq_liq_mean = @. lazy(specific(Y.c.ρq_liq, Y.c.ρ))
        ᶜq_ice_mean = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))
        return ᶜq_liq_mean, ᶜq_ice_mean
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
                    TD.has_condensate(ᶜq_liq_raiʲs.:($$j) + ᶜq_ice_snoʲs.:($$j)),
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
