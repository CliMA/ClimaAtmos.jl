import NVTX
import StaticArrays as SA
import ClimaCore.RecursiveApply: rzero, ⊞, ⊠

"""
    Helper function to populate the cloud diagnostics named tuple
"""
function make_cloud_fraction_named_tuple(t1, t2, t3)
    return NamedTuple{(:cf, :q_liq, :q_ice)}(tuple(t1, t2, t3))
end

"""
    Diagnose horizontal covariances based on vertical gradients
    (i.e. taking turbulence production as the only term)
"""
function compute_covariance(Y, p, thermo_params)
    coeff = CAP.diagnostic_covariance_coeff(p.params)
    turbconv_model = p.atmos.turbconv_model
    (; ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice) = p.precomputed

    # Compute the gradients of total water and liquid ice potential temperature
    # if they are not already provided by the turbulence model.
    if isnothing(turbconv_model)
        if p.atmos.call_cloud_diagnostics_per_stage isa
           CallCloudDiagnosticsPerStage
            (; ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
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
    # Reminder that gradients need to be precomputed when using compute_gm_mixing_length
    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field .=
        turbconv_model isa PrognosticEDMFX || turbconv_model isa DiagnosticEDMFX ?
        ᶜmixing_length(Y, p) :
        compute_gm_mixing_length(Y, p)

    # Compute covaraiance based on gradients and the mixing length
    cov_from_grad(C, L, ∇Φ, ∇Ψ) = 2 * C * L^2 * dot(∇Φ, ∇Ψ)
    ᶜq′q′ = @. lazy(
        cov_from_grad(
            coeff,
            ᶜmixing_length_field,
            Geometry.WVector(ᶜgradᵥ_q_tot),
            Geometry.WVector(ᶜgradᵥ_q_tot),
        ),
    )
    ᶜθ′θ′ = @. lazy(
        cov_from_grad(
            coeff,
            ᶜmixing_length_field,
            Geometry.WVector(ᶜgradᵥ_θ_liq_ice),
            Geometry.WVector(ᶜgradᵥ_θ_liq_ice),
        ),
    )
    ᶜθ′q′ = @. lazy(
        cov_from_grad(
            coeff,
            ᶜmixing_length_field,
            Geometry.WVector(ᶜgradᵥ_θ_liq_ice),
            Geometry.WVector(ᶜgradᵥ_q_tot),
        ),
    )

    return (ᶜq′q′, ᶜθ′θ′, ᶜθ′q′)
end

"""
   Compute f(θ, q) as a sum over inner and outer quadrature points
   that approximate the sub-grid scale variability of θ and q.

   θ - liquid ice potential temperature
   q - total water specific humidity
"""
function sum_over_quadrature_points(
    f::F,
    get_x_hat::F1,
    quad,
) where {F <: Function, F1 <: Function}
    χ = quad.a
    weights = quad.w
    quad_order = quadrature_order(quad)
    FT = eltype(χ)
    # zero outer quadrature points
    T = typeof(f(get_x_hat(χ[1], χ[1])...))
    outer_env = rzero(T)
    @inbounds for m_q in 1:quad_order
        # zero inner quadrature points
        inner_env = rzero(T)
        for m_h in 1:quad_order
            x_hat = get_x_hat(χ[m_q], χ[m_h])
            inner_env = inner_env ⊞ f(x_hat...) ⊠ weights[m_h] ⊠ FT(1 / sqrt(π))
        end
        outer_env = outer_env ⊞ inner_env ⊠ weights[m_q] ⊠ FT(1 / sqrt(π))
    end
    return outer_env
end

"""
    function compute_cloud_fraction_quadrature_diagnostics(
        thermo_params, SG_quad, p_c, q_mean, θ_mean, q′q′, θ′θ′, θ′q′
    )

where:
  - thermo params - thermodynamics parameters
  - SG_quad is a struct containing information about Gaussian quadrature order,
    sampling point values and weights
  - p_c is the air pressure
  - q_mean is the total specific humidity
  - θ_mean is the liquid-ice potential temperature
  - q′q′, θ′θ′, θ′q′ are the covariances of q_tot and liquid ice potential temperature

The function imposes additional limits on the quadrature points
to prevent negative values and enforce Cauchy-Schwarz inequality.
It returns a tuple with cloud fraction, cloud liquid and cloud ice
computed as a sum over quadrature points with weights.
"""
function compute_cloud_fraction_quadrature_diagnostics(
    thermo_params,
    SG_quad::SGSQuadrature,
    p_c,
    q_mean,
    θ_mean,
    q′q′,
    θ′θ′,
    θ′q′,
)

    # Return physical values based on quadrature points and limited covarainces
    function get_x_hat(χ1, χ2)
        FT = eltype(thermo_params)
        # Epsilon defined per typical variable fluctuation
        eps_q = eps(FT) * max(eps(FT), q_mean)
        eps_θ = eps(FT)

        # limit σ_q to prevent negative q_tot_hat
        σ_q_lim = -q_mean / (sqrt(FT(2)) * SG_quad.a[1])
        σ_q = min(sqrt(q′q′), σ_q_lim)
        # Do we also have to try to limit θ in the same way as q??
        σ_θ = sqrt(θ′θ′)

        # Enforce Cauchy-Schwarz inequality, numerically stable compute
        _corr = (θ′q′ / max(σ_q, eps_q))
        corr = max(min(_corr / max(σ_θ, eps_θ), FT(1)), FT(-1))

        # Conditionals
        σ_c = sqrt(max(1 - corr * corr, 0)) * σ_θ

        μ_c = θ_mean + sqrt(FT(2)) * corr * σ_θ * χ1
        θ_hat = μ_c + sqrt(FT(2)) * σ_c * χ2
        q_hat = q_mean + sqrt(FT(2)) * σ_q * χ1
        # The σ_q_lim limits q_tot_hat to be close to zero
        # for the negative sampling points. However due to numerical errors
        # we sometimes still get small negative numbers here
        return (θ_hat, max(FT(0), q_hat))
    end

    function f(x1_hat, x2_hat)
        FT = eltype(thermo_params)
        _ts = TD.PhaseEquil_pθq(thermo_params, p_c, x1_hat, x2_hat)
        hc = TD.has_condensate(thermo_params, _ts)

        cf = hc ? FT(1) : FT(0) # cloud fraction
        q_liq = TD.PhasePartition(thermo_params, _ts).liq # cloud liquid for radiation
        q_ice = TD.PhasePartition(thermo_params, _ts).ice # cloud ice for radiation

        return (; cf, q_liq, q_ice)
    end
    return sum_over_quadrature_points(f, get_x_hat, SG_quad)
end

"""
   Compute the grid scale cloud fraction based on sub-grid scale properties

   The options are:
   - DryModel: Cloud fraction and cloud condensate are zero.
   - GridScaleCloud: Cloud fraction is set to 1 if there is non-zero grid-scale condensate, 0 otherwise.
   - QuadratureCloud: Cloud fraction is computed by sampling over the quadrature points.
     Additional contributions from the updrafts are considered when using EDMF.
   - MLCloud: Cloud fraction is computed using a neural network for the environmental cloud fraction, while 
     cloud and liquid ice mixing ratios and EMDFX updraft contributions are computed as in `QuadratureCloud`.
"""
NVTX.@annotate function set_cloud_fraction!(Y, p, ::DryModel, _)
    FT = eltype(p.params)
    p.precomputed.cloud_diagnostics_tuple .=
        ((; cf = FT(0), q_liq = FT(0), q_ice = FT(0)),)
end
NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    moist_model::Union{EquilMoistModel, NonEquilMoistModel},
    ::GridScaleCloud,
)
    (; ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
    FT = eltype(p.params)

    if moist_model isa EquilMoistModel
        @. p.precomputed.cloud_diagnostics_tuple =
            make_cloud_fraction_named_tuple(
                ifelse(TD.has_condensate(ᶜq_liq_rai + ᶜq_ice_sno), FT(1), FT(0)),
                ᶜq_liq_rai,
                ᶜq_ice_sno,
            )
    else
        q_liq = @. lazy(specific(Y.c.ρq_liq, Y.c.ρ))
        q_ice = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))
        @. p.precomputed.cloud_diagnostics_tuple =
            make_cloud_fraction_named_tuple(
                ifelse(TD.has_condensate(ᶜq_liq_rai + ᶜq_ice_sno), FT(1), FT(0)),
                q_liq,
                q_ice,
            )
    end
end
NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::Union{EquilMoistModel, NonEquilMoistModel},
    qc::Union{QuadratureCloud, MLCloud},
)
    thermo_params = CAP.thermodynamics_params(p.params)
    turbconv_model = p.atmos.turbconv_model
    (; ᶜp, ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed

    # For PrognosticEDMFX, use environment state; otherwise use grid-scale
    if turbconv_model isa PrognosticEDMFX
        (; ᶜT⁰, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed
        ᶜρ⁰ = @. lazy(
            TD.air_density(thermo_params, ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰),
        )
        ᶜθ_mean = @. lazy(
            TD.liquid_ice_pottemp(
                thermo_params,
                ᶜT⁰,
                ᶜρ⁰,
                ᶜq_tot_safe⁰,
                ᶜq_liq_rai⁰,
                ᶜq_ice_sno⁰,
            ),
        )
    else
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

    # Compute covariance based on the gradients of q_tot and theta_liq_ice
    ᶜq′q′, ᶜθ′θ′, ᶜθ′q′ = compute_covariance(Y, p, thermo_params)

    # Compute SGS cloud fraction diagnostics based on environment quadrature points ...
    @. p.precomputed.cloud_diagnostics_tuple =
        compute_cloud_fraction_quadrature_diagnostics(
            thermo_params,
            qc.SG_quad,
            ᶜp,
            ᶜq_tot_safe,
            ᶜθ_mean,
            ᶜq′q′,
            ᶜθ′θ′,
            ᶜθ′q′,
        )

    if qc isa MLCloud
        # overwrite with the ML computed environmental cloud fraction, leaving q_liq, q_ice computed via quadrature
        set_ml_cloud_fraction!(
            Y,
            p,
            qc,
            thermo_params,
            turbconv_model,
            ᶜp,
            ᶜq_tot_safe,
            ᶜθ_mean,
        )
    end

    # ... weight by environment area fraction if using PrognosticEDMFX (assumed 1 otherwise) ...
    if turbconv_model isa PrognosticEDMFX
        ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, p.atmos.turbconv_model))
        (; ᶜT⁰, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed
        ᶜρ⁰ = @. lazy(
            TD.air_density(thermo_params, ᶜT⁰, ᶜp, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰),
        )
        @. p.precomputed.cloud_diagnostics_tuple *= NamedTuple{(:cf, :q_liq, :q_ice)}(
            tuple(
                draft_area(ᶜρa⁰, ᶜρ⁰),
                draft_area(ᶜρa⁰, ᶜρ⁰),
                draft_area(ᶜρa⁰, ᶜρ⁰),
            ),
        )
    end
    # ... and add contributions from the updrafts if using EDMF.
    if turbconv_model isa PrognosticEDMFX || turbconv_model isa DiagnosticEDMFX
        n = n_mass_flux_subdomains(turbconv_model)
        (; ᶜρʲs, ᶜtsʲs) = p.precomputed
        for j in 1:n
            ᶜρaʲ =
                turbconv_model isa PrognosticEDMFX ? Y.c.sgsʲs.:($j).ρa :
                p.precomputed.ᶜρaʲs.:($j)

            @. p.precomputed.cloud_diagnostics_tuple += NamedTuple{(:cf, :q_liq, :q_ice)}(
                tuple(
                    ifelse(
                        TD.has_condensate(thermo_params, ᶜtsʲs.:($$j)),
                        draft_area(ᶜρaʲ, ᶜρʲs.:($$j)),
                        0,
                    ),
                    draft_area(ᶜρaʲ, ᶜρʲs.:($$j)) *
                    TD.PhasePartition(thermo_params, ᶜtsʲs.:($$j)).liq,
                    draft_area(ᶜρaʲ, ᶜρʲs.:($$j)) *
                    TD.PhasePartition(thermo_params, ᶜtsʲs.:($$j)).ice,
                ),
            )
        end
    end
end

function set_ml_cloud_fraction!(
    Y,
    p,
    ml_cloud::MLCloud,
    thermo_params,
    turbconv_model,
    ᶜp,
    ᶜq_mean,
    ᶜθ_mean,
)
    FT = eltype(p.params)
    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field .=
        turbconv_model isa PrognosticEDMFX || turbconv_model isa DiagnosticEDMFX ?
        ᶜmixing_length(Y, p) :
        compute_gm_mixing_length(Y, p)

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

    p.precomputed.cloud_diagnostics_tuple.cf .=
        compute_ml_cloud_fraction.(
            Ref(ml_cloud.model),
            ᶜmixing_length_field,
            ᶜ∇q,
            ᶜ∇θ,
            ᶜq_mean,
            ᶜp,
            ᶜθ_mean,
            thermo_params,
        )
end

function compute_ml_cloud_fraction(
    nn_model,
    ᶜmixing_length_field,
    ᶜ∇q,
    ᶜ∇θ,
    q_tot,
    p_c,
    θli,
    thermo_params,
)
    FT = eltype(thermo_params)
    # Saturation state at current thermodynamic state (via θ, p, q_tot)
    ts_current = TD.PhaseEquil_pθq(thermo_params, p_c, θli, q_tot)
    q_sat = TD.q_vap_saturation(thermo_params, ts_current)

    # distance to saturation in temperature space
    Δθli, θli_sat, dqsatdθli =
        saturation_distance(q_tot, q_sat, p_c, θli, thermo_params, FT(0.1))

    # form the pi groups 
    π_1 = (q_sat - q_tot) / q_sat
    π_2 = Δθli / θli_sat
    π_3 = (((dqsatdθli * ᶜ∇θ - ᶜ∇q) * ᶜmixing_length_field) / q_sat)
    π_4 = (ᶜ∇θ * ᶜmixing_length_field) / θli_sat

    return apply_cf_nn(nn_model, π_1, π_2, π_3, π_4)
end

function saturation_distance(q_tot, q_sat, p_c, θli, thermo_params, Δθli_fd)

    # Perturbed thermodynamic states for finite-difference
    ts_perturbed = TD.PhaseEquil_pθq(
        thermo_params,
        p_c,
        θli + Δθli_fd,
        q_tot,
    )
    q_sat_perturbed = TD.q_vap_saturation(thermo_params, ts_perturbed)

    # Finite-difference derivative ∂q_sat / ∂θli
    dq_sat_dθli = (q_sat_perturbed - q_sat) / Δθli_fd

    # Newton step to saturation distance in θli-space
    Δθli = (q_sat - q_tot) / dq_sat_dθli
    θli_sat = θli + Δθli

    return Δθli, θli_sat, dq_sat_dθli
end


"""
    apply_cf_nn(model, π_1, π_2, π_3, π_4)

Compute the NN-based cloud fraction given non-dimensional groups.
"""
function apply_cf_nn(model, π_1::FT, π_2::FT, π_3::FT, π_4::FT) where {FT}
    return clamp((model(SA.SVector(π_1, π_2, π_3, π_4))[]), FT(0.0), FT(1.0))
end
