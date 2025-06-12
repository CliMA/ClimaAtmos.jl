import NVTX
import StaticArrays as SA
import ClimaCore.RecursiveApply: rzero, ⊞, ⊠

"""
    Helper function to populate the cloud diagnostics named tuple
"""
function make_named_tuple(t1, t2, t3)
    return NamedTuple{(:cf, :q_liq, :q_ice)}(tuple(t1, t2, t3))
end

# TODO: write a test with scalars that are linear with z
"""
    Diagnose horizontal covariances based on vertical gradients
    (i.e. taking turbulence production as the only term)
"""
function covariance_from_grad(coeff, mixing_length, ∇Φ, ∇Ψ)
    return 2 * coeff * mixing_length^2 * dot(∇Φ, ∇Ψ)
end

"""
   Compute the grid scale cloud fraction based on sub-grid scale properties
"""
NVTX.@annotate function set_cloud_fraction!(Y, p, ::DryModel, _)
    (; turbconv_model) = p.atmos
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
    (; params) = p
    (; turbconv_model) = p.atmos
    (; ᶜts, cloud_diagnostics_tuple) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)

    if isnothing(turbconv_model)
        if p.atmos.call_cloud_diagnostics_per_stage isa
           CallCloudDiagnosticsPerStage
            (; ᶜgradᵥ_θ_virt, ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice) = p.precomputed
            thermo_params = CAP.thermodynamics_params(p.params)
            @. ᶜgradᵥ_θ_virt =
                ᶜgradᵥ(ᶠinterp(TD.virtual_pottemp(thermo_params, ᶜts)))
            @. ᶜgradᵥ_q_tot =
                ᶜgradᵥ(ᶠinterp(TD.total_specific_humidity(thermo_params, ᶜts)))
            @. ᶜgradᵥ_θ_liq_ice =
                ᶜgradᵥ(ᶠinterp(TD.liquid_ice_pottemp(thermo_params, ᶜts)))
        end
    end
    if moist_model isa EquilMoistModel
        @. cloud_diagnostics_tuple = make_named_tuple(
            ifelse(TD.has_condensate(thermo_params, ᶜts), 1, 0),
            TD.PhasePartition(thermo_params, ᶜts).liq,
            TD.PhasePartition(thermo_params, ᶜts).ice,
        )
    else
        q_liq = @. lazy(specific(Y.c.ρq_liq, Y.c.ρ))
        q_ice = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))
        @. cloud_diagnostics_tuple =
            make_named_tuple(ifelse(q_liq + q_ice > 0, 1, 0), q_liq, q_ice)
    end
end

NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::Union{EquilMoistModel, NonEquilMoistModel},
    qc::QuadratureCloud,
)
    SG_quad = qc.SG_quad
    (; params) = p

    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜts, cloud_diagnostics_tuple) = p.precomputed
    (; turbconv_model) = p.atmos

    if isnothing(turbconv_model)
        if p.atmos.call_cloud_diagnostics_per_stage isa
           CallCloudDiagnosticsPerStage
            (; ᶜgradᵥ_θ_virt, ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice) = p.precomputed
            thermo_params = CAP.thermodynamics_params(p.params)
            @. ᶜgradᵥ_θ_virt =
                ᶜgradᵥ(ᶠinterp(TD.virtual_pottemp(thermo_params, ᶜts)))
            @. ᶜgradᵥ_q_tot =
                ᶜgradᵥ(ᶠinterp(TD.total_specific_humidity(thermo_params, ᶜts)))
            @. ᶜgradᵥ_θ_liq_ice =
                ᶜgradᵥ(ᶠinterp(TD.liquid_ice_pottemp(thermo_params, ᶜts)))
        end
    end

    # TODO - tmp fix for the Coupler test. To be removed soon.
    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field .= compute_gm_mixing_length(Y, p)

    diagnostic_covariance_coeff = CAP.diagnostic_covariance_coeff(params)
    # TODO: This is using the grid-mean gradients
    @. cloud_diagnostics_tuple = quad_loop(
        SG_quad,
        ᶜts,
        Geometry.WVector(p.precomputed.ᶜgradᵥ_q_tot),
        Geometry.WVector(p.precomputed.ᶜgradᵥ_θ_liq_ice),
        diagnostic_covariance_coeff,
        ᶜmixing_length_field,
        thermo_params,
    )
end

NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    moisture_model::Union{EquilMoistModel, NonEquilMoistModel},
    cloud_model::SGSQuadratureCloud,
)

    (; turbconv_model) = p.atmos
    set_cloud_fraction!(Y, p, moisture_model, cloud_model, turbconv_model)
end


NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::Union{EquilMoistModel, NonEquilMoistModel},
    qc::SGSQuadratureCloud,
    ::Union{EDOnlyEDMFX, DiagnosticEDMFX},
)
    SG_quad = qc.SG_quad
    (; params) = p

    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜts, cloud_diagnostics_tuple) = p.precomputed
    (; turbconv_model) = p.atmos

    # TODO - we should make this default when using diagnostic edmf
    # environment
    diagnostic_covariance_coeff = CAP.diagnostic_covariance_coeff(params)

    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field .= ᶜmixing_length(Y, p)

    @. cloud_diagnostics_tuple = quad_loop(
        SG_quad,
        ᶜts,
        Geometry.WVector(p.precomputed.ᶜgradᵥ_q_tot),
        Geometry.WVector(p.precomputed.ᶜgradᵥ_θ_liq_ice),
        diagnostic_covariance_coeff,
        ᶜmixing_length_field,
        thermo_params,
    )

    # updrafts
    n = n_mass_flux_subdomains(turbconv_model)
    if n > 0
        (; ᶜρaʲs, ᶜρʲs, ᶜtsʲs) = p.precomputed
    end

    for j in 1:n
        @. cloud_diagnostics_tuple += NamedTuple{(:cf, :q_liq, :q_ice)}(
            tuple(
                ifelse(
                    TD.has_condensate(thermo_params, ᶜtsʲs.:($$j)),
                    draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j)),
                    0,
                ),
                draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j)) *
                TD.PhasePartition(thermo_params, ᶜtsʲs.:($$j)).liq,
                draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j)) *
                TD.PhasePartition(thermo_params, ᶜtsʲs.:($$j)).ice,
            ),
        )
    end

end

NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::Union{EquilMoistModel, NonEquilMoistModel},
    qc::SGSQuadratureCloud,
    ::PrognosticEDMFX,
)
    SG_quad = qc.SG_quad
    (; params) = p

    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜts⁰, cloud_diagnostics_tuple) = p.precomputed
    (; ᶜρʲs, ᶜtsʲs) = p.precomputed
    (; turbconv_model) = p.atmos
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))

    # TODO - we should make this default when using diagnostic edmf
    # environment
    diagnostic_covariance_coeff = CAP.diagnostic_covariance_coeff(params)

    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field .= ᶜmixing_length(Y, p)

    @. cloud_diagnostics_tuple = quad_loop(
        SG_quad,
        ᶜts⁰,
        Geometry.WVector(p.precomputed.ᶜgradᵥ_q_tot),
        Geometry.WVector(p.precomputed.ᶜgradᵥ_θ_liq_ice),
        diagnostic_covariance_coeff,
        ᶜmixing_length_field,
        thermo_params,
    )

    # weight cloud diagnostics by environmental area
    @. cloud_diagnostics_tuple *= NamedTuple{(:cf, :q_liq, :q_ice)}(
        tuple(
            draft_area(ᶜρa⁰, TD.air_density(thermo_params, ᶜts⁰)),
            draft_area(ᶜρa⁰, TD.air_density(thermo_params, ᶜts⁰)),
            draft_area(ᶜρa⁰, TD.air_density(thermo_params, ᶜts⁰)),
        ),
    )
    # updrafts
    n = n_mass_flux_subdomains(turbconv_model)

    for j in 1:n
        @. cloud_diagnostics_tuple += NamedTuple{(:cf, :q_liq, :q_ice)}(
            tuple(
                ifelse(
                    TD.has_condensate(thermo_params, ᶜtsʲs.:($$j)),
                    draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)),
                    0,
                ),
                draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)) *
                TD.PhasePartition(thermo_params, ᶜtsʲs.:($$j)).liq,
                draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)) *
                TD.PhasePartition(thermo_params, ᶜtsʲs.:($$j)).ice,
            ),
        )
    end
end

"""
    function quad_loop(SG_quad, ts, ᶜ∇q, ᶜ∇θ,
                       coeff, ᶜlength_scale, thermo_params)

where:
  - SG_quad is a struct containing information about Gaussian quadrature order,
    sampling point values and weights
  - ts is the thermodynamic state
  - ᶜ∇q, ᶜ∇θ are the gradients of q_tot and liquid ice potential temperature
  - coeff - a free parameter (to be moved into params)
  - ᶜlength_scale - mixing length for simulations with EDMF and Smagorinsky
                    length scale for simulations without EDMF
  - thermo params - thermodynamics parameters

The function imposes additional limits on the quadrature points and
returns a tuple with cloud fraction, cloud liquid and cloud ice
computed as a sum over quadrature points.
"""
function quad_loop(
    SG_quad::SGSQuadrature,
    ts,
    ᶜ∇q,
    ᶜ∇θ,
    coeff,
    ᶜlength_scale,
    thermo_params,
)
    p_c = TD.air_pressure(thermo_params, ts)
    q_mean = TD.total_specific_humidity(thermo_params, ts)
    θ_mean = TD.liquid_ice_pottemp(thermo_params, ts)
    # Returns the physical values based on quadrature sampling points
    # and limited covarainces
    function get_x_hat(χ1, χ2)

        FT = eltype(χ1)

        q′q′ = covariance_from_grad(coeff, ᶜlength_scale, ᶜ∇q, ᶜ∇q)
        θ′θ′ = covariance_from_grad(coeff, ᶜlength_scale, ᶜ∇θ, ᶜ∇θ)
        θ′q′ = covariance_from_grad(coeff, ᶜlength_scale, ᶜ∇θ, ᶜ∇q)

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
        FT = eltype(x1_hat)
        _ts = thermo_state(thermo_params; p = p_c, θ = x1_hat, q_tot = x2_hat)
        hc = TD.has_condensate(thermo_params, _ts)

        cf = hc ? FT(1) : FT(0) # cloud fraction
        q_tot_sat = hc ? x2_hat : FT(0) # cloudy/dry for buoyancy in TKE
        q_liq = TD.PhasePartition(thermo_params, _ts).liq # cloud liquid for radiation
        q_ice = TD.PhasePartition(thermo_params, _ts).ice # cloud ice for radiation

        return (; cf, q_liq, q_ice)
    end
    return quad(f, get_x_hat, SG_quad)
end

"""
   Compute f(θ, q) as a sum over inner and outer quadrature points
   that approximate the sub-grid scale variability of θ and q.

   θ - liquid ice potential temperature
   q - total water specific humidity
"""
function quad(f::F, get_x_hat::F1, quad) where {F <: Function, F1 <: Function}
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
    Diagnose horizontal covariances based on vertical gradients
    (i.e. taking turbulence production as the only term)
    We'll learn the weighting coefficients in the ML closure from data
    so no need to add the diagnostic_covariance_coeff here
"""
function unweighted_covariance_from_grad(mixing_length, ∇Φ, ∇Ψ)
    return 2 * mixing_length^2 * dot(∇Φ, ∇Ψ)
end

"""
    set_cloud_fraction! for machine learning based cloud fraction closure
"""
NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    moist_model::Union{EquilMoistModel, NonEquilMoistModel},
    ::CloudML,
)
    (; params) = p
    (; ᶜts, cloud_diagnostics_tuple) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)
    FT = eltype(p.params)

    # mixing length
    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field .= ᶜmixing_length(Y, p)

    # z
    z = p.scratch.ᶜtemp_scalar_2
    z .= Fields.coordinate_field(Y.c.ρ).z

    # t
    t = p.scratch.ᶜtemp_scalar_3
    t .= TD.air_temperature.(
        CAP.thermodynamics_params(p.params),
        p.precomputed.ᶜts,
    )

    # specific humidity
    q = p.scratch.ᶜtemp_scalar_4
    @. q = Y.c.ρq_tot / Y.c.ρ

    # buoyancy_freq
    N² = p.scratch.ᶜtemp_scalar_5
    ᶜθ = TD.dry_pottemp.(CAP.thermodynamics_params(p.params), p.precomputed.ᶜts) # TODO allocation
    ᶜ∇θ = Geometry.WVector.(ᶜgradᵥ.(ᶠinterp.(ᶜθ))).components.data.:1 # TODO allocation
    N² .= CAP.grav(params) ./ ᶜθ .* ᶜ∇θ

    #TODO strain rate
    # ᶠu = p.scratch.ᶠtemp_C123
    # @. ᶠu = C123(ᶠinterp(Y.c.uₕ)) + C123(p.precomputed.ᶠu³)
    # ᶜstrain_rate = compute_strain_rate_center_vertical(ᶠu)


    # q_lengthscale
    q_lengthscale = TD.q_vap_saturation.(thermo_params, ᶜts) .- q

    function dqsatdt(thermo_params, ts)
        ts_hi = deepcopy(ts)
        ts_lo = deepcopy(ts)
        ts_hi.T .+= FT(1)
        ts_lo.T .-= FT(1)
        return (TD.q_vap_saturation.(thermo_params, ts_hi) .-
                TD.q_vap_saturation.(thermo_params, ts_lo)) ./ FT(2)
    end

    # theta lengthscale
    T_lengthscale = ᶜθ ./ t .*(TD.q_vap_saturation.(thermo_params, ᶜts) .- q) ./
                    dqsatdt(thermo_params, ᶜts)


    ᶜtke = specific.(Y.c.sgs⁰.ρatke, Y.c.ρ)

    # quantities to be used in the regression
    ᶜ∇q = Geometry.WVector.(p.precomputed.ᶜgradᵥ_q_tot⁰) # TODO computed on the fly now
    ᶜ∇θ = Geometry.WVector.(p.precomputed.ᶜgradᵥ_θ_liq_ice⁰) # TODO computed on the fly now

    q′q′ = unweighted_covariance_from_grad.(ᶜmixing_length, ᶜ∇q, ᶜ∇q)
    θ′θ′ = unweighted_covariance_from_grad.(ᶜmixing_length, ᶜ∇θ, ᶜ∇θ)
    θ′q′ = unweighted_covariance_from_grad.(ᶜmixing_length, ᶜ∇θ, ᶜ∇q)

    # get saturation specific humidity, ice and liquid water content
    q_liq = TD.PhasePartition.(thermo_params, ᶜts).liq
    q_ice = TD.PhasePartition.(thermo_params, ᶜts).ice
    q_sat = TD.q_vap_saturation.(thermo_params, ᶜts)

    # TODO: Add ML model parameters to atmos model
    # TODO: Add ML model inference here
    # For now, return zeros as placeholder
    p.precomputed.cloud_diagnostics_tuple .=
        ((; cf = FT(0), q_liq = q_liq, q_ice = q_ice),)
end
