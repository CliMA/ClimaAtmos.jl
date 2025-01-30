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
    (; ᶜmixing_length) = p.precomputed
    (; turbconv_model) = p.atmos
    FT = eltype(p.params)
    if isnothing(turbconv_model)
        compute_gm_mixing_length!(ᶜmixing_length, Y, p)
    end
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
    (; ᶜts, ᶜmixing_length, cloud_diagnostics_tuple) = p.precomputed
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
        compute_gm_mixing_length!(ᶜmixing_length, Y, p)
    end
    if moist_model isa EquilMoistModel
        @. cloud_diagnostics_tuple = make_named_tuple(
            ifelse(TD.has_condensate(thermo_params, ᶜts), 1, 0),
            TD.PhasePartition(thermo_params, ᶜts).liq,
            TD.PhasePartition(thermo_params, ᶜts).ice,
        )
    else
       @. cloud_diagnostics_tuple = make_named_tuple(
            ifelse(p.precomputed.ᶜspecific.q_liq + p.precomputed.ᶜspecific.q_ice > 0, 1, 0),
            p.precomputed.ᶜspecific.q_liq,
            p.precomputed.ᶜspecific.q_ice,
        )
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
    (; ᶜts, ᶜmixing_length, cloud_diagnostics_tuple) = p.precomputed
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
        compute_gm_mixing_length!(ᶜmixing_length, Y, p)
    end

    diagnostic_covariance_coeff = CAP.diagnostic_covariance_coeff(params)
    @. cloud_diagnostics_tuple = quad_loop(
        SG_quad,
        ᶜts,
        Geometry.WVector(p.precomputed.ᶜgradᵥ_q_tot),
        Geometry.WVector(p.precomputed.ᶜgradᵥ_θ_liq_ice),
        diagnostic_covariance_coeff,
        ᶜmixing_length,
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
    ::DiagnosticEDMFX,
)
    SG_quad = qc.SG_quad
    (; params) = p

    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜts, ᶜmixing_length, cloud_diagnostics_tuple) = p.precomputed
    (; ᶜρaʲs, ᶜρʲs, ᶜtsʲs) = p.precomputed
    (; turbconv_model) = p.atmos

    # TODO - we should make this default when using diagnostic edmf
    # environment
    diagnostic_covariance_coeff = CAP.diagnostic_covariance_coeff(params)

    @. cloud_diagnostics_tuple = quad_loop(
        SG_quad,
        ᶜts,
        Geometry.WVector(p.precomputed.ᶜgradᵥ_q_tot),
        Geometry.WVector(p.precomputed.ᶜgradᵥ_θ_liq_ice),
        diagnostic_covariance_coeff,
        ᶜmixing_length,
        thermo_params,
    )

    # updrafts
    n = n_mass_flux_subdomains(turbconv_model)

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
    (; ᶜts⁰, ᶜmixing_length, cloud_diagnostics_tuple) = p.precomputed
    (; ᶜρʲs, ᶜtsʲs, ᶜρa⁰, ᶜρ⁰) = p.precomputed
    (; turbconv_model) = p.atmos

    # TODO - we should make this default when using diagnostic edmf
    # environment
    diagnostic_covariance_coeff = CAP.diagnostic_covariance_coeff(params)

    @. cloud_diagnostics_tuple = quad_loop(
        SG_quad,
        ᶜts⁰,
        Geometry.WVector(p.precomputed.ᶜgradᵥ_q_tot⁰),
        Geometry.WVector(p.precomputed.ᶜgradᵥ_θ_liq_ice⁰),
        diagnostic_covariance_coeff,
        ᶜmixing_length,
        thermo_params,
    )

    # weight cloud diagnostics by environmental area
    @. cloud_diagnostics_tuple *= NamedTuple{(:cf, :q_liq, :q_ice)}(
        tuple(
            draft_area(ᶜρa⁰, ᶜρ⁰),
            draft_area(ᶜρa⁰, ᶜρ⁰),
            draft_area(ᶜρa⁰, ᶜρ⁰),
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
        # for the negative sampling points. However due to numerical erros
        # we sometimes still get small negative numers here
        return (θ_hat, max(FT(0), q_hat))
    end

    function f(x1_hat, x2_hat)
        FT = eltype(x1_hat)
        @assert(x1_hat >= FT(0))
        @assert(x2_hat >= FT(0))
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
