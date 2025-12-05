import NVTX
import StaticArrays as SA
import ClimaCore.RecursiveApply: rzero, ⊞, ⊠

"""
    Helper function to populate the cloud diagnostics named tuple
"""
function make_cloud_fraction_named_tuple(t1, t2, t3)
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
    compute_gradients_if_needed!(p, turbconv_model, ts)

Computes vertical gradients if they are not already computed by the turbulence model.
"""
function compute_gradients_if_needed!(p, turbconv_model, ts)
    if isnothing(turbconv_model) &&
       p.atmos.call_cloud_diagnostics_per_stage isa CallCloudDiagnosticsPerStage
        (; ᶜgradᵥ_θ_virt, ᶜgradᵥ_q_tot, ᶜgradᵥ_θ_liq_ice) = p.precomputed
        thermo_params = CAP.thermodynamics_params(p.params)
        @. ᶜgradᵥ_θ_virt =
            ᶜgradᵥ(ᶠinterp(TD.virtual_pottemp(thermo_params, ts)))
        @. ᶜgradᵥ_q_tot =
            ᶜgradᵥ(ᶠinterp(TD.total_specific_humidity(thermo_params, ts)))
        @. ᶜgradᵥ_θ_liq_ice =
            ᶜgradᵥ(ᶠinterp(TD.liquid_ice_pottemp(thermo_params, ts)))
    end
end

"""
    compute_quadrature_diagnostics(Y, p, SG_quad, ts, mixing_length_field)

Computes cloud diagnostics using Gaussian quadrature.
"""
function compute_quadrature_diagnostics(
    Y,
    p,
    SG_quad,
    ts,
    mixing_length_field,
)
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    coeff = CAP.diagnostic_covariance_coeff(params)

    ᶜ∇q = Geometry.WVector(p.precomputed.ᶜgradᵥ_q_tot)
    ᶜ∇θ = Geometry.WVector(p.precomputed.ᶜgradᵥ_θ_liq_ice)

    p_c = TD.air_pressure(thermo_params, ts)
    q_mean = TD.total_specific_humidity(thermo_params, ts)
    θ_mean = TD.liquid_ice_pottemp(thermo_params, ts)

    # Returns the physical values based on quadrature sampling points
    # and limited covarainces
    function get_x_hat(χ1, χ2)

        FT = eltype(χ1)

        q′q′ = covariance_from_grad(coeff, mixing_length_field, ᶜ∇q, ᶜ∇q)
        θ′θ′ = covariance_from_grad(coeff, mixing_length_field, ᶜ∇θ, ᶜ∇θ)
        θ′q′ = covariance_from_grad(coeff, mixing_length_field, ᶜ∇θ, ᶜ∇q)

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
    add_updraft_contributions!(cloud_diagnostics_tuple, Y, p, turbconv_model)

Adds the contribution of updrafts to the cloud diagnostics.
"""
function add_updraft_contributions!(
    cloud_diagnostics_tuple,
    Y,
    p,
    turbconv_model,
)
    n = n_mass_flux_subdomains(turbconv_model)
    if n > 0
        thermo_params = CAP.thermodynamics_params(p.params)
        (; ᶜρaʲs, ᶜρʲs, ᶜtsʲs) = p.precomputed

        for j in 1:n
            # For PrognosticEDMFX, ᶜρaʲs is not in precomputed, but we can access it from Y
            # However, the caller (set_cloud_fraction!) might have put it in precomputed or we need to handle it.
            # Let's check how it was done before.
            # In PrognosticEDMFX: draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
            # In DiagnosticEDMFX: draft_area(ᶜρaʲs.:($$j), ᶜρʲs.:($$j))
            #
            # To unify, we can check the model type or just pass the correct area field.
            # But `add_updraft_contributions!` is generic.
            # Let's use a helper to get the area.

            ᶜρaʲ = if turbconv_model isa PrognosticEDMFX
                Y.c.sgsʲs.:($j).ρa
            else
                ᶜρaʲs.:($j)
            end

            @. cloud_diagnostics_tuple += NamedTuple{(:cf, :q_liq, :q_ice)}(
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
   Compute the grid scale cloud fraction based on sub-grid scale properties

   The options are:
   - DryModel: Cloud fraction is set to 0
   - GridScaleCloud: Cloud fraction is set to 1 if condensate is present, 0 otherwise
   - QuadratureCloud: Cloud fraction is computed based on quadrature points
   - SGSQuadratureCloud: Cloud fraction is computed as an area-weighted sum of environment
     quadrature points and updraft contributions.
"""
NVTX.@annotate function set_cloud_fraction!(Y, p, ::DryModel, _, _)
    FT = eltype(p.params)
    p.precomputed.cloud_diagnostics_tuple .=
        ((; cf = FT(0), q_liq = FT(0), q_ice = FT(0)),)
end
NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    moist_model::Union{EquilMoistModel, NonEquilMoistModel},
    ::GridScaleCloud,
    _,
)
    (; ᶜts, cloud_diagnostics_tuple) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)

    compute_gradients_if_needed!(p, p.atmos.turbconv_model, ᶜts)

    if moist_model isa EquilMoistModel
        @. cloud_diagnostics_tuple = make_cloud_fraction_named_tuple(
            ifelse(TD.has_condensate(thermo_params, ᶜts), 1, 0),
            TD.PhasePartition(thermo_params, ᶜts).liq,
            TD.PhasePartition(thermo_params, ᶜts).ice,
        )
    else
        q_liq = @. lazy(specific(Y.c.ρq_liq, Y.c.ρ))
        q_ice = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))
        @. cloud_diagnostics_tuple =
            make_cloud_fraction_named_tuple(
                ifelse(q_liq + q_ice > 0, 1, 0), q_liq, q_ice
            )
    end
end
NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::Union{EquilMoistModel, NonEquilMoistModel},
    qc::QuadratureCloud,
    _,
)
    (; ᶜts, cloud_diagnostics_tuple) = p.precomputed

    compute_gradients_if_needed!(p, p.atmos.turbconv_model, ᶜts)

    # TODO - tmp fix for the Coupler test. To be removed soon.
    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field .= compute_gm_mixing_length(Y, p)

    @. cloud_diagnostics_tuple = compute_quadrature_diagnostics(
        Y,
        p,
        qc.SG_quad,
        ᶜts,
        ᶜmixing_length_field,
    )
end
NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::Union{EquilMoistModel, NonEquilMoistModel},
    qc::SGSQuadratureCloud,
    turbconv_model::Union{EDOnlyEDMFX, DiagnosticEDMFX},
)
    (; ᶜts, cloud_diagnostics_tuple) = p.precomputed

    # TODO - tmp fix for the Coupler test. To be removed soon.
    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field .= ᶜmixing_length(Y, p)

    @. cloud_diagnostics_tuple = compute_quadrature_diagnostics(
        Y,
        p,
        qc.SG_quad,
        ᶜts,
        ᶜmixing_length_field,
    )
    add_updraft_contributions!(cloud_diagnostics_tuple, Y, p, turbconv_model)
end
NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::Union{EquilMoistModel, NonEquilMoistModel},
    qc::SGSQuadratureCloud,
    turbconv_model::PrognosticEDMFX,
)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜts⁰, cloud_diagnostics_tuple) = p.precomputed

    # TODO - tmp fix for the Coupler test. To be removed soon.
    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field .= ᶜmixing_length(Y, p)

    @. cloud_diagnostics_tuple = compute_quadrature_diagnostics(
        Y,
        p,
        qc.SG_quad,
        ᶜts⁰,
        ᶜmixing_length_field,
    )

    # weight cloud diagnostics by environmental area
    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
    @. cloud_diagnostics_tuple *= NamedTuple{(:cf, :q_liq, :q_ice)}(
        tuple(
            draft_area(ᶜρa⁰, TD.air_density(thermo_params, ᶜts⁰)),
            draft_area(ᶜρa⁰, TD.air_density(thermo_params, ᶜts⁰)),
            draft_area(ᶜρa⁰, TD.air_density(thermo_params, ᶜts⁰)),
        ),
    )

    add_updraft_contributions!(cloud_diagnostics_tuple, Y, p, turbconv_model)
end
