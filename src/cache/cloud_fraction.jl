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
        SG_quad, ts, ᶜ∇q, ᶜ∇θ, coeff, ᶜlength_scale, thermo_params
    )

where:
  - SG_quad is a struct containing information about Gaussian quadrature order,
    sampling point values and weights
  - ts is the thermodynamic state
  - ᶜ∇q, ᶜ∇θ are the gradients of q_tot and liquid ice potential temperature
  - coeff - a free parameter (to be moved into params)
  - ᶜlength_scale - mixing length for simulations with EDMF and Smagorinsky
                    length scale for simulations without EDMF
  - thermo params - thermodynamics parameters

The function imposes additional limits on the quadrature points
to prevent negative values and enforce Cauchy-Schwarz inequality.
It returns a tuple with cloud fraction, cloud liquid and cloud ice
computed as a sum over quadrature points with weights.
"""
function compute_cloud_fraction_quadrature_diagnostics(
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
    (; ᶜts) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    FT = eltype(p.params)

    if moist_model isa EquilMoistModel
        @. p.precomputed.cloud_diagnostics_tuple =
            make_cloud_fraction_named_tuple(
                ifelse(TD.has_condensate(thermo_params, ᶜts), FT(1), FT(0)),
                TD.PhasePartition(thermo_params, ᶜts).liq,
                TD.PhasePartition(thermo_params, ᶜts).ice,
            )
    else
        q_liq = @. lazy(specific(Y.c.ρq_liq, Y.c.ρ))
        q_ice = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))
        @. p.precomputed.cloud_diagnostics_tuple =
            make_cloud_fraction_named_tuple(
                ifelse(q_liq + q_ice > 0, FT(1), FT(0)),
                q_liq,
                q_ice,
            )
    end
end
NVTX.@annotate function set_cloud_fraction!(
    Y,
    p,
    ::Union{EquilMoistModel, NonEquilMoistModel},
    qc::QuadratureCloud,
)
    thermo_params = CAP.thermodynamics_params(p.params)
    diagnostic_covariance_coeff = CAP.diagnostic_covariance_coeff(p.params)
    turbconv_model = p.atmos.turbconv_model

    ᶜts = turbconv_model isa PrognosticEDMFX ? p.precomputed.ᶜts⁰ : p.precomputed.ᶜts

    # Compute the gradients of total water and liquid ice potential temperature
    # if they are not already provided by the turbulence model.
    if isnothing(turbconv_model)
        if p.atmos.call_cloud_diagnostics_per_stage isa
           CallCloudDiagnosticsPerStage
            @. p.precomputed.ᶜgradᵥ_q_tot =
                ᶜgradᵥ(ᶠinterp(TD.total_specific_humidity(thermo_params, ᶜts)))
            @. p.precomputed.ᶜgradᵥ_θ_liq_ice =
                ᶜgradᵥ(ᶠinterp(TD.liquid_ice_pottemp(thermo_params, ᶜts)))
        end
    end

    ᶜmixing_length_field = p.scratch.ᶜtemp_scalar
    ᶜmixing_length_field .=
        turbconv_model isa PrognosticEDMFX || turbconv_model isa DiagnosticEDMFX ?
        ᶜmixing_length(Y, p) :
        compute_gm_mixing_length(Y, p)

    # Compute SGS cloud fraction diagnostics based on environment quadrature points ...
    @. p.precomputed.cloud_diagnostics_tuple =
        compute_cloud_fraction_quadrature_diagnostics(
            qc.SG_quad,
            ᶜts,
            Geometry.WVector(p.precomputed.ᶜgradᵥ_q_tot),
            Geometry.WVector(p.precomputed.ᶜgradᵥ_θ_liq_ice),
            diagnostic_covariance_coeff,
            ᶜmixing_length_field,
            thermo_params,
        )
    # ... weight by environment area fraction if using PrognosticEDMFX (assumed 1 otherwise) ...
    if turbconv_model isa PrognosticEDMFX
        ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, p.atmos.turbconv_model))
        @. p.precomputed.cloud_diagnostics_tuple *= NamedTuple{(:cf, :q_liq, :q_ice)}(
            tuple(
                draft_area(ᶜρa⁰, TD.air_density(thermo_params, ᶜts)),
                draft_area(ᶜρa⁰, TD.air_density(thermo_params, ᶜts)),
                draft_area(ᶜρa⁰, TD.air_density(thermo_params, ᶜts)),
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
