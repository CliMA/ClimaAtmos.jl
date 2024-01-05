import StaticArrays as SA
import ClimaCore.RecursiveApply: rzero, ⊞, ⊠

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
function set_cloud_fraction!(Y, p, ::DryModel)
    (; ᶜmixing_length) = p.precomputed
    (; turbconv_model) = p.atmos
    if isnothing(turbconv_model)
        compute_gm_mixing_length!(ᶜmixing_length, Y, p)
    end
    @. p.precomputed.ᶜcloud_fraction = 0
end
function set_cloud_fraction!(Y, p, ::Union{EquilMoistModel, NonEquilMoistModel})
    (; SG_quad, params) = p

    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜts, ᶜp, ᶜmixing_length, ᶜcloud_fraction) = p.precomputed
    (; turbconv_model) = p.atmos
    if isnothing(turbconv_model)
        compute_gm_mixing_length!(ᶜmixing_length, Y, p)
    end

    coeff = FT(2.1) # TODO - move to parameters
    @. ᶜcloud_fraction = quad_loop(
        SG_quad,
        ᶜp,
        TD.total_specific_humidity(thermo_params, ᶜts),
        TD.liquid_ice_pottemp(thermo_params, ᶜts),
        Geometry.WVector(
            ᶜgradᵥ(ᶠinterp(TD.total_specific_humidity(thermo_params, ᶜts))),
        ),
        Geometry.WVector(
            ᶜgradᵥ(ᶠinterp(TD.liquid_ice_pottemp(thermo_params, ᶜts))),
        ),
        coeff,
        ᶜmixing_length,
        thermo_params,
    )
end

"""
    function quad_loop(SG_quad, p_c, q_mean, θ_mean, ᶜ∇q, ᶜ∇θ,
                       coeff, ᶜlength_scale, thermo_params)

where:
  - SG_quad is a struct containing information about quadrature type and order
  - p_c is the atmospheric pressure
  - q_mean, θ_mean is the grid mean q_tot and liquid ice potential temperature
  - ᶜ∇q, ᶜ∇θ are the gradients of q_tot and liquid ice potential temperature
  - coeff - a free parameter (to be moved into params)
  - ᶜlength_scale - mixing length for simulations with EDMF and Smagorinsky
                    length scale for simulations without EDMF
  - thermo params - thermodynamics parameters

The function imposes additional limits on the quadrature points and
returns cloud fraction computed as a sum over quadrature points.
"""
function quad_loop(
    SG_quad::SGSQuadrature,
    p_c,
    q_mean,
    θ_mean,
    ᶜ∇q,
    ᶜ∇θ,
    coeff,
    ᶜlength_scale,
    thermo_params,
)
    # Returns the physical values based on quadrature sampling points
    # and limited covarainces
    function get_x_hat(χ1, χ2)

        @assert SG_quad.quadrature_type isa GaussianQuad
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
        ts = thermo_state(thermo_params; p = p_c, θ = x1_hat, q_tot = x2_hat)
        hc = TD.has_condensate(thermo_params, ts)
        return (;
            cf = hc ? FT(1) : FT(0), # cloud fraction
            q_tot_sat = hc ? x2_hat : FT(0), # cloudy/dry for buoyancy in TKE
        )
    end

    return quad(f, get_x_hat, SG_quad).cf
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
