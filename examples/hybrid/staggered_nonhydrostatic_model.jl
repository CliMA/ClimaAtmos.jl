using LinearAlgebra: ×, norm, norm_sqr, dot

import ClimaAtmos.Parameters as CAP
using ClimaCore: Operators, Fields, Limiters

using ClimaCore.Geometry: ⊗

import Thermodynamics as TD

using ClimaCore.Utilities: half

include("schur_complement_W.jl")
include("hyperdiffusion.jl")

# Note: FT must be defined before `include("staggered_nonhydrostatic_model.jl")`

# Functions on which the model depends:
# CAP.R_d(params)         # dry specific gas constant
# CAP.kappa_d(params)     # dry adiabatic exponent
# CAP.T_triple(params)    # triple point temperature of water
# CAP.MSLP(params)        # reference pressure
# CAP.grav(params)        # gravitational acceleration
# CAP.Omega(params)       # rotation rate (only used if space is spherical)
# CAP.cv_d(params)        # dry isochoric specific heat capacity
# The value of cv_d is implied by the values of R_d and kappa_d

# The model also depends on f_plane_coriolis_frequency(params)
# This is a constant Coriolis frequency that is only used if space is flat

# To add additional terms to the explicit part of the tendency, define new
# methods for `additional_cache` and `additional_tendency!`.

const divₕ = Operators.Divergence()
const wdivₕ = Operators.WeakDivergence()
const gradₕ = Operators.Gradient()
const wgradₕ = Operators.WeakGradient()
const curlₕ = Operators.Curl()
const wcurlₕ = Operators.WeakCurl()

const ᶜinterp = Operators.InterpolateF2C()
const ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶜdivᵥ = Operators.DivergenceF2C(
    top = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
    bottom = Operators.SetValue(Geometry.Contravariant3Vector(FT(0))),
)
const ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
    top = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
)
const ᶜgradᵥ = Operators.GradientF2C()
const ᶠcurlᵥ = Operators.CurlC2F(
    bottom = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
    top = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
)
const ᶠupwind1 = Operators.UpwindBiasedProductC2F()
const ᶠupwind3 = Operators.Upwind3rdOrderBiasedProductC2F(
    bottom = Operators.ThirdOrderOneSided(),
    top = Operators.ThirdOrderOneSided(),
)
const ᶠfct_boris_book = Operators.FCTBorisBook(
    bottom = Operators.FirstOrderOneSided(),
    top = Operators.FirstOrderOneSided(),
)
const ᶠfct_zalesak = Operators.FCTZalesak(
    bottom = Operators.FirstOrderOneSided(),
    top = Operators.FirstOrderOneSided(),
)

const ᶜinterp_stencil = Operators.Operator2Stencil(ᶜinterp)
const ᶠinterp_stencil = Operators.Operator2Stencil(ᶠinterp)
const ᶜdivᵥ_stencil = Operators.Operator2Stencil(ᶜdivᵥ)
const ᶠgradᵥ_stencil = Operators.Operator2Stencil(ᶠgradᵥ)

const C123 = Geometry.Covariant123Vector

include("thermo_state.jl")

get_cache(Y, params, spaces, model_spec, numerics, simulation) = merge(
    default_cache(Y, params, model_spec, spaces, numerics, simulation),
    additional_cache(Y, params, model_spec, simulation.dt),
)

function default_cache(Y, params, model_spec, spaces, numerics, simulation)
    (; energy_upwinding, tracer_upwinding, apply_limiter) = numerics
    ᶜcoord = Fields.local_geometry_field(Y.c).coordinates
    ᶠcoord = Fields.local_geometry_field(Y.f).coordinates
    gⁱʲ = Fields.level(Fields.local_geometry_field(Y.f), ClimaCore.Utilities.half).gⁱʲ
    g¹³ = gⁱʲ.components.data.:3
    g²³ = gⁱʲ.components.data.:6
    g³³ = gⁱʲ.components.data.:9 
    ᶜΦ = CAP.grav(params) .* ᶜcoord.z
    z_sfc = Fields.level(ᶠcoord.z, half)
    if eltype(ᶜcoord) <: Geometry.LatLongZPoint
        Ω = CAP.Omega(params)
        ᶜf = @. 2 * Ω * sind(ᶜcoord.lat)
        lat_sfc = Fields.level(ᶜcoord.lat, 1)
    else
        f = CAP.f_plane_coriolis_frequency(params)
        ᶜf = map(_ -> f, ᶜcoord)
        lat_sfc = map(_ -> FT(0), Fields.level(ᶜcoord, 1))
    end
    ᶜf = @. Geometry.Contravariant3Vector(Geometry.WVector(ᶜf))
    T_sfc = @. 29 * exp(-lat_sfc^2 / (2 * 26^2)) + 271
    ts_type = thermo_state_type(Y.c, FT)
    ghost_buffer = (
        c = Spaces.create_ghost_buffer(Y.c),
        f = Spaces.create_ghost_buffer(Y.f),
        χ = Spaces.create_ghost_buffer(Y.c.ρ), # for hyperdiffusion
        χw = Spaces.create_ghost_buffer(Y.f.w.components.data.:1), # for hyperdiffusion
        χuₕ = Spaces.create_ghost_buffer(Y.c.uₕ), # for hyperdiffusion
    )
    (:ρq_tot in propertynames(Y.c)) && (
        ghost_buffer =
            (ghost_buffer..., ᶜχρq_tot = Spaces.create_ghost_buffer(Y.c.ρ))
    )
    if apply_limiter
        tracers = filter(is_tracer_var, propertynames(Y.c))
        make_limiter =
            ᶜρc_name ->
                Limiters.QuasiMonotoneLimiter(getproperty(Y.c, ᶜρc_name), Y.c.ρ)
        limiters = NamedTuple{tracers}(map(make_limiter, tracers))
    else
        limiters = nothing
    end
    pnc = propertynames(Y.c)
    ᶜρh_kwargs =
        :ρe_tot in pnc || :ρe_int in pnc ? (; ᶜρh = similar(Y.c, FT)) : ()
    return (;
        simulation,
        spaces,
        Yₜ = similar(Y), # only needed when using increment formulation
        limiters,
        ᶜρh_kwargs...,
        ᶜuvw = similar(Y.c, Geometry.Covariant123Vector{FT}),
        ᶜK = similar(Y.c, FT),
        ᶜΦ,
        ᶠgradᵥ_ᶜΦ = ᶠgradᵥ.(ᶜΦ),
        ᶜts = similar(Y.c, ts_type),
        ᶜp = similar(Y.c, FT),
        ᶜT = similar(Y.c, FT),
        ᶜω³ = similar(Y.c, Geometry.Contravariant3Vector{FT}),
        ᶠω¹² = similar(Y.f, Geometry.Contravariant12Vector{FT}),
        ᶠu¹² = similar(Y.f, Geometry.Contravariant12Vector{FT}),
        ᶠu³ = similar(Y.f, Geometry.Contravariant3Vector{FT}),
        ᶜω¹² = similar(Y.c, Geometry.Contravariant12Vector{FT}),
        ᶜu³ = similar(Y.c, Geometry.Contravariant3Vector{FT}),
        ᵇᶜu₃ = similar(gⁱʲ, Geometry.Covariant3Vector{FT}),
        ᶜw = similar(Y.c, Geometry.Covariant3Vector{FT}),
        ᶜf,
        z_sfc,
        T_sfc,
        gⁱʲ, 
        g¹³,
        g²³,
        g³³,
        ρ_sfc = similar(T_sfc, FT),
        q_sfc = similar(T_sfc, FT),
        ∂ᶜK∂ᶠw_data = similar(
            Y.c,
            Operators.StencilCoefs{-half, half, NTuple{2, FT}},
        ),
        params,
        energy_upwinding,
        tracer_upwinding,
        ghost_buffer = ghost_buffer,
    )
end

# TODO: All of these should use dtγ instead of dt, but dtγ is not available in
# the implicit tendency function. Since dt >= dtγ, we can safely use dt for now.
vertical_transport!(ᶜρcₜ, ᶠw, ᶜρ, ᶜρc, dt, ::Val{:none}) =
    @. ᶜρcₜ = -(ᶜdivᵥ(ᶠinterp(ᶜρc) * ᶠw))
vertical_transport!(ᶜρcₜ, ᶠw, ᶜρ, ᶜρc, dt, ::Val{:first_order}) =
    @. ᶜρcₜ = -(ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind1(ᶠw, ᶜρc / ᶜρ)))
vertical_transport!(ᶜρcₜ, ᶠw, ᶜρ, ᶜρc, dt, ::Val{:third_order}) =
    @. ᶜρcₜ = -(ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind3(ᶠw, ᶜρc / ᶜρ)))
vertical_transport!(ᶜρcₜ, ᶠw, ᶜρ, ᶜρc, dt, ::Val{:boris_book}) = @. ᶜρcₜ =
    -(ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind1(ᶠw, ᶜρc / ᶜρ))) - ᶜdivᵥ(
        ᶠinterp(ᶜρ) * ᶠfct_boris_book(
            ᶠupwind3(ᶠw, ᶜρc / ᶜρ) - ᶠupwind1(ᶠw, ᶜρc / ᶜρ),
            (ᶜρc / dt - ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind1(ᶠw, ᶜρc / ᶜρ))) / ᶜρ,
        ),
    )
vertical_transport!(ᶜρcₜ, ᶠw, ᶜρ, ᶜρc, dt, ::Val{:zalesak}) = @. ᶜρcₜ =
    -(ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind1(ᶠw, ᶜρc / ᶜρ))) - ᶜdivᵥ(
        ᶠinterp(ᶜρ) * ᶠfct_zalesak(
            ᶠupwind3(ᶠw, ᶜρc / ᶜρ) - ᶠupwind1(ᶠw, ᶜρc / ᶜρ),
            ᶜρc / ᶜρ / dt,
            (ᶜρc / dt - ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind1(ᶠw, ᶜρc / ᶜρ))) / ᶜρ,
        ),
    )

# Used for automatically computing the Jacobian ∂Yₜ/∂Y. Currently requires
# allocation because the cache is stored separately from Y, which means that
# similar(Y, <:Dual) doesn't allocate an appropriate cache for computing Yₜ.
function implicit_cache_vars(
    Y::Fields.FieldVector{T},
    p,
) where {T <: AbstractFloat}
    (; ᶜK, ᶜts, ᶜp) = p
    return (; ᶜK, ᶜts, ᶜp)
end
function implicit_cache_vars(Y::Fields.FieldVector{T}, p) where {T <: Dual}
    ᶜρ = Y.c.ρ
    ᶜK = similar(ᶜρ)
    ᶜts = similar(ᶜρ, eltype(p.ts).name.wrapper{eltype(ᶜρ)})
    ᶜp = similar(ᶜρ)
    return (; ᶜK, ᶜts, ᶜp)
end

function implicit_tendency!(Yₜ, Y, p, t)
    @nvtx "implicit tendency" color = colorant"yellow" begin
        _implicit_tendency!(Yₜ, Y, p, t)
    end
end

function _implicit_tendency!(Yₜ, Y, p, t)
    Fields.bycolumn(axes(Y.c)) do colidx
        ᶜρ = Y.c.ρ
        ᶜuₕ = Y.c.uₕ
        ᶠw = Y.f.w
        (; ᶜK, ᶠgradᵥ_ᶜΦ, ᶜts, ᶜp, params, thermo_dispatcher) = p
        (; energy_upwinding, tracer_upwinding, simulation) = p

        thermo_params = CAP.thermodynamics_params(params)
        dt = simulation.dt
        @. ᶜK[colidx] =
            norm_sqr(C123(ᶜuₕ[colidx]) + C123(ᶜinterp(ᶠw[colidx]))) / 2
        thermo_state!(
            ᶜts[colidx],
            Y.c[colidx],
            thermo_params,
            thermo_dispatcher,
            ᶜinterp,
            ᶜK[colidx],
            Y.f.w[colidx],
        )
        @. ᶜp[colidx] = TD.air_pressure(thermo_params, ᶜts[colidx])

        if p.tendency_knobs.has_turbconv
            parent(Yₜ.c.turbconv[colidx]) .= FT(0)
            parent(Yₜ.f.turbconv[colidx]) .= FT(0)
        end

        vertical_transport!(
            Yₜ.c.ρ[colidx],
            ᶠw[colidx],
            ᶜρ[colidx],
            ᶜρ[colidx],
            dt,
            Val(:none),
        )

        if :ρθ in propertynames(Y.c)
            vertical_transport!(
                Yₜ.c.ρθ[colidx],
                ᶠw[colidx],
                ᶜρ[colidx],
                Y.c.ρθ[colidx],
                dt,
                energy_upwinding,
            )
        elseif :ρe_tot in propertynames(Y.c)
            (; ᶜρh) = p
            @. ᶜρh[colidx] = Y.c.ρe_tot[colidx] + ᶜp[colidx]
            vertical_transport!(
                Yₜ.c.ρe_tot[colidx],
                ᶠw[colidx],
                ᶜρ[colidx],
                ᶜρh[colidx],
                dt,
                energy_upwinding,
            )
        elseif :ρe_int in propertynames(Y.c)
            (; ᶜρh) = p
            @. ᶜρh[colidx] = Y.c.ρe_int[colidx] + ᶜp[colidx]
            vertical_transport!(
                Yₜ.c.ρe_int[colidx],
                ᶠw[colidx],
                ᶜρ[colidx],
                ᶜρh[colidx],
                dt,
                energy_upwinding,
            )
            @. Yₜ.c.ρe_int[colidx] += ᶜinterp(
                dot(
                    ᶠgradᵥ(ᶜp[colidx]),
                    Geometry.Contravariant3Vector(ᶠw[colidx]),
                ),
            )
        end

        Yₜ.c.uₕ[colidx] .= Ref(zero(eltype(Yₜ.c.uₕ[colidx])))

        @. Yₜ.f.w[colidx] =
            -(ᶠgradᵥ(ᶜp[colidx]) / ᶠinterp(ᶜρ[colidx]) + ᶠgradᵥ_ᶜΦ[colidx])
        if p.tendency_knobs.rayleigh_sponge
            @. Yₜ.f.w[colidx] -= p.ᶠβ_rayleigh_w[colidx] * Y.f.w[colidx]
        end

        for ᶜρc_name in filter(is_tracer_var, propertynames(Y.c))
            ᶜρcₜ = getproperty(Yₜ.c, ᶜρc_name)
            ᶜρc = getproperty(Y.c, ᶜρc_name)
            vertical_transport!(
                ᶜρcₜ[colidx],
                ᶠw[colidx],
                ᶜρ[colidx],
                ᶜρc[colidx],
                dt,
                tracer_upwinding,
            )
        end
    end
    return nothing
end

function remaining_tendency!(Yₜ, Y, p, t)
    (; compressibility_model) = p
    @nvtx "remaining tendency" color = colorant"yellow" begin
        Yₜ .= zero(eltype(Yₜ))
        if compressibility_model isa CA.CompressibleFluid
            @nvtx "precomputed quantities" color = colorant"orange" begin
                precomputed_quantities!(Y, p, t)
            end
            @nvtx "horizontal" color = colorant"orange" begin
                horizontal_advection_tendency!(Yₜ, Y, p, t)
            end
            explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
        end
        @nvtx "additional_tendency!" color = colorant"orange" begin
            additional_tendency!(Yₜ, Y, p, t)
        end
        @nvtx "dss_remaining_tendency" color = colorant"blue" begin
            Spaces.weighted_dss_start!(Yₜ.c, p.ghost_buffer.c)
            Spaces.weighted_dss_start!(Yₜ.f, p.ghost_buffer.f)
            Spaces.weighted_dss_internal!(Yₜ.c, p.ghost_buffer.c)
            Spaces.weighted_dss_internal!(Yₜ.f, p.ghost_buffer.f)
            Spaces.weighted_dss_ghost!(Yₜ.c, p.ghost_buffer.c)
            Spaces.weighted_dss_ghost!(Yₜ.f, p.ghost_buffer.f)
        end
    end
    return Yₜ
end

function remaining_tendency_increment!(Y⁺, Y, p, t, dtγ)
    (; Yₜ, limiters) = p
    (; compressibility_model) = p
    @nvtx "remaining tendency increment" color = colorant"yellow" begin
        Yₜ .= zero(eltype(Yₜ))
        if compressibility_model isa CA.CompressibleFluid
            @nvtx "precomputed quantities" color = colorant"orange" begin
                precomputed_quantities!(Y, p, t)
            end
            @nvtx "horizontal" color = colorant"orange" begin
                horizontal_advection_tendency!(Yₜ, Y, p, t)
            end
            # Apply limiter
            if !isnothing(limiters)
                @. Y⁺ += dtγ * Yₜ
                for ᶜρc_name in filter(is_tracer_var, propertynames(Y.c))
                    ρc_limiter = getproperty(limiters, ᶜρc_name)
                    ᶜρc = getproperty(Y.c, ᶜρc_name)
                    ᶜρc⁺ = getproperty(Y⁺.c, ᶜρc_name)
                    Limiters.compute_bounds!(ρc_limiter, ᶜρc, Y.c.ρ)
                    Limiters.apply_limiter!(ᶜρc⁺, Y⁺.c.ρ, ρc_limiter)
                end
                Yₜ .= zero(eltype(Yₜ))
            end
            explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
        end
        @nvtx "additional_tendency! increment" color = colorant"orange" begin
            additional_tendency!(Yₜ, Y, p, t)
            @. Y⁺ += dtγ * Yₜ
        end
        @nvtx "dss_remaining_tendency increment" color = colorant"blue" begin
            Spaces.weighted_dss_start!(Y⁺.c, p.ghost_buffer.c)
            Spaces.weighted_dss_start!(Y⁺.f, p.ghost_buffer.f)
            Spaces.weighted_dss_internal!(Y⁺.c, p.ghost_buffer.c)
            Spaces.weighted_dss_internal!(Y⁺.f, p.ghost_buffer.f)
            Spaces.weighted_dss_ghost!(Y⁺.c, p.ghost_buffer.c)
            Spaces.weighted_dss_ghost!(Y⁺.f, p.ghost_buffer.f)
        end
    end
    return Y⁺
end

function precomputed_quantities!(Y, p, t)
    Fields.bycolumn(axes(Y.c)) do colidx
        precomputed_quantities!(Y, p, t, colidx)
    end
end
function precomputed_quantities!(Y, p, t, colidx)
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜuvw, ᶜK, ᶜts, ᶜp, ᵇᶜu₃, g¹³, g²³, g³³, params, thermo_dispatcher) = p
    
    # Impose boundary condition on vertical velocity term.
    ᵇᶜuₕ = Fields.level(ᶠinterp.(ᶜuₕ[colidx]), ClimaCore.Utilities.half)
    @. ᵇᶜu₃ = Geometry.Covariant3Vector(-1 * g¹³ / g³³ * ᵇᶜuₕ.components.data.:1)
    enforce_boundary = Operators.SetBoundaryOperator(bottom = Operators.SetValue(ᵇᶜu₃))
    @. ᶠw = apply_boundary_w(fw)
    

    @. ᶜuvw[colidx] = C123(ᶜuₕ[colidx]) + C123(ᶜinterp(ᶠw[colidx]))
    @. ᶜK[colidx] = norm_sqr(ᶜuvw[colidx]) / 2
    thermo_params = CAP.thermodynamics_params(params)
    thermo_state!(
        ᶜts[colidx],
        Y.c[colidx],
        thermo_params,
        thermo_dispatcher,
        ᶜinterp,
        ᶜK[colidx],
        Y.f.w[colidx],
    )
    @. ᶜp[colidx] = TD.air_pressure(thermo_params, ᶜts[colidx])
    return nothing
end

function horizontal_advection_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜuvw, ᶜK, ᶜΦ, ᶜts, ᶜp, ᶜω³, ᶠω¹², ᶜω¹², ᶜw, ᶜu³, params) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)

    # Mass conservation
    @. Yₜ.c.ρ -= divₕ(ᶜρ * ᶜuvw)

    # Energy conservation
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ -= divₕ(Y.c.ρθ * ᶜuvw)
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot -= divₕ((Y.c.ρe_tot + ᶜp) * ᶜuvw)
    elseif :ρe_int in propertynames(Y.c)
        if point_type <: Geometry.Abstract3DPoint
            @. Yₜ.c.ρe_int -=
                divₕ((Y.c.ρe_int + ᶜp) * ᶜuvw) -
                dot(gradₕ(ᶜp), Geometry.Contravariant12Vector(ᶜuₕ))
        else
            @. Yₜ.c.ρe_int -=
                divₕ((Y.c.ρe_int + ᶜp) * ᶜuvw) -
                dot(gradₕ(ᶜp), Geometry.Contravariant1Vector(ᶜuₕ))
        end
    end

    # Momentum conservation
    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = curlₕ(ᶜuₕ)
        @. ᶠω¹² = curlₕ(ᶠw)
        @. ᶜw = ᶜinterp(ᶠw)
        @. ᶜω¹² = curlₕ(ᶜw)
        @. Yₜ.c.uₕ -= gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + ᶜΦ)
    elseif point_type <: Geometry.Abstract2DPoint
        ᶜω³ .= Ref(zero(eltype(ᶜω³)))
        @. ᶜw = ᶜinterp(ᶠw)
        @. ᶠω¹² = Geometry.Contravariant12Vector(curlₕ(ᶠw))
        @. ᶜω¹² = Geometry.Contravariant12Vector(curlₕ(ᶜw))
        @. Yₜ.c.uₕ -=
            Geometry.Covariant12Vector(gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + ᶜΦ))
    end

    # Tracer conservation
    for ᶜρc_name in filter(is_tracer_var, propertynames(Y.c))
        ᶜρc = getproperty(Y.c, ᶜρc_name)
        ᶜρcₜ = getproperty(Yₜ.c, ᶜρc_name)
        @. ᶜρcₜ -= divₕ(ᶜρc * ᶜuvw)
    end
    return nothing
end

function explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    @nvtx "vertical" color = colorant"orange" begin
        _explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    end
    return nothing
end
function _explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    Fields.bycolumn(axes(Y.c)) do colidx
        ᶜρ = Y.c.ρ
        ᶜuₕ = Y.c.uₕ
        ᶠw = Y.f.w
        (; ᶜuvw, ᶜK, ᶜp, ᶜω³, ᶠω¹², ᶜω¹², ᶜw, ᶜu³, ᶠu¹², ᶠu³, ᶜf) = p
        # Mass conservation
        @. Yₜ.c.ρ[colidx] -= ᶜdivᵥ(ᶠinterp(ᶜρ[colidx] * ᶜuₕ[colidx]))

        # Energy conservation
        if :ρθ in propertynames(Y.c)
            @. Yₜ.c.ρθ[colidx] -= ᶜdivᵥ(ᶠinterp(Y.c.ρθ[colidx] * ᶜuₕ[colidx]))
        elseif :ρe_tot in propertynames(Y.c)
            @. Yₜ.c.ρe_tot[colidx] -=
                ᶜdivᵥ(ᶠinterp((Y.c.ρe_tot[colidx] + ᶜp[colidx]) * ᶜuₕ[colidx]))
        elseif :ρe_int in propertynames(Y.c)
            @. Yₜ.c.ρe_int[colidx] -=
                ᶜdivᵥ(ᶠinterp((Y.c.ρe_int[colidx] + ᶜp[colidx]) * ᶜuₕ[colidx]))
        end

        # Momentum conservation
        @. ᶠω¹²[colidx] += ᶠcurlᵥ(ᶜuₕ[colidx])
        @. ᶜω¹²[colidx] += ᶜinterp(ᶠcurlᵥ(ᶜuₕ[colidx]))
        @. ᶠu¹²[colidx] = Geometry.project(
            Geometry.Contravariant12Axis(),
            ᶠinterp(ᶜuvw[colidx]),
        )
        @. ᶠu³[colidx] = Geometry.project(
            Geometry.Contravariant3Axis(),
            C123(ᶠinterp(ᶜuₕ[colidx])) + C123(ᶠw[colidx]),
        )
        # Oct19
        @. Yₜ.c.uₕ[colidx] -=
            ᶜω¹²[colidx] × ᶜinterp(ᶠu³[colidx]) +
            (ᶜf[colidx] + ᶜω³[colidx]) ×
            (Geometry.project(Geometry.Contravariant12Axis(), ᶜuvw[colidx]))
        # Oct19
        #@. Yₜ.c.uₕ[colidx] -=
        #    ᶜinterp(ᶠω¹²[colidx] × ᶠu³[colidx]) +
        #    (ᶜf[colidx] + ᶜω³[colidx]) ×
        #    (Geometry.project(Geometry.Contravariant12Axis(), ᶜuvw[colidx]))
        @. Yₜ.f.w[colidx] -= ᶠω¹²[colidx] × ᶠu¹²[colidx] + ᶠgradᵥ(ᶜK[colidx])

        # Tracer conservation
        for ᶜρc_name in filter(is_tracer_var, propertynames(Y.c))
            ᶜρc = getproperty(Y.c, ᶜρc_name)
            ᶜρcₜ = getproperty(Yₜ.c, ᶜρc_name)
            @. ᶜρcₜ[colidx] -= ᶜdivᵥ(ᶠinterp(ᶜρc[colidx] * ᶜuₕ[colidx]))
        end
        nothing
    end
    return nothing
end

# Allow one() to be called on vectors.
Base.one(::T) where {T <: Geometry.AxisTensor} = one(T)
Base.one(::Type{T}) where {T′, A, S, T <: Geometry.AxisTensor{T′, 1, A, S}} =
    T(axes(T), S(one(T′)))

# Allow StencilCoefs to be expanded.
function Base.convert(
    T::Type{<:Operators.StencilCoefs{lbw′, ubw′}},
    coefs::Operators.StencilCoefs{lbw, ubw},
) where {lbw, ubw, lbw′, ubw′}
    if lbw′ <= lbw && ubw′ >= ubw
        zero_val = zero(eltype(T))
        lpadding = ntuple(_ -> zero_val, lbw - lbw′)
        rpadding = ntuple(_ -> zero_val, ubw′ - ubw)
        return T((lpadding..., coefs.coefs..., rpadding...))
    else
        error("Cannot convert a StencilCoefs object with bandwidths $lbw and \
              $ubw to a StencilCoefs object with bandwidths $lbw′ and $ubw′")
    end
end

# In vertical_transport_jac!, we assume that ∂(ᶜρc)/∂(ᶠw_data) = 0; if
# this is not the case, the additional term should be added to the
# result of this function.
# In addition, we approximate the Jacobian for vertical transport with
# FCT using the Jacobian for third-order upwinding (since only FCT
# requires dt, we do not need to pass dt to this function).
function vertical_transport_jac!(∂ᶜρcₜ∂ᶠw, ᶠw, ᶜρ, ᶜρc, ::Val{:none})
    @. ∂ᶜρcₜ∂ᶠw = -(ᶜdivᵥ_stencil(ᶠinterp(ᶜρc) * one(ᶠw)))
    return nothing
end
function vertical_transport_jac!(∂ᶜρcₜ∂ᶠw, ᶠw, ᶜρ, ᶜρc, ::Val{:first_order})
    # To convert ᶠw to ᶠw_data, we extract the third vector component.
    to_scalar(vector) = vector.u₃
    FT = Spaces.undertype(axes(ᶜρ))
    ref_εw = Ref(Geometry.Covariant3Vector(eps(FT)))
    @. ∂ᶜρcₜ∂ᶠw = -(ᶜdivᵥ_stencil(
        ᶠinterp(ᶜρ) * ᶠupwind1(ᶠw + ref_εw, ᶜρc / ᶜρ) / to_scalar(ᶠw + ref_εw),
    ))
    return nothing
end
function vertical_transport_jac!(∂ᶜρcₜ∂ᶠw, ᶠw, ᶜρ, ᶜρc, ::Val)
    # To convert ᶠw to ᶠw_data, we extract the third vector component.
    to_scalar(vector) = vector.u₃
    FT = Spaces.undertype(axes(ᶜρ))
    ref_εw = Ref(Geometry.Covariant3Vector(eps(FT)))
    @. ∂ᶜρcₜ∂ᶠw = -(ᶜdivᵥ_stencil(
        ᶠinterp(ᶜρ) * ᶠupwind3(ᶠw + ref_εw, ᶜρc / ᶜρ) / to_scalar(ᶠw + ref_εw),
    ))
    return nothing
end

function validate_flags!(Y, flags, energy_upwinding)
    if :ρe_tot in propertynames(Y.c)
        if energy_upwinding === Val(:none) && flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode != :no_∂ᶜp∂ᶜK
            error(
                "∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :exact or :no_∂ᶜp∂ᶜK when using ρe_tot \
                without upwinding",
            )
        elseif flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode != :no_∂ᶜp∂ᶜK
            # TODO: Add Operator2Stencil for UpwindBiasedProductC2F to ClimaCore
            # to allow exact Jacobian calculation.
            error("∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :no_∂ᶜp∂ᶜK when using ρe_tot with \
                  upwinding")
        end
    elseif :ρe_int in propertynames(Y.c) && flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode != :exact
        error("∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :exact when using ρe_int")
    end
    # TODO: If we end up using :gradΦ_shenanigans, optimize it to
    # `cached_stencil / ᶠinterp(ᶜρ)`.
    if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode != :exact && flags.∂ᶠ𝕄ₜ∂ᶜρ_mode != :gradΦ_shenanigans
        error("∂ᶠ𝕄ₜ∂ᶜρ_mode must be :exact or :gradΦ_shenanigans")
    end
end

call_verify_wfact_matrix() = false

function Wfact!(W, Y, p, dtγ, t)
    @nvtx "Wfact!" color = colorant"green" begin
        _Wfact!(W, Y, p, dtγ, t)
    end
end

function _Wfact!(W, Y, p, dtγ, t)
    p.apply_moisture_filter && affect_filter!(Y)
    (; flags, dtγ_ref) = W
    (; ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶠ𝕄ₜ∂ᶠ𝕄, ∂ᶜ𝕋ₜ∂ᶠ𝕄_field) = W
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶠgradᵥ_ᶜΦ, ᶜts, ᶜp, ∂ᶜK∂ᶠw_data, params) = p
    (; energy_upwinding, tracer_upwinding, thermo_dispatcher) = p

    validate_flags!(Y, flags, energy_upwinding)


    R_d = FT(CAP.R_d(params))
    κ_d = FT(CAP.kappa_d(params))
    cv_d = FT(CAP.cv_d(params))
    T_tri = FT(CAP.T_triple(params))
    MSLP = FT(CAP.MSLP(params))

    dtγ_ref[] = dtγ

    # If we let ᶠw_data = ᶠw.components.data.:1 and ᶠw_unit = one.(ᶠw), then
    # ᶠw == ᶠw_data .* ᶠw_unit. The Jacobian blocks involve ᶠw_data, not ᶠw.
    ᶠw_data = ᶠw.components.data.:1

    # To convert ∂(ᶠwₜ)/∂(ᶜ𝔼) to ∂(ᶠw_data)ₜ/∂(ᶜ𝔼) and ∂(ᶠwₜ)/∂(ᶠw_data) to
    # ∂(ᶠw_data)ₜ/∂(ᶠw_data), we extract the third component of each vector-
    # valued stencil coefficient.
    to_scalar_coefs(vector_coefs) =
        map(vector_coef -> vector_coef.u₃, vector_coefs)

    Fields.bycolumn(axes(Y.c)) do colidx
        # If ᶜρcₜ = -ᶜdivᵥ(ᶠinterp(ᶜρc) * ᶠw), then
        # ∂(ᶜρcₜ)/∂(ᶠw_data) =
        #     -ᶜdivᵥ_stencil(ᶠinterp(ᶜρc) * ᶠw_unit) -
        #     ᶜdivᵥ_stencil(ᶠw) * ᶠinterp_stencil(1) * ∂(ᶜρc)/∂(ᶠw_data)
        # If ᶜρcₜ = -ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind(ᶠw, ᶜρc / ᶜρ)), then
        # ∂(ᶜρcₜ)/∂(ᶠw_data) =
        #     -ᶜdivᵥ_stencil(ᶠinterp(ᶜρc) *
        #     ᶠupwind(ᶠw + εw, ᶜρc) / to_scalar(ᶠw + εw)) -
        #     ᶜdivᵥ_stencil(ᶠinterp(ᶜρ)) * ᶠupwind_stencil(ᶠw, 1 / ᶜρ) *
        #     ∂(ᶜρc)/∂(ᶠw_data)
        # The εw is only necessary in case w = 0.
        # Since Operator2Stencil has not yet been extended to upwinding
        # operators, ᶠupwind_stencil is not available.
        @. ᶜK[colidx] =
            norm_sqr(C123(ᶜuₕ[colidx]) + C123(ᶜinterp(ᶠw[colidx]))) / 2
        thermo_params = CAP.thermodynamics_params(params)
        thermo_state!(
            ᶜts[colidx],
            Y.c[colidx],
            thermo_params,
            thermo_dispatcher,
            ᶜinterp,
            ᶜK[colidx],
            Y.f.w[colidx],
        )
        @. ᶜp[colidx] = TD.air_pressure(thermo_params, ᶜts[colidx])

        # ᶜinterp(ᶠw) =
        #     ᶜinterp(ᶠw)_data * ᶜinterp(ᶠw)_unit =
        #     ᶜinterp(ᶠw_data) * ᶜinterp(ᶠw)_unit
        # norm_sqr(ᶜinterp(ᶠw)) =
        #     norm_sqr(ᶜinterp(ᶠw_data) * ᶜinterp(ᶠw)_unit) =
        #     ᶜinterp(ᶠw_data)^2 * norm_sqr(ᶜinterp(ᶠw)_unit)
        # ᶜK =
        #     norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2 =
        #     norm_sqr(ᶜuₕ) / 2 + norm_sqr(ᶜinterp(ᶠw)) / 2 =
        #     norm_sqr(ᶜuₕ) / 2 + ᶜinterp(ᶠw_data)^2 * norm_sqr(ᶜinterp(ᶠw)_unit) / 2
        # ∂(ᶜK)/∂(ᶠw_data) =
        #     ∂(ᶜK)/∂(ᶜinterp(ᶠw_data)) * ∂(ᶜinterp(ᶠw_data))/∂(ᶠw_data) =
        #     ᶜinterp(ᶠw_data) * norm_sqr(ᶜinterp(ᶠw)_unit) * ᶜinterp_stencil(1)
        @. ∂ᶜK∂ᶠw_data[colidx] =
            ᶜinterp(ᶠw_data[colidx]) *
            norm_sqr(one(ᶜinterp(ᶠw[colidx]))) *
            ᶜinterp_stencil(one(ᶠw_data[colidx]))

        # vertical_transport!(Yₜ.c.ρ, ᶠw, ᶜρ, ᶜρ, dt, Val(:none))
        vertical_transport_jac!(
            ∂ᶜρₜ∂ᶠ𝕄[colidx],
            ᶠw[colidx],
            ᶜρ[colidx],
            ᶜρ[colidx],
            Val(:none),
        )

        if :ρθ in propertynames(Y.c)
            ᶜρθ = Y.c.ρθ
            # vertical_transport!(Yₜ.c.ρθ, ᶠw, ᶜρ, ᶜρθ, dt, energy_upwinding)
            vertical_transport_jac!(
                ∂ᶜ𝔼ₜ∂ᶠ𝕄[colidx],
                ᶠw[colidx],
                ᶜρ[colidx],
                ᶜρθ[colidx],
                energy_upwinding,
            )
        elseif :ρe_tot in propertynames(Y.c)
            ᶜρe = Y.c.ρe_tot
            (; ᶜρh) = p
            @. ᶜρh[colidx] = ᶜρe[colidx] + ᶜp[colidx]
            # vertical_transport!(Yₜ.c.ρe_tot, ᶠw, ᶜρ, ᶜρh, dt, energy_upwinding)
            vertical_transport_jac!(
                ∂ᶜ𝔼ₜ∂ᶠ𝕄[colidx],
                ᶠw[colidx],
                ᶜρ[colidx],
                ᶜρh[colidx],
                energy_upwinding,
            )
            if energy_upwinding === Val(:none)
                if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact
                    # ∂(ᶜρh)/∂(ᶠw_data) = ∂(ᶜp)/∂(ᶠw_data) =
                    #     ∂(ᶜp)/∂(ᶜK) * ∂(ᶜK)/∂(ᶠw_data)
                    # If we ignore the dependence of pressure on moisture,
                    # ∂(ᶜp)/∂(ᶜK) = -ᶜρ * R_d / cv_d
                    @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 -= compose(
                        ᶜdivᵥ_stencil(ᶠw[colidx]),
                        compose(
                            ᶠinterp_stencil(one(ᶜp[colidx])),
                            -(ᶜρ[colidx] * R_d / cv_d) * ∂ᶜK∂ᶠw_data[colidx],
                        ),
                    )
                end
            end
        elseif :ρe_int in propertynames(Y.c)
            (; ᶜρh) = p
            @. ᶜρh[colidx] = Y.c.ρe_int[colidx] + ᶜp[colidx]
            # vertical_transport!(Yₜ.c.ρe_int, ᶠw, ᶜρ, ᶜρh, dt, energy_upwinding)
            # ᶜρe_intₜ += ᶜinterp(dot(ᶠgradᵥ(ᶜp), Geometry.Contravariant3Vector(ᶠw))
            vertical_transport_jac!(
                ∂ᶜ𝔼ₜ∂ᶠ𝕄[colidx],
                ᶠw[colidx],
                ᶜρ[colidx],
                ᶜρh[colidx],
                energy_upwinding,
            )
            @. ∂ᶜ𝔼ₜ∂ᶠ𝕄[colidx] += ᶜinterp_stencil(
                dot(
                    ᶠgradᵥ(ᶜp[colidx]),
                    Geometry.Contravariant3Vector(one(ᶠw[colidx])),
                ),
            )
        end

        if :ρθ in propertynames(Y.c)
            # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ_ᶜΦ
            # ∂(ᶠwₜ)/∂(ᶜρθ) = ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρθ)
            # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
            # If we ignore the dependence of pressure on moisture,
            # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρθ) =
            #     ᶠgradᵥ_stencil(
            #         R_d / (1 - κ_d) * (ᶜρθ * R_d / MSLP)^(κ_d / (1 - κ_d))
            #     )
            @. ∂ᶠ𝕄ₜ∂ᶜ𝔼[colidx] = to_scalar_coefs(
                -1 / ᶠinterp(ᶜρ[colidx]) * ᶠgradᵥ_stencil(
                    R_d / (1 - κ_d) *
                    (ᶜρθ[colidx] * R_d / MSLP)^(κ_d / (1 - κ_d)),
                ),
            )

            if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
                # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ_ᶜΦ
                # ∂(ᶠwₜ)/∂(ᶜρ) = ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) * ∂(ᶠinterp(ᶜρ))/∂(ᶜρ)
                # ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) = ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2
                # ∂(ᶠinterp(ᶜρ))/∂(ᶜρ) = ᶠinterp_stencil(1)
                @. ∂ᶠ𝕄ₜ∂ᶜρ[colidx] = to_scalar_coefs(
                    ᶠgradᵥ(ᶜp[colidx]) / ᶠinterp(ᶜρ[colidx])^2 *
                    ᶠinterp_stencil(one(ᶜρ[colidx])),
                )
            elseif flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :gradΦ_shenanigans
                # ᶠwₜ = (
                #     -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ′) -
                #     ᶠgradᵥ_ᶜΦ / ᶠinterp(ᶜρ′) * ᶠinterp(ᶜρ)
                # ), where ᶜρ′ = ᶜρ but we approximate ∂(ᶜρ′)/∂(ᶜρ) = 0
                @. ∂ᶠ𝕄ₜ∂ᶜρ[colidx] = to_scalar_coefs(
                    -(ᶠgradᵥ_ᶜΦ[colidx]) / ᶠinterp(ᶜρ[colidx]) *
                    ᶠinterp_stencil(one(ᶜρ[colidx])),
                )
            end
        elseif :ρe_tot in propertynames(Y.c)
            # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ_ᶜΦ
            # ∂(ᶠwₜ)/∂(ᶜρe) = ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe)
            # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
            # If we ignore the dependence of pressure on moisture,
            # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe) = ᶠgradᵥ_stencil(R_d / cv_d)
            @. ∂ᶠ𝕄ₜ∂ᶜ𝔼[colidx] = to_scalar_coefs(
                -1 / ᶠinterp(ᶜρ[colidx]) *
                ᶠgradᵥ_stencil(R_d / cv_d * one(ᶜρe[colidx])),
            )

            if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
                # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ_ᶜΦ
                # ∂(ᶠwₜ)/∂(ᶜρ) =
                #     ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) +
                #     ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) * ∂(ᶠinterp(ᶜρ))/∂(ᶜρ)
                # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
                # If we ignore the dependence of pressure on moisture,
                # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) =
                #     ᶠgradᵥ_stencil(R_d * (-(ᶜK + ᶜΦ) / cv_d + T_tri))
                # ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) = ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2
                # ∂(ᶠinterp(ᶜρ))/∂(ᶜρ) = ᶠinterp_stencil(1)
                @. ∂ᶠ𝕄ₜ∂ᶜρ[colidx] = to_scalar_coefs(
                    -1 / ᶠinterp(ᶜρ[colidx]) * ᶠgradᵥ_stencil(
                        R_d * (-(ᶜK[colidx] + ᶜΦ[colidx]) / cv_d + T_tri),
                    ) +
                    ᶠgradᵥ(ᶜp[colidx]) / ᶠinterp(ᶜρ[colidx])^2 *
                    ᶠinterp_stencil(one(ᶜρ[colidx])),
                )
            elseif flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :gradΦ_shenanigans
                # ᶠwₜ = (
                #     -ᶠgradᵥ(ᶜp′) / ᶠinterp(ᶜρ′) -
                #     ᶠgradᵥ_ᶜΦ / ᶠinterp(ᶜρ′) * ᶠinterp(ᶜρ)
                # ), where ᶜρ′ = ᶜρ but we approximate ∂ᶜρ′/∂ᶜρ = 0, and where
                # ᶜp′ = ᶜp but with ᶜK = 0
                @. ∂ᶠ𝕄ₜ∂ᶜρ[colidx] = to_scalar_coefs(
                    -1 / ᶠinterp(ᶜρ[colidx]) *
                    ᶠgradᵥ_stencil(R_d * (-(ᶜΦ[colidx]) / cv_d + T_tri)) -
                    ᶠgradᵥ_ᶜΦ[colidx] / ᶠinterp(ᶜρ[colidx]) *
                    ᶠinterp_stencil(one(ᶜρ[colidx])),
                )
            end
        elseif :ρe_int in propertynames(Y.c)
            ᶜρe_int = Y.c.ρe_int
            # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ_ᶜΦ
            # ∂(ᶠwₜ)/∂(ᶜρe_int) = ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe_int)
            # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
            # If we ignore the dependence of pressure on moisture,
            # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe_int) = ᶠgradᵥ_stencil(R_d / cv_d)
            @. ∂ᶠ𝕄ₜ∂ᶜ𝔼[colidx] = to_scalar_coefs(
                -1 / ᶠinterp(ᶜρ[colidx]) *
                ᶠgradᵥ_stencil(R_d / cv_d * one(ᶜρe_int[colidx])),
            )

            if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
                # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ_ᶜΦ
                # ∂(ᶠwₜ)/∂(ᶜρ) =
                #     ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) +
                #     ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) * ∂(ᶠinterp(ᶜρ))/∂(ᶜρ)
                # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
                # If we ignore the dependence of pressure on moisture,
                # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) = ᶠgradᵥ_stencil(R_d * T_tri)
                # ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) = ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2
                # ∂(ᶠinterp(ᶜρ))/∂(ᶜρ) = ᶠinterp_stencil(1)
                @. ∂ᶠ𝕄ₜ∂ᶜρ[colidx] = to_scalar_coefs(
                    -1 / ᶠinterp(ᶜρ[colidx]) *
                    ᶠgradᵥ_stencil(R_d * T_tri * one(ᶜρe_int[colidx])) +
                    ᶠgradᵥ(ᶜp[colidx]) / ᶠinterp(ᶜρ[colidx])^2 *
                    ᶠinterp_stencil(one(ᶜρ[colidx])),
                )
            elseif flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :gradΦ_shenanigans
                # ᶠwₜ = (
                #     -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ′) -
                #     ᶠgradᵥ_ᶜΦ / ᶠinterp(ᶜρ′) * ᶠinterp(ᶜρ)
                # ), where ᶜp′ = ᶜp but we approximate ∂ᶜρ′/∂ᶜρ = 0
                @. ∂ᶠ𝕄ₜ∂ᶜρ[colidx] = to_scalar_coefs(
                    -1 / ᶠinterp(ᶜρ[colidx]) *
                    ᶠgradᵥ_stencil(R_d * T_tri * one(ᶜρe_int[colidx])) -
                    ᶠgradᵥ_ᶜΦ[colidx] / ᶠinterp(ᶜρ[colidx]) *
                    ᶠinterp_stencil(one(ᶜρ[colidx])),
                )
            end
        end

        # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ_ᶜΦ
        # ∂(ᶠwₜ)/∂(ᶠw_data) =
        #     ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶠw_dataₜ) =
        #     ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜK) * ∂(ᶜK)/∂(ᶠw_dataₜ)
        # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
        # If we ignore the dependence of pressure on moisture,
        # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜK) =
        #     ᶜ𝔼_name == :ρe_tot ? ᶠgradᵥ_stencil(-ᶜρ * R_d / cv_d) : 0
        if :ρθ in propertynames(Y.c) || :ρe_int in propertynames(Y.c)
            ∂ᶠ𝕄ₜ∂ᶠ𝕄[colidx] .=
                Ref(Operators.StencilCoefs{-1, 1}((FT(0), FT(0), FT(0))))
        elseif :ρe_tot in propertynames(Y.c)
            @. ∂ᶠ𝕄ₜ∂ᶠ𝕄[colidx] = to_scalar_coefs(
                compose(
                    -1 / ᶠinterp(ᶜρ[colidx]) *
                    ᶠgradᵥ_stencil(-(ᶜρ[colidx] * R_d / cv_d)),
                    ∂ᶜK∂ᶠw_data[colidx],
                ),
            )
        end

        if p.tendency_knobs.rayleigh_sponge
            # ᶠwₜ -= p.ᶠβ_rayleigh_w * ᶠw
            # ∂(ᶠwₜ)/∂(ᶠw_data) -= p.ᶠβ_rayleigh_w
            @. ∂ᶠ𝕄ₜ∂ᶠ𝕄[colidx].coefs.:2 -= p.ᶠβ_rayleigh_w[colidx]
        end

        for ᶜρc_name in filter(is_tracer_var, propertynames(Y.c))
            ∂ᶜρcₜ∂ᶠ𝕄 = getproperty(∂ᶜ𝕋ₜ∂ᶠ𝕄_field, ᶜρc_name)
            ᶜρc = getproperty(Y.c, ᶜρc_name)
            # vertical_transport!(ᶜρcₜ, ᶠw, ᶜρ, ᶜρc, dt, tracer_upwinding)
            vertical_transport_jac!(
                ∂ᶜρcₜ∂ᶠ𝕄[colidx],
                ᶠw[colidx],
                ᶜρ[colidx],
                ᶜρc[colidx],
                tracer_upwinding,
            )
        end
    end

    # TODO: Figure out a way to test the Jacobian when the thermodynamic
    # state is PhaseEquil (i.e., when the implicit tendency calls saturation
    # adjustment).
    if call_verify_wfact_matrix()
        verify_wfact_matrix(W, Y, p, dtγ, t)
    end
end

function verify_wfact_matrix(W, Y, p, dtγ, t)
    (; ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶠ𝕄ₜ∂ᶠ𝕄, ∂ᶜ𝕋ₜ∂ᶠ𝕄_field) = W
    (; ᶜts) = p

    if eltype(ᶜts) <: TD.PhaseEquil
        error("This function is incompatible with $(typeof(ᶜts))")
    end

    # Checking every column takes too long, so just check one.
    i, j, h = 1, 1, 1
    args = (implicit_tendency!, Y, p, t, i, j, h)
    ᶜ𝔼_name = filter(is_energy_var, propertynames(Y.c))[1]

    @assert matrix_column(∂ᶜρₜ∂ᶠ𝕄, axes(Y.f), i, j, h) ≈
            exact_column_jacobian_block(args..., (:c, :ρ), (:f, :w))
    @assert matrix_column(∂ᶠ𝕄ₜ∂ᶜ𝔼, axes(Y.c), i, j, h) ≈
            exact_column_jacobian_block(args..., (:f, :w), (:c, ᶜ𝔼_name))
    @assert matrix_column(∂ᶠ𝕄ₜ∂ᶠ𝕄, axes(Y.f), i, j, h) ≈
            exact_column_jacobian_block(args..., (:f, :w), (:f, :w))
    for ᶜρc_name in filter(is_tracer_var, propertynames(Y.c))
        ∂ᶜρcₜ∂ᶠ𝕄 = getproperty(∂ᶜ𝕋ₜ∂ᶠ𝕄_field, ᶜρc_name)
        ᶜρc_tuple = (:c, ᶜρc_name)
        @assert matrix_column(∂ᶜρcₜ∂ᶠ𝕄, axes(Y.f), i, j, h) ≈
                exact_column_jacobian_block(args..., ᶜρc_tuple, (:f, :w))
    end

    ∂ᶜ𝔼ₜ∂ᶠ𝕄_approx = matrix_column(∂ᶜ𝔼ₜ∂ᶠ𝕄, axes(Y.f), i, j, h)
    ∂ᶜ𝔼ₜ∂ᶠ𝕄_exact =
        exact_column_jacobian_block(args..., (:c, ᶜ𝔼_name), (:f, :w))
    if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact
        @assert ∂ᶜ𝔼ₜ∂ᶠ𝕄_approx ≈ ∂ᶜ𝔼ₜ∂ᶠ𝕄_exact
    else
        err = norm(∂ᶜ𝔼ₜ∂ᶠ𝕄_approx .- ∂ᶜ𝔼ₜ∂ᶠ𝕄_exact) / norm(∂ᶜ𝔼ₜ∂ᶠ𝕄_exact)
        @assert err < 1e-6
        # Note: the highest value seen so far is ~3e-7 (only applies to ρe_tot)
    end

    ∂ᶠ𝕄ₜ∂ᶜρ_approx = matrix_column(∂ᶠ𝕄ₜ∂ᶜρ, axes(Y.c), i, j, h)
    ∂ᶠ𝕄ₜ∂ᶜρ_exact = exact_column_jacobian_block(args..., (:f, :w), (:c, :ρ))
    if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
        @assert ∂ᶠ𝕄ₜ∂ᶜρ_approx ≈ ∂ᶠ𝕄ₜ∂ᶜρ_exact
    else
        err = norm(∂ᶠ𝕄ₜ∂ᶜρ_approx .- ∂ᶠ𝕄ₜ∂ᶜρ_exact) / norm(∂ᶠ𝕄ₜ∂ᶜρ_exact)
        @assert err < 0.03
        # Note: the highest value seen so far for ρe_tot is ~0.01, and the
        # highest value seen so far for ρθ is ~0.02
    end
end
