using LinearAlgebra: ×, norm, norm_sqr, dot

import ClimaAtmos.Parameters as CAP
using ClimaCore: Operators, Fields

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
const ᶠcurlᵥ = Operators.CurlC2F(
    bottom = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
    top = Operators.SetCurl(Geometry.Contravariant12Vector(FT(0), FT(0))),
)
const ᶜFC = Operators.FluxCorrectionC2C(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
const ᶠupwind_product1 = Operators.UpwindBiasedProductC2F()
const ᶠupwind_product3 = Operators.Upwind3rdOrderBiasedProductC2F(
    bottom = Operators.FirstOrderOneSided(),
    top = Operators.FirstOrderOneSided(),
)

const ᶜinterp_stencil = Operators.Operator2Stencil(ᶜinterp)
const ᶠinterp_stencil = Operators.Operator2Stencil(ᶠinterp)
const ᶜdivᵥ_stencil = Operators.Operator2Stencil(ᶜdivᵥ)
const ᶠgradᵥ_stencil = Operators.Operator2Stencil(ᶠgradᵥ)

const C123 = Geometry.Covariant123Vector

include("thermo_state.jl")

get_cache(Y, params, spaces, model_spec, numerics, simulation, dt) = merge(
    default_cache(Y, params, spaces, numerics, simulation),
    additional_cache(Y, params, model_spec, dt),
)

function default_cache(Y, params, spaces, numerics, simulation)
    (; upwinding_mode) = numerics
    ᶜcoord = Fields.local_geometry_field(Y.c).coordinates
    ᶠcoord = Fields.local_geometry_field(Y.f).coordinates
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
    return (;
        simulation,
        spaces,
        ᶜuvw = similar(Y.c, Geometry.Covariant123Vector{FT}),
        ᶜK = similar(Y.c, FT),
        ᶜΦ = CAP.grav(params) .* ᶜcoord.z,
        ᶜts = similar(Y.c, ts_type),
        ᶜp = similar(Y.c, FT),
        ᶜω³ = similar(Y.c, Geometry.Contravariant3Vector{FT}),
        ᶠω¹² = similar(Y.f, Geometry.Contravariant12Vector{FT}),
        ᶠu¹² = similar(Y.f, Geometry.Contravariant12Vector{FT}),
        ᶠu³ = similar(Y.f, Geometry.Contravariant3Vector{FT}),
        ᶜf,
        z_sfc,
        T_sfc,
        ∂ᶜK∂ᶠw_data = similar(
            Y.c,
            Operators.StencilCoefs{-half, half, NTuple{2, FT}},
        ),
        params,
        ᶠupwind_product = upwinding_mode == :first_order ? ᶠupwind_product1 :
                          upwinding_mode == :third_order ? ᶠupwind_product3 :
                          nothing,
        ghost_buffer = ghost_buffer,
    )
end

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

function implicit_tendency_special!(Yₜ, Y, p, t)
    (; apply_moisture_filter) = p
    apply_moisture_filter && affect_filter!(Y)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜΦ, params, ᶠupwind_product) = p
    thermo_params = CAP.thermodynamics_params(params)
    # Used for automatically computing the Jacobian ∂Yₜ/∂Y. Currently requires
    # allocation because the cache is stored separately from Y, which means that
    # similar(Y, <:Dual) doesn't allocate an appropriate cache for computing Yₜ.
    (; ᶜK, ᶜts, ᶜp) = implicit_cache_vars(Y, p)
    @nvtx "implicit tendency special" color = colorant"yellow" begin
        Fields.bycolumn(axes(Y.c)) do colidx

            @. ᶜK[colidx] =
                norm_sqr(C123(ᶜuₕ[colidx]) + C123(ᶜinterp(ᶠw[colidx]))) / 2

            @. Yₜ.c.ρ[colidx] = -(ᶜdivᵥ(ᶠinterp(ᶜρ[colidx]) * ᶠw[colidx]))

            thermo_state!(
                ᶜts[colidx],
                Y.c[colidx],
                params,
                ᶜinterp,
                ᶜK[colidx],
                Y.f.w[colidx],
            )
            @. ᶜp[colidx] = TD.air_pressure(thermo_params, ᶜts[colidx])
            if isnothing(ᶠupwind_product)
                @. Yₜ.c.ρe_tot[colidx] = -(ᶜdivᵥ(
                    ᶠinterp(Y.c.ρe_tot[colidx] + ᶜp[colidx]) * ᶠw[colidx],
                ))
            else
                @. Yₜ.c.ρe_tot[colidx] = -(ᶜdivᵥ(
                    ᶠinterp(Y.c.ρ[colidx]) * ᶠupwind_product(
                        ᶠw[colidx],
                        (Y.c.ρe_tot[colidx] + ᶜp[colidx]) / Y.c.ρ[colidx],
                    ),
                ))
            end

            # TODO: Add flux correction to the Jacobian
            # @. Yₜ.c.ρ += ᶜFC(ᶠw, ᶜρ)
            # if :ρθ in propertynames(Y.c)
            #     @. Yₜ.c.ρθ += ᶜFC(ᶠw, ᶜρθ)
            # elseif :ρe_tot in propertynames(Y.c)
            #     @. Yₜ.c.ρe_tot += ᶜFC(ᶠw, ᶜρe)
            # elseif :ρe_int in propertynames(Y.c)
            #     @. Yₜ.c.ρe_int += ᶜFC(ᶠw, ᶜρe_int)
            # end

            Yₜ.c.uₕ[colidx] .= Ref(zero(eltype(Yₜ.c.uₕ)))

            @. Yₜ.f.w[colidx] = -(
                ᶠgradᵥ(ᶜp[colidx]) / ᶠinterp(ᶜρ[colidx]) +
                ᶠgradᵥ(ᶜK[colidx] + ᶜΦ[colidx])
            )

            for ᶜ𝕋_name in filter(is_tracer_var, propertynames(Y.c))
                ᶜ𝕋 = getproperty(Y.c, ᶜ𝕋_name)
                ᶜ𝕋ₜ = getproperty(Yₜ.c, ᶜ𝕋_name)
                if isnothing(ᶠupwind_product)
                    @. ᶜ𝕋ₜ[colidx] = -(ᶜdivᵥ(ᶠinterp(ᶜ𝕋[colidx]) * ᶠw[colidx]))
                else
                    @. ᶜ𝕋ₜ[colidx] = -(ᶜdivᵥ(
                        ᶠinterp(Y.c.ρ[colidx]) * ᶠupwind_product(
                            ᶠw[colidx],
                            ᶜ𝕋[colidx] / Y.c.ρ[colidx],
                        ),
                    ))
                end
            end
        end
    end
    return Yₜ
end

function implicit_tendency_generic!(Yₜ, Y, p, t)
    (; apply_moisture_filter) = p
    apply_moisture_filter && affect_filter!(Y)
    @nvtx "implicit tendency" color = colorant"yellow" begin
        ᶜρ = Y.c.ρ
        ᶜuₕ = Y.c.uₕ
        ᶠw = Y.f.w
        (; ᶜK, ᶜΦ, ᶜts, ᶜp, params, ᶠupwind_product) = p
        thermo_params = CAP.thermodynamics_params(params)
        # Used for automatically computing the Jacobian ∂Yₜ/∂Y. Currently requires
        # allocation because the cache is stored separately from Y, which means that
        # similar(Y, <:Dual) doesn't allocate an appropriate cache for computing Yₜ.
        if eltype(Y) <: Dual
            ᶜK = similar(ᶜρ)
            ᶜts = similar(ᶜρ, eltype(ᶜts).name.wrapper{eltype(ᶜρ)})
            ᶜp = similar(ᶜρ)
        end

        @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2

        @. Yₜ.c.ρ = -(ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠw))

        thermo_state!(ᶜts, Y, params, ᶜinterp, ᶜK)
        @. ᶜp = TD.air_pressure(thermo_params, ᶜts)
        if :ρθ in propertynames(Y.c)
            if isnothing(ᶠupwind_product)
                @. Yₜ.c.ρθ = -(ᶜdivᵥ(ᶠinterp(Y.c.ρθ) * ᶠw))
            else
                @. Yₜ.c.ρθ = -(ᶜdivᵥ(
                    ᶠinterp(Y.c.ρ) * ᶠupwind_product(ᶠw, Y.c.ρθ / Y.c.ρ),
                ))
            end
        elseif :ρe_tot in propertynames(Y.c)
            if isnothing(ᶠupwind_product)
                @. Yₜ.c.ρe_tot = -(ᶜdivᵥ(ᶠinterp(Y.c.ρe_tot + ᶜp) * ᶠw))
            else
                @. Yₜ.c.ρe_tot = -(ᶜdivᵥ(
                    ᶠinterp(Y.c.ρ) *
                    ᶠupwind_product(ᶠw, (Y.c.ρe_tot + ᶜp) / Y.c.ρ),
                ))
            end
        elseif :ρe_int in propertynames(Y.c)
            if isnothing(ᶠupwind_product)
                @. Yₜ.c.ρe_int = -(
                    ᶜdivᵥ(ᶠinterp(Y.c.ρe_int + ᶜp) * ᶠw) - ᶜinterp(
                        dot(ᶠgradᵥ(ᶜp), Geometry.Contravariant3Vector(ᶠw)),
                    )
                )
                # or, equivalently,
                # Yₜ.c.ρe_int = -(ᶜdivᵥ(ᶠinterp(Y.c.ρe_int) * ᶠw) + ᶜp * ᶜdivᵥ(ᶠw))
            else
                @. Yₜ.c.ρe_int = -(
                    ᶜdivᵥ(
                        ᶠinterp(Y.c.ρ) *
                        ᶠupwind_product(ᶠw, (Y.c.ρe_int + ᶜp) / Y.c.ρ),
                    ) - ᶜinterp(
                        dot(ᶠgradᵥ(ᶜp), Geometry.Contravariant3Vector(ᶠw)),
                    )
                )
            end
        end

        # TODO: Add flux correction to the Jacobian
        # @. Yₜ.c.ρ += ᶜFC(ᶠw, ᶜρ)
        # if :ρθ in propertynames(Y.c)
        #     @. Yₜ.c.ρθ += ᶜFC(ᶠw, ᶜρθ)
        # elseif :ρe_tot in propertynames(Y.c)
        #     @. Yₜ.c.ρe_tot += ᶜFC(ᶠw, ᶜρe)
        # elseif :ρe_int in propertynames(Y.c)
        #     @. Yₜ.c.ρe_int += ᶜFC(ᶠw, ᶜρe_int)
        # end

        Yₜ.c.uₕ .= Ref(zero(eltype(Yₜ.c.uₕ)))

        @. Yₜ.f.w = -(ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) + ᶠgradᵥ(ᶜK + ᶜΦ))

        for ᶜ𝕋_name in filter(is_tracer_var, propertynames(Y.c))
            ᶜ𝕋 = getproperty(Y.c, ᶜ𝕋_name)
            ᶜ𝕋ₜ = getproperty(Yₜ.c, ᶜ𝕋_name)
            if isnothing(ᶠupwind_product)
                @. ᶜ𝕋ₜ = -(ᶜdivᵥ(ᶠinterp(ᶜ𝕋) * ᶠw))
            else
                @. ᶜ𝕋ₜ =
                    -(ᶜdivᵥ(ᶠinterp(Y.c.ρ) * ᶠupwind_product(ᶠw, ᶜ𝕋 / Y.c.ρ)))
            end
        end
    end
    return Yₜ
end

function remaining_tendency!(Yₜ, Y, p, t)
    @nvtx "remaining tendency" color = colorant"yellow" begin
        (; enable_default_remaining_tendency) = p
        Yₜ .= zero(eltype(Yₜ))
        if enable_default_remaining_tendency
            default_remaining_tendency!(Yₜ, Y, p, t)
        end
        additional_tendency!(Yₜ, Y, p, t)
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

function default_remaining_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜuvw, ᶜK, ᶜΦ, ᶜts, ᶜp, ᶜω³, ᶠω¹², ᶠu¹², ᶠu³, ᶜf, params) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)
    thermo_params = CAP.thermodynamics_params(params)

    @. ᶜuvw = C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))
    @. ᶜK = norm_sqr(ᶜuvw) / 2

    # Mass conservation

    @. Yₜ.c.ρ -= divₕ(ᶜρ * ᶜuvw)
    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(ᶜρ * ᶜuₕ))

    # Energy conservation

    thermo_state!(ᶜts, Y, params, ᶜinterp, ᶜK)
    @. ᶜp = TD.air_pressure(thermo_params, ᶜts)
    if :ρθ in propertynames(Y.c)
        @. Yₜ.c.ρθ -= divₕ(Y.c.ρθ * ᶜuvw)
        @. Yₜ.c.ρθ -= ᶜdivᵥ(ᶠinterp(Y.c.ρθ * ᶜuₕ))
    elseif :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot -= divₕ((Y.c.ρe_tot + ᶜp) * ᶜuvw)
        @. Yₜ.c.ρe_tot -= ᶜdivᵥ(ᶠinterp((Y.c.ρe_tot + ᶜp) * ᶜuₕ))
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
        @. Yₜ.c.ρe_int -= ᶜdivᵥ(ᶠinterp((Y.c.ρe_int + ᶜp) * ᶜuₕ))
        # or, equivalently,
        # @. Yₜ.c.ρe_int -= divₕ(Y.c.ρe_int * ᶜuvw) + ᶜp * divₕ(ᶜuvw)
        # @. Yₜ.c.ρe_int -=
        #     ᶜdivᵥ(ᶠinterp(Y.c.ρe_int * ᶜuₕ)) + ᶜp * ᶜdivᵥ(ᶠinterp(ᶜuₕ))
    end

    # Momentum conservation

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜω³ = curlₕ(ᶜuₕ)
        @. ᶠω¹² = curlₕ(ᶠw)
    elseif point_type <: Geometry.Abstract2DPoint
        ᶜω³ .= Ref(zero(eltype(ᶜω³)))
        @. ᶠω¹² = Geometry.Contravariant12Vector(curlₕ(ᶠw))
    end
    @. ᶠω¹² += ᶠcurlᵥ(ᶜuₕ)

    @. ᶠu¹² = Geometry.project(Geometry.Contravariant12Axis(), ᶠinterp(ᶜuvw))
    @. ᶠu³ = Geometry.project(
        Geometry.Contravariant3Axis(),
        C123(ᶠinterp(ᶜuₕ)) + C123(ᶠw),
    )

    @. Yₜ.c.uₕ -=
        ᶜinterp(ᶠω¹² × ᶠu³) +
        (ᶜf + ᶜω³) × (Geometry.project(Geometry.Contravariant12Axis(), ᶜuvw))
    if point_type <: Geometry.Abstract3DPoint
        @. Yₜ.c.uₕ -= gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + ᶜΦ)
    elseif point_type <: Geometry.Abstract2DPoint
        @. Yₜ.c.uₕ -=
            Geometry.Covariant12Vector(gradₕ(ᶜp) / ᶜρ + gradₕ(ᶜK + ᶜΦ))
    end

    @. Yₜ.f.w -= ᶠω¹² × ᶠu¹²

    # Tracer conservation

    for ᶜ𝕋_name in filter(is_tracer_var, propertynames(Y.c))
        ᶜ𝕋 = getproperty(Y.c, ᶜ𝕋_name)
        ᶜ𝕋ₜ = getproperty(Yₜ.c, ᶜ𝕋_name)
        @. ᶜ𝕋ₜ -= divₕ(ᶜ𝕋 * ᶜuvw)
        @. ᶜ𝕋ₜ -= ᶜdivᵥ(ᶠinterp(ᶜ𝕋 * ᶜuₕ))
    end
end

# Allow one() to be called on vectors.
Base.one(::T) where {T <: Geometry.AxisTensor} = one(T)
Base.one(::Type{T}) where {T′, A, S, T <: Geometry.AxisTensor{T′, 1, A, S}} =
    T(axes(T), S(one(T′)))

# :ρe_tot in propertynames(Y.c) && flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :no_∂ᶜp∂ᶜK && flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
function Wfact_special!(W, Y, p, dtγ, t)
    (; apply_moisture_filter) = p
    apply_moisture_filter && affect_filter!(Y)
    (; flags, dtγ_ref) = W
    (; ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶠ𝕄ₜ∂ᶠ𝕄, ∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple) = W
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶜts, ᶜp, ∂ᶜK∂ᶠw_data, params, ᶠupwind_product) = p
    @nvtx "Wfact!" color = colorant"green" begin
        thermo_params = CAP.thermodynamics_params(params)

        R_d = FT(CAP.R_d(params))
        κ_d = FT(CAP.kappa_d(params))
        cv_d = FT(CAP.cv_d(params))
        T_tri = FT(CAP.T_triple(params))
        MSLP = FT(CAP.MSLP(params))

        dtγ_ref[] = dtγ

        # If we let ᶠw_data = ᶠw.components.data.:1 and ᶠw_unit = one.(ᶠw), then
        # ᶠw == ᶠw_data .* ᶠw_unit. The Jacobian blocks involve ᶠw_data, not ᶠw.
        ᶠw_data = ᶠw.components.data.:1

        # If ∂(ᶜarg)/∂(ᶠw_data) = 0, then
        # ∂(ᶠupwind_product(ᶠw, ᶜarg))/∂(ᶠw_data) =
        #     ᶠupwind_product(ᶠw + εw, arg) / to_scalar(ᶠw + εw).
        # The εw is only necessary in case w = 0.
        εw = Ref(Geometry.Covariant3Vector(eps(FT)))
        to_scalar(vector) = vector.u₃

        to_scalar_coefs(vector_coefs) =
            map(vector_coef -> vector_coef.u₃, vector_coefs)


        Fields.bycolumn(axes(Y.c)) do colidx
            @. ∂ᶜK∂ᶠw_data[colidx] =
                ᶜinterp(ᶠw_data[colidx]) *
                norm_sqr(one(ᶜinterp(ᶠw[colidx]))) *
                ᶜinterp_stencil(one(ᶠw_data[colidx]))
            @. ∂ᶜρₜ∂ᶠ𝕄[colidx] =
                -(ᶜdivᵥ_stencil(ᶠinterp(ᶜρ[colidx]) * one(ᶠw[colidx])))

            # elseif :ρe_tot in propertynames(Y.c)
            ᶜρe = Y.c.ρe_tot
            @. ᶜK[colidx] =
                norm_sqr(C123(ᶜuₕ[colidx]) + C123(ᶜinterp(ᶠw[colidx]))) / 2
            thermo_state!(
                ᶜts[colidx],
                Y.c[colidx],
                params,
                ᶜinterp,
                ᶜK[colidx],
                ᶠw[colidx],
            )
            @. ᶜp[colidx] = TD.air_pressure(thermo_params, ᶜts[colidx])

            if isnothing(ᶠupwind_product)
                #         elseif flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :no_∂ᶜp∂ᶜK
                #             # same as above, but we approximate ∂(ᶜp)/∂(ᶜK) = 0, so that
                #             # ∂ᶜ𝔼ₜ∂ᶠ𝕄 has 3 diagonals instead of 5
                @. ∂ᶜ𝔼ₜ∂ᶠ𝕄[colidx] = -(ᶜdivᵥ_stencil(
                    ᶠinterp(ᶜρe[colidx] + ᶜp[colidx]) * one(ᶠw[colidx]),
                ))
            else
                #         if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :no_∂ᶜp∂ᶜK
                @. ∂ᶜ𝔼ₜ∂ᶠ𝕄[colidx] = -(ᶜdivᵥ_stencil(
                    ᶠinterp(ᶜρ[colidx]) * ᶠupwind_product(
                        ᶠw[colidx] + εw,
                        (ᶜρe[colidx] + ᶜp[colidx]) / ᶜρ[colidx],
                    ) / to_scalar(ᶠw[colidx] + εw),
                ))
            end
            # elseif :ρe_tot in propertynames(Y.c)
            # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
            # ∂(ᶠwₜ)/∂(ᶜρe) = ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe)
            # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
            # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe) = ᶠgradᵥ_stencil(R_d / cv_d)
            @. ∂ᶠ𝕄ₜ∂ᶜ𝔼[colidx] = to_scalar_coefs(
                -1 / ᶠinterp(ᶜρ[colidx]) *
                ᶠgradᵥ_stencil(R_d / cv_d * one(ᶜρe[colidx])),
            )


            # if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
            # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
            # ∂(ᶠwₜ)/∂(ᶜρ) =
            #     ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) +
            #     ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) * ∂(ᶠinterp(ᶜρ))/∂(ᶜρ)
            # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
            # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) =
            #     ᶠgradᵥ_stencil(R_d * (-(ᶜK + ᶜΦ) / cv_d + T_tri))
            # ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) = ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2
            # ∂(ᶠinterp(ᶜρ))/∂(ᶜρ) = ᶠinterp_stencil(1)
            @. ∂ᶠ𝕄ₜ∂ᶜρ[colidx] = to_scalar_coefs(
                -1 / ᶠinterp(ᶜρ[colidx]) * ᶠgradᵥ_stencil(
                    R_d * (-(ᶜK[colidx] + ᶜΦ[colidx]) / cv_d + T_tri),
                ) +
                ᶠgradᵥ(ᶜp[colidx]) / abs2(ᶠinterp(ᶜρ[colidx])) *
                ᶠinterp_stencil(one(ᶜρ[colidx])),
            )

            # elseif :ρe_tot in propertynames(Y.c)
            @. ∂ᶠ𝕄ₜ∂ᶠ𝕄[colidx] = to_scalar_coefs(
                compose(
                    -1 / ᶠinterp(ᶜρ[colidx]) *
                    ᶠgradᵥ_stencil(-(ᶜρ[colidx] * R_d / cv_d)) +
                    -1 * ᶠgradᵥ_stencil(one(ᶜK[colidx])),
                    ∂ᶜK∂ᶠw_data[colidx],
                ),
            )

            for ᶜ𝕋_name in filter(is_tracer_var, propertynames(Y.c))
                ᶜ𝕋 = getproperty(Y.c, ᶜ𝕋_name)
                ∂ᶜ𝕋ₜ∂ᶠ𝕄 = getproperty(∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple, ᶜ𝕋_name)
                if isnothing(ᶠupwind_product)
                    # ᶜ𝕋ₜ = -ᶜdivᵥ(ᶠinterp(ᶜ𝕋) * ᶠw)
                    # ∂(ᶜ𝕋ₜ)/∂(ᶠw_data) = -ᶜdivᵥ_stencil(ᶠinterp(ᶜ𝕋) * ᶠw_unit)
                    @. ∂ᶜ𝕋ₜ∂ᶠ𝕄[colidx] =
                        -(ᶜdivᵥ_stencil(ᶠinterp(ᶜ𝕋[colidx]) * one(ᶠw[colidx])))
                else
                    # ᶜ𝕋ₜ = -ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind_product(ᶠw, ᶜ𝕋 / ᶜρ))
                    # ∂(ᶜ𝕋ₜ)/∂(ᶠw_data) =
                    #     -ᶜdivᵥ_stencil(
                    #         ᶠinterp(ᶜρ) * ∂(ᶠupwind_product(ᶠw, ᶜ𝕋 / ᶜρ))/∂(ᶠw_data),
                    #     )
                    @. ∂ᶜ𝕋ₜ∂ᶠ𝕄[colidx] = -(ᶜdivᵥ_stencil(
                        ᶠinterp(ᶜρ[colidx]) * ᶠupwind_product(
                            ᶠw[colidx] + εw,
                            ᶜ𝕋[colidx] / ᶜρ[colidx],
                        ) / to_scalar(ᶠw[colidx] + εw),
                    ))
                end
            end
        end
    end
end


function Wfact_generic!(W, Y, p, dtγ, t)
    (; apply_moisture_filter) = p
    apply_moisture_filter && affect_filter!(Y)
    (; flags, dtγ_ref) = W
    (; ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶠ𝕄ₜ∂ᶠ𝕄, ∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple) = W
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶜts, ᶜp, ∂ᶜK∂ᶠw_data, params, ᶠupwind_product) = p
    @nvtx "Wfact!" color = colorant"green" begin
        thermo_params = CAP.thermodynamics_params(params)

        R_d = FT(CAP.R_d(params))
        κ_d = FT(CAP.kappa_d(params))
        cv_d = FT(CAP.cv_d(params))
        T_tri = FT(CAP.T_triple(params))
        MSLP = FT(CAP.MSLP(params))

        dtγ_ref[] = dtγ

        # If we let ᶠw_data = ᶠw.components.data.:1 and ᶠw_unit = one.(ᶠw), then
        # ᶠw == ᶠw_data .* ᶠw_unit. The Jacobian blocks involve ᶠw_data, not ᶠw.
        ᶠw_data = ᶠw.components.data.:1

        # If ∂(ᶜarg)/∂(ᶠw_data) = 0, then
        # ∂(ᶠupwind_product(ᶠw, ᶜarg))/∂(ᶠw_data) =
        #     ᶠupwind_product(ᶠw + εw, arg) / to_scalar(ᶠw + εw).
        # The εw is only necessary in case w = 0.
        εw = Ref(Geometry.Covariant3Vector(eps(FT)))
        to_scalar(vector) = vector.u₃

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
        @. ∂ᶜK∂ᶠw_data =
            ᶜinterp(ᶠw_data) *
            norm_sqr(one(ᶜinterp(ᶠw))) *
            ᶜinterp_stencil(one(ᶠw_data))

        # ᶜρₜ = -ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠw)
        # ∂(ᶜρₜ)/∂(ᶠw_data) = -ᶜdivᵥ_stencil(ᶠinterp(ᶜρ) * ᶠw_unit)
        @. ∂ᶜρₜ∂ᶠ𝕄 = -(ᶜdivᵥ_stencil(ᶠinterp(ᶜρ) * one(ᶠw)))

        @. ᶜK = norm_sqr(C123(ᶜuₕ) + C123(ᶜinterp(ᶠw))) / 2
        thermo_state!(ᶜts, Y, params, ᶜinterp, ᶜK)
        @. ᶜp = TD.air_pressure(thermo_params, ᶜts)

        if :ρθ in propertynames(Y.c)
            ᶜρθ = Y.c.ρθ

            if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode != :exact
                error("∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :exact when using ρθ")
            end

            if isnothing(ᶠupwind_product)
                # ᶜρθₜ = -ᶜdivᵥ(ᶠinterp(ᶜρθ) * ᶠw)
                # ∂(ᶜρθₜ)/∂(ᶠw_data) = -ᶜdivᵥ_stencil(ᶠinterp(ᶜρθ) * ᶠw_unit)
                @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 = -(ᶜdivᵥ_stencil(ᶠinterp(ᶜρθ) * one(ᶠw)))
            else
                # ᶜρθₜ = -ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind_product(ᶠw, ᶜρθ / ᶜρ))
                # ∂(ᶜρθₜ)/∂(ᶠw_data) =
                #     -ᶜdivᵥ_stencil(
                #         ᶠinterp(ᶜρ) * ∂(ᶠupwind_product(ᶠw, ᶜρθ / ᶜρ))/∂(ᶠw_data),
                #     )
                @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 = -(ᶜdivᵥ_stencil(
                    ᶠinterp(ᶜρ) * ᶠupwind_product(ᶠw + εw, ᶜρθ / ᶜρ) /
                    to_scalar(ᶠw + εw),
                ))
            end
        elseif :ρe_tot in propertynames(Y.c)
            ᶜρe = Y.c.ρe_tot

            if isnothing(ᶠupwind_product)
                if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact
                    # ᶜρeₜ = -ᶜdivᵥ(ᶠinterp(ᶜρe + ᶜp) * ᶠw)
                    # ∂(ᶜρeₜ)/∂(ᶠw_data) =
                    #     -ᶜdivᵥ_stencil(ᶠinterp(ᶜρe + ᶜp) * ᶠw_unit) -
                    #     ᶜdivᵥ_stencil(ᶠw) * ∂(ᶠinterp(ᶜρe + ᶜp))/∂(ᶠw_data)
                    # ∂(ᶠinterp(ᶜρe + ᶜp))/∂(ᶠw_data) =
                    #     ∂(ᶠinterp(ᶜρe + ᶜp))/∂(ᶜp) * ∂(ᶜp)/∂(ᶠw_data)
                    # ∂(ᶠinterp(ᶜρe + ᶜp))/∂(ᶜp) = ᶠinterp_stencil(1)
                    # ∂(ᶜp)/∂(ᶠw_data) = ∂(ᶜp)/∂(ᶜK) * ∂(ᶜK)/∂(ᶠw_data)
                    # ∂(ᶜp)/∂(ᶜK) = -ᶜρ * R_d / cv_d
                    @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
                        -(ᶜdivᵥ_stencil(ᶠinterp(ᶜρe + ᶜp) * one(ᶠw))) - compose(
                            ᶜdivᵥ_stencil(ᶠw),
                            compose(
                                ᶠinterp_stencil(one(ᶜp)),
                                -(ᶜρ * R_d / cv_d) * ∂ᶜK∂ᶠw_data,
                            ),
                        )
                elseif flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :no_∂ᶜp∂ᶜK
                    # same as above, but we approximate ∂(ᶜp)/∂(ᶜK) = 0, so that
                    # ∂ᶜ𝔼ₜ∂ᶠ𝕄 has 3 diagonals instead of 5
                    @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 = -(ᶜdivᵥ_stencil(ᶠinterp(ᶜρe + ᶜp) * one(ᶠw)))
                else
                    error(
                        "∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :exact or :no_∂ᶜp∂ᶜK when using ρe_tot \
                        without upwinding",
                    )
                end
            else
                # TODO: Add Operator2Stencil for UpwindBiasedProductC2F to ClimaCore
                # to allow exact Jacobian calculation.
                if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :no_∂ᶜp∂ᶜK
                    # ᶜρeₜ =
                    #     -ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind_product(ᶠw, (ᶜρe + ᶜp) / ᶜρ))
                    # ∂(ᶜρeₜ)/∂(ᶠw_data) =
                    #     -ᶜdivᵥ_stencil(
                    #         ᶠinterp(ᶜρ) *
                    #         ∂(ᶠupwind_product(ᶠw, (ᶜρe + ᶜp) / ᶜρ))/∂(ᶠw_data),
                    #     )
                    @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 = -(ᶜdivᵥ_stencil(
                        ᶠinterp(ᶜρ) *
                        ᶠupwind_product(ᶠw + εw, (ᶜρe + ᶜp) / ᶜρ) /
                        to_scalar(ᶠw + εw),
                    ))
                else
                    error(
                        "∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :no_∂ᶜp∂ᶜK when using ρe_tot with \
                      upwinding",
                    )
                end
            end
        elseif :ρe_int in propertynames(Y.c)
            ᶜρe_int = Y.c.ρe_int

            if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode != :exact
                error("∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :exact when using ρe_int")
            end

            if isnothing(ᶠupwind_product)
                # ᶜρe_intₜ =
                #     -(
                #         ᶜdivᵥ(ᶠinterp(ᶜρe_int + ᶜp) * ᶠw) -
                #         ᶜinterp(dot(ᶠgradᵥ(ᶜp), Geometry.Contravariant3Vector(ᶠw))
                #     )
                # ∂(ᶜρe_intₜ)/∂(ᶠw_data) =
                #     -(
                #         ᶜdivᵥ_stencil(ᶠinterp(ᶜρe_int + ᶜp) * ᶠw_unit) -
                #         ᶜinterp_stencil(dot(
                #             ᶠgradᵥ(ᶜp),
                #             Geometry.Contravariant3Vector(ᶠw_unit),
                #         ),)
                #     )
                @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 = -(
                    ᶜdivᵥ_stencil(ᶠinterp(ᶜρe_int + ᶜp) * one(ᶠw)) -
                    ᶜinterp_stencil(
                        dot(ᶠgradᵥ(ᶜp), Geometry.Contravariant3Vector(one(ᶠw))),
                    )
                )
            else
                # ᶜρe_intₜ =
                #     -(
                #         ᶜdivᵥ(
                #             ᶠinterp(ᶜρ) * ᶠupwind_product(ᶠw, (ᶜρe_int + ᶜp) / ᶜρ),
                #         ) -
                #         ᶜinterp(dot(ᶠgradᵥ(ᶜp), Geometry.Contravariant3Vector(ᶠw)))
                #     )
                # ∂(ᶜρe_intₜ)/∂(ᶠw_data) =
                #     -(
                #         ᶜdivᵥ_stencil(
                #             ᶠinterp(ᶜρ) *
                #             ∂(ᶠupwind_product(ᶠw, (ᶜρe_int + ᶜp) / ᶜρ))/∂(ᶠw_data),
                #         ) -
                #         ᶜinterp_stencil(dot(
                #             ᶠgradᵥ(ᶜp),
                #             Geometry.Contravariant3Vector(ᶠw_unit),
                #         ),)
                #     )
                @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 = -(
                    ᶜdivᵥ_stencil(
                        ᶠinterp(ᶜρ) *
                        ᶠupwind_product(ᶠw + εw, (ᶜρe_int + ᶜp) / ᶜρ) /
                        to_scalar(ᶠw + εw),
                    ) - ᶜinterp_stencil(
                        dot(ᶠgradᵥ(ᶜp), Geometry.Contravariant3Vector(one(ᶠw))),
                    )
                )
            end
        end

        # To convert ∂(ᶠwₜ)/∂(ᶜ𝔼) to ∂(ᶠw_data)ₜ/∂(ᶜ𝔼) and ∂(ᶠwₜ)/∂(ᶠw_data) to
        # ∂(ᶠw_data)ₜ/∂(ᶠw_data), we must extract the third component of each
        # vector-valued stencil coefficient.
        to_scalar_coefs(vector_coefs) =
            map(vector_coef -> vector_coef.u₃, vector_coefs)

        # TODO: If we end up using :gradΦ_shenanigans, optimize it to
        # `cached_stencil / ᶠinterp(ᶜρ)`.
        if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode != :exact &&
           flags.∂ᶠ𝕄ₜ∂ᶜρ_mode != :gradΦ_shenanigans
            error("∂ᶠ𝕄ₜ∂ᶜρ_mode must be :exact or :gradΦ_shenanigans")
        end
        if :ρθ in propertynames(Y.c)
            # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
            # ∂(ᶠwₜ)/∂(ᶜρθ) = ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρθ)
            # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
            # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρθ) =
            #     ᶠgradᵥ_stencil(
            #         R_d / (1 - κ_d) * (ᶜρθ * R_d / MSLP)^(κ_d / (1 - κ_d))
            #     )
            @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 = to_scalar_coefs(
                -1 / ᶠinterp(ᶜρ) * ᶠgradᵥ_stencil(
                    R_d / (1 - κ_d) * (ᶜρθ * R_d / MSLP)^(κ_d / (1 - κ_d)),
                ),
            )

            if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
                # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
                # ∂(ᶠwₜ)/∂(ᶜρ) = ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) * ∂(ᶠinterp(ᶜρ))/∂(ᶜρ)
                # ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) = ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2
                # ∂(ᶠinterp(ᶜρ))/∂(ᶜρ) = ᶠinterp_stencil(1)
                @. ∂ᶠ𝕄ₜ∂ᶜρ = to_scalar_coefs(
                    ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2 * ᶠinterp_stencil(one(ᶜρ)),
                )
            elseif flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :gradΦ_shenanigans
                # ᶠwₜ = (
                #     -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ′) -
                #     ᶠgradᵥ(ᶜΦ) / ᶠinterp(ᶜρ′) * ᶠinterp(ᶜρ)
                # ), where ᶜρ′ = ᶜρ but we approximate ∂(ᶜρ′)/∂(ᶜρ) = 0
                @. ∂ᶠ𝕄ₜ∂ᶜρ = to_scalar_coefs(
                    -(ᶠgradᵥ(ᶜΦ)) / ᶠinterp(ᶜρ) * ᶠinterp_stencil(one(ᶜρ)),
                )
            end
        elseif :ρe_tot in propertynames(Y.c)
            # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
            # ∂(ᶠwₜ)/∂(ᶜρe) = ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe)
            # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
            # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe) = ᶠgradᵥ_stencil(R_d / cv_d)
            @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 = to_scalar_coefs(
                -1 / ᶠinterp(ᶜρ) * ᶠgradᵥ_stencil(R_d / cv_d * one(ᶜρe)),
            )

            if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
                # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
                # ∂(ᶠwₜ)/∂(ᶜρ) =
                #     ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) +
                #     ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) * ∂(ᶠinterp(ᶜρ))/∂(ᶜρ)
                # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
                # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) =
                #     ᶠgradᵥ_stencil(R_d * (-(ᶜK + ᶜΦ) / cv_d + T_tri))
                # ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) = ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2
                # ∂(ᶠinterp(ᶜρ))/∂(ᶜρ) = ᶠinterp_stencil(1)
                @. ∂ᶠ𝕄ₜ∂ᶜρ = to_scalar_coefs(
                    -1 / ᶠinterp(ᶜρ) *
                    ᶠgradᵥ_stencil(R_d * (-(ᶜK + ᶜΦ) / cv_d + T_tri)) +
                    ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2 * ᶠinterp_stencil(one(ᶜρ)),
                )
            elseif flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :gradΦ_shenanigans
                # ᶠwₜ = (
                #     -ᶠgradᵥ(ᶜp′) / ᶠinterp(ᶜρ′) -
                #     ᶠgradᵥ(ᶜΦ) / ᶠinterp(ᶜρ′) * ᶠinterp(ᶜρ)
                # ), where ᶜρ′ = ᶜρ but we approximate ∂ᶜρ′/∂ᶜρ = 0, and where
                # ᶜp′ = ᶜp but with ᶜK = 0
                @. ∂ᶠ𝕄ₜ∂ᶜρ = to_scalar_coefs(
                    -1 / ᶠinterp(ᶜρ) *
                    ᶠgradᵥ_stencil(R_d * (-(ᶜΦ) / cv_d + T_tri)) -
                    ᶠgradᵥ(ᶜΦ) / ᶠinterp(ᶜρ) * ᶠinterp_stencil(one(ᶜρ)),
                )
            end
        elseif :ρe_int in propertynames(Y.c)
            # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
            # ∂(ᶠwₜ)/∂(ᶜρe_int) = ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe_int)
            # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
            # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρe_int) = ᶠgradᵥ_stencil(R_d / cv_d)
            @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 = to_scalar_coefs(
                -1 / ᶠinterp(ᶜρ) * ᶠgradᵥ_stencil(R_d / cv_d * one(ᶜρe_int)),
            )

            if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
                # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
                # ∂(ᶠwₜ)/∂(ᶜρ) =
                #     ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) +
                #     ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) * ∂(ᶠinterp(ᶜρ))/∂(ᶜρ)
                # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
                # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜρ) = ᶠgradᵥ_stencil(R_d * T_tri)
                # ∂(ᶠwₜ)/∂(ᶠinterp(ᶜρ)) = ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2
                # ∂(ᶠinterp(ᶜρ))/∂(ᶜρ) = ᶠinterp_stencil(1)
                @. ∂ᶠ𝕄ₜ∂ᶜρ = to_scalar_coefs(
                    -1 / ᶠinterp(ᶜρ) *
                    ᶠgradᵥ_stencil(R_d * T_tri * one(ᶜρe_int)) +
                    ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ)^2 * ᶠinterp_stencil(one(ᶜρ)),
                )
            elseif flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :gradΦ_shenanigans
                # ᶠwₜ = (
                #     -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ′) -
                #     ᶠgradᵥ(ᶜΦ) / ᶠinterp(ᶜρ′) * ᶠinterp(ᶜρ)
                # ), where ᶜp′ = ᶜp but we approximate ∂ᶜρ′/∂ᶜρ = 0
                @. ∂ᶠ𝕄ₜ∂ᶜρ = to_scalar_coefs(
                    -1 / ᶠinterp(ᶜρ) *
                    ᶠgradᵥ_stencil(R_d * T_tri * one(ᶜρe_int)) -
                    ᶠgradᵥ(ᶜΦ) / ᶠinterp(ᶜρ) * ᶠinterp_stencil(one(ᶜρ)),
                )
            end
        end

        # ᶠwₜ = -ᶠgradᵥ(ᶜp) / ᶠinterp(ᶜρ) - ᶠgradᵥ(ᶜK + ᶜΦ)
        # ∂(ᶠwₜ)/∂(ᶠw_data) =
        #     ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶠw_dataₜ) +
        #     ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜK + ᶜΦ)) * ∂(ᶠgradᵥ(ᶜK + ᶜΦ))/∂(ᶠw_dataₜ) =
        #     (
        #         ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) * ∂(ᶠgradᵥ(ᶜp))/∂(ᶜK) +
        #         ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜK + ᶜΦ)) * ∂(ᶠgradᵥ(ᶜK + ᶜΦ))/∂(ᶜK)
        #     ) * ∂(ᶜK)/∂(ᶠw_dataₜ)
        # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜp)) = -1 / ᶠinterp(ᶜρ)
        # ∂(ᶠgradᵥ(ᶜp))/∂(ᶜK) =
        #     ᶜ𝔼_name == :ρe_tot ? ᶠgradᵥ_stencil(-ᶜρ * R_d / cv_d) : 0
        # ∂(ᶠwₜ)/∂(ᶠgradᵥ(ᶜK + ᶜΦ)) = -1
        # ∂(ᶠgradᵥ(ᶜK + ᶜΦ))/∂(ᶜK) = ᶠgradᵥ_stencil(1)
        # ∂(ᶜK)/∂(ᶠw_data) =
        #     ᶜinterp(ᶠw_data) * norm_sqr(ᶜinterp(ᶠw)_unit) * ᶜinterp_stencil(1)
        if :ρθ in propertynames(Y.c) || :ρe_int in propertynames(Y.c)
            @. ∂ᶠ𝕄ₜ∂ᶠ𝕄 = to_scalar_coefs(
                compose(-1 * ᶠgradᵥ_stencil(one(ᶜK)), ∂ᶜK∂ᶠw_data),
            )
        elseif :ρe_tot in propertynames(Y.c)
            @. ∂ᶠ𝕄ₜ∂ᶠ𝕄 = to_scalar_coefs(
                compose(
                    -1 / ᶠinterp(ᶜρ) * ᶠgradᵥ_stencil(-(ᶜρ * R_d / cv_d)) +
                    -1 * ᶠgradᵥ_stencil(one(ᶜK)),
                    ∂ᶜK∂ᶠw_data,
                ),
            )
        end

        for ᶜ𝕋_name in filter(is_tracer_var, propertynames(Y.c))
            ᶜ𝕋 = getproperty(Y.c, ᶜ𝕋_name)
            ∂ᶜ𝕋ₜ∂ᶠ𝕄 = getproperty(∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple, ᶜ𝕋_name)
            if isnothing(ᶠupwind_product)
                # ᶜ𝕋ₜ = -ᶜdivᵥ(ᶠinterp(ᶜ𝕋) * ᶠw)
                # ∂(ᶜ𝕋ₜ)/∂(ᶠw_data) = -ᶜdivᵥ_stencil(ᶠinterp(ᶜ𝕋) * ᶠw_unit)
                @. ∂ᶜ𝕋ₜ∂ᶠ𝕄 = -(ᶜdivᵥ_stencil(ᶠinterp(ᶜ𝕋) * one(ᶠw)))
            else
                # ᶜ𝕋ₜ = -ᶜdivᵥ(ᶠinterp(ᶜρ) * ᶠupwind_product(ᶠw, ᶜ𝕋 / ᶜρ))
                # ∂(ᶜ𝕋ₜ)/∂(ᶠw_data) =
                #     -ᶜdivᵥ_stencil(
                #         ᶠinterp(ᶜρ) * ∂(ᶠupwind_product(ᶠw, ᶜ𝕋 / ᶜρ))/∂(ᶠw_data),
                #     )
                @. ∂ᶜ𝕋ₜ∂ᶠ𝕄 = -(ᶜdivᵥ_stencil(
                    ᶠinterp(ᶜρ) * ᶠupwind_product(ᶠw + εw, ᶜ𝕋 / ᶜρ) /
                    to_scalar(ᶠw + εw),
                ))
            end
        end

        # TODO: Figure out a way to test the Jacobian when the thermodynamic state
        # is PhaseEquil (i.e., when implicit_tendency! calls saturation adjustment).
        if W.test && !(eltype(ᶜts) <: TD.PhaseEquil)
            # Checking every column takes too long, so just check one.
            i, j, h = 1, 1, 1
            args = (implicit_tendency!, Y, p, t, i, j, h)
            ᶜ𝔼_name = filter(is_energy_var, propertynames(Y.c))[1]

            @assert matrix_column(∂ᶜρₜ∂ᶠ𝕄, axes(Y.f), i, j, h) ==
                    exact_column_jacobian_block(args..., (:c, :ρ), (:f, :w))
            @assert matrix_column(∂ᶠ𝕄ₜ∂ᶜ𝔼, axes(Y.c), i, j, h) ≈
                    exact_column_jacobian_block(
                args...,
                (:f, :w),
                (:c, ᶜ𝔼_name),
            )
            @assert matrix_column(∂ᶠ𝕄ₜ∂ᶠ𝕄, axes(Y.f), i, j, h) ≈
                    exact_column_jacobian_block(args..., (:f, :w), (:f, :w))
            for ᶜ𝕋_name in filter(is_tracer_var, propertynames(Y.c))
                ∂ᶜ𝕋ₜ∂ᶠ𝕄 = getproperty(∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple, ᶜ𝕋_name)
                ᶜ𝕋_tuple = (:c, ᶜ𝕋_name)
                @assert matrix_column(∂ᶜ𝕋ₜ∂ᶠ𝕄, axes(Y.f), i, j, h) ≈
                        exact_column_jacobian_block(args..., ᶜ𝕋_tuple, (:f, :w))
            end

            ∂ᶜ𝔼ₜ∂ᶠ𝕄_approx = matrix_column(∂ᶜ𝔼ₜ∂ᶠ𝕄, axes(Y.f), i, j, h)
            ∂ᶜ𝔼ₜ∂ᶠ𝕄_exact =
                exact_column_jacobian_block(args..., (:c, ᶜ𝔼_name), (:f, :w))
            if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact
                @assert ∂ᶜ𝔼ₜ∂ᶠ𝕄_approx ≈ ∂ᶜ𝔼ₜ∂ᶠ𝕄_exact
            else
                err =
                    norm(∂ᶜ𝔼ₜ∂ᶠ𝕄_approx .- ∂ᶜ𝔼ₜ∂ᶠ𝕄_exact) / norm(∂ᶜ𝔼ₜ∂ᶠ𝕄_exact)
                @assert err < 1e-6
                # Note: the highest value seen so far is ~3e-7 (only applies to ρe_tot)
            end

            ∂ᶠ𝕄ₜ∂ᶜρ_approx = matrix_column(∂ᶠ𝕄ₜ∂ᶜρ, axes(Y.c), i, j, h)
            ∂ᶠ𝕄ₜ∂ᶜρ_exact =
                exact_column_jacobian_block(args..., (:f, :w), (:c, :ρ))
            if flags.∂ᶠ𝕄ₜ∂ᶜρ_mode == :exact
                @assert ∂ᶠ𝕄ₜ∂ᶜρ_approx ≈ ∂ᶠ𝕄ₜ∂ᶜρ_exact
            else
                err =
                    norm(∂ᶠ𝕄ₜ∂ᶜρ_approx .- ∂ᶠ𝕄ₜ∂ᶜρ_exact) / norm(∂ᶠ𝕄ₜ∂ᶜρ_exact)
                @assert err < 0.03
                # Note: the highest value seen so far for ρe_tot is ~0.01, and the
                # highest value seen so far for ρθ is ~0.02
            end
        end
    end
end
