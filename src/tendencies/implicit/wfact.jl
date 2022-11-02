#####
##### Wfact
#####

using LinearAlgebra: norm_sqr
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

# In vertical_transport_jac!, we assume that ∂(ᶜρc)/∂(ᶠw_data) = 0; if
# this is not the case, the additional term should be added to the
# result of this function.
# In addition, we approximate the Jacobian for vertical transport with
# FCT using the Jacobian for third-order upwinding (since only FCT
# requires dt, we do not need to pass dt to this function).

# TODO: store operators in `energy_upwinding` so that not all of them are always needed
function vertical_transport_jac!(∂ᶜρcₜ∂ᶠw, ᶠw, ᶜρ, ᶜρc, operators, ::Val{:none})
    (; ᶜdivᵥ_stencil, ᶠinterp) = operators
    @. ∂ᶜρcₜ∂ᶠw = -(ᶜdivᵥ_stencil(ᶠinterp(ᶜρc) * one(ᶠw)))
    return nothing
end
function vertical_transport_jac!(
    ∂ᶜρcₜ∂ᶠw,
    ᶠw,
    ᶜρ,
    ᶜρc,
    operators,
    ::Val{:first_order},
)
    (; ᶜdivᵥ_stencil, ᶠinterp, ᶠupwind1) = operators
    # To convert ᶠw to ᶠw_data, we extract the third vector component.
    to_scalar(vector) = vector.u₃
    FT = Spaces.undertype(axes(ᶜρ))
    ref_εw = Ref(Geometry.Covariant3Vector(eps(FT)))
    @. ∂ᶜρcₜ∂ᶠw = -(ᶜdivᵥ_stencil(
        ᶠinterp(ᶜρ) * ᶠupwind1(ᶠw + ref_εw, ᶜρc / ᶜρ) / to_scalar(ᶠw + ref_εw),
    ))
    return nothing
end
function vertical_transport_jac!(∂ᶜρcₜ∂ᶠw, ᶠw, ᶜρ, ᶜρc, operators, ::Val)
    (; ᶜdivᵥ_stencil, ᶠinterp, ᶠupwind3) = operators
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

function Wfact!(W, Y, p, dtγ, t)
    # @nvtx "Wfact!" color = colorant"green" begin
    _Wfact!(W, Y, p, dtγ, t)
    # end
end

function _Wfact!(W, Y, p, dtγ, t)
    # p.apply_moisture_filter && affect_filter!(Y)
    (; ᶠgradᵥ, ᶠinterp, ᶠinterp_stencil, ᶠupwind1, ᶠgradᵥ_stencil) = p.operators
    (; ᶜinterp, ᶜinterp_stencil, ᶠupwind3, ᶜdivᵥ_stencil) = p.operators

    (; flags, dtγ_ref) = W
    (; ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶠ𝕄ₜ∂ᶠ𝕄, ∂ᶜ𝕋ₜ∂ᶠ𝕄_field) = W
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜK, ᶜΦ, ᶠgradᵥ_ᶜΦ, ᶜts, ᶜp, ∂ᶜK∂ᶠw_data, params) = p
    (; energy_upwinding, tracer_upwinding, thermo_dispatcher) = p

    validate_flags!(Y, flags, energy_upwinding)
    FT = Spaces.undertype(axes(Y.c))
    C123 = Geometry.Covariant123Vector
    compose = Operators.ComposeStencils()

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
        thermo_state!(Y, p, ᶜinterp, colidx)
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
            p.operators,
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
                p.operators,
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
                p.operators,
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
                p.operators,
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
                p.operators,
                tracer_upwinding,
            )
        end
    end
end
