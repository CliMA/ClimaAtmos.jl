#####
##### Wfact
#####

using LinearAlgebra: norm_sqr
import ClimaCore.Spaces as Spaces
import ClimaCore.Operators as Operators
import ClimaCore.Fields as Fields

# From the chain rule, we know that
# ∂(ᶜρχₜ)/∂(ᶠu₃_data) = ∂(ᶜρχₜ)/∂(ᶠu³_data) * ∂(ᶠu³_data)/∂(ᶠu₃_data),
# where ∂(ᶠu³_data)/∂(ᶠu₃_data) = ᶠg³³.
# If ᶜρχₜ = -ᶜadvdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠu³ * ᶠinterp(ᶜχ)), then
# ∂(ᶜρχₜ)/∂(ᶠu³_data) =
#     -ᶜadvdivᵥ_stencil(ᶠwinterp(ᶜJ, ᶜρ) * ᶠu³_unit * ᶠinterp(ᶜχ)) -
#     ᶜadvdivᵥ_stencil(ᶠwinterp(ᶜJ, ᶜρ) * ᶠu³) * ᶠinterp_stencil(1) *
#     ∂(ᶜχ)/∂(ᶠu₃_data).
# If ᶜρχₜ = -ᶜadvdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind(ᶠu³, ᶜχ)), then
# ∂(ᶜρχₜ)/∂(ᶠu₃_data) =
#     -ᶜadvdivᵥ_stencil(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind(ᶠu³, ᶜχ) / ᶠu³_data) -
#     ᶜadvdivᵥ_stencil(ᶠwinterp(ᶜJ, ᶜρ)) * ᶠupwind_stencil(ᶠu³, 1) *
#     ∂(ᶜχ)/∂(ᶠu₃_data).
# Since ᶠu³_data can be 0, we need to modify the last derivative by replacing
# ᶠu³ with CT3(ᶠu³_data + eps(ᶠu³_data)), which lets us avoid divisions by 0.
# Since Operator2Stencil has not yet been extended to upwinding operators,
# ᶠupwind_stencil is not available.
# For simplicity, we approximate the value of ∂(ᶜρχₜ)/∂(ᶠu³_data) for FCT
# (both Boris-Book and Zalesak) using the value for first-order upwinding.
# In the following function, we assume that ∂(ᶜχ)/∂(ᶠu₃_data) = 0; if this is
# not the case, the additional term should be added to this function's result.
get_data_plus_ε(vector) = vector.u³ + eps(vector.u³)
set_∂ᶜρχₜ∂ᶠu₃!(∂ᶜρχₜ∂ᶠu₃_data, ᶜJ, ᶜρ, ᶠu³, ᶜχ, ᶠg³³, ::Val{:none}) =
    @. ∂ᶜρχₜ∂ᶠu₃_data =
        -(ᶜadvdivᵥ_stencil(ᶠwinterp(ᶜJ, ᶜρ) * one(ᶠu³) * ᶠinterp(ᶜχ) * ᶠg³³))
set_∂ᶜρχₜ∂ᶠu₃!(∂ᶜρχₜ∂ᶠu₃_data, ᶜJ, ᶜρ, ᶠu³, ᶜχ, ᶠg³³, ::Val{:first_order}) =
    @. ∂ᶜρχₜ∂ᶠu₃_data = -(ᶜadvdivᵥ_stencil(
        ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(CT3(get_data_plus_ε(ᶠu³)), ᶜχ) /
        get_data_plus_ε(ᶠu³) * ᶠg³³,
    ))
set_∂ᶜρχₜ∂ᶠu₃!(∂ᶜρχₜ∂ᶠu₃_data, ᶜJ, ᶜρ, ᶠu³, ᶜχ, ᶠg³³, ::Val{:third_order}) =
    @. ∂ᶜρχₜ∂ᶠu₃_data = -(ᶜadvdivᵥ_stencil(
        ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind3(CT3(get_data_plus_ε(ᶠu³)), ᶜχ) /
        get_data_plus_ε(ᶠu³) * ᶠg³³,
    ))
set_∂ᶜρχₜ∂ᶠu₃!(∂ᶜρχₜ∂ᶠu₃_data, ᶜJ, ᶜρ, ᶠu³, ᶜχ, ᶠg³³, ::Val{:boris_book}) =
    set_∂ᶜρχₜ∂ᶠu₃!(∂ᶜρχₜ∂ᶠu₃_data, ᶜJ, ᶜρ, ᶠu³, ᶜχ, ᶠg³³, Val(:first_order))
set_∂ᶜρχₜ∂ᶠu₃!(∂ᶜρχₜ∂ᶠu₃_data, ᶜJ, ᶜρ, ᶠu³, ᶜχ, ᶠg³³, ::Val{:zalesak}) =
    set_∂ᶜρχₜ∂ᶠu₃!(∂ᶜρχₜ∂ᶠu₃_data, ᶜJ, ᶜρ, ᶠu³, ᶜχ, ᶠg³³, Val(:first_order))

function validate_flags!(Y, flags, energy_upwinding)
    # TODO: Add Operator2Stencil for UpwindBiasedProductC2F to ClimaCore
    # to allow exact Jacobian calculation.
    :ρe_tot in propertynames(Y.c) &&
        energy_upwinding !== Val(:none) &&
        flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact &&
        error(
            "∂ᶜ𝔼ₜ∂ᶠ𝕄_mode must be :no_∂ᶜp∂ᶜK when using ρe_tot with upwinding",
        )
end

NVTX.@annotate function Wfact!(W, Y, p, dtγ, t)
    fill_with_nans!(p)
    # set_precomputed_quantities!(Y, p, t)
    Fields.bycolumn(axes(Y.c)) do colidx
        Wfact!(W, Y, p, dtγ, t, colidx)
    end
end

function Wfact!(W, Y, p, dtγ, t, colidx)
    (; flags, dtγ_ref) = W
    (; ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶠ𝕄ₜ∂ᶠ𝕄, ∂ᶜ𝕋ₜ∂ᶠ𝕄_field) = W
    (; ᶜspecific, ᶠu³, ᶜK, ᶜp) = p
    (; ᶜΦ, ᶠgradᵥ_ᶜΦ, ᶜρ_ref, ᶜp_ref, params, ∂ᶜK∂ᶠu₃_data) = p
    (; energy_upwinding, tracer_upwinding, density_upwinding) = p

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠg³³ = g³³_field(Y.f)

    validate_flags!(Y, flags, energy_upwinding)
    FT = Spaces.undertype(axes(Y.c))
    compose = Operators.ComposeStencils()

    R_d = FT(CAP.R_d(params))
    κ_d = FT(CAP.kappa_d(params))
    cv_d = FT(CAP.cv_d(params))
    T_tri = FT(CAP.T_triple(params))
    p_ref_theta = FT(CAP.p_ref_theta(params))

    dtγ_ref[] = dtγ

    # We can express the pressure as
    # ᶜp = R_d * (ᶜρe_tot / cv_d + ᶜρ * (-(ᶜK + ᶜΦ) / cv_d + T_tri)) + O(ᶜq_tot)
    # We will ignore all O(ᶜq_tot) terms when computing derivatives of pressure.

    # ᶜK =
    #     (
    #         dot(C123(ᶜuₕ), CT123(ᶜuₕ)) +
    #         ᶜinterp(dot(C123(ᶠu₃), CT123(ᶠu₃))) +
    #         2 * dot(CT123(ᶜuₕ), ᶜinterp(C123(ᶠu₃)))
    #     ) / 2 =
    #     (
    #         dot(C123(ᶜuₕ), CT123(ᶜuₕ)) +
    #         ᶜinterp(ᶠu₃_data^2 * dot(C123(ᶠu₃_unit), CT123(ᶠu₃_unit))) +
    #         2 * dot(CT123(ᶜuₕ), ᶜinterp(ᶠu₃_data * C123(ᶠu₃_unit)))
    #     ) / 2 =
    # ∂(ᶜK)/∂(ᶠu₃_data) =
    #     (
    #         ᶜinterp_stencil(2 * ᶠu₃_data * dot(C123(ᶠu₃_unit), CT123(ᶠu₃_unit))) +
    #         2 * dot(CT123(ᶜuₕ), ᶜinterp_stencil(C123(ᶠu₃_unit)))
    #     ) / 2 =
    #     ᶜinterp_stencil(dot(C123(ᶠu₃_unit), CT123(ᶠu₃))) +
    #     dot(CT123(ᶜuₕ), ᶜinterp_stencil(C123(ᶠu₃_unit)))
    @. ∂ᶜK∂ᶠu₃_data[colidx] =
        ᶜinterp_stencil(dot(C123(one(ᶠu₃[colidx])), CT123(ᶠu₃[colidx])))
    @. ∂ᶜK∂ᶠu₃_data.coefs.:1[colidx] += dot(
        CT123(ᶜuₕ[colidx]),
        getindex(ᶜinterp_stencil(C123(one(ᶠu₃[colidx]))), 1),
    )
    @. ∂ᶜK∂ᶠu₃_data.coefs.:2[colidx] += dot(
        CT123(ᶜuₕ[colidx]),
        getindex(ᶜinterp_stencil(C123(one(ᶠu₃[colidx]))), 2),
    )
    # TODO: Figure out why rewriting this as shown below incurs allocations:
    # @inline map_dot(vector, vectors) =
    #     map(vector_coef -> dot(vector, vector_coef), vectors)
    # @. ∂ᶜK∂ᶠu₃_data[colidx] =
    #     ᶜinterp_stencil(dot(C123(one(ᶠu₃[colidx])), CT123(ᶠu₃[colidx]))) +
    #     map_dot(CT123(ᶜuₕ[colidx]), ᶜinterp_stencil(C123(one(ᶠu₃[colidx]))))

    ᶜ1 = p.ᶜtemp_scalar
    @. ᶜ1[colidx] = one(ᶜρ[colidx])
    set_∂ᶜρχₜ∂ᶠu₃!(
        ∂ᶜρₜ∂ᶠ𝕄[colidx],
        ᶜJ[colidx],
        ᶜρ[colidx],
        ᶠu³[colidx],
        ᶜ1[colidx],
        ᶠg³³[colidx],
        density_upwinding,
    )

    if :ρθ in propertynames(Y.c)
        set_∂ᶜρχₜ∂ᶠu₃!(
            ∂ᶜ𝔼ₜ∂ᶠ𝕄[colidx],
            ᶜJ[colidx],
            ᶜρ[colidx],
            ᶠu³[colidx],
            ᶜspecific.θ[colidx],
            ᶠg³³[colidx],
            energy_upwinding,
        )
    elseif :ρe_tot in propertynames(Y.c)
        (; ᶜh_tot) = p
        set_∂ᶜρχₜ∂ᶠu₃!(
            ∂ᶜ𝔼ₜ∂ᶠ𝕄[colidx],
            ᶜJ[colidx],
            ᶜρ[colidx],
            ᶠu³[colidx],
            ᶜh_tot[colidx],
            ᶠg³³[colidx],
            energy_upwinding,
        )
        if flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact
            # ∂(ᶜh_tot)/∂(ᶠu₃_data) =
            #     ∂(ᶜp / ᶜρ)/∂(ᶠu₃_data) =
            #     ∂(ᶜp / ᶜρ)/∂(ᶜK) * ∂(ᶜK)/∂(ᶠu₃_data)
            # If we ignore the dependence of pressure on moisture,
            # ∂(ᶜp / ᶜρ)/∂(ᶜK) = -R_d / cv_d
            if energy_upwinding === Val(:none)
                @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 -= compose(
                    ᶜadvdivᵥ_stencil(
                        ᶠwinterp(ᶜJ[colidx], ᶜρ[colidx]) * ᶠu³[colidx],
                    ),
                    compose(
                        ᶠinterp_stencil(ᶜ1[colidx]),
                        -R_d / cv_d * ∂ᶜK∂ᶠu₃_data[colidx],
                    ),
                )
            end
        end
    end
    for (∂ᶜ𝕋ₜ∂ᶠ𝕄, ᶜχ, _) in matching_subfields(∂ᶜ𝕋ₜ∂ᶠ𝕄_field, ᶜspecific)
        set_∂ᶜρχₜ∂ᶠu₃!(
            ∂ᶜ𝕋ₜ∂ᶠ𝕄[colidx],
            ᶜJ[colidx],
            ᶜρ[colidx],
            ᶠu³[colidx],
            ᶜχ[colidx],
            ᶠg³³[colidx],
            tracer_upwinding,
        )
    end

    # We use map_get_data to convert ∂(ᶠu₃ₜ)/∂(X) to ∂(ᶠu₃_data)ₜ/∂(X).
    @inline map_get_data(vectors) = map(vector -> vector.u₃, vectors)

    # ᶠu₃ₜ = -(ᶠgradᵥ(ᶜp - ᶜp_ref) + ᶠinterp(ᶜρ - ᶜρ_ref) * ᶠgradᵥ_ᶜΦ) / ᶠinterp(ᶜρ)
    if :ρθ in propertynames(Y.c)
        # ∂(ᶠu₃ₜ)/∂(ᶜρθ) =
        #     ∂(ᶠu₃ₜ)/∂(ᶠgradᵥ(ᶜp - ᶜp_ref)) * ∂(ᶠgradᵥ(ᶜp - ᶜp_ref))/∂(ᶜρθ)
        # ∂(ᶠu₃ₜ)/∂(ᶠgradᵥ(ᶜp - ᶜp_ref)) = -1 / ᶠinterp(ᶜρ)
        # If we ignore the dependence of pressure on moisture,
        # ∂(ᶠgradᵥ(ᶜp - ᶜp_ref))/∂(ᶜρθ) =
        #     ᶠgradᵥ_stencil(
        #         R_d / (1 - κ_d) * (ᶜρθ * R_d / p_ref_theta)^(κ_d / (1 - κ_d))
        #     )
        ᶜρθ = Y.c.ρθ
        @. ∂ᶠ𝕄ₜ∂ᶜ𝔼[colidx] = map_get_data(
            -1 / ᶠinterp(ᶜρ[colidx]) * ᶠgradᵥ_stencil(
                R_d / (1 - κ_d) *
                (ᶜρθ[colidx] * R_d / p_ref_theta)^(κ_d / (1 - κ_d)),
            ),
        )

        # ∂(ᶠu₃ₜ)/∂(ᶜρ) =
        #     ∂(ᶠu₃ₜ)/∂(ᶠinterp(ᶜρ - ᶜρ_ref)) * ∂(ᶠinterp(ᶜρ - ᶜρ_ref))/∂(ᶜρ) +
        #     ∂(ᶠu₃ₜ)/∂(ᶠinterp(ᶜρ)) * ∂(ᶠinterp(ᶜρ))/∂(ᶜρ)
        # ∂(ᶠu₃ₜ)/∂(ᶠinterp(ᶜρ - ᶜρ_ref)) = -ᶠgradᵥ_ᶜΦ / ᶠinterp(ᶜρ)
        # ∂(ᶠinterp(ᶜρ - ᶜρ_ref))/∂(ᶜρ) = ᶠinterp_stencil(1)
        # ∂(ᶠu₃ₜ)/∂(ᶠinterp(ᶜρ)) =
        #     (ᶠgradᵥ(ᶜp - ᶜp_ref) + ᶠinterp(ᶜρ - ᶜρ_ref) * ᶠgradᵥ_ᶜΦ) / ᶠinterp(ᶜρ)^2
        # ∂(ᶠinterp(ᶜρ))/∂(ᶜρ) = ᶠinterp_stencil(1)
        @. ∂ᶠ𝕄ₜ∂ᶜρ[colidx] = map_get_data(
            (
                ᶠgradᵥ(ᶜp[colidx] - ᶜp_ref[colidx]) -
                ᶠinterp(ᶜρ_ref[colidx]) * ᶠgradᵥ_ᶜΦ[colidx]
            ) / abs2(ᶠinterp(ᶜρ[colidx])) * ᶠinterp_stencil(ᶜ1[colidx]),
        )

        # ∂(ᶠu₃ₜ)/∂(ᶠu₃_data) = 0
        ∂ᶠ𝕄ₜ∂ᶠ𝕄[colidx] .=
            tuple(Operators.StencilCoefs{-1, 1}((FT(0), FT(0), FT(0))))
    elseif :ρe_tot in propertynames(Y.c)
        # ∂(ᶠu₃ₜ)/∂(ᶜρe_tot) =
        #     ∂(ᶠu₃ₜ)/∂(ᶠgradᵥ(ᶜp - ᶜp_ref)) * ∂(ᶠgradᵥ(ᶜp - ᶜp_ref))/∂(ᶜρe_tot)
        # ∂(ᶠu₃ₜ)/∂(ᶠgradᵥ(ᶜp - ᶜp_ref)) = -1 / ᶠinterp(ᶜρ)
        # If we ignore the dependence of pressure on moisture,
        # ∂(ᶠgradᵥ(ᶜp - ᶜp_ref))/∂(ᶜρe_tot) = ᶠgradᵥ_stencil(R_d / cv_d)
        @. ∂ᶠ𝕄ₜ∂ᶜ𝔼[colidx] = map_get_data(
            -1 / ᶠinterp(ᶜρ[colidx]) * ᶠgradᵥ_stencil(R_d / cv_d * ᶜ1[colidx]),
        )

        # ∂(ᶠu₃ₜ)/∂(ᶜρ) =
        #     ∂(ᶠu₃ₜ)/∂(ᶠgradᵥ(ᶜp - ᶜp_ref)) * ∂(ᶠgradᵥ(ᶜp - ᶜp_ref))/∂(ᶜρ) +
        #     ∂(ᶠu₃ₜ)/∂(ᶠinterp(ᶜρ - ᶜρ_ref)) * ∂(ᶠinterp(ᶜρ - ᶜρ_ref))/∂(ᶜρ) +
        #     ∂(ᶠu₃ₜ)/∂(ᶠinterp(ᶜρ)) * ∂(ᶠinterp(ᶜρ))/∂(ᶜρ)
        # ∂(ᶠu₃ₜ)/∂(ᶠgradᵥ(ᶜp - ᶜp_ref)) = -1 / ᶠinterp(ᶜρ)
        # If we ignore the dependence of pressure on moisture,
        # ∂(ᶠgradᵥ(ᶜp - ᶜp_ref))/∂(ᶜρ) =
        #     ᶠgradᵥ_stencil(R_d * (-(ᶜK + ᶜΦ) / cv_d + T_tri))
        # ∂(ᶠu₃ₜ)/∂(ᶠinterp(ᶜρ - ᶜρ_ref)) = -ᶠgradᵥ_ᶜΦ / ᶠinterp(ᶜρ)
        # ∂(ᶠinterp(ᶜρ - ᶜρ_ref))/∂(ᶜρ) = ᶠinterp_stencil(1)
        # ∂(ᶠu₃ₜ)/∂(ᶠinterp(ᶜρ)) =
        #     (ᶠgradᵥ(ᶜp - ᶜp_ref) + ᶠinterp(ᶜρ - ᶜρ_ref) * ᶠgradᵥ_ᶜΦ) / ᶠinterp(ᶜρ)^2
        # ∂(ᶠinterp(ᶜρ))/∂(ᶜρ) = ᶠinterp_stencil(1)
        @. ∂ᶠ𝕄ₜ∂ᶜρ[colidx] = map_get_data(
            -1 / ᶠinterp(ᶜρ[colidx]) * ᶠgradᵥ_stencil(
                R_d * (-(ᶜK[colidx] + ᶜΦ[colidx]) / cv_d + T_tri),
            ) +
            (
                ᶠgradᵥ(ᶜp[colidx] - ᶜp_ref[colidx]) -
                ᶠinterp(ᶜρ_ref[colidx]) * ᶠgradᵥ_ᶜΦ[colidx]
            ) / abs2(ᶠinterp(ᶜρ[colidx])) * ᶠinterp_stencil(ᶜ1[colidx]),
        )

        # ∂(ᶠu₃ₜ)/∂(ᶠu₃_data) =
        #     ∂(ᶠu₃ₜ)/∂(ᶠgradᵥ(ᶜp - ᶜp_ref)) * ∂(ᶠgradᵥ(ᶜp - ᶜp_ref))/∂(ᶠu₃_data) =
        #     ∂(ᶠu₃ₜ)/∂(ᶠgradᵥ(ᶜp - ᶜp_ref)) * ∂(ᶠgradᵥ(ᶜp - ᶜp_ref))/∂(ᶜK) * ∂(ᶜK)/∂(ᶠu₃_data)
        # ∂(ᶠu₃ₜ)/∂(ᶠgradᵥ(ᶜp - ᶜp_ref)) = -1 / ᶠinterp(ᶜρ)
        # If we ignore the dependence of pressure on moisture,
        # ∂(ᶠgradᵥ(ᶜp - ᶜp_ref))/∂(ᶜK) = ᶠgradᵥ_stencil(-ᶜρ * R_d / cv_d)
        @. ∂ᶠ𝕄ₜ∂ᶠ𝕄[colidx] = map_get_data(
            compose(
                -1 / ᶠinterp(ᶜρ[colidx]) *
                ᶠgradᵥ_stencil(-(ᶜρ[colidx] * R_d / cv_d)),
                ∂ᶜK∂ᶠu₃_data[colidx],
            ),
        )
    end

    if p.atmos.rayleigh_sponge isa RayleighSponge
        # ᶠu₃ₜ -= p.ᶠβ_rayleigh_w * ᶠu₃
        # ∂(ᶠu₃ₜ)/∂(ᶠu₃_data) -= p.ᶠβ_rayleigh_w * ᶠu₃_unit
        @. ∂ᶠ𝕄ₜ∂ᶠ𝕄[colidx].coefs.:2 -= p.ᶠβ_rayleigh_w[colidx]
    end

    return nothing
end
