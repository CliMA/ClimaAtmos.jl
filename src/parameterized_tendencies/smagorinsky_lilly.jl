#####
##### Smagornsky Lilly Diffusion
#####

import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators
import LinearAlgebra as la
import ClimaCore: Geometry

smagorinsky_lilly_cache(Y, atmos::AtmosModel) =
    smagorinsky_lilly_cache(Y, atmos.smagorinsky_lilly)

smagorinsky_lilly_cache(Y, ::Nothing) = (;)

function smagorinsky_lilly_cache(Y, sl::SmagorinskyLilly)
    # D = v/Pr, where v = (Cs * cbrt(J))^2 * sqrt(2*Sij*Sij). The cube root of the
    # volume is the average length of the sides of the cell.
    (; Cs) = sl
    FT = eltype(Y)
    h_space = Spaces.horizontal_space(axes(Y.c))
    Δ_filter = Spaces.node_horizontal_length_scale(h_space)
    ᶜtemp_scalar_3 = zero(Fields.Field(Float32, axes(Y.c.ρ)))
    v_t = ᶜtemp_scalar_3
    return (; v_t, Δ_filter)
end

horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl::SmagorinskyLilly) 
    if !(hasproperty(p.precomputed, :ᶜspecific))
        throw(ErrorException("p does not have the property ᶜspecific."))
    end

    (; Cs) = sl
    (; v_t, Δ_filter) = p.smagorinsky_lilly
    (; ᶜu, ᶠu³) = p.precomputed 

    @show "start horizontal tendency_ SUCCESS"
    @show Y.c.ρq_tot
    @show "start horizontal tendency_ SUCCESS"

    # Operators
    FT = eltype(v_t)
    ᶜJ = Fields.local_geometry_field(Y.c).J 

    ᶜS  = zero(p.scratch.ᶜtemp_strain)
    ᶠS  = zero(p.scratch.ᶠtemp_strain)
    ᶜϵ = zero(p.scratch.ᶜtemp_UVWxUVW)
    ᶠϵ = zero(p.scratch.ᶠtemp_UVWxUVW)
    
    localu = @. Geometry.UVWVector(ᶜu)
    localfu = @. Geometry.UVWVector(ᶠinterp(ᶜu))
    any(isnan, localu) && error("Found NaN in horizontal localu")
    any(isnan, localfu) && error("Found NaN in horizontal localfu")

    c1 = @. Geometry.UVWVector(gradₕ(localu.components.data.:1))
    c2 = @. Geometry.UVWVector(gradₕ(localu.components.data.:2))
    c3 = @. Geometry.UVWVector(gradₕ(localu.components.data.:3))
    @. ᶜS.components.data.:1 = c1.components.data.:1
    @. ᶜS.components.data.:2 = c2.components.data.:1
    @. ᶜS.components.data.:3 = c3.components.data.:1
    @. ᶜS.components.data.:4 = c1.components.data.:2
    @. ᶜS.components.data.:5 = c2.components.data.:2
    @. ᶜS.components.data.:6 = c3.components.data.:2
    @. ᶜS.components.data.:7 = zero(c3.components.data.:2)
    @. ᶜS.components.data.:8 = zero(c3.components.data.:2)
    @. ᶜS.components.data.:9 = zero(c3.components.data.:2)
    CA.compute_strain_rate_center!(ᶜϵ, Geometry.Covariant123Vector.(localfu))
    @. ᶜS = (ᶜS + adjoint(ᶜS)) + ᶜϵ
    any(isnan, ᶜS) && error("Found NaN in horizontal ᶜS")

    t1 = @. Geometry.UVWVector(gradₕ(localfu.components.data.:1))
    t2 = @. Geometry.UVWVector(gradₕ(localfu.components.data.:2))
    t3 = @. Geometry.UVWVector(gradₕ(localfu.components.data.:3))
    @. ᶠS.components.data.:1 = t1.components.data.:1
    @. ᶠS.components.data.:2 = t2.components.data.:1
    @. ᶠS.components.data.:3 = t3.components.data.:1
    @. ᶠS.components.data.:4 = t1.components.data.:2
    @. ᶠS.components.data.:5 = t2.components.data.:2
    @. ᶠS.components.data.:6 = t3.components.data.:2
    @. ᶠS.components.data.:7 = zero(t3.components.data.:2)
    @. ᶠS.components.data.:8 = zero(t3.components.data.:2)
    @. ᶠS.components.data.:9 = zero(t3.components.data.:2)
    CA.compute_strain_rate_face!(ᶠϵ, Geometry.Covariant123Vector.(localu))
    @. ᶠS = (ᶠS + adjoint(ᶠS)) + ᶠϵ
    any(isnan, ᶠS) && error("Found NaN in horizontal ᶠS")

    ᶜv_t = @. (Cs * Δ_filter)^2 * sqrt(2 * CA.norm_sqr(ᶜS))
    ᶠv_t = @. ᶠinterp(ᶜv_t)
    ᶜD = @. FT(3) * ᶜv_t

    @. v_t = ᶜv_t
    any(isnan, ᶜv_t) && error("Found NaN in horizontal ᶜv_t")

    ᶠρ = @. ᶠwinterp(ᶜJ, Y.c.ρ)
    any(isnan, ᶠρ) && error("Found NaN in horizontal ᶠρ")
    
    any(isnan, ᶜS) && error("Found NaN in horizontal ᶜS")
    @. Yₜ.c.uₕ += C12(wdivₕ(Y.c.ρ * ᶜv_t * ᶜS)) / Y.c.ρ
    
    @. Yₜ.f.u₃ += C3(wdivₕ(ᶠρ * ᶠv_t * ᶠS)) / ᶠρ
    any(isnan, Yₜ) && error("Found NaN in horizontal Yₜ") 

    # energy adjustment
    (; ᶜspecific) = p.precomputed

    if :ρe_tot in propertynames(Yₜ.c)
       (; ᶜh_tot) = p.precomputed
        @. Yₜ.c.ρe_tot += wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜh_tot)) 
    end
    any(isnan, Yₜ) && error("Found NaN in horizontal Yₜ") 

    # q_tot and other tracer adjustment (moisture affects mass terms 
    # as well)
    for (ᶜρχₜ, ᶜχ, χ_name) in CA.matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        @show ("--------h-------") 
        @show (ᶜρχₜ)
        @show (ᶜχ)
        @show (χ_name) 
        @show ("--------h-------") 
        @show (Y.c.ρ) 
        @show (Y.c.ρq_tot)
        @show (Y.c.ρq_tot ./ Y.c.ρ)
        @show ("ρq_tot") 
        @show (ᶜD)
        @show ("--------h------") 
        #@. Yₜ.c.ρq_tot = wdivₕ(Y.c.ρ * ᶜD * gradₕ(Y.c.ρq_tot ./ Y.c.ρ))
        @. ᶜρχₜ += wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜχ)) 
        @. Yₜ.c.ρ += wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜχ)) 
    end
    any(isnan, Yₜ) && error("Found NaN in horizontal Yₜ") 

end

function vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl::SmagorinskyLilly) 
    if !(hasproperty(p.precomputed, :ᶜspecific))
        throw(ErrorException("p does not have the property ᶜspecific."))
    end
    (; Cs) = sl
    (; v_t, Δ_filter) = p.smagorinsky_lilly
    (; ᶜu, ᶠu³, sfc_conditions, ᶜspecific) = p.precomputed 

    # Operators
    FT = eltype(v_t)
    ᶜJ = Fields.local_geometry_field(Y.c).J 

    ᶜS  = zero(p.scratch.ᶜtemp_strain)
    ᶠS  = zero(p.scratch.ᶠtemp_strain)
    ᶜϵ = zero(p.scratch.ᶜtemp_UVWxUVW)
    ᶠϵ = zero(p.scratch.ᶠtemp_UVWxUVW)
    
    localu = @. Geometry.UVWVector(ᶜu)
    ᶠu = @. Geometry.UVWVector(ᶠwinterp(ᶜJ * Y.c.ρ, Y.c.uₕ)) + Geometry.UVWVector(Y.f.u₃)

    c1 = @. Geometry.UVWVector(gradₕ(localu.components.data.:1))
    c2 = @. Geometry.UVWVector(gradₕ(localu.components.data.:2))
    c3 = @. Geometry.UVWVector(gradₕ(localu.components.data.:3))
    @. ᶜS.components.data.:1 = c1.components.data.:1
    @. ᶜS.components.data.:2 = c2.components.data.:1
    @. ᶜS.components.data.:3 = c3.components.data.:1
    @. ᶜS.components.data.:4 = c1.components.data.:2
    @. ᶜS.components.data.:5 = c2.components.data.:2
    @. ᶜS.components.data.:6 = c3.components.data.:2
    @. ᶜS.components.data.:7 = zero(c3.components.data.:2)
    @. ᶜS.components.data.:8 = zero(c3.components.data.:2)
    @. ᶜS.components.data.:9 = zero(c3.components.data.:2)
    CA.compute_strain_rate_center!(ᶜϵ, Geometry.Covariant123Vector.(ᶠu))
    @. ᶜS = (ᶜS + adjoint(ᶜS)) + ᶜϵ

    t1 = @. Geometry.UVWVector(gradₕ(ᶠu.components.data.:1))
    t2 = @. Geometry.UVWVector(gradₕ(ᶠu.components.data.:2))
    t3 = @. Geometry.UVWVector(gradₕ(ᶠu.components.data.:3))
    @. ᶠS.components.data.:1 = t1.components.data.:1
    @. ᶠS.components.data.:2 = t2.components.data.:1
    @. ᶠS.components.data.:3 = t3.components.data.:1
    @. ᶠS.components.data.:4 = t1.components.data.:2
    @. ᶠS.components.data.:5 = t2.components.data.:2
    @. ᶠS.components.data.:6 = t3.components.data.:2
    @. ᶠS.components.data.:7 = zero(t3.components.data.:2)
    @. ᶠS.components.data.:8 = zero(t3.components.data.:2)
    @. ᶠS.components.data.:9 = zero(t3.components.data.:2)
    CA.compute_strain_rate_face!(ᶠϵ, ᶜu)
    @. ᶠS = (ᶠS + adjoint(ᶠS)) + ᶠϵ

    ᶜv_t = @. (Cs * Δ_filter)^2 * sqrt(2 * CA.norm_sqr(ᶜS))
    ᶠv_t = @. ᶠinterp(ᶜv_t)
    ᶜD = @. FT(3) * ᶜv_t
    @. v_t = ᶜv_t
    
    @. Yₜ.c.uₕ -= C12(
        ᶜdivᵥ(
            -2 *
            ᶠinterp(Y.c.ρ) *
            ᶠv_t *
            ᶠS,
        ) / Y.c.ρ,
    )
    any(isnan, Yₜ) && error("Found NaN in vertical sgs tendency after uh")
    
     #apply boundary condition for momentum flux
    ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
        bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ),
    )
    @. Yₜ.c.uₕ +=
     C12(ᶜdivᵥ_uₕ((FT(0) * ᶠgradᵥ(Y.c.uₕ))) / Y.c.ρ)

     any(isnan, Yₜ) && error("Found NaN in vertical sgs tendency after uh bc")
    ᶜdivᵥ_u3 = Operators.DivergenceC2F(
        top = Operators.SetValue(C3(FT(0)) ⊗ C3(FT(0))),
        bottom = Operators.SetValue(C3(FT(0)) ⊗ C3(FT(0))), 
    )
    
    @. Yₜ.f.u₃ -= C3(
        ᶠinterp(ᶜdivᵥ(
            -2 *
            ᶠinterp(Y.c.ρ) *
            ᶠv_t *
            ᶠS,
        ) / Y.c.ρ),
    )

    any(isnan, Yₜ) && error("Found NaN in vertical sgs tendency after u3")
    
    if :ρe_tot in propertynames(Yₜ.c)
        (; ᶜh_tot) = p.precomputed
        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_h_tot), 
        )
        @. Yₜ.c.ρe_tot -= ᶜdivᵥ_ρe_tot(
					-(ᶠinterp(Y.c.ρ) * 
					ᶠinterp(ᶜD) * 
					ᶠgradᵥ(ᶜh_tot))
				  )
    end
    any(isnan, Yₜ) && error("Found NaN in vertical sgs tendency after etot")

    ρ_flux_χ = zero(p.scratch.sfc_temp_C3)
    for (ᶜρχₜ, ᶜχ, χ_name) in CA.matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        if χ_name == :q_tot
            @. ρ_flux_χ = sfc_conditions.ρ_flux_q_tot
        else
            @. ρ_flux_χ = C3(FT(0))
        end
        ᶜdivᵥ_ρχ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(ρ_flux_χ), 
        )
        # The code below adjusts the tendency by -div(\rho * d_qtot), where
        # d_qtot = -(D * grad(qtot)). The two negatives cancel out, so we have a +=
        @. ᶜρχₜ -= ᶜdivᵥ_ρχ(-(ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜD) * ᶠgradᵥ(ᶜχ)))
        @. Yₜ.c.ρ -= ᶜdivᵥ_ρχ(-(ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜD) * ᶠgradᵥ(ᶜχ)))
    end
    any(isnan, Yₜ) && error("Found NaN in vertical sgs tendency after qtot")
    @show "end vertical tendency_ SUCCESS"
    @show Y.c.ρq_tot
    @show "end vertical tendency_ SUCCESS"
end

# function vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, colidx, sl::SmagorinskyLilly) 
#     if !(hasproperty(p.precomputed, :ᶜspecific))
#         throw(ErrorException("p does not have the property ᶜspecific."))
#     end
#     (; Cs) = sl
#     (; v_t, ᶜD, Δ_filter) = p.smagorinsky_lilly
#     (; ᶜu, ᶠu³, sfc_conditions, ᶜspecific) = p.precomputed 

#     # Operators
#     FT = eltype(v_t)
#     wdivₕ = Operators.WeakDivergence()
#     ᶜJ = Fields.local_geometry_field(Y.c).J 

#     ᶜS  = p.scratch.ᶜtemp_strain
#     ᶠS  = p.scratch.ᶠtemp_strain
#     ᶜϵ = p.scratch.ᶜtemp_UVWxUVW
#     ᶠϵ = p.scratch.ᶠtemp_UVWxUVW
    
#     localu = @. Geometry.UVWVector(ᶜu[colidx])
#     ᶠu = @. Geometry.UVWVector(ᶠwinterp(ᶜJ[colidx] * Y.c.ρ[colidx], Y.c.uₕ[colidx])) + Geometry.UVWVector(Y.f.u₃[colidx])

#     c1 = @. Geometry.UVWVector(gradₕ(localu[colidx].components.data.:1))
#     c2 = @. Geometry.UVWVector(gradₕ(localu[colidx].components.data.:2))
#     c3 = @. Geometry.UVWVector(gradₕ(localu[colidx].components.data.:3))
#     @. ᶜS[colidx].components.data.:1 = c1[colidx].components.data.:1
#     @. ᶜS[colidx].components.data.:2 = c2[colidx].components.data.:1
#     @. ᶜS[colidx].components.data.:3 = c3[colidx].components.data.:1
#     @. ᶜS[colidx].components.data.:4 = c1[colidx].components.data.:2
#     @. ᶜS[colidx].components.data.:5 = c2[colidx].components.data.:2
#     @. ᶜS[colidx].components.data.:6 = c3[colidx].components.data.:2
#     CA.compute_strain_rate_center!(ᶜϵ[colidx], Geometry.Covariant123Vector.(ᶠu[colidx]))
#     @. ᶜS[colidx] = (ᶜS[colidx] + adjoint(ᶜS[colidx])) + ᶜϵ[colidx]

#     t1 = @. Geometry.UVWVector(gradₕ(ᶠu[colidx].components.data.:1))
#     t2 = @. Geometry.UVWVector(gradₕ(ᶠu[colidx].components.data.:2))
#     t3 = @. Geometry.UVWVector(gradₕ(ᶠu[colidx].components.data.:3))
#     @. ᶠS[colidx].components.data.:1 = t1[colidx].components.data.:1
#     @. ᶠS[colidx].components.data.:2 = t2[colidx].components.data.:1
#     @. ᶠS[colidx].components.data.:3 = t3[colidx].components.data.:1
#     @. ᶠS[colidx].components.data.:4 = t1[colidx].components.data.:2
#     @. ᶠS[colidx].components.data.:5 = t2[colidx].components.data.:2
#     @. ᶠS[colidx].components.data.:6 = t3[colidx].components.data.:2
#     CA.compute_strain_rate_face!(ᶠϵ[colidx], ᶜu[colidx])
#     @. ᶠS[colidx] = (ᶠS[colidx] + adjoint(ᶠS[colidx])) + ᶠϵ[colidx]

#     ᶜv_t = @. (Cs * Δ_filter)^2 * sqrt(2 * CA.norm_sqr(ᶜS[colidx]))
#     ᶠv_t = @. ᶠinterp(ᶜv_t[colidx])
#     @. ᶜD[colidx] = @. FT(3) * ᶜv_t[colidx]
#     @. v_t[colidx] = ᶜv_t[colidx]
    
#     @. Yₜ.c.uₕ[colidx] -= C12(
#         ᶜdivᵥ(
#             -2 *
#             ᶠinterp(Y.c.ρ[colidx]) *
#             ᶠv_t[colidx] *
#             ᶠS[colidx],
#         ) / Y.c.ρ[colidx],
#     )

#     any(isnan, Yₜ) && error("Found NaN in vertical sgs tendency")
    
#     # apply boundary condition for momentum flux
#     #ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
#     #    top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
#     #    bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ[colidx]),
#     #)
#     #@. Yₜ.c.uₕ[colidx] -=
#     #C12(ᶜdivᵥ_uₕ(-(FT(0) * ᶠgradᵥ(Y.c.uₕ[colidx]))) / Y.c.ρ[colidx])

#     ᶜdivᵥ_u3 = Operators.DivergenceC2F(
#         top = Operators.SetValue(C3(FT(0)) ⊗ C3(FT(0))),
#         bottom = Operators.SetValue(C3(FT(0)) ⊗ C3(FT(0))), 
#     )
    
#     @. Yₜ.f.u₃[colidx] -= C3(
#         ᶠinterp(ᶜdivᵥ(
#             -2 *
#             ᶠinterp(Y.c.ρ[colidx]) *
#             ᶠv_t[colidx] *
#             ᶠS[colidx],
#         ) / Y.c.ρ[colidx]),
#     )

#     any(isnan, Yₜ) && error("Found NaN in vertical sgs tendency")
    
#     if :ρe_tot in propertynames(Yₜ.c)
#         (; ᶜh_tot) = p.precomputed
#         ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
#             top = Operators.SetValue(C3(FT(0))),
#             bottom = Operators.SetValue(sfc_conditions.ρ_flux_h_tot[colidx]), 
#         )
#         @. Yₜ.c.ρe_tot[colidx] -= ᶜdivᵥ_ρe_tot(
# 					-(ᶠinterp(Y.c.ρ[colidx]) * 
# 					ᶠinterp(ᶜD[colidx]) * 
# 					ᶠgradᵥ(ᶜh_tot[colidx]))
# 				  )
#     end
#     any(isnan, Yₜ) && error("Found NaN in vertical sgs tendency")

#     ρ_flux_χ = p.scratch.sfc_temp_C3
#     for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
#         χ_name == :e_tot && continue
#         if χ_name == :q_tot
#             @. ρ_flux_χ[colidx] = sfc_conditions.ρ_flux_q_tot[colidx]
#         else
#             @. ρ_flux_χ[colidx] = C3(FT(0))
#         end
#         ᶜdivᵥ_ρχ = Operators.DivergenceF2C(
#             top = Operators.SetValue(C3(FT(0))),
#             bottom = Operators.SetValue(ρ_flux_χ[colidx]), 
#         )
#         # The code below adjusts the tendency by -div(\rho * d_qtot), where
#         # d_qtot = -(D * grad(qtot)). The two negatives cancel out, so we have a +=
#         @. ᶜρχₜ[colidx] -= ᶜdivᵥ_ρχ(-(ᶠinterp(Y.c.ρ[colidx]) * ᶠinterp(ᶜD[colidx]) * ᶠgradᵥ(ᶜχ[colidx])))
#     end
#     any(isnan, Yₜ) && error("Found NaN in vertical sgs tendency")

# end
