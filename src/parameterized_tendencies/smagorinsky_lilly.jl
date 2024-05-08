#####
##### Smagorinsky Lilly Diffusion
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

    ᶜtemp_scalar = similar(Y.c.ρ, FT)
    ᶜtemp_scalar_2 = similar(Y.c.ρ, FT)
    ᶜtemp_scalar_3 = similar(Y.c.ρ, FT)
    ᶠtemp_C123 = similar(Y.f, C123{FT})
    ᶜtemp_CT3 = similar(Y.c, CT3{FT})

    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜshear² = ᶜtemp_scalar
    ᶠu = ᶠtemp_C123
    @. ᶠu = C123(ᶠinterp(Y.c.uₕ)) + C123(Y.f.u₃)
    ct3_unit = ᶜtemp_CT3
    @. ct3_unit = CT3(Geometry.WVector(FT(1)), ᶜlg)
    @. ᶜshear² = norm_sqr(adjoint(CA.ᶜgradᵥ(ᶠu)) * ct3_unit)

    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶜD = ᶜtemp_scalar_2
    v_t = ᶜtemp_scalar_3
    Pr = 1/3
    @. ᶜD = (1/Pr)*v_t
    return (; v_t, ᶜD, Δ_filter)
end

horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) = nothing

function horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl::SmagorinskyLilly) 
    if !(hasproperty(p.precomputed, :ᶜspecific))
        throw(ErrorException("p does not have the property ᶜspecific."))
    end

    (; Cs) = sl
    (; v_t, ᶜD, Δ_filter) = p.smagorinsky_lilly
    (; ᶜu, ᶠu³) = p.precomputed 

    # Operators
    FT = eltype(v_t)
    wdivₕ = Operators.WeakDivergence()
    hgrad = Operators.WeakGradient()
    ᶜJ = Fields.local_geometry_field(Y.c).J 

    ᶜS  = p.scratch.ᶜtemp_strain
    ᶠS  = p.scratch.ᶠtemp_strain
    ᶜϵ = p.scratch.ᶜtemp_UVWxUVW
    ᶠϵ = p.scratch.ᶠtemp_UVWxUVW
    
    localu = @. Geometry.UVWVector(ᶜu)
    ᶠu = @. Geometry.UVWVector(ᶠwinterp(ᶜJ * Y.c.ρ, Y.c.uₕ)) + Geometry.UVWVector(Y.f.u₃)
    co_ᶠu = @. Geometry.Covariant123Vector(ᶠu)

    c1 = @. Geometry.UVWVector(hgrad(localu.components.data.:1))
    c2 = @. Geometry.UVWVector(hgrad(localu.components.data.:2))
    c3 = @. Geometry.UVWVector(hgrad(localu.components.data.:3))
    @. ᶜS.components.data.:1 = c1.components.data.:1
    @. ᶜS.components.data.:2 = c2.components.data.:1
    @. ᶜS.components.data.:3 = c3.components.data.:1
    @. ᶜS.components.data.:4 = c1.components.data.:2
    @. ᶜS.components.data.:5 = c2.components.data.:2
    @. ᶜS.components.data.:6 = c3.components.data.:2
    CA.compute_strain_rate_center!(ᶜϵ, co_ᶠu)
    @. ᶜS = 1/2 * (ᶜS + adjoint(ᶜS)) + ᶜϵ

    t1 = @. Geometry.UVWVector(hgrad(ᶠu.components.data.:1))
    t2 = @. Geometry.UVWVector(hgrad(ᶠu.components.data.:2))
    t3 = @. Geometry.UVWVector(hgrad(ᶠu.components.data.:3))
    @. ᶠS.components.data.:1 = t1.components.data.:1
    @. ᶠS.components.data.:2 = t2.components.data.:1
    @. ᶠS.components.data.:3 = t3.components.data.:1
    @. ᶠS.components.data.:4 = t1.components.data.:2
    @. ᶠS.components.data.:5 = t2.components.data.:2
    @. ᶠS.components.data.:6 = t3.components.data.:2
    CA.compute_strain_rate_face!(ᶠϵ, ᶜu)
    @. ᶠS = 1/2 * (ᶠS + adjoint(ᶠS)) + ᶠϵ
    any(isnan, ᶠS) && error("Found NaN in horizontal ᶠS")

    ᶜv_t = @. (Cs * Δ_filter)^2 * sqrt(2 * CA.norm_sqr(ᶜS))
    ᶠv_t = @. ᶠinterp(ᶜv_t)
    ᶜD = @. FT(3) * ᶜv_t
    

    @. v_t = ᶜv_t
    any(isnan, ᶜv_t) && error("Found NaN in horizontal ᶜv_t")

    ᶠρ = @. ᶠwinterp(ᶜJ, Y.c.ρ)
    any(isnan, ᶠρ) && error("Found NaN in horizontal ᶠρ")
    
    any(isnan, ᶜS) && error("Found NaN in horizontal ᶜS")
    #@. Yₜ.c.uₕ += @. C12(wdivₕ(Y.c.ρ * ᶜv_t * ᶜS)) / Y.c.ρ
    @. Yₜ.c.uₕ += @. C12(wdivₕ(ᶜv_t * ᶜS))
    
    #@. Yₜ.f.u₃ += @. C3(wdivₕ(ᶠρ * ᶠv_t * ᶠS)) / ᶠρ
    @. Yₜ.f.u₃ += @. C3(wdivₕ(ᶠv_t * ᶠS))
    any(isnan, Yₜ) && error("Found NaN in horizontal Yₜ") 

    # energy adjustment
    (; ᶜspecific) = p.precomputed

    if :ρe_tot in propertynames(Yₜ.c)
       (; ᶜh_tot) = p.precomputed
        @. Yₜ.c.ρe_tot += wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜh_tot)) 
    end

    # q_tot and other tracer adjustment
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue

        # The code below adjusts the tendency by -div(\rho * d_qtot), where
        # d_qtot = -(D * grad(qtot)). The two negatives cancel out, so we have a +=

        @. ᶜρχₜ += wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜχ)) 
    end
end

function vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, colidx, sl::SmagorinskyLilly) 
    if !(hasproperty(p.precomputed, :ᶜspecific))
        throw(ErrorException("p does not have the property ᶜspecific."))
    end
    (; Cs) = sl
    (; v_t, ᶜD, Δ_filter) = p.smagorinsky_lilly
    (; ᶜu, ᶠu³, sfc_conditions, ᶜspecific) = p.precomputed 

    # Operators
    FT = eltype(v_t)
    wdivₕ = Operators.WeakDivergence()
    hgrad = Operators.WeakGradient()
    ᶜJ = Fields.local_geometry_field(Y.c).J 

    ᶜS  = p.scratch.ᶜtemp_strain
    ᶠS  = p.scratch.ᶠtemp_strain
    ᶜϵ = p.scratch.ᶜtemp_UVWxUVW
    ᶠϵ = p.scratch.ᶠtemp_UVWxUVW
    
    localu = @. Geometry.UVWVector(ᶜu)
    ᶠu = @. Geometry.UVWVector(ᶠwinterp(ᶜJ * Y.c.ρ, Y.c.uₕ)) + Geometry.UVWVector(Y.f.u₃)
    co_ᶠu = @. Geometry.Covariant123Vector(ᶠu)

    c1 = @. Geometry.UVWVector(hgrad(localu.components.data.:1))
    c2 = @. Geometry.UVWVector(hgrad(localu.components.data.:2))
    c3 = @. Geometry.UVWVector(hgrad(localu.components.data.:3))
    @. ᶜS.components.data.:1 = c1.components.data.:1
    @. ᶜS.components.data.:2 = c2.components.data.:1
    @. ᶜS.components.data.:3 = c3.components.data.:1
    @. ᶜS.components.data.:4 = c1.components.data.:2
    @. ᶜS.components.data.:5 = c2.components.data.:2
    @. ᶜS.components.data.:6 = c3.components.data.:2
    CA.compute_strain_rate_center!(ᶜϵ, co_ᶠu)
    @. ᶜS = 1/2 * (ᶜS + adjoint(ᶜS)) + ᶜϵ

    t1 = @. Geometry.UVWVector(hgrad(ᶠu.components.data.:1))
    t2 = @. Geometry.UVWVector(hgrad(ᶠu.components.data.:2))
    t3 = @. Geometry.UVWVector(hgrad(ᶠu.components.data.:3))
    @. ᶠS.components.data.:1 = t1.components.data.:1
    @. ᶠS.components.data.:2 = t2.components.data.:1
    @. ᶠS.components.data.:3 = t3.components.data.:1
    @. ᶠS.components.data.:4 = t1.components.data.:2
    @. ᶠS.components.data.:5 = t2.components.data.:2
    @. ᶠS.components.data.:6 = t3.components.data.:2
    CA.compute_strain_rate_face!(ᶠϵ, ᶜu)
    @. ᶠS = 1/2 * (ᶠS + adjoint(ᶠS)) + ᶠϵ

    ᶜv_t = @. (Cs * Δ_filter)^2 * sqrt(2 * CA.norm_sqr(ᶜS))
    ᶠv_t = @. ᶠinterp(ᶜv_t)
    ᶜD = @. FT(3) * ᶜv_t
    @. v_t = ᶜv_t
    ᶜD = @. FT(3) * ᶜv_t
    
    # Smagorinsky Computations ####
    ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
        bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ[colidx]),
    )
    #@. Yₜ.c.uₕ[colidx] -= C12(
    #    ᶜdivᵥ(
    #        -2 *
    #        ᶠinterp(Y.c.ρ[colidx]) *
    #        ᶠv_t[colidx] *
    #        ᶠS[colidx],
    #    ) / Y.c.ρ[colidx],
    #)
    @. Yₜ.c.uₕ[colidx] -= C12(
        ᶜdivᵥ(
            -2 *
            ᶠv_t[colidx] *
            ᶠS[colidx],
        ) 
    )

    #@. Yₜ.c.uₕ[colidx] -=
    #    ᶜdivᵥ_uₕ((-FT(0) * ᶠgradᵥ(Y.c.uₕ[colidx])))

    ᶜdivᵥ_u3 = Operators.DivergenceC2F(
        top = Operators.SetValue(C3(FT(0)) ⊗ C3(FT(0))),
        bottom = Operators.SetValue(C3(FT(0)) ⊗ C3(FT(0))), 
    )
    
    #@. Yₜ.f.u₃[colidx] -= C3(
    #    ᶠinterp(ᶜdivᵥ(
    #        -2 *
    #        ᶠinterp(Y.c.ρ[colidx]) *
    #        ᶠv_t[colidx] *
    #        ᶠS[colidx],
    #    ) / Y.c.ρ[colidx]),
    #)
    #
    
    @. Yₜ.f.u₃[colidx] -= C3(
        ᶠinterp(ᶜdivᵥ(
            -2 *
            ᶠv_t[colidx] *
            ᶠS[colidx],),
               )
       )
    
    if :ρe_tot in propertynames(Yₜ.c)
        (; ᶜh_tot) = p.precomputed
        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_h_tot[colidx]), 
        )
        @. Yₜ.c.ρe_tot[colidx] -= ᶜdivᵥ_ρe_tot(
					-(ᶠinterp(Y.c.ρ[colidx]) * 
					ᶠinterp(ᶜD[colidx]) * 
					ᶠgradᵥ(ᶜh_tot[colidx]))
				  )
    end

    ρ_flux_χ = p.scratch.sfc_temp_C3
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        if χ_name == :q_tot
            @. ρ_flux_χ[colidx] = sfc_conditions.ρ_flux_q_tot[colidx]
        else
            @. ρ_flux_χ[colidx] = C3(FT(0))
        end
        ᶜdivᵥ_ρχ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(ρ_flux_χ[colidx]), 
        )
        # The code below adjusts the tendency by -div(\rho * d_qtot), where
        # d_qtot = -(D * grad(qtot)). The two negatives cancel out, so we have a +=
        @. ᶜρχₜ[colidx] -= ᶜdivᵥ_ρχ(-(ᶠinterp(Y.c.ρ[colidx]) * ᶠinterp(ᶜD[colidx]) * ᶠgradᵥ(ᶜχ[colidx])))
    end
    any(isnan, Yₜ) && error("Found NaN in horizontal sgs tendency")

end
