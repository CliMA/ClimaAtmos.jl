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
    ᶜtemp_scalar_2 = similar(Y.c.ρ, FT)
    ᶜtemp_scalar_3 = similar(Y.c.ρ, FT)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶜD = ᶜtemp_scalar_2
    νₜ = ᶜtemp_scalar_3
    @. νₜ *= FT(0)
    @. ᶜD *= FT(0)
    Δ_filter = @. cbrt(ᶜJ)
    return (; νₜ, ᶜD, Δ_filter)
end

horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) = nothing

function horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl::SmagorinskyLilly) 
    if !(hasproperty(p.precomputed, :ᶜspecific))
        throw(ErrorException("p does not have the property ᶜspecific."))
    end

    (; Cs) = sl
    (; νₜ, ᶜD, Δ_filter) = p.smagorinsky_lilly
    (; ᶜu, ᶠu³) = p.precomputed 

    # Operators
    FT = eltype(νₜ)
    wdivₕ = Operators.WeakDivergence()
    hgrad = Operators.Gradient()

    ᶜS  = p.scratch.ᶜtemp_strain
    ᶠS  = p.scratch.ᶠtemp_strain
    ᶜϵ = p.scratch.ᶜtemp_UVWxUVW
    ᶠϵ = p.scratch.ᶠtemp_UVWxUVW
    
    localu = @. Geometry.UVWVector(ᶜu)
    ᶠu = @. Geometry.UVWVector(ᶠinterp(Y.c.uₕ)) + Geometry.UVWVector(ᶠu³)
    co_ᶠu = @. Geometry.Covariant123Vector(ᶠinterp(Y.c.uₕ)) + Geometry.Covariant123Vector(ᶠu³)

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

    Spaces.weighted_dss!(ᶜS)
    Spaces.weighted_dss!(ᶠS)
    ᶜνₜ = @. (Cs * Δ_filter)^2 * sqrt(2 * CA.norm_sqr(ᶜS))
    ᶠνₜ = @. ᶠinterp(ᶜνₜ)
    ᶜD = @. FT(3) * ᶜνₜ

    @. νₜ = ᶜνₜ

    ᶠρ = @. ᶠinterp(Y.c.ρ)
    
    @. Yₜ.c.uₕ -= @. C12(wdivₕ(Y.c.ρ * ᶜνₜ * ᶜS)) / Y.c.ρ
    @. Yₜ.f.u₃ -= @. C3(wdivₕ(ᶠρ * ᶠνₜ * ᶠS)) / ᶠρ

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
    (; νₜ, ᶜD, Δ_filter) = p.smagorinsky_lilly
    (; ᶜspecific, sfc_conditions) = p.precomputed
    (; ᶜu, ᶠu³) = p.precomputed 
    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ
    
    wdivₕ = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    
    ρ_flux_χ = p.scratch.ᶜtemp_scalar

    FT = eltype(Y)
    
    ᶜϵ = p.scratch.ᶜtemp_UVWxUVW
    
    (; Cs) = sl
    (; νₜ, ᶜD, Δ_filter) = p.smagorinsky_lilly
    (; ᶜu, ᶠu³) = p.precomputed 

    # Operators
    FT = eltype(νₜ)
    wdivₕ = Operators.WeakDivergence()
    hgrad = Operators.Gradient()
    
    ᶜS  = p.scratch.ᶜtemp_strain
    ᶠS  = p.scratch.ᶠtemp_strain
    ᶜϵ = p.scratch.ᶜtemp_UVWxUVW
    ᶠϵ = p.scratch.ᶠtemp_UVWxUVW
    
    localu = @. Geometry.UVWVector(ᶜu)
    ᶠu = @. Geometry.UVWVector(ᶠinterp(Y.c.uₕ)) + Geometry.UVWVector(ᶠu³)
    co_ᶠu = @. Geometry.Covariant123Vector(ᶠinterp(Y.c.uₕ)) + Geometry.Covariant123Vector(ᶠu³)

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

    Spaces.weighted_dss!(ᶜS)
    Spaces.weighted_dss!(ᶠS)
    ᶜνₜ = @. (Cs * Δ_filter)^2 * sqrt(2 * CA.norm_sqr(ᶜS))
    #ᶠνₜ = @. (Cs * Δ_filter)^2 * sqrt(2 * CA.norm_sqr(ᶠS))
    ᶠνₜ = @. ᶠinterp(ᶜνₜ)
    ᶜD = @. FT(3) * ᶜνₜ
    
    # Smagorinsky Computations ####
    ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
        bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ[colidx]),
    )
    @. Yₜ.c.uₕ[colidx] -= C12(
        ᶜdivᵥ(
            -2 *
            ᶠinterp(Y.c.ρ[colidx]) *
            ᶠνₜ[colidx] *
            ᶠϵ[colidx],
        ) / Y.c.ρ[colidx],
    )
    @. Yₜ.c.uₕ[colidx] -=
        ᶜdivᵥ_uₕ(-(FT(0) * ᶠgradᵥ(Y.c.uₕ[colidx]))) / Y.c.ρ[colidx]

    ᶜdivᵥ_u3 = Operators.DivergenceC2F(
        top = Operators.SetValue(C3(FT(0)) ⊗ C3(FT(0))),
        bottom = Operators.SetValue(C3(FT(0)) ⊗ C3(FT(0))), 
    )
    @. Yₜ.f.u₃[colidx] -= C3(
        ᶠinterp(ᶜdivᵥ(
            -2 *
	    ᶠinterp(Y.c.ρ[colidx]) *
            ᶠνₜ[colidx] *
            ᶠϵ[colidx],
        ) / Y.c.ρ[colidx]),
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
    
end
