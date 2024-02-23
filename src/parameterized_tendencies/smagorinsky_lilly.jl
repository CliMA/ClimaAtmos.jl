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
    νₜ = ᶜtemp_scalar_3
    @. νₜ = ((Cs * cbrt(ᶜJ))^2)*sqrt(2 * (ᶜshear²))
    return (; νₜ, ᶜD)
end

horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) = nothing


function horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl::SmagorinskyLilly) 
    if !(hasproperty(p.precomputed, :ᶜspecific))
        throw(ErrorException("p does not have the property ᶜspecific."))
    end

    (; Cs) = sl
    (; νₜ, ᶜD) = p.smagorinsky_lilly
    (; ᶜu, ᶠu³) = p.precomputed 

    # momentum balance adjustment
 
    # Velocity composition onto cell faces #
    ᶠu = p.scratch.ᶠtemp_C123
    @. ᶠu = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³)

    # Set up scratch space for strain rates
    ᶜϵ = p.scratch.ᶜtemp_UVWxUVW
    ᶠϵ = p.scratch.ᶠtemp_UVWxUVW
    # Compute strain rates
    compute_strain_rate_center!(ᶜϵ, ᶠu)
    compute_strain_rate_face!(ᶠϵ, ᶜu)
    Δ = eltype(Cs)(300)
    ᶜνₜ = @. (Cs * Δ)^2 * sqrt(norm_sqr(ᶜϵ))
    ᶠνₜ = @. (Cs * Δ)^2 * sqrt(norm_sqr(ᶠϵ))
    @. ᶜD = 3 * ᶜνₜ
    
    # Smagorinsky Operators #
    wdivₕ = Operators.WeakDivergence()
    hgrad = Operators.Gradient()

    # Compute the 3D Cartesian Components
    #@. Yₜ.c.uₕ -= -2 * ᶜνₜ * (wgradₕ(divₕ(-Y.c.uₕ)) - C12(wcurlₕ(C3(curlₕ(-Y.c.uₕ)))))
    #@. Yₜ.f.u₃ -= -2 * ᶠνₜ * (-C3(wcurlₕ(C12(curlₕ(-Y.f.u₃)))))

    ρτ = @. -2 * ᶠinterp(Y.c.ρ) * ᶠνₜ * ᶠϵ
    ρτc = @. -2 * Y.c.ρ * ᶜνₜ * ᶜϵ

    @show ᶜνₜ
    @show ᶠνₜ

    ρτ11 = ρτ.components.data.:1
    ρτ12 = ρτ.components.data.:4
    ρτ13 = ρτ.components.data.:7
    ρτ22 = ρτ.components.data.:5
    ρτ23 = ρτ.components.data.:8
    ρτ33 = ρτ.components.data.:9

    # Assert stress tensor symmetry. 
    ρτc11 = ρτc.components.data.:1
    ρτc12 = ρτc.components.data.:4
    ρτc13 = ρτc.components.data.:7
    ρτc22 = ρτc.components.data.:5
    ρτc23 = ρτc.components.data.:8
    ρτc33 = ρτc.components.data.:9

    @. Yₜ.c.uₕ.components.data.:1 -= hgrad(ρτc11).components.data.:1 / Y.c.ρ
    @. Yₜ.c.uₕ.components.data.:1 -= hgrad(ρτc12).components.data.:2 / Y.c.ρ
    
    @. Yₜ.c.uₕ.components.data.:2 -= hgrad(ρτc12).components.data.:1 / Y.c.ρ
    @. Yₜ.c.uₕ.components.data.:2 -= hgrad(ρτc22).components.data.:2 / Y.c.ρ
    
    @. Yₜ.f.u₃.components.data.:1 -= hgrad(ρτ13).components.data.:1 / ᶠinterp(Y.c.ρ)
    @. Yₜ.f.u₃.components.data.:1 -= hgrad(ρτ23).components.data.:2 / ᶠinterp(Y.c.ρ)
    
    # energy adjustment
    (; ᶜspecific) = p.precomputed

    if :ρe_tot in propertynames(Yₜ.c)
        (; ᶜh_tot) = p.precomputed
        @. Yₜ.c.ρe_tot += divₕ(Y.c.ρ * ᶜD * gradₕ(ᶜh_tot)) 
    end

    # q_tot and other tracer adjustment
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue

        # The code below adjusts the tendency by -div(\rho * d_qtot), where
        # d_qtot = -(D * grad(qtot)). The two negatives cancel out, so we have a +=

        @. ᶜρχₜ += divₕ(Y.c.ρ * ᶜD * gradₕ(ᶜχ)) 
    end

end

function vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, colidx, sl::SmagorinskyLilly) 
    if !(hasproperty(p.precomputed, :ᶜspecific))
        throw(ErrorException("p does not have the property ᶜspecific."))
    end

    (; Cs) = sl
    (; νₜ, ᶜD) = p.smagorinsky_lilly
    (; ᶜspecific, sfc_conditions) = p.precomputed
    (; ᶜu, ᶠu³) = p.precomputed 
    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ
    
    ρ_flux_χ = p.scratch.ᶜtemp_scalar

    FT = eltype(Y)
    
    # Smagorinsky Computations ####
    ᶠu = p.scratch.ᶠtemp_C123
    @. ᶠu = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³)
    ᶜϵ = p.scratch.ᶜtemp_UVWxUVW
    ᶠϵ = p.scratch.ᶠtemp_UVWxUVW
    compute_strain_rate_center!(ᶜϵ, ᶠu)
    compute_strain_rate_face!(ᶠϵ, ᶜu)
    Δ = eltype(Cs)(300)
    ᶜνₜ = @. (Cs * Δ)^2 * sqrt(norm_sqr(ᶜϵ))
    ᶠνₜ = @. (Cs * Δ)^2 * sqrt(norm_sqr(ᶠϵ))
    @. ᶜD = 3 * ᶜνₜ
    
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

    # TODO: is the code below doing what you think it does?
    divᵥ_u3 = Operators.DivergenceC2F(
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

    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        if χ_name == :q_tot
            @. ρ_flux_χ[colidx] = sfc_conditions.ρ_flux_q_tot[colidx]
        elseif χ_name == :θ
            @. ρ_flux_χ[colidx] = sfc_conditions.ρ_flux_θ[colidx]
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
