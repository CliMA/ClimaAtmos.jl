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
    #
    # p.scratch.ᶜtemp_UVWxUVW can be used for strain-rate calculations using the 
    # ClimaAtmos utilities functions
 
    # Smagorinsky Computations ####
    ᶠu = p.scratch.ᶠtemp_C123
    @. ᶠu = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³)
    ᶜϵ = p.scratch.ᶜtemp_UVWxUVW
    ᶠϵ = p.scratch.ᶠtemp_UVWxUVW
    compute_strain_rate_center!(ᶜϵ, ᶠu)
    compute_strain_rate_face!(ᶠϵ, ᶜu)
    Δ = eltype(Cs)(100)
    ᶜνₜ = @. (Cs * Δ)^2 * sqrt(norm(ᶜϵ))
    ᶠνₜ = @. (Cs * Δ)^2 * sqrt(norm(ᶠϵ))
    @. ᶜD = 3 * ᶜνₜ
    # Smagorinsky Computations ####

    # construct 3D cartesian component

    @. Yₜ.c.uₕ -=
    2 * ᶜνₜ * (wgradₕ(divₕ(-Y.c.uₕ)) - C12(wcurlₕ(C3(curlₕ(-Y.c.uₕ)))))

    @. Yₜ.f.u₃ -= 2 * ᶠνₜ * (-C3(wcurlₕ(C12(curlₕ(-Y.f.u₃)))))
    
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
    
    ρ_flux_χ = p.scratch.ᶜtemp_scalar

    FT = eltype(Y)
    
    # Smagorinsky Computations ####
    ᶠu = p.scratch.ᶠtemp_C123
    @. ᶠu = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³)
    ᶜϵ = p.scratch.ᶜtemp_UVWxUVW
    ᶠϵ = p.scratch.ᶠtemp_UVWxUVW
    compute_strain_rate_center!(ᶜϵ, ᶠu)
    compute_strain_rate_face!(ᶠϵ, ᶜu)
    Δ = eltype(Cs)(100)
    ᶜνₜ = @. (Cs * Δ)^2 * sqrt(norm(ᶜϵ))
    ᶠνₜ = @. (Cs * Δ)^2 * sqrt(norm(ᶠϵ))
    @. ᶜD = 3 * ᶜνₜ
    # Smagorinsky Computations ####

    ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
        bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ[colidx]),
    )
    @. Yₜ.c.uₕ[colidx] -=
        ᶜdivᵥ_uₕ(-(ᶠinterp(Y.c.ρ[colidx]) * ᶠinterp(2 * ᶜνₜ[colidx]) * ᶠgradᵥ(Y.c.uₕ[colidx]))) / Y.c.ρ[colidx]

    # TODO: is the code below doing what you think it does?
    divᵥ_u3 = Operators.DivergenceC2F(
        top = Operators.SetValue(C3(FT(0)) ⊗ C3(FT(0))),
        bottom = Operators.SetValue(C3(FT(0)) ⊗ C3(FT(0))), 
    )

    @. Yₜ.f.u₃[colidx] -= 
        divᵥ_u3(-(Y.c.ρ[colidx] * 2 * ᶜνₜ[colidx] * ᶜgradᵥ(Y.f.u₃[colidx]))) / ᶠinterp(Y.c.ρ[colidx])
    
    if :ρe_tot in propertynames(Yₜ.c)
        (; ᶜh_tot) = p.precomputed
        
        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_h_tot[colidx]), 
        )

        @. Yₜ.c.ρe_tot[colidx] -= ᶜdivᵥ_ρe_tot(-(ᶠinterp(Y.c.ρ[colidx]) * ᶠinterp(ᶜD[colidx]) * ᶠgradᵥ(ᶜh_tot[colidx])))
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
