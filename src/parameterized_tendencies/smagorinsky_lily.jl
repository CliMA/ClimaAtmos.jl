#####
##### Smagorinsky Lily Diffusion
#####

import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators
import LinearAlgebra as la


horizontal_smagorinsky_lily_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
vertical_smagorinsky_lily_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) = nothing


function horizontal_smagorinsky_lily_tendency!(Yₜ, Y, p, t, sl::SmagorinskyLily) 
    if !(hasproperty(p, :ᶜspecific))
        throw(ErrorException("p does not have the property ᶜspecific."))
    end

    # Cs is the Smagorinsky coefficient, ^cJ is the volume of the cell. ^cD is 
    # calculated from the Smagorinsky-Lily model for eddy viscosity as described
    # in (Sridhar et al. 2022).

    # D = v/Pr, where v = (Cs * cbrt(J))^2 * sqrt(2*Sij*Sij). The cube root of the
    # volume is the average length of the sides of the cell.

    (; Cs) = sl

    FT = eltype(Y)
    (; ᶠu³, ᶜu, ᶜD) = p
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜshear² = p.ᶜtemp_scalar
    ᶠu = p.ᶠtemp_C123
    @. ᶠu = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³)
    ct3_unit = p.ᶜtemp_CT3
    @. ct3_unit = CT3(Geometry.WVector(FT(1)), ᶜlg)
    @. ᶜshear² = norm_sqr(adjoint(CA.ᶜgradᵥ(ᶠu)) * ct3_unit)

    ᶜJ = Fields.local_geometry_field(Y.c).J
    # ᶜD = p.ᶜtemp_scalar
    two_v_t = p.ᶜtemp_scalar_2
    @. two_v_t = 2*((Cs * cbrt(ᶜJ))^2)*sqrt(2 * (ᶜshear²))
    Pr = 1/3
    @. ᶜD = (1/Pr)*((Cs * cbrt(ᶜJ))^2)*sqrt(2 * (ᶜshear²))


    @. Yₜ.c.uₕ[colidx] -=
    divₕ(-(Y.c.ρ[colidx] * two_v_t[colidx] * gradₕ(Y.c.uₕ[colidx]))) / Y.c.ρ[colidx]

    (; ᶜspecific) = p

    if :ρe_tot in propertynames(Yₜ.c)
        (; ᶜh_tot) = p
        @. Yₜ.c.ρe_tot += divₕ(Y.c.ρ * ᶜD * gradₕ(ᶜh_tot)) 
    end
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue

        # The code below adjusts the tendency by -div(\rho * d_qtot), where
        # d_qtot = -(D * grad(qtot)). The two negatives cancel out, so we have a +=

        @. ᶜρχₜ += divₕ(Y.c.ρ * ᶜD * gradₕ(ᶜχ)) 
    end

    
end

function vertical_smagorinsky_lily_tendency!(Yₜ, Y, p, t, colidx, sl::SmagorinskyLily) 
    if !(hasproperty(p, :ᶜspecific))
        throw(ErrorException("p does not have the property ᶜspecific."))
    end

    # Cs is the Smagorinsky coefficient, ^cJ is the volume of the cell. ^cD is 
    # calculated from the Smagorinsky-Lily model for eddy viscosity as described
    # in (Sridhar et al. 2022).

    # D = v/Pr, where v = (Cs * cbrt(J))^2 * sqrt(2*Sij*Sij). The cube root of the
    # volume is the average length of the sides of the cell.

    (; Cs) = sl

    FT = eltype(Y)
    (; ᶠu³, ᶜu, ᶜD) = p
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜshear² = p.ᶜtemp_scalar
    ᶠu = p.ᶠtemp_C123
    @. ᶠu = C123(ᶠinterp(Y.c.uₕ)) + C123(ᶠu³)
    ct3_unit = p.ᶜtemp_CT3
    @. ct3_unit = CT3(Geometry.WVector(FT(1)), ᶜlg)
    @. ᶜshear² = norm_sqr(adjoint(CA.ᶜgradᵥ(ᶠu)) * ct3_unit)

    ᶜJ = Fields.local_geometry_field(Y.c).J
    # ᶜD = p.ᶜtemp_scalar
    two_v_t = p.ᶜtemp_scalar_2
    @. two_v_t = 2*((Cs * cbrt(ᶜJ))^2)*sqrt(2 * (ᶜshear²))
    Pr = 1/3
    @. ᶜD = (1/Pr)*((Cs * cbrt(ᶜJ))^2)*sqrt(2 * (ᶜshear²))


    (; ᶜspecific, sfc_conditions) = p
    
    ρ_flux_χ = p.sfc_temp_C3

    
    ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
        bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ[colidx]),
    )
    @. Yₜ.c.uₕ[colidx] -=
        ᶜdivᵥ_uₕ(-(ᶠinterp(Y.c.ρ[colidx]) * ᶠinterp(two_v_t[colidx]) * ᶠgradᵥ(Y.c.uₕ[colidx]))) / Y.c.ρ[colidx]
    

    if :ρe_tot in propertynames(Yₜ.c)
        (; ᶜh_tot) = p
        
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