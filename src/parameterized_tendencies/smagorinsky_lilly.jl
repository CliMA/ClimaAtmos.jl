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
    (;ᶜΦ) = p.core

    # Operators
    FT = eltype(v_t)
    grav = FT(9.81)
    ᶜJ = Fields.local_geometry_field(Y.c).J 

    ᶜS  = zero(p.scratch.ᶜtemp_strain)
    ᶠS  = zero(p.scratch.ᶠtemp_strain)
    ᶜϵ = zero(p.scratch.ᶜtemp_UVWxUVW)
    ᶠϵ = zero(p.scratch.ᶠtemp_UVWxUVW)
    ᶠfb  = p.scratch.ᶠtemp_scalar
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z

    localu = @. Geometry.UVWVector(ᶜu)
    ᶠu = @. Geometry.UVWVector(ᶠwinterp(ᶜJ * Y.c.ρ, Y.c.uₕ)) + Geometry.UVWVector(Y.f.u₃)
    @. ᶠS = Geometry.project(Geometry.UVWAxis(), gradₕ(ᶠu))
    @. ᶜS = Geometry.project(Geometry.UVWAxis(), gradₕ(localu))
    CA.compute_strain_rate_face!(ᶠϵ, Geometry.Covariant123Vector.(localu))
    CA.compute_strain_rate_center!(ᶜϵ, Geometry.Covariant123Vector.(ᶠu))
    @. ᶠS = (ᶠS + adjoint(ᶠS)) + ᶠϵ
    @. ᶜS = (ᶜS + adjoint(ᶜS)) + ᶜϵ

    (; sfc_conditions, ᶜts) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    ᶠts_sfc = sfc_conditions.ts
    #θ_v = @. TD.virtual_pottemp(thermo_params, ᶜts)
    #@. ᶠfb = (max(FT(0), 1 - 3*(grav / ᶠinterp(θ_v) * Geometry.WVector(ᶠgradᵥ(θ_v)).components.data.:1) / CA.norm_sqr(ᶠS)))^(1/4)

    ᶠρ = @. ᶠwinterp(ᶜJ, Y.c.ρ)
    ᶠv_t = @. (Cs * Δ_filter)^2 * sqrt(2 * CA.norm_sqr(ᶠS)) #* ᶠfb
    ᶜv_t = @. ᶜinterp(ᶠv_t)
    ᶜD = @. FT(3) * ᶜv_t

    @. v_t = ᶜv_t

    @. Yₜ.c.uₕ += C12(wdivₕ(Y.c.ρ * ᶜv_t * ᶜS)) / Y.c.ρ
    
    @. Yₜ.f.u₃ += C3(wdivₕ(ᶠρ * ᶠv_t * ᶠS)) / ᶠρ

    # energy adjustment
    (; ᶜspecific) = p.precomputed

    if :ρe_tot in propertynames(Yₜ.c)
       (; ᶜh_tot) = p.precomputed
        @. Yₜ.c.ρe_tot += wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜh_tot)) 
    end

    # q_tot and other tracer adjustment (moisture affects mass terms 
    # as well)
    for (ᶜρχₜ, ᶜχ, χ_name) in CA.matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        @. ᶜρχₜ += wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜχ)) 
        @. Yₜ.c.ρ += wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜχ)) 
    end

end

function vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl::SmagorinskyLilly) 
    if !(hasproperty(p.precomputed, :ᶜspecific))
        throw(ErrorException("p does not have the property ᶜspecific."))
    end
    (; Cs) = sl
    (; v_t, Δ_filter) = p.smagorinsky_lilly
    (; ᶜu, ᶠu³, sfc_conditions, ᶜspecific) = p.precomputed 
    (;ᶜΦ) = p.core
    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ

    # Operators
    FT = eltype(v_t)
    ᶜJ = Fields.local_geometry_field(Y.c).J 

    ᶜS  = zero(p.scratch.ᶜtemp_strain)
    ᶠS  = zero(p.scratch.ᶠtemp_strain)
    ᶜϵ = zero(p.scratch.ᶜtemp_UVWxUVW)
    ᶠϵ = zero(p.scratch.ᶠtemp_UVWxUVW)
    
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z

    ᶠfb  = p.scratch.ᶠtemp_scalar
    grav = FT(9.81)

    localu = @. Geometry.UVWVector(ᶜu)
    ᶠu = @. Geometry.UVWVector(ᶠwinterp(ᶜJ * Y.c.ρ, Y.c.uₕ)) + Geometry.UVWVector(Y.f.u₃)
    @. ᶠS = Geometry.project(Geometry.UVWAxis(), gradₕ(ᶠu))
    @. ᶜS = Geometry.project(Geometry.UVWAxis(), gradₕ(localu))
    CA.compute_strain_rate_face!(ᶠϵ, ᶜu)
    CA.compute_strain_rate_center!(ᶜϵ, Geometry.Covariant123Vector.(ᶠu))
    @. ᶠS = (ᶠS + adjoint(ᶠS)) + ᶠϵ
    @. ᶜS = (ᶜS + adjoint(ᶜS)) + ᶜϵ
    
    (; sfc_conditions, ᶜts) = p.precomputed
    ᶠts_sfc = sfc_conditions.ts
    thermo_params = CAP.thermodynamics_params(p.params)
    #θ_v = @. TD.virtual_pottemp(thermo_params, ᶜts)
    #@. ᶠfb = (max(FT(0), 1 - 3*(grav / ᶠinterp(θ_v) * Geometry.WVector(ᶠgradᵥ(θ_v)).components.data.:1) / CA.norm_sqr(ᶠS)))^(1/4)

    ᶠρ = @. ᶠwinterp(ᶜJ, Y.c.ρ)
    ᶠv_t = @. (Cs * Δ_filter)^2 * sqrt(2 * CA.norm_sqr(ᶠS)) #* ᶠfb
    ᶜv_t = @. ᶜinterp(ᶠv_t)
    ᶜD = @. FT(3) * ᶜv_t
    
    @. Yₜ.c.uₕ -= C12(
        ᶜdivᵥ(
            -2 *
            ᶠinterp(Y.c.ρ) *
            ᶠv_t *
            ᶠS,
        ) / Y.c.ρ,
    )

    # apply boundary condition for momentum flux
    ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
        bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ),
    )
    @. Yₜ.c.uₕ -= ᶜdivᵥ_uₕ(-(FT(0) * ᶠgradᵥ(Y.c.uₕ))) / Y.c.ρ
    
    @. Yₜ.f.u₃ -= C3(
        ᶠinterp(ᶜdivᵥ(
            -2 *
            ᶠinterp(Y.c.ρ) *
            ᶠv_t *
            ᶠS,
        ) / Y.c.ρ),
    )
    
    # BC verified ✓
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
        @. ᶜρχₜ -= ᶜdivᵥ_ρχ(-(ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜD) * ᶠgradᵥ(ᶜχ)))
        if !(χ_name in (:q_rai, :q_sno))
            @. Yₜ.c.ρ -= ᶜdivᵥ_ρχ(-(ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜD) * ᶠgradᵥ(ᶜχ)))
        end
    end
end
