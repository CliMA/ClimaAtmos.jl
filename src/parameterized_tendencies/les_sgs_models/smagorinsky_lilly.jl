#####
##### Smagornsky Lilly Diffusion
#####

import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators
import ClimaCore: Geometry
import ClimaCore.Utilities: half

smagorinsky_lilly_cache(Y, atmos::AtmosModel) =
    smagorinsky_lilly_cache(Y, atmos.smagorinsky_lilly)

smagorinsky_lilly_cache(Y, ::Nothing) = (;)

function smagorinsky_lilly_cache(Y, sl::SmagorinskyLilly)
    (; Cs) = sl
    FT = eltype(Y)
    h_space = Spaces.horizontal_space(axes(Y.c))
    Δᵥ_filter = Fields.Δz_field(axes(Y.c))
    Δₕ_filter = Spaces.node_horizontal_length_scale(h_space)
    ᶜtemp_scalar_3 = zero(Fields.Field(Float32, axes(Y.c.ρ)))
    v_t = ᶜtemp_scalar_3
    return (; v_t, Δₕ_filter, Δᵥ_filter)
end

horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl::SmagorinskyLilly) 
    if !(hasproperty(p.precomputed, :ᶜspecific))
        throw(ErrorException("p does not have the property ᶜspecific."))
    end

    (; Cs) = sl
    (; params) = p
    (; v_t, Δₕ_filter) = p.smagorinsky_lilly
    (; ᶜu, ᶠu³, sfc_conditions) = p.precomputed 
    
    Δ_filter = Δₕ_filter

    # Operators
    FT = eltype(v_t)
    grav = CAP.grav(params)
    ᶜJ = Fields.local_geometry_field(Y.c).J 
    Pr_t_neutral = FT(1/3)

    ᶜS  = zero(p.scratch.ᶜtemp_strain)
    ᶠS  = zero(p.scratch.ᶠtemp_strain)
    ᶜϵ = zero(p.scratch.ᶜtemp_UVWxUVW)
    ᶠϵ = zero(p.scratch.ᶠtemp_UVWxUVW)
    ᶠfb  = p.scratch.ᶠtemp_scalar
    ᶜfb  = p.scratch.ᶜtemp_scalar
    
    u_phys = @. Geometry.UVWVector(ᶜu)
    ᶠu = @. Geometry.UVWVector(ᶠinterp(Y.c.uₕ)) + Geometry.UVWVector(ᶠu³)
    @. ᶜS = Geometry.project(Geometry.UVWAxis(), gradₕ(u_phys))
    @. ᶠS = Geometry.project(Geometry.UVWAxis(), gradₕ(ᶠu))
    CA.compute_strain_rate_center!(ᶜϵ, Geometry.Covariant123Vector.(ᶠu))
    CA.compute_strain_rate_face!(ᶠϵ, Geometry.Covariant123Vector.(ᶜu))
    @. ᶜS = (ᶜS + adjoint(ᶜS))/2 + ᶜϵ
    @. ᶠS = (ᶠS + adjoint(ᶠS))/2 + ᶠϵ

    (; ᶜts) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)

    ᶠts_sfc = sfc_conditions.ts
    θ_v = @. TD.virtual_pottemp(thermo_params, ᶜts)
    θ_v_sfc = @. TD.virtual_pottemp(thermo_params, ᶠts_sfc)
    ᶜ∇ = Operators.GradientF2C();
    θc2f = Operators.InterpolateC2F(;top = Operators.Extrapolate(),
                                     bottom = Operators.SetValue(θ_v_sfc));
    ᶜ∇θ = @. ᶜ∇(ᶠinterp(θ_v))

    ᶜfb .= zero(ᶜfb) .+ FT(1)
    N² = @. grav / θ_v * Geometry.WVector(ᶜ∇θ).components.data.:1
    @. ᶜfb = (max(FT(0), 
                  1 - (N²) / Pr_t_neutral / (CA.norm_sqr(ᶜS) + eps(FT))))^(1/2)
    ᶜfb .= ifelse.(N² .<= FT(0), zero(ᶜfb) .+ FT(1),ᶜfb)

    ᶠρ = @. ᶠinterp(Y.c.ρ)
    ᶜv_t = @. (Cs * Δ_filter)^2 * sqrt(2 * CA.norm_sqr(ᶜS)) * ᶜfb
    ᶠv_t = @. ᶠinterp(ᶜv_t)
    ᶜD = @. ᶜv_t / Pr_t_neutral

    ᶜτ = @. FT(-2) * ᶜv_t * ᶜS
    ᶠτ = @. FT(-2) * ᶠv_t * ᶠS

    @. Yₜ.c.uₕ -= C12(wdivₕ(Y.c.ρ * ᶜτ) / Y.c.ρ)
    @. Yₜ.f.u₃ -= C3(wdivₕ(ᶠρ * ᶠτ) / ᶠρ)

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
        if !(χ_name in (:q_rai, :q_sno))
            @. Yₜ.c.ρ += wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜχ)) 
        end
    end

end

function vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl::SmagorinskyLilly) 
    if !(hasproperty(p.precomputed, :ᶜspecific))
        throw(ErrorException("p does not have the property ᶜspecific."))
    end
    
    (; params) = p
    (; Cs) = sl
    (; v_t, Δᵥ_filter) = p.smagorinsky_lilly
    (; ᶜu, ᶠu³, sfc_conditions, ᶜspecific, ᶜts) = p.precomputed 
    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ

    FT = eltype(v_t)
    Δ_filter = Δᵥ_filter
    Pr_t_neutral = FT(1/3)

    grav = CAP.grav(params)
    ᶜJ = Fields.local_geometry_field(Y.c).J 

    ᶜS  = zero(p.scratch.ᶜtemp_strain)
    ᶠS  = zero(p.scratch.ᶠtemp_strain)
    ᶜϵ = zero(p.scratch.ᶜtemp_UVWxUVW)
    ᶠϵ = zero(p.scratch.ᶠtemp_UVWxUVW)
    
    ᶠfb  = p.scratch.ᶠtemp_scalar
    ᶜfb  = p.scratch.ᶜtemp_scalar
    u_phys = @. Geometry.UVWVector(ᶜu)
    ᶠu = @. Geometry.UVWVector(ᶠinterp(Y.c.uₕ)) + Geometry.UVWVector(ᶠu³)
    @. ᶜS = Geometry.project(Geometry.UVWAxis(), gradₕ(u_phys))
    @. ᶠS = Geometry.project(Geometry.UVWAxis(), gradₕ(ᶠu))
    CA.compute_strain_rate_face!(ᶠϵ, Geometry.Covariant123Vector.(ᶜu))
    CA.compute_strain_rate_center!(ᶜϵ, Geometry.Covariant123Vector.(ᶠu))
    @. ᶜS = (ᶜS + adjoint(ᶜS))/2 + ᶜϵ
    @. ᶠS = (ᶠS + adjoint(ᶠS))/2 + ᶠϵ
    
    thermo_params = CAP.thermodynamics_params(p.params)
    θ_v = @. TD.virtual_pottemp(thermo_params, ᶜts)
    ᶠts_sfc = sfc_conditions.ts
    θ_v_sfc = @. TD.virtual_pottemp(thermo_params, ᶠts_sfc)
    ᶜ∇ = Operators.GradientF2C();
    θc2f = Operators.InterpolateC2F(;top = Operators.Extrapolate(),
                                     bottom = Operators.SetValue(θ_v_sfc));
    ᶜ∇θ = @. ᶜ∇(ᶠinterp(θ_v))
    N² = @. grav / θ_v * Geometry.WVector(ᶜ∇θ).components.data.:1
    ᶜfb .= zero(ᶜfb) .+ FT(1)
    @. ᶜfb = (max(FT(0), 
                  1 - (N²) / Pr_t_neutral / (CA.norm_sqr(ᶜS) + eps(FT))))^(1/2)
    ᶜfb .= ifelse.(N² .<= FT(0), zero(ᶜfb) .+ FT(1),ᶜfb)

    ᶜv_t = @. (Cs * Δ_filter)^2 * sqrt(2 * CA.norm_sqr(ᶜS)) * ᶜfb
    ᶠv_t = @. ᶠinterp(ᶜv_t)
    ᶜD = @. ᶜv_t / Pr_t_neutral
    
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

    ᶠdivᵥ = Operators.DivergenceC2F(
                         bottom = Operators.SetDivergence(FT(0)),
                         top = Operators.SetDivergence(FT(0))
                       )
    
    @. Yₜ.f.u₃ -= C3(
        ᶠdivᵥ(
            -2 *
            Y.c.ρ *
            ᶜv_t *
            ᶜS,
           ) / ᶠinterp(Y.c.ρ),
    )

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
