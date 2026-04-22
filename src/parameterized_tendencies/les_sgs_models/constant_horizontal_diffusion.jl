#####
##### Constant Horizontal Diffusion
#####

horizontal_constant_diffusion_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

"""
    horizontal_constant_diffusion_tendency!(Yₜ,Y, p, t, ::ConstantHorizontalDiffusion)

"""
function horizontal_constant_diffusion_tendency!(
    Yₜ,
    Y,
    p,
    t,
    chd::ConstantHorizontalDiffusion,
)
    FT = eltype(Y)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜtemp_scalar, ᶠtemp_scalar) = p.scratch

    ᶜD = @. ᶜtemp_scalar = FT(chd.D)

    # # Total energy diffusion
    # (; ᶜh_tot) = p.precomputed
    # @. Yₜ.c.ρe_tot += wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜh_tot))

    # # Tracer diffusion
    # foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
    #     ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
    #     ᶜρχₜ_diffusion = @. lazy(wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜχ)))
    #     @. ᶜρχₜ += ᶜρχₜ_diffusion
    #     if ρχ_name == @name(ρq_tot)
    #         @. Yₜ.c.ρ += ᶜρχₜ_diffusion
    #     end
    # end

    # Sub-grid scale diffusion for prognostic EDMFX
    turbconv_model = p.atmos.turbconv_model
    if turbconv_model isa PrognosticEDMFX
        n = n_mass_flux_subdomains(turbconv_model)
        (; ᶜρʲs) = p.precomputed
        ᶜJ = Fields.local_geometry_field(Y.c).J
        ᶜρ = Y.c.ρ
        for j in 1:n
            # Area fraction diffusion: ∂(ρa)/∂t += ∇·(ρⱼ D ∇aⱼ)
            ᶜaʲ = @. lazy(draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)))
            @. Yₜ.c.sgsʲs.:($$j).ρa += wdivₕ(ᶜρʲs.:($$j) * ᶜD * gradₕ(ᶜaʲ))

            # Vertical velocity diffusion
            ᶜuʲ = p.precomputed.ᶜuʲs.:($j)
            ᶜ∇²uʲ = @. p.hyperdiff.ᶜ∇²u = C123(wgradₕ(divₕ(ᶜuʲ))) - C123(wcurlₕ(C123(curlₕ(ᶜuʲ))))
            @. Yₜ.f.sgsʲs.:($$j).u₃ += ᶠwinterp(ᶜJ * ᶜρ, C3(ᶜD * ᶜ∇²uʲ))
        end
    end
end
