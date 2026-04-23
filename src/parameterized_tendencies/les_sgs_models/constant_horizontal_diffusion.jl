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
        for j in 1:n
            # Area fraction diffusion: ∂(ρa)/∂t += ∇·(ρⱼ D ∇aⱼ)
            ᶜaʲ = @. lazy(draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j)))
            @. Yₜ.c.sgsʲs.:($$j).ρa += wdivₕ(ᶜρʲs.:($$j) * ᶜD * gradₕ(ᶜaʲ))

            # Vertical velocity diffusion (scalar Laplacian on faces, no F→C→F interpolation)
            @. Yₜ.f.sgsʲs.:($$j).u₃.components.data.:1 +=
                $(FT(chd.D)) * wdivₕ(gradₕ(Y.f.sgsʲs.:($$j).u₃.components.data.:1))
        end
    end
end
