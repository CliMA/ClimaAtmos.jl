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
    (; ᶜts) = p.precomputed

    # Scalar diffusivity and reusable density factor
    D = FT(chd.D)
    ρ = Y.c.ρ
    ρD = @. lazy(ρ * D)

    # Total energy diffusion
    ᶜh_tot = @. lazy(
        TD.total_specific_enthalpy(
            thermo_params,
            ᶜts,
            specific(Y.c.ρe_tot, ρ),
        ),
    )
    @. Yₜ.c.ρe_tot += wdivₕ(ρD * gradₕ(ᶜh_tot))

    # Tracer diffusion
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜχ = @. lazy(specific(ᶜρχ, ρ))
        ᶜρχₜ_diffusion = @. lazy(wdivₕ(ρD * gradₕ(ᶜχ)))
        @. ᶜρχₜ += ᶜρχₜ_diffusion
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ += ᶜρχₜ_diffusion
        end
    end
end
