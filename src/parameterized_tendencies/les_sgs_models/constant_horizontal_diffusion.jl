#####
##### Constant Horizontal Diffusion
#####

horizontal_constant_diffusion_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

"""
    horizontal_constant_diffusion_tendency!(Yₜ, Y, p, t, chd)

Add horizontal diffusion of total energy and tracers with a spatially uniform diffusivity
to `Yₜ` in place; return `nothing`.

Total energy receives the divergence of the split enthalpy flux
`-ρ D [∇ₕs_d + (h_eff + Φ) ∇ₕq_tot_eff]` (see `ᶜh_eff_plus_Φ!`), whose water part is
absent in dry configurations, and each grid-scale tracer `χ` receives `+∇ₕ·(ρ D ∇ₕχ)`,
with the constant diffusivity `D = chd.D` [m²/s]. The `ρq_tot` diffusion is also added to
`Yₜ.c.ρ` so that moisture diffusion conserves mass. This model diffuses scalars only.

Reads `ᶜT` from `p.precomputed` and writes `p.scratch.ᶜtemp_scalar` and `ᶜtemp_scalar_2`.
The tendency is always applied explicitly, from `remaining_tendency!`; the `::Nothing`
method is a no-op.
"""
function horizontal_constant_diffusion_tendency!(
    Yₜ,
    Y,
    p,
    t,
    chd::ConstantHorizontalDiffusion,
)
    FT = eltype(Y)
    (; ᶜtemp_scalar, ᶜtemp_scalar_2) = p.scratch

    ᶜD = @. ᶜtemp_scalar = FT(chd.D)

    # Total energy diffusion: the dry-static-energy part of the enthalpy flux,
    # then the enthalpy carried by the diffusing water
    ᶜs_d = ᶜdry_static_energy(p)
    @. Yₜ.c.ρe_tot += wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜs_d))
    if !(p.atmos.microphysics_model isa DryModel)
        ᶜh_eff_plus_Φ = ᶜh_eff_plus_Φ!(ᶜtemp_scalar_2, Y, p)
        ᶜq_tot_eff = ᶜdiffusing_water(Y, p)
        @. Yₜ.c.ρe_tot += wdivₕ(Y.c.ρ * ᶜD * ᶜh_eff_plus_Φ * gradₕ(ᶜq_tot_eff))
    end

    # Tracer diffusion
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        ᶜρχₜ_diffusion = @. lazy(wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜχ)))
        @. ᶜρχₜ += ᶜρχₜ_diffusion
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ += ᶜρχₜ_diffusion
        end
    end
end
