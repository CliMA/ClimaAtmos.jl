#####
##### Constant Horizontal Diffusion
#####

horizontal_constant_diffusion_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

"""
    horizontal_constant_diffusion_tendency!(Yₜ, Y, p, t, chd)

Add horizontal diffusion of total energy and tracers with a spatially uniform diffusivity
to `Yₜ` in place; return `nothing`.

Total energy receives the divergence of the enthalpy flux
`-ρ D [∇ₕs_d + (h_v + Φ) ∇ₕq_tot_eff + (h_l - h_v) ∇ₕq_lcl + (h_i - h_v) ∇ₕq_icl]`,
whose water terms are absent in dry configurations. Total water diffuses on `q_tot_eff`
(with the tendency mirrored to `Yₜ.c.ρ` for mass conservation while suspended cloud
species diffuse down their own gradients and rain and snow do not diffuse), and each
passive grid-scale tracer `χ` receives `+∇ₕ·(ρ D ∇ₕχ)`, with the constant diffusivity
`D = chd.D` [m²/s]. This model diffuses scalars only.

Reads `ᶜT` from `p.precomputed` and writes `p.scratch` scalar fields. The tendency is
always applied explicitly, from `remaining_tendency!`; the `::Nothing` method is a no-op.
"""
function horizontal_constant_diffusion_tendency!(
    Yₜ,
    Y,
    p,
    t,
    chd::ConstantHorizontalDiffusion,
)
    FT = eltype(Y)
    (; ᶜtemp_scalar, ᶜtemp_scalar_2, ᶜtemp_scalar_3) = p.scratch
    ᶜρ = Y.c.ρ

    ᶜD = @. ᶜtemp_scalar = FT(chd.D)
    ᶜρD = @. lazy(ᶜρ * ᶜD)

    # Total energy diffusion: dry static energy plus the moisture enthalpy flux
    # -ρ D [(h_v + Φ) ∇ₕq_tot_eff + (h_l - h_v) ∇ₕq_lcl + (h_i - h_v) ∇ₕq_icl]
    ᶜs_d = ᶜdry_static_energy(p)
    @. Yₜ.c.ρe_tot += wdivₕ(ᶜρD * gradₕ(ᶜs_d))
    if !(p.atmos.microphysics_model isa DryModel)
        ᶜq_tot_eff = ᶜdiffusing_water(Y, p)
        ᶜρq_tot_diff = ᶜtemp_scalar_2
        @. ᶜρq_tot_diff = wdivₕ(ᶜρD * gradₕ(ᶜq_tot_eff))
        @. Yₜ.c.ρq_tot += ᶜρq_tot_diff
        @. Yₜ.c.ρ += ᶜρq_tot_diff
        add_water_enthalpy_diffusionₕ!(Yₜ.c.ρe_tot, ᶜρD, Y, p, ᶜtemp_scalar_3)
        diffuse_suspended_cloud_speciesₕ!(Yₜ, Y, ᶜρD)
    end

    # Passive (non-microphysics) grid-scale tracers
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ρχ_name in microphysics_tracer_names(Y) && return
        ᶜχ = @. lazy(specific(ᶜρχ, ᶜρ))
        @. ᶜρχₜ += wdivₕ(ᶜρ * ᶜD * gradₕ(ᶜχ))
    end
end
