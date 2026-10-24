#####
##### Constant Horizontal Diffusion
#####

horizontal_constant_diffusion_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

"""
    horizontal_constant_diffusion_tendency!(Yₜ, Y, p, t, chd)

Add horizontal diffusion of total energy and tracers with a spatially uniform diffusivity
to `Yₜ` in place; return `nothing`.

Total energy receives the divergence of the enthalpy flux
`-ρ D [c_p ∇ₕT + ∇ₕΦ + (h_v + Φ) ∇ₕq_tot + (h_l - h_v) ∇ₕq_liq + (h_i - h_v) ∇ₕq_ice]`,
where `c_p` is the heat capacity of the diffusing mixture (`add_sensible_heat_diffusionₕ!`),
`q_liq` and `q_ice` include rain and snow; its water terms are absent in dry
configurations. Total water diffuses into `Yₜ.c.ρq_tot` and, for mass conservation,
`Yₜ.c.ρ`; each cloud and precipitation species diffuses down its own gradient
(`add_moisture_diffusionₕ!` with `precipitation = true`); and each passive grid-scale
tracer `χ` receives `+∇ₕ·(ρ D ∇ₕχ)`, with the constant diffusivity `D = chd.D`
[m²/s]. This model diffuses scalars only.

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

    # Total energy diffusion: sensible heat, then the water enthalpy flux
    # with the water species, precipitation included
    add_sensible_heat_diffusionₕ!(
        Yₜ.c.ρe_tot,
        ᶜρD,
        p.precomputed.ᶜT,
        ᶜdiffusing_water_phases(Y, p; precipitation = true),
        p,
        ᶜtemp_scalar_3,
    )
    add_moisture_diffusionₕ!(
        Yₜ,
        Y,
        p,
        ᶜρD,
        ᶜtemp_scalar_3,
        ᶜtemp_scalar_2;
        precipitation = true,
    )

    # Passive (non-microphysics) grid-scale tracers
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ρχ_name in microphysics_tracer_names(Y) && return
        ᶜχ = @. lazy(specific(ᶜρχ, ᶜρ))
        @. ᶜρχₜ += wdivₕ(ᶜρ * ᶜD * gradₕ(ᶜχ))
    end
end
