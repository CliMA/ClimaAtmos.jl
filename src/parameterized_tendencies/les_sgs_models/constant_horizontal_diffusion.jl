#####
##### Constant Horizontal Diffusion
#####

horizontal_constant_diffusion_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

"""
    horizontal_constant_diffusion_tendency!(Yₜ, Y, p, t, chd)

Add horizontal diffusion of total energy and tracers with a spatially uniform diffusivity
to `Yₜ` in place; return `nothing`.

Total energy receives `+∇ₕ·(ρ D ∇ₕh_tot)` and each grid-scale tracer `χ` receives
`+∇ₕ·(ρ D ∇ₕχ)`, with the constant diffusivity `D = chd.D` [m²/s]. The `ρq_tot` diffusion
is also added to `Yₜ.c.ρ` so that moisture diffusion conserves mass. Momentum is left
untouched: unlike the Smagorinsky-Lilly and AMD closures, this model diffuses scalars only.

Reads `ᶜh_tot` from `p.precomputed` and uses `p.scratch.ᶜtemp_scalar`. The tendency is
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
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜtemp_scalar) = p.scratch

    ᶜD = @. ᶜtemp_scalar = FT(chd.D)

    # Total energy diffusion
    (; ᶜh_tot) = p.precomputed
    @. Yₜ.c.ρe_tot += wdivₕ(Y.c.ρ * ᶜD * gradₕ(ᶜh_tot))

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
