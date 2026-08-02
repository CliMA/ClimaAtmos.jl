#####
##### Vertical diffusion boundary layer parameterization
#####

import ClimaCore.Geometry: ⊗

"""
    vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t)
    vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, vert_diff_model)

Add the K-theory vertical turbulent diffusion tendencies, representing mixing in
the boundary layer and free atmosphere.

The three-argument method dispatches on `p.atmos.vertical_diffusion`; the method
for `::Nothing` is a no-op. For a `VerticalDiffusion` or
`DecayWithHeightDiffusion` model, the eddy diffusivity `ᶜK_h` comes from
`ᶜcompute_eddy_diffusivity_coefficient`, and the same value is used as the eddy
viscosity `K_u`. Face diffusivities are formed as a harmonic mean (the reciprocal
of the interpolated reciprocal), so the flux collapses at a face separating a
turbulent layer from quiescent, strongly stratified air, where an arithmetic mean
would leave about `K/2`.

Increments:

  - `Yₜ.c.uₕ`: `∇⋅τ / ρ` with the stress `τ = 2 ρ K_u S`, `S` the vertical strain
    rate. Skipped when momentum vertical diffusion is disabled. The default
    (zero-flux) boundary conditions apply, because the surface stress is added by
    `surface_flux_tendency!`.
  - `Yₜ.c.ρe_tot`: divergence of the enthalpy flux in dry-static-energy plus
    water-enthalpy form, `F_h = -ρ K_h (∇ᵥs_d + Σ_μ h_tot,μ ∇ᵥq_μ)`, with
    `s_d` the dry static energy and `h_tot,μ` the total enthalpy of water
    constituent `μ ∈ (vap, liq, ice)`. The water terms deliberately use the
    unscaled `K_h`, which keeps the energy budget consistent with the unscaled
    `ρq_tot` diffusion, preserves total-water invariance, and matches the implicit
    solver's Jacobian.
  - Every grid-mean tracer `Yₜ.c.ρχ`: divergence of `F_χ = -ρ K_h ∇ᵥχ`, where `K_h`
    is rescaled by the parameter `α_vert_diff_tracer` for the sedimenting
    microphysics species (`gs_sedimenting_tracer_candidates`).
  - `Yₜ.c.ρ`: the `ρq_tot` diffusion tendency only, so condensate and precipitation
    diffusion do not move dry mass.

Reads the precomputed `ᶜu`, `ᶜp`, `ᶜT`, `ᶜq_liq`, `ᶜq_ice`, `ᶜq_tot_nonneg`, the
geopotential `ᶜΦ` from `p.core`, and scratch space; `t` is unused. Called from
`additional_tendency!` when `p.atmos.diff_mode == Explicit()`, and from
`implicit_tendency!` otherwise. Returns `nothing`.
"""
vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t) =
    vertical_diffusion_boundary_layer_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.vertical_diffusion,
    )

vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function vertical_diffusion_boundary_layer_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::Union{VerticalDiffusion, DecayWithHeightDiffusion},
)
    FT = eltype(Y)
    (; vertical_diffusion) = p.atmos
    α_vert_diff_microphysics = CAP.α_vert_diff_tracer(p.params)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜu, ᶜp, ᶜT, ᶜq_liq, ᶜq_ice) = p.precomputed
    ᶜK_h = p.scratch.ᶜtemp_scalar
    if vertical_diffusion isa DecayWithHeightDiffusion
        ᶜK_h .= ᶜcompute_eddy_diffusivity_coefficient(Y.c.ρ, vertical_diffusion)
    elseif vertical_diffusion isa VerticalDiffusion
        ᶜK_h .= ᶜcompute_eddy_diffusivity_coefficient(
            Y.c.uₕ,
            ᶜp,
            vertical_diffusion,
        )
    end

    # Face diffusivities use a harmonic mean (reciprocal of interpolated
    # reciprocal), so the diffusive flux collapses at faces separating a
    # turbulent layer from quiescent, strongly stratified air (e.g., a
    # capping inversion), where arithmetic averaging would assign ≈ K/2.
    ϵK = eps(FT)
    if !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
        ᶠstrain_rate = compute_strain_rate_face_vertical(ᶜu)
        @. Yₜ.c.uₕ -= C12(
            ᶜdivᵥ(-2 * ᶠinterp(Y.c.ρ) / ᶠinterp(1 / max(ᶜK_h, ϵK)) * ᶠstrain_rate) / Y.c.ρ,
        ) # assumes ᶜK_u = ᶜK_h
    end

    # Total enthalpy diffusion, using the dry-static-energy + water-enthalpy
    # decomposition F_h = -K_h ∇s_d + Σ_μ h_tot,μ (-K_h ∇q_μ); see the
    # matching term in `edmfx_sgs_diffusive_flux_tendency!` for details.
    # Note: F_qμ for liquid and ice uses unscaled K_h (omitting the tracer
    # vertical diffusion factor α_vert_diff_tracer) to maintain exact energetic
    # consistency with the unscaled ρq_tot diffusion equation, preserve total
    # water invariance, and align with the implicit solver's Jacobian.
    (; ᶜΦ) = p.core
    (; ᶜq_tot_nonneg) = p.precomputed
    ᶜq_vap = @. lazy(TD.vapor_specific_humidity(ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
    ᶠgrad_h = ᶠtotal_enthalpy_gradientᵥ(thermo_params, ᶜT, ᶜΦ, ᶜq_vap, ᶜq_liq, ᶜq_ice)
    @. Yₜ.c.ρe_tot -=
        ᶜdiffdivᵥ(-(ᶠinterp(Y.c.ρ) / ᶠinterp(1 / max(ᶜK_h, ϵK)) * ᶠgrad_h))

    ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar_2
    ᶜK_h_scaled = p.scratch.ᶜtemp_scalar_3

    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        if ρχ_name in gs_sedimenting_tracer_candidates
            @. ᶜK_h_scaled = α_vert_diff_microphysics * ᶜK_h
        else
            @. ᶜK_h_scaled = ᶜK_h
        end
        ᶠρK = @. lazy(ᶠinterp(Y.c.ρ) / ᶠinterp(1 / max(ᶜK_h_scaled, ϵK)))
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        ᶜ∇ᵥρD∇χₜ = ᶜdiffusive_flux_divergenceᵥ(ᶠρK, ᶜχ)
        @. ᶜρχₜ_diffusion = ᶜ∇ᵥρD∇χₜ
        @. ᶜρχₜ -= ᶜρχₜ_diffusion
        # Only add contribution from total water diffusion to mass tendency
        # (exclude contributions from diffusion of condensate, precipitation)
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ -= ᶜρχₜ_diffusion
        end
    end
end
