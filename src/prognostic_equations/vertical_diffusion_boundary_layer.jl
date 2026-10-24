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
  - `Yₜ.c.ρe_tot`: divergence of the enthalpy flux
    `F_E = -ρ K_h [c_p ∇ᵥT + ∇ᵥΦ + (h_l - h_v) ∇ᵥq_lcl + (h_i - h_v) ∇ᵥq_icl + (h_v + Φ) ∇ᵥq_tot_eff]`,
    the water part of Eq. (33) of Yatunin et al. (2026) with the sensible heat
    carried at the heat capacity `c_p` of the diffusing mixture
    (`add_sensible_heat_diffusionᵥ!`), where
    `q_tot_eff = q_tot - q_rai - q_sno` is the water that diffuses. The
    sensible-heat term is added for every configuration; the water terms only
    when `ρq_tot` is prognostic.
  - `Yₜ.c.ρq_tot` and `Yₜ.c.ρ`: the total-water flux `F = -ρ K_h ∇ᵥq_tot_eff`,
    applied to both with the same sign, so that diffusing water carries the
    corresponding moist-air mass.
  - Suspended cloud species (`suspended_cloud_species`: `ρq_lcl`, `ρq_icl`,
    `ρn_lcl`, `ρn_ice`, and the P3 `ρq_rim` and `ρb_rim`): each diffuses down its
    own vertical gradient, `F_χ = -ρ K_h ∇ᵥχ` (`add_scalar_diffusionᵥ!`).
  - Rain, snow, and rain number density (`ρq_rai`, `ρq_sno`, `ρn_rai`) are not
    diffused, since they sediment rather than mix.
  - Passive (non-microphysics) grid-mean tracers: divergence of
    `F_χ = -ρ K_h ∇ᵥχ`, diffused independently with the full `K_h`.

Reads the precomputed `ᶜu`, `ᶜp`, `ᶜT`, the
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
    ϵ_FT = eps(FT)
    (; vertical_diffusion) = p.atmos
    (; ᶜu, ᶜp) = p.precomputed
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
    ᶠρK = @. lazy(ᶠinterp(Y.c.ρ) / ᶠinterp(1 / max(ᶜK_h, ϵ_FT)))
    if !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
        ᶠstrain_rate = compute_strain_rate_face_vertical(ᶜu)
        @. Yₜ.c.uₕ -= C12(
            ᶜdivᵥ(-2 * ᶠρK * ᶠstrain_rate) / Y.c.ρ,
        ) # assumes ᶜK_u = ᶜK_h
    end

    # Energy, water, and passive tracers; rain and snow do not diffuse.
    add_scalar_diffusionᵥ!(
        Yₜ,
        Y,
        p,
        ᶠρK,
        p.scratch.ᶜtemp_scalar_3,
        p.scratch.ᶜtemp_scalar_2,
    )
end
