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
  - `Yₜ.c.ρe_tot`: divergence of the single-gradient enthalpy flux
    `F_E = -ρ K_h [∇ᵥs_d + (h_eff + Φ) ∇ᵥq_tot_eff]`, where `s_d = h_d + Φ` is the
    dry static energy, `q_tot_eff = q_tot - q_rai - q_sno` is the water that
    actually diffuses, and `h_eff = (h_v q_v + h_l q_lcl + h_i q_icl) / max(q_water_nonneg, ε)` is the mass-weighted enthalpy of that water. The dry
    term is added for every configuration; the water term only when `ρq_tot` is
    prognostic.
  - `Yₜ.c.ρq_tot` and `Yₜ.c.ρ`: the total-water flux `F = -ρ K_h ∇ᵥq_tot_eff`,
    applied to both with the same sign, so that diffusing water carries the
    corresponding moist-air mass.
  - `Yₜ.c.ρq_lcl` and `Yₜ.c.ρq_icl`: no flux of their own. Each takes a share of
    the aggregate water tendency, scaled by the clipped ratio
    `min(q_μ / q_tot_eff, 1)`, and the matching number densities `ρn_lcl` and
    `ρn_icl` scale with it too, which preserves the mean particle mass.
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
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜu, ᶜp, ᶜT) = p.precomputed
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

    # Enthalpy diffusion. Dry static energy piece applies in all
    # configurations (including dry); the water enthalpy piece is added
    # below when ρq_tot is prognostic.
    (; ᶜΦ) = p.core
    @. Yₜ.c.ρe_tot -=
        ᶜdiffdivᵥ(-(ᶠρK * ᶠgradᵥ(TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ))))

    # Water diffusion on q_tot_eff = q_tot - q_rai - q_sno (rain/snow
    # excluded). Cloud species (lcl, icl) inherit their share of the
    # aggregate q_tot diffusion via clipped ratio; rain, snow, and n_rai
    # do not diffuse. Enthalpy water contribution uses h_eff-weighted
    # single-gradient form.
    if !(p.atmos.microphysics_model isa DryModel)
        ᶜq_tot_eff = ᶜdiffusing_water(Y, p)
        ᶜρq_tot_diff = p.scratch.ᶜtemp_scalar_2
        @. ᶜρq_tot_diff = ᶜdiffdivᵥ(-(ᶠρK * ᶠgradᵥ(ᶜq_tot_eff)))
        @. Yₜ.c.ρq_tot -= ᶜρq_tot_diff
        @. Yₜ.c.ρ -= ᶜρq_tot_diff

        # Water enthalpy contribution: -ρK·(h_eff+Φ)·∇q_tot_eff.
        ᶜq_vap, ᶜq_lcl, ᶜq_icl = ᶜsuspended_water(Y, p)
        ᶜh_eff_plus_Φ = ᶜh_eff_plus_Φ!(
            p.scratch.ᶜtemp_scalar_3,
            thermo_params,
            ᶜT,
            ᶜΦ,
            ᶜq_vap,
            ᶜq_lcl,
            ᶜq_icl,
        )
        @. Yₜ.c.ρe_tot -=
            ᶜdiffdivᵥ(-(ᶠρK * ᶠinterp(ᶜh_eff_plus_Φ) * ᶠgradᵥ(ᶜq_tot_eff)))

        # Distribute ρq_tot_diff to cloud mass (and number) species.
        ᶜratio = p.scratch.ᶜtemp_scalar_4
        for (ρq_name, ρn_name) in
            ((@name(c.ρq_lcl), @name(c.ρn_lcl)), (@name(c.ρq_icl), @name(c.ρn_icl)))
            MatrixFields.has_field(Y, ρq_name) || continue
            ᶜρq = MatrixFields.get_field(Y, ρq_name)
            ᶜρqₜ = MatrixFields.get_field(Yₜ, ρq_name)
            @. ᶜratio =
                max(FT(0), min(FT(1), specific(ᶜρq, Y.c.ρ) / max(ᶜq_tot_eff, ϵ_FT)))
            @. ᶜρqₜ -= ᶜratio * ᶜρq_tot_diff
            if MatrixFields.has_field(Y, ρn_name)
                ᶜρn = MatrixFields.get_field(Y, ρn_name)
                ᶜρnₜ = MatrixFields.get_field(Yₜ, ρn_name)
                @. ᶜρnₜ -= ᶜratio * max(FT(0), ᶜρn) / max(ᶜρq, ϵ_FT) * ᶜρq_tot_diff
            end
        end
    end

    # Passive (non-microphysics) grid-scale tracers: independent diffusion at
    # full K_h. Skip microphysics species (handled above or no diffusion).
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ρχ_name in microphysics_tracer_names(Y) && return
        @. ᶜρχₜ -= ᶜdiffdivᵥ(-(ᶠρK * ᶠgradᵥ(specific(ᶜρχ, Y.c.ρ))))
    end
end
