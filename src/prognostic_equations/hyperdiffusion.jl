#####
##### Hyperdiffusion
#####

import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces

"""
    ν₄(hyperdiff, Y)

Return a `NamedTuple` with the scalar hyperdiffusivity `ν₄_scalar` and the
hyperviscosity `ν₄_vorticity` [m⁴/s].

Both coefficients scale with `h^3`, where `h` is the mean nodal distance of the
horizontal grid, following the empirical results of Lauritzen et al. (2018,
https://doi.org/10.1029/2017MS001257): `ν₄_vorticity = ν₄_vorticity_coeff * h^3`
and `ν₄_scalar = ν₄_vorticity / prandtl_number`, with both parameters taken from
`hyperdiff` (a `Hyperdiffusion` model).
"""
function ν₄(hyperdiff, Y)
    h = Spaces.node_horizontal_length_scale(Spaces.horizontal_space(axes(Y.c)))
    # Vorticity coefficient unchanged
    ν₄_vorticity = hyperdiff.ν₄_vorticity_coeff * h^3
    # Scalar coefficient = vorticity coefficient / Prandtl number
    ν₄_scalar = ν₄_vorticity / hyperdiff.prandtl_number
    return (; ν₄_scalar, ν₄_vorticity)
end

"""
    hyperdiffusion_cache(Y, atmos)
    hyperdiffusion_cache(Y, hyperdiff::Hyperdiffusion, turbconv_model)

Allocate the cache fields that hold the DSSed Laplacians (`∇²`) used by the
hyperdiffusion tendencies.

Returns an empty `NamedTuple` when `atmos.hyperdiff` is `nothing`. Otherwise
allocates `ᶜ∇²u`, the energy-split fields `ᶜ∇²s_d` and `ᶜ∇²q_tot_eff` (energy
hyperdiffusion acts on dry static energy plus an enthalpy-weighted effective
total water, never on a lumped `h_tot`), `ᶜ∇²specific_tracers`, the corresponding
per-updraft fields for `PrognosticEDMFX` (`ᶜ∇²uʲs`, `ᶜ∇²s_dʲs`, and
`ᶜ∇²q_tot_effʲs`, plus `ᶜ∇²sgs_tracerʲs` when the state carries SGS tracers),
`ᶜ∇²tke` when the turbulence-convection model carries prognostic TKE, and DSS
ghost buffers when the space requires DSS.
"""
function hyperdiffusion_cache(Y, atmos)
    (; hyperdiff, turbconv_model) = atmos
    isnothing(hyperdiff) && return (;)  # No hyperdiffiusion
    hyperdiffusion_cache(Y, hyperdiff, turbconv_model)
end

function hyperdiffusion_cache(Y, ::Hyperdiffusion, turbconv_model)
    FT = eltype(Y)
    n = n_mass_flux_subdomains(turbconv_model)

    # Grid-scale hyperdiffusion inputs. Energy hyperdiffusion is applied as
    # dry static energy plus a total-water enthalpy contribution weighted
    # by the aggregate enthalpy `h_eff = (h_v·q_v + h_l·q_liq + h_i·q_ice) /
    # q_tot_eff`, so only `∇²s_d` and `∇²q_tot_eff` are DSSed. `q_tot_eff`
    # is the effective total-water field that undergoes hyperdiffusion
    gs_quantities = (;
        ᶜ∇²u = similar(Y.c, C123{FT}),
        ᶜ∇²s_d = similar(Y.c, FT),
        ᶜ∇²q_tot_eff = similar(Y.c, FT),
        # Allocate without materializing the fused NamedTuple broadcast, which
        # does not compile on a GPU once there are more than about 32 tracers.
        # `prep_tracer_hyperdiffusion_tendency!` fills this one tracer at a time.
        ᶜ∇²specific_tracers = allocate_ᶜspecific_gs_tracers(Y, FT),
    )

    # Sub-grid scale quantities. `ᶜ∇²uʲs` is DSSed as a full C123 vector
    # (matches the grid-mean pattern for `ᶜ∇²u`); only its C3 component is
    # used in `apply_hyperdiffusion_tendency!` for the u₃ⱼ tendency, but the
    # C12 components are still needed in the outer `wcurlₕ(C123(curlₕ(⋅)))`
    # under non-orthogonal metrics (topography).
    sgs_tracer_hyperdiff =
        turbconv_model isa PrognosticEDMFX && !isempty(sgs_tracer_names(Y)) ?
        (; ᶜ∇²sgs_tracerʲs = similar(Y.c, NTuple{n, FT})) : (;)
    sgs_quantities =
        turbconv_model isa PrognosticEDMFX ?
        (;
            ᶜ∇²uʲs = similar(Y.c, NTuple{n, C123{FT}}),
            ᶜ∇²s_dʲs = similar(Y.c, NTuple{n, FT}),
            ᶜ∇²q_tot_effʲs = similar(Y.c, NTuple{n, FT}),
            sgs_tracer_hyperdiff...,
        ) : (;)
    maybe_ᶜ∇²tke =
        use_prognostic_tke(turbconv_model) ? (; ᶜ∇²tke = similar(Y.c, FT)) : (;)
    sgs_quantities = (; sgs_quantities..., maybe_ᶜ∇²tke...)
    quantities = (; gs_quantities..., sgs_quantities...)
    if do_dss(axes(Y.c))
        quantities = (;
            quantities...,
            hyperdiffusion_ghost_buffer = map(Spaces.create_dss_buffer, quantities),
        )
    end
    return quantities
end

"""
    prep_hyperdiffusion_tendency!(Yₜ, Y, p, t)

Compute the horizontal Laplacians that feed the dynamics/energy hyperdiffusion and
store them in `p.hyperdiff`, ready to be DSSed.

Fills `ᶜ∇²u` (grad-div minus curl-curl vector Laplacian), the energy-split fields
`ᶜ∇²s_d` and `ᶜ∇²q_tot_eff`, `ᶜ∇²tke` when prognostic TKE is active, and the
analogous per-updraft fields for `PrognosticEDMFX`. The dry static energy and the
effective total water are diffused separately (rather than a lumped `h_tot`) so
the `∇⁴` operator never mixes dry-air enthalpy with water enthalpy;
`apply_hyperdiffusion_tendency!` reassembles the pieces into a total enthalpy
flux.

Does nothing when `p.atmos.hyperdiff` is `nothing`. `Yₜ` and `t` are unused. The
fields written here must be DSSed (see `dss_hyperdiffusion_tendency_pairs`) before
`apply_hyperdiffusion_tendency!` is called. Called from `hyperdiffusion_tendency!`.
Returns `nothing`.
"""
NVTX.@annotate function prep_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    (; params) = p
    (; ᶜΦ) = p.core
    thermo_params = CAP.thermodynamics_params(params)

    isnothing(hyperdiff) && return nothing

    n = n_mass_flux_subdomains(turbconv_model)
    diffuse_tke = use_prognostic_tke(turbconv_model)
    (; ᶜu, ᶜT, ᶜp) = p.precomputed
    (; ᶜ∇²u, ᶜ∇²s_d, ᶜ∇²q_tot_eff) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²uʲs, ᶜ∇²s_dʲs, ᶜ∇²q_tot_effʲs) = p.hyperdiff
        (; ᶜuʲs, ᶜTʲs) = p.precomputed
    end

    # Grid scale hyperdiffusion. Scalars are hyperdiffused as perturbations
    # from a smooth hydrostatic reference profile: on tilted terrain-following
    # surfaces, `gradₕ` of a horizontally uniform `s_d(z)` or `q_tot(z)` is
    # nonzero purely from the coordinate tilt (dominated by `Φ` for `s_d`), so
    # `∇⁴` on the raw field would spuriously mix scalars across topography.
    # The reference-state pair `(sd_r, q_tot_r)` cancels this leading-order
    # geometric term, matching the split-form pressure-gradient treatment in
    # `advection.jl`.
    @. ᶜ∇²u = C123(wgradₕ(divₕ(ᶜu))) - C123(wcurlₕ(C123(curlₕ(ᶜu))))
    @. ᶜ∇²s_d = wdivₕ(
        gradₕ(
            TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ) -
            sd_r(thermo_params, ᶜp),
        ),
    )
    if MatrixFields.has_field(Y, @name(c.ρq_tot))
        if p.atmos.microphysics_model isa Union{NonEquilibriumMicrophysics1M,
            NonEquilibriumMicrophysics2M}
            @. ᶜ∇²q_tot_eff = wdivₕ(
                gradₕ(
                    specific(Y.c.ρq_tot - Y.c.ρq_rai - Y.c.ρq_sno, Y.c.ρ) -
                    q_tot_r(thermo_params, ᶜp),
                ),
            )
        else
            @. ᶜ∇²q_tot_eff = wdivₕ(
                gradₕ(
                    specific(Y.c.ρq_tot, Y.c.ρ) -
                    q_tot_r(thermo_params, ᶜp),
                ),
            )
        end
    end

    if diffuse_tke
        ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
        (; ᶜ∇²tke) = p.hyperdiff
        @. ᶜ∇²tke = wdivₕ(gradₕ(ᶜtke))
    end

    # Sub-grid scale hyperdiffusion. SGS mseⱼ uses the same
    # dry-static-energy + q_tot_eff split as the grid mean (assembled with
    # subdomain thermodynamics in `apply_hyperdiffusion_tendency!`).
    if turbconv_model isa PrognosticEDMFX
        # Config-level (not per-j) check: all subdomains carry the same
        # prognostic fields, so hoist the has_field lookup.
        for j in 1:n
            # Full vector Laplacian identity, matching the grid-mean pattern
            # for `ᶜ∇²u`. Under non-orthogonal metrics (topography), the C12
            # components matter because they couple into the C3 of the outer
            # `wcurlₕ(C123(curlₕ(⋅)))` via g_{13}, g_{23}.
            @. ᶜ∇²uʲs.:($$j) =
                C123(wgradₕ(divₕ(ᶜuʲs.:($$j)))) -
                C123(wcurlₕ(C123(curlₕ(ᶜuʲs.:($$j)))))
            # Same reference-state subtraction as grid mean. Updrafts share
            # the grid-mean pressure, so `sd_r(ᶜp)` and `q_tot_r(ᶜp)` reuse the
            # same profiles.
            @. ᶜ∇²s_dʲs.:($$j) = wdivₕ(
                gradₕ(
                    TD.dry_static_energy(thermo_params, ᶜTʲs.:($$j), ᶜΦ) -
                    sd_r(thermo_params, ᶜp),
                ),
            )
            if p.atmos.microphysics_model isa Union{NonEquilibriumMicrophysics1M,
                NonEquilibriumMicrophysics2M}
                @. ᶜ∇²q_tot_effʲs.:($$j) = wdivₕ(
                    gradₕ(
                        Y.c.sgsʲs.:($$j).q_tot -
                        Y.c.sgsʲs.:($$j).q_rai -
                        Y.c.sgsʲs.:($$j).q_sno -
                        q_tot_r(thermo_params, ᶜp),
                    ),
                )
            else
                @. ᶜ∇²q_tot_effʲs.:($$j) = wdivₕ(
                    gradₕ(
                        Y.c.sgsʲs.:($$j).q_tot - q_tot_r(thermo_params, ᶜp),
                    ),
                )
            end
        end
    end
end

"""
    apply_hyperdiffusion_tendency!(Yₜ, Y, p, t)

Add the fourth-order (`∇⁴`) hyperdiffusion tendencies for momentum, energy, and
TKE, built from the DSSed Laplacians in `p.hyperdiff`.

Increments (each term enters with a minus sign, so hyperdiffusion damps
grid-scale noise):

  - `Yₜ.c.uₕ` and `Yₜ.f.u₃`: `ν₄_vorticity` times the vector `∇⁴u`, with the grad-div
    part scaled by `hyperdiff.divergence_damping_factor`.
  - `Yₜ.c.ρe_tot`: `ν₄_scalar` times the divergence of the total enthalpy
    hyperdiffusion flux, `∇⋅(ρ ∇∇²s_d) + ∇⋅(ρ (h_eff + Φ) ∇∇²q_tot_eff)`. The
    second term is applied only when `ρq_tot` is prognostic. Here `q_tot_eff` is
    the water that actually hyperdiffuses, vapor plus cloud liquid plus cloud ice
    with rain and snow excluded, and `h_eff = (h_v q_v + h_l q_lcl + h_i q_icl) / q_tot_eff` is its aggregate specific enthalpy; because the phase shares sum
    to one, the geopotential collapses into the single `+ Φ`. This mirrors the
    vertical boundary-layer flux in
    `vertical_diffusion_boundary_layer_tendency!`.
  - `Yₜ.c.ρtke`: `ν₄_vorticity` times `∇⋅(ρ ∇∇²tke)`, when prognostic TKE is active.
  - For `PrognosticEDMFX`, `Yₜ.f.sgsʲs.:(j).u₃` (curl-curl part only) and
    `Yₜ.c.sgsʲs.:(j).mse` (same energy split, with subdomain thermodynamics and no
    density weighting).

The two coefficients come from `ν₄`: `ν₄_vorticity` for momentum and TKE, and
`ν₄_scalar = ν₄_vorticity / prandtl_number` for scalars. Requires DSS to have been
applied to the pairs from `dss_hyperdiffusion_tendency_pairs`; does nothing when
`p.atmos.hyperdiff` is `nothing`. Called from `hyperdiffusion_tendency!`. Returns
`nothing`.
"""
NVTX.@annotate function apply_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜΦ) = p.core
    (; divergence_damping_factor) = hyperdiff
    (; ν₄_scalar, ν₄_vorticity) = ν₄(hyperdiff, Y)

    n = n_mass_flux_subdomains(turbconv_model)
    diffuse_tke = use_prognostic_tke(turbconv_model)
    ᶜρ = Y.c.ρ
    ᶜJ = Fields.local_geometry_field(Y.c).J
    point_type = eltype(Fields.coordinate_field(Y.c))
    (; ᶜT) = p.precomputed
    (; ᶜ∇²u, ᶜ∇²s_d, ᶜ∇²q_tot_eff) = p.hyperdiff
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²uʲs, ᶜ∇²s_dʲs, ᶜ∇²q_tot_effʲs) = p.hyperdiff
        (; ᶜTʲs, ᶜq_tot_nonnegʲs, ᶜq_liqʲs, ᶜq_iceʲs) = p.precomputed
    end
    if use_prognostic_tke(turbconv_model)
        (; ᶜ∇²tke) = p.hyperdiff
    end

    # re-use to store the curl-curl part
    ᶜ∇⁴u = @. ᶜ∇²u =
        divergence_damping_factor * C123(wgradₕ(divₕ(ᶜ∇²u))) -
        C123(wcurlₕ(C123(curlₕ(ᶜ∇²u))))
    @. Yₜ.c.uₕ -= ν₄_vorticity * C12(ᶜ∇⁴u)
    @. Yₜ.f.u₃ -= ν₄_vorticity * ᶠwinterp(ᶜJ * ᶜρ, C3(ᶜ∇⁴u))

    # Total enthalpy hyperdiffusion. The flux has two pieces:
    #
    #   F_h_dry   = ν · ρ · grad(∇²s_d)                       (dry static)
    #   F_h_water = ν · ρ · (h_eff + Φ) · grad(∇²q_tot_eff)   (water)
    #
    # where `h_eff = (h_v·q_v + h_l·q_lcl + h_i·q_icl) / q_tot_eff` is the
    # local aggregate specific enthalpy of the water that actually
    # hyperdiffuses (vapor + cloud liquid + cloud ice; rain and snow are
    # excluded). Since the vapor/cloud shares sum to 1, the Φ per phase
    # collapses to a single +Φ term in the flux.
    # Dry static energy contribution (always applied).
    ᶜh_flux_div = p.scratch.ᶜtemp_scalar
    @. ᶜh_flux_div = wdivₕ(ᶜρ * gradₕ(ᶜ∇²s_d))
    # Water enthalpy contribution — only when ρq_tot is prognostic (dry
    # configurations skip this entirely).
    if MatrixFields.has_field(Y, @name(c.ρq_tot))
        ᶜq_vap, ᶜq_lcl, ᶜq_icl = ᶜsuspended_water(Y, p)
        ᶜh_eff_plus_Φ = ᶜh_eff_plus_Φ!(
            p.scratch.ᶜtemp_scalar_2,
            thermo_params,
            ᶜT,
            ᶜΦ,
            ᶜq_vap,
            ᶜq_lcl,
            ᶜq_icl,
        )
        @. ᶜh_flux_div += wdivₕ(ᶜρ * ᶜh_eff_plus_Φ * gradₕ(ᶜ∇²q_tot_eff))
    end
    @. Yₜ.c.ρe_tot -= ν₄_scalar * ᶜh_flux_div

    if (turbconv_model isa AbstractEDMF) && diffuse_tke
        @. Yₜ.c.ρtke -= ν₄_vorticity * wdivₕ(ᶜρ * gradₕ(ᶜ∇²tke))
    end
    # Sub-grid scale hyperdiffusion continued
    if turbconv_model isa PrognosticEDMFX
        # Hoist config-level checks so branches don't run per j.
        for j in 1:n
            if point_type <: Geometry.Abstract3DPoint
                # Only the C3 component of ∇⁴uⱼ contributes to Yₜ.f.u₃ⱼ,
                # so drop the grad-div term (its C3 is zero).
                ᶜ∇⁴uⱼ = @. ᶜ∇²uʲs.:($$j) =
                    -C123(wcurlₕ(C123(curlₕ(ᶜ∇²uʲs.:($$j)))))
                @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                    ν₄_vorticity * ᶠwinterp(ᶜJ * ᶜρ, C3(ᶜ∇⁴uⱼ))
            end
            # SGS mse hyperdiff — matches the grid-mean expression above,
            # using SGS thermodynamics (Tʲ, q_μʲ). No density weighting:
            # hyperdiff on the specific mseʲ [J/kg/s], symmetric with the
            # SGS q_totʲ hyperdiff below. q_tot_effⱼ excludes rain/snow;
            # h_effⱼ is built from vapor + cloud only.
            ᶜq_vapʲ = @. lazy(
                TD.vapor_specific_humidity(
                    ᶜq_tot_nonnegʲs.:($$j),
                    ᶜq_liqʲs.:($$j),
                    ᶜq_iceʲs.:($$j),
                ),
            )
            ᶜq_lclʲ, ᶜq_iclʲ =
                p.atmos.microphysics_model isa Union{NonEquilibriumMicrophysics1M,
                    NonEquilibriumMicrophysics2M} ?
                (
                    (@. lazy(Y.c.sgsʲs.:($$j).q_lcl)),
                    (@. lazy(Y.c.sgsʲs.:($$j).q_icl)),
                ) : (ᶜq_liqʲs.:($j), ᶜq_iceʲs.:($j))
            ᶜh_effⱼ_plus_Φ = ᶜh_eff_plus_Φ!(
                p.scratch.ᶜtemp_scalar_2,
                thermo_params,
                ᶜTʲs.:($j),
                ᶜΦ,
                ᶜq_vapʲ,
                ᶜq_lclʲ,
                ᶜq_iclʲ,
            )
            @. ᶜh_flux_div = wdivₕ(gradₕ(ᶜ∇²s_dʲs.:($$j)))
            @. ᶜh_flux_div +=
                wdivₕ(ᶜh_effⱼ_plus_Φ * gradₕ(ᶜ∇²q_tot_effʲs.:($$j)))
            @. Yₜ.c.sgsʲs.:($$j).mse -= ν₄_scalar * ᶜh_flux_div
        end
    end
end

"""
    dss_hyperdiffusion_tendency_pairs(p)

Return the tuple of `field => ghost_buffer` pairs that must be DSSed between the
`prep_*` and `apply_*` stages of the hyperdiffusion tendencies.

Covers the dynamics fields (`ᶜ∇²u`, the energy-split `ᶜ∇²s_d` and
`ᶜ∇²q_tot_eff`, `ᶜ∇²tke` when active, and the per-updraft counterparts `ᶜ∇²uʲs`,
`ᶜ∇²s_dʲs`, and `ᶜ∇²q_tot_effʲs` for `PrognosticEDMFX`) and the grid-scale tracer
field `ᶜ∇²specific_tracers`.
Called from `hyperdiffusion_tendency!`, which passes the pairs to
`ClimaCore.Spaces.weighted_dss!`.
"""
function dss_hyperdiffusion_tendency_pairs(p)
    (; turbconv_model) = p.atmos
    buffer = p.hyperdiff.hyperdiffusion_ghost_buffer
    (; ᶜ∇²u, ᶜ∇²s_d, ᶜ∇²q_tot_eff) = p.hyperdiff
    diffuse_tke = use_prognostic_tke(turbconv_model)
    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²uʲs, ᶜ∇²s_dʲs, ᶜ∇²q_tot_effʲs) = p.hyperdiff
    end
    if use_prognostic_tke(turbconv_model)
        (; ᶜ∇²tke) = p.hyperdiff
    end

    core_dynamics_pairs = (
        ᶜ∇²u => buffer.ᶜ∇²u,
        ᶜ∇²s_d => buffer.ᶜ∇²s_d,
        ᶜ∇²q_tot_eff => buffer.ᶜ∇²q_tot_eff,
        (diffuse_tke ? (ᶜ∇²tke => buffer.ᶜ∇²tke,) : ())...,
    )
    tc_dynamics_pairs =
        turbconv_model isa PrognosticEDMFX ?
        (
            ᶜ∇²uʲs => buffer.ᶜ∇²uʲs,
            ᶜ∇²s_dʲs => buffer.ᶜ∇²s_dʲs,
            ᶜ∇²q_tot_effʲs => buffer.ᶜ∇²q_tot_effʲs,
        ) : ()
    dynamics_pairs = (core_dynamics_pairs..., tc_dynamics_pairs...)

    (; ᶜ∇²specific_tracers) = p.hyperdiff
    core_tracer_pairs =
        !isempty(propertynames(ᶜ∇²specific_tracers)) ?
        (ᶜ∇²specific_tracers => buffer.ᶜ∇²specific_tracers,) : ()
    tracer_pairs = core_tracer_pairs
    return (dynamics_pairs..., tracer_pairs...)
end

"""
    prep_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)

Compute the horizontal Laplacians of the specific grid-scale tracers and store
them in `p.hyperdiff`, ready to be DSSed.

Fills `ᶜ∇²specific_tracers` with `∇²(ρχ/ρ)` for every grid-scale tracer. Updraft
tracers are handled separately, through the shared `ᶜ∇²sgs_tracerʲs` scratch
field. Does nothing when `p.atmos.hyperdiff` is `nothing`. `Yₜ` and `t` are unused. The fields written here must be DSSed (see
`dss_hyperdiffusion_tendency_pairs`) before `apply_tracer_hyperdiffusion_tendency!`
is called. Called from `hyperdiffusion_tendency!`. Returns `nothing`.
"""
NVTX.@annotate function prep_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; ᶜ∇²specific_tracers) = p.hyperdiff

    # TODO: Fix RecursiveApply bug in gradₕ to fuse this operation.
    # ᶜ∇²specific_tracers .= wdivₕ.(gradₕ.(ᶜspecific_gs_tracers(Y)))
    foreach_gs_tracer(Y, ᶜ∇²specific_tracers) do ᶜρχ, ᶜ∇²χ, _
        @. ᶜ∇²χ = wdivₕ(gradₕ(specific(ᶜρχ, Y.c.ρ)))
    end
    return nothing
end

"""
    apply_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)

Add the fourth-order (`∇⁴`) hyperdiffusion tendencies for tracers, built from the
DSSed Laplacians in `p.hyperdiff`.

Increments (each with a minus sign):

  - `ρq_tot` is hyperdiffused on `q_tot_eff = q_tot - q_rai - q_sno` and the
    resulting mass tendency is applied to `Yₜ.c.ρq_tot` and `Yₜ.c.ρ` (so water
    hyperdiffusion moves mass consistently). Cloud mass species (`ρq_lcl`,
    `ρq_icl`) receive their share by pure scaling of that tendency with the
    clipped ratio `min(q_μ/q_tot_eff, 1)`; their number densities scale
    proportionally. Rain, snow, and rain number density (`ρq_rai`, `ρq_sno`,
    `ρn_rai`) receive no hyperdiffusion.
  - Every passive (non-microphysics) grid-mean tracer `Yₜ.c.ρχ`:
    `ν₄_scalar * ∇⋅(ρ ∇∇²χ)`.
  - For `PrognosticEDMFX`, `Yₜ.c.sgsʲs.:(j).q_tot` and the compensating
    `Yₜ.c.sgsʲs.:(j).ρa` term, plus every auto-discovered SGS tracer (with its own
    prep → DSS → apply cycle through the shared scratch field `ᶜ∇²sgs_tracerʲs`).

Requires DSS to have been applied to the pairs from
`dss_hyperdiffusion_tendency_pairs`; does nothing when `p.atmos.hyperdiff` is
`nothing`. Called from `hyperdiffusion_tendency!` with the limited tendency vector
`Yₜ_lim`. Returns `nothing`.
"""
NVTX.@annotate function apply_tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    (; hyperdiff, turbconv_model) = p.atmos
    isnothing(hyperdiff) && return nothing

    (; ν₄_scalar) = ν₄(hyperdiff, Y)
    FT = eltype(p.params)
    ϵ_FT = eps(FT)
    n = n_mass_flux_subdomains(turbconv_model)
    (; ᶜ∇²specific_tracers) = p.hyperdiff

    # Grid-mean total-water hyperdiff via the single ∇²q_tot_eff field.
    # The resulting `ρq_tot_hyperdiff` is applied to ρq_tot and ρ (in
    # divergence form for total-water mass conservation) and then
    # *distributed* to each cloud mass prognostic by pure scaling with the
    # clipped ratio `min(q_μ/q_tot_eff, 1)`. This intentionally drops the
    # per-species divergence form — the alternative flux distribution
    # implies advecting each species at ν₀·∇³q_t/q_t, which explodes in
    # dry regions. Rain, snow and n_rai do not hyperdiffuse.
    if MatrixFields.has_field(Y, @name(c.ρq_tot))
        (; ᶜ∇²q_tot_eff) = p.hyperdiff
        ᶜρ = Y.c.ρ
        ᶜρq_tot_hyperdiff = p.scratch.ᶜtemp_scalar
        @. ᶜρq_tot_hyperdiff = ν₄_scalar * wdivₕ(ᶜρ * gradₕ(ᶜ∇²q_tot_eff))
        @. Yₜ.c.ρq_tot -= ᶜρq_tot_hyperdiff
        @. Yₜ.c.ρ -= ᶜρq_tot_hyperdiff
        # Species distribution: pure scaling of the q_tot tendency by the
        # clipped mass fraction `min(q_μ/q_t_eff, 1)`. This is NOT
        # divergence form for the individual species.
        #
        # Number density tendencies (ρn_lcl, ρn_icl if present) are made
        # proportional to their corresponding mass tendencies via
        # dρn/n = dρq/q, i.e. dρn = (ρn/ρq) · dρq. That preserves the mean
        # particle mass across the hyperdiff.
        ᶜratio = p.scratch.ᶜtemp_scalar_2
        ᶜρq_tot_eff =
            p.atmos.microphysics_model isa Union{NonEquilibriumMicrophysics1M,
                NonEquilibriumMicrophysics2M} ?
            (@. lazy(Y.c.ρq_tot - Y.c.ρq_rai - Y.c.ρq_sno)) : (@. lazy(Y.c.ρq_tot))
        for (ρq_name, ρn_name) in (
            (@name(c.ρq_lcl), @name(c.ρn_lcl)),
            (@name(c.ρq_icl), @name(c.ρn_icl)),
        )
            MatrixFields.has_field(Y, ρq_name) || continue
            ᶜρq = MatrixFields.get_field(Y, ρq_name)
            ᶜρqₜ = MatrixFields.get_field(Yₜ, ρq_name)
            @. ᶜratio =
                max(FT(0), min(FT(1), ᶜρq / max(ᶜρq_tot_eff, ϵ_FT)))
            @. ᶜρqₜ -= ᶜratio * ᶜρq_tot_hyperdiff
            if MatrixFields.has_field(Y, ρn_name)
                ᶜρn = MatrixFields.get_field(Y, ρn_name)
                ᶜρnₜ = MatrixFields.get_field(Yₜ, ρn_name)
                @. ᶜρnₜ -=
                    ᶜratio * max(FT(0), ᶜρn) / max(ᶜρq, ϵ_FT) * ᶜρq_tot_hyperdiff
            end
        end
    end

    _microphysics_names = (
        @name(ρq_lcl), @name(ρq_icl), @name(ρq_rai),
        @name(ρq_sno), @name(ρn_lcl), @name(ρn_rai),
    )
    # TODO: Since we are not applying the limiter to density (or area-weighted
    # density), the mass redistributed by hyperdiffusion will not be conserved
    # by the limiter. Is this a significant problem?
    foreach_gs_tracer(Yₜ, ᶜ∇²specific_tracers) do ᶜρχₜ, ᶜ∇²χ, ρχ_name
        # ρq_tot is handled via the flux above; cloud species (lcl, icl,
        # n_lcl) are handled via the tendency split above; rain/snow/n_rai do
        # not hyperdiffuse. Everything else falls through and gets its
        # standard ∇⁴ tendency.
        ρχ_name == @name(ρq_tot) && return
        ρχ_name in _microphysics_names && return
        @. ᶜρχₜ -= ν₄_scalar * wdivₕ(Y.c.ρ * gradₕ(ᶜ∇²χ))
    end

    if turbconv_model isa PrognosticEDMFX
        (; ᶜ∇²q_tot_effʲs) = p.hyperdiff
        for j in 1:n
            ᶜq_totʲ_hyperdiff = p.scratch.ᶜtemp_scalar
            @. ᶜq_totʲ_hyperdiff =
                ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²q_tot_effʲs.:($$j)))
            @. Yₜ.c.sgsʲs.:($$j).q_tot -= ᶜq_totʲ_hyperdiff
            @. Yₜ.c.sgsʲs.:($$j).ρa -=
                Y.c.sgsʲs.:($$j).ρa / (1 - Y.c.sgsʲs.:($$j).q_tot) *
                ᶜq_totʲ_hyperdiff
            # SGS species distribution — same tendency-scaling scheme as
            # the grid mean above.
            ᶜratioⱼ = p.scratch.ᶜtemp_scalar_2
            ᶜq_tot_effⱼ =
                p.atmos.microphysics_model isa Union{NonEquilibriumMicrophysics1M,
                    NonEquilibriumMicrophysics2M} ?
                (@. lazy(
                    Y.c.sgsʲs.:($$j).q_tot -
                    Y.c.sgsʲs.:($$j).q_rai -
                    Y.c.sgsʲs.:($$j).q_sno,
                )) : (@. lazy(Y.c.sgsʲs.:($$j).q_tot))
            for (χⱼ_name, nⱼ_name) in (
                (@name(q_lcl), @name(n_lcl)),
                (@name(q_icl), @name(n_icl)),
            )
                MatrixFields.has_field(Y.c.sgsʲs.:($j), χⱼ_name) || continue
                ᶜχⱼ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χⱼ_name)
                ᶜχⱼₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χⱼ_name)
                @. ᶜratioⱼ =
                    max(FT(0), min(FT(1), ᶜχⱼ / max(ᶜq_tot_effⱼ, ϵ_FT)))
                @. ᶜχⱼₜ -= ᶜratioⱼ * ᶜq_totʲ_hyperdiff
                if MatrixFields.has_field(Y.c.sgsʲs.:($j), nⱼ_name)
                    ᶜnⱼ = MatrixFields.get_field(Y.c.sgsʲs.:($j), nⱼ_name)
                    ᶜnⱼₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), nⱼ_name)
                    @. ᶜnⱼₜ -=
                        ᶜratioⱼ * max(FT(0), ᶜnⱼ) / max(ᶜχⱼ, ϵ_FT) *
                        ᶜq_totʲ_hyperdiff
                end
            end
        end
        # Passive (non-microphysics) SGS tracers keep their independent ∇⁴
        # tendency. Cloud SGS species are handled via the q_tot_effⱼ flux
        # split above; rain/snow/n_rai do not hyperdiffuse.
        _microphysics_sgs_names = (
            @name(q_lcl), @name(q_icl), @name(q_rai),
            @name(q_sno), @name(n_lcl), @name(n_rai),
        )
        if !isempty(sgs_tracer_names(Y))
            (; ᶜ∇²sgs_tracerʲs) = p.hyperdiff
            for χ_name in sgs_tracer_names(Y)
                # `continue`, not `return`: this is a plain for-loop, not a
                # do-block, so `return` would exit the whole function and
                # skip any passive tracers after a microphysics one.
                χ_name in _microphysics_sgs_names && continue
                for j in 1:n
                    ᶜχʲ = MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name)
                    @. ᶜ∇²sgs_tracerʲs.:($$j) = wdivₕ(gradₕ(ᶜχʲ))
                end
                if do_dss(axes(Y.c))
                    Spaces.weighted_dss!(
                        ᶜ∇²sgs_tracerʲs =>
                            p.hyperdiff.hyperdiffusion_ghost_buffer.ᶜ∇²sgs_tracerʲs,
                    )
                end
                for j in 1:n
                    ᶜχʲₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:($j), χ_name)
                    @. ᶜχʲₜ -=
                        ν₄_scalar * wdivₕ(gradₕ(ᶜ∇²sgs_tracerʲs.:($$j)))
                end
            end
        end
    end
    return nothing
end
