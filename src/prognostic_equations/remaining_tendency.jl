
"""
    hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)

Orchestrates the calculation and application of hyperdiffusion tendencies to the
state vector `Y`.

This function follows a sequence:
1. Prepares hyperdiffusion tendencies for tracers (stored in `Yₜ_lim`).
2. Prepares hyperdiffusion tendencies for other state variables (e.g., momentum, energy, stored in `Yₜ`).
3. If Direct Stiffness Summation (DSS) is required and hyperdiffusion is active, performs DSS on the 
   prepared hyperdiffusion tendencies.
4. Applies the (potentially DSSed) hyperdiffusion tendencies to `Yₜ_lim` and `Yₜ`.

The distinction between `Yₜ` and `Yₜ_lim` allows for separate handling, often
because tracers might be subject to limiters applied via `Yₜ_lim`.

Arguments:
- `Yₜ`: The main tendency state vector, modified in place.
- `Yₜ_lim`: The tendency state vector for tracers (often subject to limiters), modified in place.
- `Y`: The current state vector.
- `p`: Cache containing parameters, atmospheric model configuration (e.g., `p.atmos.hyperdiff`),
       and data for DSS.
- `t`: Current simulation time.

Helper functions `prep_..._tendency!`, `dss_hyperdiffusion_tendency_pairs`,
and `apply_..._tendency!` implement the specific details of hyperdiffusion.
"""
NVTX.@annotate function hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    prep_tracer_hyperdiffusion_tendency!(Yₜ_lim, Y, p, t)
    prep_hyperdiffusion_tendency!(Yₜ, Y, p, t)
    if do_dss(axes(Y.c)) && !isnothing(p.atmos.hyperdiff)
        pairs = dss_hyperdiffusion_tendency_pairs(p)
        Spaces.weighted_dss!(pairs...)
    end
    apply_tracer_hyperdiffusion_tendency!(Yₜ_lim, Y, p, t)
    apply_hyperdiffusion_tendency!(Yₜ, Y, p, t)
end

using ClimaCore.RecursiveApply: rzero

#####
##### Cell center tendencies
#####

"""
    ᶜremaining_tendency(ᶜY, ᶠY, p, t)

Returns a Broadcasted object, for evaluating the cell center remaining
tendency. This method calls `ᶜremaining_tendency(Val(name), ᶜY, ᶠY, p, t)` for
all `propertynames` of `ᶜY`.
"""
function ᶜremaining_tendency(ᶜY, ᶠY, p, t)
    names = propertynames(ᶜY)
    tends = construct_tendencies(Val(names), ᶜremaining_tendency, ᶜY, ᶠY, p, t)
    # We cannot broadcast over a NamedTuple, so we need to check that edge case
    # first.
    if all(t -> !(t isa Base.Broadcast.Broadcasted), tends)
        return make_named_tuple(Val(names), tends...)
    else
        return lazy.(make_named_tuple.(Val(names), tends...))
    end
end

#####
##### Cell face tendencies
#####

"""
    ᶠremaining_tendency(ᶜY, ᶠY, p, t)

Returns a Broadcasted object, for evaluating the cell center remaining
tendency. This method calls `ᶠremaining_tendency(Val(name), ᶜY, ᶠY, p, t)` for
all `propertynames` of `ᶠY`.
"""
function ᶠremaining_tendency(ᶜY, ᶠY, p, t)
    names = propertynames(ᶠY)
    tends = construct_tendencies(Val(names), ᶠremaining_tendency, ᶜY, ᶠY, p, t)
    # We cannot broadcast over a NamedTuple, so we need to check that edge case
    # first.
    if all(t -> !(t isa Base.Broadcast.Broadcasted), tends)
        return make_named_tuple(Val(names), tends...)
    else
        return lazy.(make_named_tuple.(Val(names), tends...))
    end
end

#####
##### Individual tendencies
#####

function ᶜremaining_tendency(::Val{:ρ}, ᶜY, ᶠY, p, t)
    ∑tendencies = zero(eltype(ᶜY.ρ))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:uₕ}, ᶜY, ᶠY, p, t)
    ∑tendencies = zero(eltype(ᶜY.uₕ))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρe_tot}, ᶜY, ᶠY, p, t)
    ∑tendencies = zero(eltype(ᶜY.ρe_tot))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρq_tot}, ᶜY, ᶠY, p, t)
    ∑tendencies = zero(eltype(ᶜY.ρq_tot))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρq_liq}, ᶜY, ᶠY, p, t)
    ∑tendencies = zero(eltype(ᶜY.ρq_liq))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρq_ice}, ᶜY, ᶠY, p, t)
    ∑tendencies = zero(eltype(ᶜY.ρq_ice))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρn_liq}, ᶜY, ᶠY, p, t)
    ∑tendencies = zero(eltype(ᶜY.ρn_liq))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρn_rai}, ᶜY, ᶠY, p, t)
    ∑tendencies = zero(eltype(ᶜY.ρn_rai))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρq_rai}, ᶜY, ᶠY, p, t)
    ∑tendencies = zero(eltype(ᶜY.ρq_rai))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:ρq_sno}, ᶜY, ᶠY, p, t)
    ∑tendencies = zero(eltype(ᶜY.ρq_sno))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:sgsʲs}, ᶜY, ᶠY, p, t)
    ∑tendencies = rzero(eltype(ᶜY.sgsʲs))
    return ∑tendencies
end
function ᶜremaining_tendency(::Val{:sgs⁰}, ᶜY, ᶠY, p, t)
    ∑tendencies = rzero(eltype(ᶜY.sgs⁰))
    return ∑tendencies
end
function ᶠremaining_tendency(::Val{:u₃}, ᶜY, ᶠY, p, t)
    ∑tendencies = zero(eltype(ᶠY.u₃))
    return ∑tendencies
end
function ᶠremaining_tendency(::Val{:sgsʲs}, ᶜY, ᶠY, p, t)
    ∑tendencies = rzero(eltype(ᶠY.sgsʲs))
    return ∑tendencies
end

"""
    remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)

Computes a set of explicit tendencies for the atmospheric model.

This function acts as a high-level orchestrator, zeroing out the tendency vectors
`Yₜ` (for main model variables) and `Yₜ_lim` (for tracers) and then sequentially
calling various component tendency functions to accumulate contributions from:
- Horizontal advection (tracers and dynamics).
- Hyperdiffusion.
- Explicit vertical advection.
- Other specialized vertical advection (e.g., for water).
- A wide range of "additional" tendencies including sponge layers, physical
  parameterizations, forcings, and EDMFX subgrid-scale processes.

Arguments:
- `Yₜ`: The main tendency state vector, modified in place.
- `Yₜ_lim`: The tendency state vector for tracers, modified in place.
- `Y`: The current state vector.
- `p`: Cache containing parameters, precomputed fields, and model configurations.
- `t`: Current simulation time.

Returns:
- `Yₜ`: The populated main tendency state vector.
"""
NVTX.@annotate function remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ_lim .= zero(eltype(Yₜ_lim))
    device = ClimaComms.device(axes(Y.c))
    p_kernel = (;
        zmax = Spaces.z_max(axes(Y.f)),
        atmos = p.atmos,
        params = p.params,
        dt = p.dt,
    )
    if :sfc in propertynames(Y) # columnwise! does not yet handle .sfc
        parent(Yₜ.sfc) .= zero(Spaces.undertype(axes(Y.c)))
    end
    Operators.columnwise!(
        device,
        ᶜremaining_tendency,
        ᶠremaining_tendency,
        Yₜ.c,
        Yₜ.f,
        Y.c,
        Y.f,
        p_kernel,
        t,
    )
    horizontal_tracer_advection_tendency!(Yₜ_lim, Y, p, t)
    fill_with_nans!(p)  # TODO: would be better to limit this to debug mode (e.g., if p.debug_mode...)
    horizontal_dynamics_tendency!(Yₜ, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    vertical_advection_of_water_tendency!(Yₜ, Y, p, t)
    additional_tendency!(Yₜ, Y, p, t)
    return Yₜ
end

import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces

"""
    z_coordinate_fields(space::Spaces.AbstractSpace)

Extracts the `z` (vertical) coordinate fields from the cell centers (`ᶜz`)
and cell faces (`ᶠz`) of a given `ClimaCore.Spaces.AbstractSpace`.

Arguments:
- `space`: An `AbstractSpace` from which to derive center and face spaces.

Returns:
- A `NamedTuple` with fields `ᶜz` and `ᶠz`, containing the vertical coordinate fields.
"""
function z_coordinate_fields(space::Spaces.AbstractSpace)
    ᶜz = Fields.coordinate_field(Spaces.center_space(space)).z
    ᶠz = Fields.coordinate_field(Spaces.face_space(space)).z
    return (; ᶜz, ᶠz)
end

"""
    additional_tendency!(Yₜ, Y, p, t)

Aggregates various "additional" physical parameterization, forcing, and
subgrid-scale (SGS) tendency contributions into the main tendency vector `Yₜ`
and tracer tendency vector `Yₜ_lim` (implicitly via calls to functions that might use it,
though this function primarily modifies `Yₜ`).

This function is a central hub for incorporating tendencies from:
- Sponge layers (viscous and Rayleigh).
- Idealized forcings (e.g., Held-Suarez).
- Single Column Model (SCM) specific terms (e.g., SCM Coriolis).
- Large-scale advection (often prescribed for test cases).
- Subsidence (prescribed in single-column configurations).
- External forcings.
- Explicitly handled components of the EDMFX SGS scheme (vertical advection,
  diffusive fluxes, entrainment/detrainment, mass fluxes, non-hydrostatic pressure).
- EDMFX filter and TKE tendencies.
- Surface fluxes.
- Radiation.
- Cloud microphysics (condensation/evaporation).
- Precipitation processes (grid-scale and EDMFX).
- Surface temperature evolution.
- Pressure work terms.
- Smagorinsky-Lilly SGS turbulence.
- Gravity wave drag (orographic and non-orographic).
- Optional zeroing of velocity tendencies for specific tests.

Arguments:
- `Yₜ`: The main tendency state vector, modified in place.
- `Y`: The current state vector.
- `p`: Cache containing parameters, precomputed fields, and extensive model configurations.
- `t`: Current simulation time.

This function relies on numerous specialized sub-functions to calculate each
distinct tendency component. The order of calls can be important due to
dependencies or operator splitting assumptions.
"""
NVTX.@annotate function additional_tendency!(Yₜ, Y, p, t)

    (; ᶜh_tot, ᶜspecific) = p.precomputed
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜρ = Y.c.ρ
    (; forcing_type, moisture_model, rayleigh_sponge, viscous_sponge) = p.atmos
    (; ls_adv, scm_coriolis) = p.atmos
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜp, sfc_conditions, ᶜts) = p.precomputed

    vst_uₕ = viscous_sponge_tendency_uₕ(ᶜuₕ, viscous_sponge)
    vst_u₃ = viscous_sponge_tendency_u₃(ᶠu₃, viscous_sponge)
    vst_ρe_tot = viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, viscous_sponge)
    rst_uₕ = rayleigh_sponge_tendency_uₕ(ᶜuₕ, rayleigh_sponge)
    hs_args = (ᶜuₕ, ᶜp, params, sfc_conditions.ts, moisture_model, forcing_type)
    hs_tendency_uₕ = held_suarez_forcing_tendency_uₕ(hs_args...)
    hs_tendency_ρe_tot = held_suarez_forcing_tendency_ρe_tot(ᶜρ, hs_args...)
    edmf_cor_tend_uₕ = scm_coriolis_tendency_uₕ(ᶜuₕ, scm_coriolis)
    lsa_args = (ᶜρ, thermo_params, ᶜts, t, ls_adv)
    bc_lsa_tend_ρe_tot = large_scale_advection_tendency_ρe_tot(lsa_args...)

    # TODO: fuse, once we fix
    #       https://github.com/CliMA/ClimaCore.jl/issues/2165
    @. Yₜ.c.uₕ += vst_uₕ
    @. Yₜ.c.uₕ += rst_uₕ
    @. Yₜ.f.u₃.components.data.:1 += vst_u₃
    @. Yₜ.c.ρe_tot += vst_ρe_tot

    # TODO: can we write this out explicitly?
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        vst_tracer = viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, viscous_sponge)
        @. ᶜρχₜ += vst_tracer
        @. Yₜ.c.ρ += vst_tracer  # TODO: This doesn't look right for all tracers here. Remove?
    end

    # Held Suarez tendencies
    @. Yₜ.c.uₕ += hs_tendency_uₕ
    @. Yₜ.c.ρe_tot += hs_tendency_ρe_tot

    subsidence_tendency!(Yₜ, Y, p, t, p.atmos.subsidence)

    @. Yₜ.c.ρe_tot += bc_lsa_tend_ρe_tot
    if moisture_model isa AbstractMoistModel
        bc_lsa_tend_ρq_tot = large_scale_advection_tendency_ρq_tot(lsa_args...)
        @. Yₜ.c.ρq_tot += bc_lsa_tend_ρq_tot
    end

    @. Yₜ.c.uₕ += edmf_cor_tend_uₕ

    external_forcing_tendency!(Yₜ, Y, p, t, p.atmos.external_forcing)

    if p.atmos.sgs_adv_mode == Explicit()
        edmfx_sgs_vertical_advection_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.turbconv_model,
        )
    end

    if p.atmos.diff_mode == Explicit()
        vertical_diffusion_boundary_layer_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.vert_diff,
        )
        edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end

    surface_flux_tendency!(Yₜ, Y, p, t)

    radiation_tendency!(Yₜ, Y, p, t, p.atmos.radiation_mode)
    if p.atmos.sgs_entr_detr_mode == Explicit()
        edmfx_entr_detr_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    if p.atmos.sgs_mf_mode == Explicit()
        edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    if p.atmos.sgs_nh_pressure_mode == Explicit()
        edmfx_nh_pressure_drag_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    edmfx_filter_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    edmfx_tke_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    if p.atmos.noneq_cloud_formation_mode == Explicit()
        cloud_condensate_tendency!(
            Yₜ,
            Y,
            p,
            p.atmos.moisture_model,
            p.atmos.precip_model,
        )
    end

    edmfx_precipitation_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.turbconv_model,
        p.atmos.precip_model,
    )
    precipitation_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.moisture_model,
        p.atmos.precip_model,
        p.atmos.turbconv_model,
    )

    # TODO: make bycolumn-able
    non_orographic_gravity_wave_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.non_orographic_gravity_wave,
    )
    orographic_gravity_wave_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.orographic_gravity_wave,
    )

    # NOTE: Precipitation tendencies should be applied before calling this function,
    # because precipitation cache is used in this function
    surface_temp_tendency!(Yₜ, Y, p, t, p.atmos.surface_model)

    # NOTE: All ρa tendencies should be applied before calling this function
    pressure_work_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    sl = p.atmos.smagorinsky_lilly
    horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl)
    vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl)

    # NOTE: This will zero out all momentum tendencies in the EDMFX advection test, 
    # where velocities do not evolve
    # DO NOT add additional velocity tendencies after this function
    zero_velocity_tendency!(Yₜ, Y, p, t)
end
