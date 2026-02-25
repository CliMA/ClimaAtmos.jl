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
    check_state_nans(Y, "remaining_tendency!")
    Yₜ_lim .= zero(eltype(Yₜ_lim))
    Yₜ .= zero(eltype(Yₜ))
    horizontal_tracer_advection_tendency!(Yₜ_lim, Y, p, t)
    fill_with_nans!(p)  # TODO: would be better to limit this to debug mode (e.g., if p.debug_mode...)
    horizontal_dynamics_tendency!(Yₜ, Y, p, t)
    check_tendency_nans(Yₜ, "horizontal_dynamics_tendency!")
    hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    check_tendency_nans(Yₜ, "hyperdiffusion_tendency!")
    explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    check_tendency_nans(Yₜ, "explicit_vertical_advection_tendency!")
    additional_tendency!(Yₜ, Y, p, t)
    check_tendency_nans(Yₜ, "additional_tendency!")
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

    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜρ = Y.c.ρ
    (; radiation_mode, moisture_model, turbconv_model, microphysics_model) =
        p.atmos
    (; rayleigh_sponge, viscous_sponge) = p.atmos
    (; ls_adv, scm_coriolis) = p.atmos
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜp, ᶜK, ᶜT, ᶜh_tot, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
    (; sfc_conditions) = p.precomputed

    vst_uₕ = viscous_sponge_tendency_uₕ(ᶜuₕ, viscous_sponge)
    vst_u₃ = viscous_sponge_tendency_u₃(ᶠu₃, viscous_sponge)
    vst_ρe_tot = viscous_sponge_tendency_ρe_tot(ᶜρ, ᶜh_tot, viscous_sponge)
    rst_uₕ = rayleigh_sponge_tendency_uₕ(ᶜuₕ, rayleigh_sponge)

    if use_prognostic_tke(turbconv_model)
        rst_ρtke = rayleigh_sponge_tendency_sgs_tracer(Y.c.ρtke, rayleigh_sponge)
        @. Yₜ.c.ρtke += rst_ρtke
    end
    if turbconv_model isa PrognosticEDMFX
        ᶜmse = @. lazy(ᶜh_tot - ᶜK)
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
        n = n_mass_flux_subdomains(p.atmos.turbconv_model)
        for j in 1:n
            rst_sgs_mse = rayleigh_sponge_tendency_sgs_tracer(
                Y.c.sgsʲs.:($j).mse, ᶜmse, rayleigh_sponge,
            )
            @. Yₜ.c.sgsʲs.:($$j).mse += rst_sgs_mse
            rst_sgs_q_tot = rayleigh_sponge_tendency_sgs_tracer(
                Y.c.sgsʲs.:($j).q_tot, ᶜq_tot, rayleigh_sponge,
            )
            @. Yₜ.c.sgsʲs.:($$j).q_tot += rst_sgs_q_tot
        end
        if moisture_model isa NonEquilMoistModel &&
           microphysics_model isa Microphysics1Moment
            # TODO: This doesn't work for multiple updrafts
            moisture_species = (
                (@name(c.sgsʲs.:(1).q_liq), @name(c.ρq_liq)),
                (@name(c.sgsʲs.:(1).q_ice), @name(c.ρq_ice)),
                (@name(c.sgsʲs.:(1).q_rai), @name(c.ρq_rai)),
                (@name(c.sgsʲs.:(1).q_sno), @name(c.ρq_sno)),
            )
            MatrixFields.unrolled_foreach(moisture_species) do (sgs_q_name, ρq_name)
                ᶜρq = MatrixFields.get_field(Y, ρq_name)
                ᶜq = @. lazy(specific(ᶜρq, Y.c.ρ))
                ᶜsgs_q = MatrixFields.get_field(Y, sgs_q_name)
                ᶜsgs_qₜ = MatrixFields.get_field(Yₜ, sgs_q_name)
                rst_sgs_q = rayleigh_sponge_tendency_sgs_tracer(ᶜsgs_q, ᶜq, rayleigh_sponge)
                @. ᶜsgs_qₜ += rst_sgs_q
            end
        end
    end
    # For HeldSuarezForcing, the radiation_mode is used as the forcing parameter
    forcing = radiation_mode isa HeldSuarezForcing ? radiation_mode : nothing
    hs_args = (ᶜuₕ, ᶜp, params, sfc_conditions.T_sfc, moisture_model, forcing)
    hs_tendency_uₕ = held_suarez_forcing_tendency_uₕ(hs_args...)
    hs_tendency_ρe_tot = held_suarez_forcing_tendency_ρe_tot(ᶜρ, hs_args...)
    edmf_cor_tend_uₕ = scm_coriolis_tendency_uₕ(ᶜuₕ, scm_coriolis)
    lsa_args = (ᶜρ, thermo_params, ᶜT, ᶜp, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno, t, ls_adv)
    bc_lsa_tend_ρe_tot = large_scale_advection_tendency_ρe_tot(lsa_args...)

    # TODO: fuse, once we fix
    #       https://github.com/CliMA/ClimaCore.jl/issues/2165
    @. Yₜ.c.uₕ += vst_uₕ
    @. Yₜ.c.uₕ += rst_uₕ
    @. Yₜ.f.u₃.components.data.:1 += vst_u₃
    @. Yₜ.c.ρe_tot += vst_ρe_tot

    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        vst_tracer = viscous_sponge_tendency_tracer(ᶜρ, ᶜχ, viscous_sponge)
        @. ᶜρχₜ += vst_tracer
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ += vst_tracer
        end
    end

    if turbconv_model isa PrognosticEDMFX
        n = n_mass_flux_subdomains(turbconv_model)
        for j in 1:n
            ᶠu₃ʲ = Y.f.sgsʲs.:($j).u₃
            vst_u₃ʲ = viscous_sponge_tendency_u₃(ᶠu₃ʲ, viscous_sponge)
            @. Yₜ.f.sgsʲs.:($$j).u₃.components.data.:1 += vst_u₃ʲ
        end
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
            p.atmos.vertical_diffusion,
        )
        edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end

    surface_flux_tendency!(Yₜ, Y, p, t)

    radiation_tendency!(Yₜ, Y, p, t, p.atmos.radiation_mode)
    if p.atmos.sgs_entr_detr_mode == Explicit()
        edmfx_entr_detr_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
        edmfx_first_interior_entr_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    if p.atmos.sgs_mf_mode == Explicit()
        edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    if p.atmos.sgs_nh_pressure_mode == Explicit()
        edmfx_nh_pressure_drag_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    if p.atmos.sgs_vertdiff_mode == Explicit()
        edmfx_vertical_diffusion_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    edmfx_filter_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    edmfx_tke_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    if p.atmos.noneq_cloud_formation_mode == Explicit()
        cloud_condensate_tendency!(
            Yₜ,
            Y,
            p,
            p.atmos.moisture_model,
            p.atmos.microphysics_model,
            p.atmos.turbconv_model,
        )
    end

    edmfx_precipitation_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.turbconv_model,
        p.atmos.microphysics_model,
    )
    precipitation_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.moisture_model,
        p.atmos.microphysics_model,
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

    amd = p.atmos.amd_les
    horizontal_amd_tendency!(Yₜ, Y, p, t, amd)
    vertical_amd_tendency!(Yₜ, Y, p, t, amd)

    chd = p.atmos.constant_horizontal_diffusion
    horizontal_constant_diffusion_tendency!(Yₜ, Y, p, t, chd)

    # Optional tendency to bring negative small tracers back from negative
    # at the cost of water vapor.
    tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t, moisture_model, microphysics_model)

    # NOTE: This will zero out all momentum tendencies in the EDMFX advection test,
    # where velocities do not evolve
    # DO NOT add additional velocity tendencies after this function
    zero_velocity_tendency!(Yₜ, Y, p, t)
end
