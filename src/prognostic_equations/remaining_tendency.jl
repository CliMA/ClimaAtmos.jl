"""
    hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)

Orchestrate the prep → DSS → apply sequence of the hyperdiffusion tendencies.

Calls `prep_tracer_hyperdiffusion_tendency!` and `prep_hyperdiffusion_tendency!`
to fill the `∇²` cache fields, DSSes them (via `dss_hyperdiffusion_tendency_pairs`)
when the space requires DSS and hyperdiffusion is active, and then calls
`apply_tracer_hyperdiffusion_tendency!` and `apply_hyperdiffusion_tendency!`.

Tracer tendencies accumulate into `Yₜ_lim`, which is subject to the tracer
limiters, while momentum, energy, and TKE tendencies accumulate into `Yₜ`. Called
from `remaining_tendency!`.
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

Compute the explicit ("remaining") tendencies of the IMEX splitting.

Zeroes `Yₜ`, then calls `add_remaining_tendency!`, which zeroes `Yₜ_lim` and
accumulates, in order:

 1. `horizontal_advection_tendency!`: `horizontal_tracer_advection_tendency!` (into
    `Yₜ_lim`) and `horizontal_dynamics_tendency!` in one slab loop.
 2. `hyperdiffusion_tendency!` (tracer part into `Yₜ_lim`).
 3. `explicit_vertical_advection_tendency!`.
 4. `additional_tendency!` (sponges, forcings, parameterizations, EDMFX SGS terms).

`Yₜ_lim` collects the tracer tendencies that are subject to the horizontal
limiters (see `limiters_func!`); everything else accumulates into `Yₜ`. The
implicitly treated vertical acoustic and advective terms live in
`implicit_tendency!`; vertical diffusion, surface fluxes, and microphysics run
in the column physics (`step_column_physics!`).

# Returns

`Yₜ`, the populated main tendency vector.
"""
NVTX.@annotate function remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    return add_remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
end

"""
    add_remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)

Add the tendencies of `remaining_tendency!` to `Yₜ`, overwriting `Yₜ_lim`.
Returns `Yₜ`.
"""
function add_remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ_lim .= zero(eltype(Yₜ_lim))
    fill_with_nans!(p)  # TODO: would be better to limit this to debug mode (e.g., if p.debug_mode...)
    horizontal_advection_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    explicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    additional_tendency!(Yₜ, Y, p, t)
    return Yₜ
end

import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Spaces as Spaces

"""
    z_coordinate_fields(space::Spaces.AbstractSpace)

Return a `NamedTuple` `(; ᶜz, ᶠz)` with the vertical coordinate fields [m] at the
cell centers and cell faces of `space`.
"""
function z_coordinate_fields(space::Spaces.AbstractSpace)
    ᶜz = Fields.coordinate_field(Spaces.center_space(space)).z
    ᶠz = Fields.coordinate_field(Spaces.face_space(space)).z
    return (; ᶜz, ᶠz)
end

"""
    additional_tendency!(Yₜ, Y, p, t)

Aggregate the explicit physical-parameterization, forcing, and subgrid-scale
tendencies into `Yₜ`.

Accumulates contributions from:

  - Sponge layers (viscous and Rayleigh), including grid-scale tracers, TKE, and
    EDMFX updraft fields.
  - Idealized Held-Suarez forcing (when `radiation_mode isa HeldSuarezForcing`).
  - Single-column Coriolis forcing, prescribed large-scale advection, subsidence,
    and external forcings (see `external_forcing_tendency!`).
  - Surface fluxes, radiation, EDMFX TKE, and chemistry.
  - Non-orographic and orographic gravity-wave drag.
  - Surface temperature evolution, horizontal Smagorinsky-Lilly and the
    vertical Smagorinsky-Lilly momentum terms not solved in the column
    physics, AMD LES, and constant horizontal diffusion.
  - Tracer nonnegativity restoration at the cost of water vapor, and finally
    `zero_velocity_tendency!` for advection tests (which must remain last so no
    later call reintroduces velocity tendencies).

Vertical diffusion, surface fluxes, microphysics, and surface precipitation run
in the column physics (`step_column_physics!`), not here. Called from
`remaining_tendency!`. Returns `nothing`.
"""
NVTX.@annotate function additional_tendency!(Yₜ, Y, p, t)

    ᶜuₕ = Y.c.uₕ
    ᶜρ = Y.c.ρ
    (; radiation_mode, microphysics_model, turbconv_model) = p.atmos
    (; rayleigh_sponge) = p.atmos
    (; ls_adv, scm_coriolis) = p.atmos
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜp, ᶜK, ᶜT, ᶜh_tot, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; sfc_conditions) = p.precomputed

    if !isnothing(rayleigh_sponge)
        (; ᶜz, ᶠz) = z_coordinate_fields(axes(Y.c))
        zmax = Spaces.z_max(axes(ᶠz))
        draft_indices = StaticOneTo(n_mass_flux_subdomains(turbconv_model))
        χ_names = sgs_tracer_names(Y)
        ρχ_names = unrolled_map(get_ρχ_name, χ_names)
        DataLayouts.foreach_column(
            Yₜ.c, Y.c, ᶜh_tot, ᶜK, ᶜz,
        ) do Yₜc, Yc, ᶜh_tot, ᶜK, ᶜz
            @. Yₜc.uₕ -= β_rayleigh_uₕ(rayleigh_sponge, ᶜz, zmax) * Yc.uₕ
            ᶜβ_tr = @. β_rayleigh_tracer(rayleigh_sponge, ᶜz, zmax)
            if MatrixFields.has_field(Yc, @name(ρtke))
                @. Yₜc.ρtke -= ᶜβ_tr * Yc.ρtke
            end
            # Updrafts relax toward the grid mean.
            for j in draft_indices
                ᶜsgsʲ = Yc.sgsʲs.:($j)
                ᶜsgsʲₜ = Yₜc.sgsʲs.:($j)
                @. ᶜsgsʲₜ.mse -= ᶜβ_tr * (ᶜsgsʲ.mse - (ᶜh_tot - ᶜK))
                @. ᶜsgsʲₜ.q_tot -= ᶜβ_tr * (ᶜsgsʲ.q_tot - specific(Yc.ρq_tot, Yc.ρ))
                MatrixFields.unrolled_foreach(
                    map(tuple, χ_names, ρχ_names),
                ) do (χ_name, ρχ_name)
                    @inline
                    ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
                    ᶜsgs_χ = MatrixFields.get_field(ᶜsgsʲ, χ_name)
                    ᶜsgs_χₜ = MatrixFields.get_field(ᶜsgsʲₜ, χ_name)
                    @. ᶜsgs_χₜ -= ᶜβ_tr * (ᶜsgs_χ - specific(ᶜρχ, Yc.ρ))
                end
            end
        end
    end
    # For HeldSuarezForcing, the radiation_mode is used as the forcing parameter
    forcing = radiation_mode isa HeldSuarezForcing ? radiation_mode : nothing
    hs_args = (ᶜuₕ, ᶜp, params, sfc_conditions.T_sfc, microphysics_model, forcing)
    hs_tendency_uₕ = held_suarez_forcing_tendency_uₕ(hs_args...)
    hs_tendency_ρe_tot = held_suarez_forcing_tendency_ρe_tot(ᶜρ, hs_args...)
    edmf_cor_tend_uₕ = scm_coriolis_tendency_uₕ(ᶜuₕ, scm_coriolis)
    lsa_args =
        (ᶜρ, thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, t, ls_adv)
    bc_lsa_tend_ρe_tot = large_scale_advection_tendency_ρe_tot(lsa_args...)

    viscous_sponge_tendency!(Yₜ, Y, p)

    # Held Suarez tendencies
    @. Yₜ.c.uₕ += hs_tendency_uₕ
    @. Yₜ.c.ρe_tot += hs_tendency_ρe_tot

    subsidence_tendency!(Yₜ, Y, p, t, p.atmos.subsidence)

    @. Yₜ.c.ρe_tot += bc_lsa_tend_ρe_tot
    if microphysics_model isa MoistMicrophysics
        bc_lsa_tend_ρq_tot = large_scale_advection_tendency_ρq_tot(lsa_args...)
        @. Yₜ.c.ρq_tot += bc_lsa_tend_ρq_tot
    end

    @. Yₜ.c.uₕ += edmf_cor_tend_uₕ

    external_forcing_tendency!(Yₜ, Y, p, t, p.atmos.external_forcing)

    radiation_tendency!(Yₜ, Y, p, t, p.atmos.radiation_mode)

    # Chemistry tendencies
    chemistry_tendency!(Yₜ, Y, p, t, p.atmos.chemistry_model)

    non_orographic_gravity_wave_apply_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.non_orographic_gravity_wave,
    )
    orographic_gravity_wave_apply_tendency!(
        Yₜ,
        Y,
        p,
        p.atmos.orographic_gravity_wave,
    )

    surface_temp_tendency!(Yₜ, Y, p, t, p.atmos.surface.temperature)

    sl = p.atmos.smagorinsky_lilly
    horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, sl)
    vertical_smagorinsky_lilly_explicit_tendency!(Yₜ, Y, p, t, sl)

    amd_tendency!(Yₜ, Y, p, t, p.atmos.amd_les)

    chd = p.atmos.constant_horizontal_diffusion
    horizontal_constant_diffusion_tendency!(Yₜ, Y, p, t, chd)

    edmfx_sgs_horizontal_diffusive_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    # Optional tendency to bring negative small tracers back from negative
    # at the cost of water vapor.
    tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t, microphysics_model)

    # NOTE: This will zero out all momentum tendencies in the EDMFX advection test,
    # where velocities do not evolve
    # DO NOT add additional velocity tendencies after this function
    zero_velocity_tendency!(Yₜ, Y, p, t)
end

"""
    prescribed_flow_vertical_advection_tendency!(Yₜ, Y, p, t)

Compute explicit vertical transport, vertical momentum PGF/buoyancy/Rayleigh
sponge, and `zero_velocity_tendency!` for `PrescribedFlow`. Microphysics and
sedimentation run in `post_step_callback!`. Mutates `Yₜ`.
"""
function prescribed_flow_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; rayleigh_sponge) = p.atmos
    (; energy_humidity_upwinding, tracer_upwinding) = p.atmos.numerics
    (; params) = p
    ᶜJ = Fields.local_geometry_field(axes(Y.c)).J
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J
    (; ᶜΦ, ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶠu³, ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)
    cp_d = CAP.cp_d(params)
    zero_dt = zero(eltype(params))
    sedimenting_names = sedimenting_tracer_names(Y)
    passive_names = passive_gs_tracer_names(Y)

    # Explicit vertical transport by the prescribed flow: continuity, central
    # geopotential and pressure work paired with `energy_humidity_upwinding` on
    # `e_tot - Φ`, `q_tot`, and all microphysics tracers, `tracer_upwinding` on
    # passive grid-scale tracers, and first-order upwind on `ρtke`.
    # Microphysics and sedimentation run in `post_step_callback!`.
    DataLayouts.foreach_column(
        Yₜ.c, Y.c, ᶜJ, ᶠJ, ᶠu³, ᶜp, ᶜΦ,
    ) do Yₜc, Yc, ᶜJ, ᶠJ, ᶠu³, ᶜp, ᶜΦ
        ᶠρ = @. ᶠinterp(Yc.ρ * ᶜJ) / ᶠJ
        @. Yₜc.ρ -= ᶜadvdivᵥ(ᶠρ * ᶠu³)

        ᶜχ_e = @. lazy(specific(Yc.ρe_tot, Yc.ρ) - ᶜΦ)
        @. Yₜc.ρe_tot -= ᶜadvdivᵥ(ᶠρ * ᶠu³ * ᶠinterp(ᶜΦ) + ᶠu³ * ᶠinterp(ᶜp))
        add_vertical_transport!(
            Yₜc.ρe_tot, ᶠρ, ᶠu³, ᶜχ_e, zero_dt, energy_humidity_upwinding,
        )
        if MatrixFields.has_field(Yc, @name(ρq_tot))
            ᶜq_tot = @. lazy(specific(Yc.ρq_tot, Yc.ρ))
            add_vertical_transport!(
                Yₜc.ρq_tot, ᶠρ, ᶠu³, ᶜq_tot, zero_dt, energy_humidity_upwinding,
            )
        end
        MatrixFields.unrolled_foreach(sedimenting_names) do ρχ_name
            @inline
            ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
            ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
            ᶜχ = @. lazy(specific(ᶜρχ, Yc.ρ))
            add_vertical_transport!(
                ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, zero_dt, energy_humidity_upwinding,
            )
        end
        MatrixFields.unrolled_foreach(passive_names) do ρχ_name
            @inline
            ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
            ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
            ᶜχ = @. lazy(specific(ᶜρχ, Yc.ρ))
            add_vertical_transport!(
                ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, zero_dt, tracer_upwinding,
            )
        end
        if MatrixFields.has_field(Yc, @name(ρtke))
            ᶜtke = @. lazy(specific(Yc.ρtke, Yc.ρ))
            add_vertical_transport!(
                Yₜc.ρtke, ᶠρ, ᶠu³, ᶜtke, zero_dt, Val(:first_order),
            )
        end
    end

    rst_u₃ = rayleigh_sponge_tendency_u₃(Y.f.u₃, rayleigh_sponge)
    DataLayouts.foreach_column(
        Yₜ.f, ᶠgradᵥ_ᶜΦ, ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice,
        rst_u₃ isa NullBroadcasted ? () : rst_u₃,
    ) do Yₜf, ᶠgradᵥ_ᶜΦ, ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, rst_u₃
        add_implicit_u₃_tendency!(
            Yₜf, ᶠgradᵥ_ᶜΦ, ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice,
            thermo_params, cp_d,
        )
        rst_u₃ isa Tuple || @. Yₜf.u₃ += rst_u₃
    end

    zero_velocity_tendency!(Yₜ, Y, p, t)
    return nothing
end

"""
    fully_explicit_tendency!(Yₜ, Yₜ_lim, Y, p, t)

Compute the full tendency explicitly for `PrescribedFlow` by evaluating
`remaining_tendency!` and `prescribed_flow_vertical_advection_tendency!`.
Mutates `Yₜ` and `Yₜ_lim`.
"""
function fully_explicit_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    prescribed_flow_vertical_advection_tendency!(Yₜ, Y, p, t)
    return nothing
end
