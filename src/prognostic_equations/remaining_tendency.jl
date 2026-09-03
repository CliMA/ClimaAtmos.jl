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

Zeroes `Yₜ` and `Yₜ_lim`, then accumulates, in order:

 1. `horizontal_tracer_advection_tendency!` (into `Yₜ_lim`).
 2. `horizontal_dynamics_tendency!`.
 3. `hyperdiffusion_tendency!` (tracer part into `Yₜ_lim`).
 4. `explicit_vertical_advection_tendency!`.
 5. `additional_tendency!` (sponges, forcings, parameterizations, EDMFX SGS terms).

`Yₜ_lim` collects the tracer tendencies that are subject to the horizontal
limiters (see `limiters_func!`); everything else accumulates into `Yₜ`. The
implicitly treated counterparts (vertical acoustic/advective terms, diffusion when
`p.atmos.diff_mode == Implicit()`) live in `implicit_tendency!`.

# Returns

`Yₜ`, the populated main tendency vector.
"""
NVTX.@annotate function remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ_lim .= zero(eltype(Yₜ_lim))
    Yₜ .= zero(eltype(Yₜ))
    horizontal_tracer_advection_tendency!(Yₜ_lim, Y, p, t)
    fill_with_nans!(p)  # TODO: would be better to limit this to debug mode (e.g., if p.debug_mode...)
    horizontal_dynamics_tendency!(Yₜ, Y, p, t)
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
  - Vertical diffusion and EDMFX SGS diffusive fluxes, only when
    `p.atmos.diff_mode == Explicit()` (implicit diffusion is applied in
    `implicit_tendency!`).
  - Surface fluxes, radiation, EDMFX TKE, and chemistry.
  - Microphysics (condensation, precipitation) and surface precipitation, only when
    `p.atmos.microphysics_tendency_timestepping == Explicit()`.
  - Non-orographic and orographic gravity-wave drag.
  - Surface temperature evolution, pressure work, Smagorinsky-Lilly, AMD LES, and
    constant horizontal diffusion.
  - Tracer nonnegativity restoration at the cost of water vapor, and finally
    `zero_velocity_tendency!` for advection tests (which must remain last so no
    later call reintroduces velocity tendencies).

The order of calls matters: microphysics must precede `surface_temp_tendency!`
(which reads the precipitation cache), and all `ρa` tendencies must precede
`pressure_work_tendency!`. Called from `remaining_tendency!`. Returns `nothing`.
"""
NVTX.@annotate function additional_tendency!(Yₜ, Y, p, t)

    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜρ = Y.c.ρ
    (; radiation_mode, microphysics_model, turbconv_model) = p.atmos
    (; rayleigh_sponge, viscous_sponge) = p.atmos
    (; ls_adv, scm_coriolis) = p.atmos
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜp, ᶜK, ᶜT, ᶜh_tot, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; sfc_conditions) = p.precomputed

    rst_uₕ = rayleigh_sponge_tendency_uₕ(ᶜuₕ, rayleigh_sponge)

    if use_prognostic_tke(turbconv_model)
        rst_ρtke = rayleigh_sponge_tendency_tracer(Y.c.ρtke, rayleigh_sponge)
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
        # Auto-discovered SGS tracers (microphysics species and any
        # user-defined passive tracers)
        for χ_name in sgs_tracer_names(Y)
            ρχ_name = get_ρχ_name(χ_name)
            ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
            ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
            for j in 1:n
                ᶜsgs_χ = MatrixFields.get_field(Y.c.sgsʲs.:(1), χ_name)
                ᶜsgs_χₜ = MatrixFields.get_field(Yₜ.c.sgsʲs.:(1), χ_name)
                rst_sgs_χ = rayleigh_sponge_tendency_sgs_tracer(ᶜsgs_χ, ᶜχ, rayleigh_sponge)
                @. ᶜsgs_χₜ += rst_sgs_χ
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

    # TODO: fuse, once we fix
    #       https://github.com/CliMA/ClimaCore.jl/issues/2165
    @. Yₜ.c.uₕ += rst_uₕ
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
    edmfx_tke_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    # Chemistry tendencies
    chemistry_tendency!(Yₜ, Y, p, t, p.atmos.chemistry_model)

    # Unified microphysics tendencies (cloud condensation + precipitation)
    if p.atmos.microphysics_tendency_timestepping == Explicit()
        microphysics_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.microphysics_model,
            p.atmos.turbconv_model,
        )
    end

    non_orographic_gravity_wave_apply_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.non_orographic_gravity_wave,
    )
    orographic_gravity_wave_apply_tendency!(
        Yₜ,
        p,
        p.atmos.orographic_gravity_wave,
    )

    # NOTE: Microphysics tendencies should be applied before calling this function,
    # because precipitation cache is used in this function
    surface_temp_tendency!(Yₜ, Y, p, t, p.atmos.surface.temperature)
    if p.atmos.microphysics_tendency_timestepping == Explicit()
        surface_precipitation_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.surface.temperature,
            p.atmos.microphysics_model,
        )
    end

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

    edmfx_sgs_horizontal_diffusive_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    # Optional tendency to bring negative small tracers back from negative
    # at the cost of water vapor.
    tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t, microphysics_model)

    aerosol_deposition_tendency!(Yₜ, Y, p, t)

    # NOTE: This will zero out all momentum tendencies in the EDMFX advection test,
    # where velocities do not evolve
    # DO NOT add additional velocity tendencies after this function
    zero_velocity_tendency!(Yₜ, Y, p, t)
end

"""
    fully_explicit_tendency!(Yₜ, Yₜ_lim, Y, p, t)

Compute the full tendency explicitly, by evaluating `implicit_tendency!` into
scratch space and adding it to `remaining_tendency!`.

Experimental timestepping mode used by `args_integrator` when `prescribed_flow` is
set, where the flow is imposed and implicit treatment of sound waves is
unnecessary. Mutates `Yₜ`, `Yₜ_lim`, and `p.scratch.temp_Yₜ_imp`.
"""
function fully_explicit_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    (; temp_Yₜ_imp) = p.scratch
    implicit_tendency!(temp_Yₜ_imp, Y, p, t)
    remaining_tendency!(Yₜ, Yₜ_lim, Y, p, t)
    Yₜ .+= temp_Yₜ_imp
end
