#####
##### Implicit tendencies
#####

import ClimaCore
import ClimaCore: DataLayouts, Fields, Geometry

"""
    implicit_tendency!(Yₜ, Y, p, t)

Compute the implicitly treated tendency of the state `Y` at time `t`,
overwriting `Yₜ`.

This is the stiff part of the IMEX splitting, containing the terms that are
fast relative to the timestep: vertical advection and the vertical momentum
equation (`implicit_vertical_advection_tendency!`), EDMFX subdomain vertical
advection, entrainment/detrainment, mass flux, boundary conditions, pressure
work, and the cached updraft `u₃`/`ρa` stage tendencies. Microphysics sources
and surface precipitation deposition are always applied here; the
`implicit_microphysics` config only decides whether 1M cond/subl flow through
`ᶜmp_tendency_implicit` (refreshed per Newton iterate with a `−dtγ/τ`
Jacobian) or stay in the frozen bulk `ᶜmp_tendency`. Vertical diffusion is
included only when `p.atmos.diff_mode == Implicit()`. All non-EDMFX
configurations dispatch the SGS tendencies to no-ops.

The linearization of this tendency is the Jacobian ``∂Yₜ/∂Y`` computed by the
[`JacobianAlgorithm`](@ref) subtypes; the residual it enters contains the
central (`Val(:none)`) vertical transport of `ρe_tot` and `ρq_tot`, and the
first-order upwind (`Val(:first_order)`) vertical transport of the
non-equilibrium microphysics tracers and `ρtke` — the Courant-limited
upwind correction on `ρe_tot` and `ρq_tot` is applied after the Newton solve by
`correct_implicit_advection_tendency!`.

Reads the precomputed quantities set by `set_implicit_precomputed_quantities!`
(`ᶠu³`, `ᶜh_tot`, sedimentation and precipitation terminal velocities, and the
EDMFX subdomain states). Returns `nothing`.

See [Implicit Solver](@ref) for the IMEX formulation.
"""
NVTX.@annotate function implicit_tendency!(Yₜ, Y, p, t)
    fill_with_nans!(p)
    Yₜ .= zero(eltype(Yₜ))
    implicit_vertical_advection_tendency!(Yₜ, Y, p, t)

    # Microphysics tendencies are always applied here. `implicit_microphysics`
    # only decides whether cond/subl flows through `ᶜmp_tendency_implicit`
    # (refreshed per Newton iter via `set_implicit_precomputed_quantities!`
    # with a `-dtγ/τ` Jacobian) or stays in the frozen bulk `ᶜmp_tendency`.
    microphysics_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.microphysics_model,
        p.atmos.turbconv_model,
    )
    surface_precipitation_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.surface.temperature,
        p.atmos.microphysics_model,
    )

    edmfx_sgs_vertical_advection_tendency!(
        Yₜ,
        Y,
        p,
        t,
        p.atmos.turbconv_model,
    )

    if p.atmos.diff_mode == Implicit()
        vertical_diffusion_boundary_layer_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.vertical_diffusion,
        )
        vertical_smagorinsky_lilly_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.smagorinsky_lilly,
        )
        edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
        edmfx_tke_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end

    edmfx_entr_detr_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    edmfx_boundary_condition_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    # NOTE: All ρa tendencies should be applied before calling this function
    pressure_work_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    sgs_u₃_implicit_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    sgs_ρa_implicit_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    # NOTE: This will zero out all momentum tendencies in the edmfx advection test
    # DO NOT add additional velocity tendencies after this function
    zero_velocity_tendency!(Yₜ, Y, p, t)

    return nothing
end

"""
    vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, upwinding)

Return a lazy center-space broadcast of the flux-form vertical transport
tendency `-∂ᵥ(ρ u³ χ)`, with zero-flux boundary conditions imposed by
`ᶜadvdivᵥ`.

# Arguments

  - `ᶜρ`: Center-space density [kg/m³].
  - `ᶠu³`: Face-space contravariant vertical velocity [1/s].
  - `ᶜχ`: Center-space specific (per-mass) transported quantity.
  - `dt`: Timestep [s]; `ᶠlin_vanleer` receives `zero(dt)` to evaluate the
    Courant-independent MUSCL reconstruction.
  - `upwinding`: Reconstruction of `χ` on faces; one of `Val(:none)` (central
    interpolation), `Val(:first_order)`, `Val(:third_order)`, or
    `Val(:vanleer_limiter)`.
"""
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:none})
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    ᶠJ = Fields.local_geometry_field(axes(ᶠu³)).J
    return @. lazy(-(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠu³ * ᶠinterp(ᶜχ))))
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:first_order})
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    ᶠJ = Fields.local_geometry_field(axes(ᶠu³)).J
    return @. lazy(-(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠupwind1(ᶠu³, ᶜχ))))
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:vanleer_limiter})
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    ᶠJ = Fields.local_geometry_field(axes(ᶠu³)).J
    # `ᶠlin_vanleer` receives `zero(dt)`, so it evaluates the
    # Courant-independent MUSCL reconstruction with the `MonotoneLocalExtrema`
    # (Lin et al. 1994, eq. 5) slope limiter.
    return @. lazy(
        -(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠlin_vanleer(ᶠu³, ᶜχ, zero(dt)))),
    )
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:third_order})
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    ᶠJ = Fields.local_geometry_field(axes(ᶠu³)).J
    return @. lazy(-(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠupwind3(ᶠu³, ᶜχ))))
end

"""
    add_vertical_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dt, upwinding)

Add the vertical transport of `vertical_transport` to `ᶜρχₜ`, with the face
density `ᶠρ = ᶠinterp(ᶜρ * ᶜJ) / ᶠJ` already computed, so that a fused column loop can
share it between several tracers. Inlined, since it runs inside loop bodies (a lazy
broadcast returned from a function is not inferred there).
"""
@inline add_vertical_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dt, ::Val{:none}) =
    (@. ᶜρχₜ += -(ᶜadvdivᵥ(ᶠρ * ᶠu³ * ᶠinterp(ᶜχ))); nothing)
@inline add_vertical_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dt, ::Val{:first_order}) =
    (@. ᶜρχₜ += -(ᶜadvdivᵥ(ᶠρ * ᶠupwind1(ᶠu³, ᶜχ))); nothing)
@inline add_vertical_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dt, ::Val{:vanleer_limiter}) =
    (@. ᶜρχₜ += -(ᶜadvdivᵥ(ᶠρ * ᶠlin_vanleer(ᶠu³, ᶜχ, zero(dt)))); nothing)
@inline add_vertical_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dt, ::Val{:third_order}) =
    (@. ᶜρχₜ += -(ᶜadvdivᵥ(ᶠρ * ᶠupwind3(ᶠu³, ᶜχ))); nothing)

"""
    vertical_advection(ᶠu³, ᶜχ, upwinding)

Return a lazy center-space broadcast of the advective-form vertical advection
tendency `-u³ ∂ᵥχ`, computed as the difference `-(∂ᵥ(u³ χ) - χ ∂ᵥu³)` so that
the same divergence operator (and reconstruction) is used as in
`vertical_transport`. `upwinding` is `Val(:none)`, `Val(:first_order)`,
or `Val(:third_order)`.
"""
vertical_advection(ᶠu³, ᶜχ, ::Val{:none}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))
vertical_advection(ᶠu³, ᶜχ, ::Val{:first_order}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))
vertical_advection(ᶠu³, ᶜχ, ::Val{:third_order}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))

"""
    implicit_vertical_advection_tendency!(Yₜ, Y, p, t)

Add the implicit vertical-advection and vertical-momentum tendencies to `Yₜ`.

Adds, in order:

  - The mass flux divergence `-∂ᵥ(ρ u³)` to `Yₜ.c.ρ`, with zero flux through the
    top and bottom boundaries (consistent with the state filter that zeroes `ᶠu³`
    there and with the `ᶜadvdivᵥ_matrix()` used in the manual Jacobian).
  - Central (`Val(:none)`) vertical transport of `h_tot` and `q_tot` to
    `Yₜ.c.ρe_tot` and `Yₜ.c.ρq_tot`; the Courant-limited upwind dissipation is
    applied post-Newton by `correct_implicit_advection_tendency!`.
  - First-order upwind vertical transport of the non-equilibrium microphysics
    tracers and `ρtke` with the grid-mean velocity `ᶠu³`, plus vertical
    transport of the microphysics tracers with their terminal velocities using
    downward (`ᶠtop_bias`) biasing and a free-outflow bottom boundary
    (`ᶜprecipdivᵥ`), and the water sedimentation contributions to `ρ`, `ρe_tot`,
    and `ρq_tot` from `vertical_advection_of_water_tendency!`.
  - The Exner-form pressure gradient and buoyancy tendency
    `-(∂ᵥΦ - ∂ᵥΦ_r + cp_d (θ_v - θ_vr) ∂ᵥΠ)` and the Rayleigh sponge tendency to
    `Yₜ.f.u₃`.

Vertical advection of passive non-water tracers by the mean flow is treated
explicitly. Returns `nothing`.
"""
function implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; rayleigh_sponge) = p.atmos
    (; params) = p
    ᶜJ = Fields.local_geometry_field(axes(Y.c)).J
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J
    (; ᶠgradᵥ_ᶜΦ, ᶠ∂ξ³∂x³) = p.core
    (; ᶠu³, ᶜp, ᶜh_tot, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)
    cp_d = CAP.cp_d(params)

    # Non-equilibrium microphysics tracers are advected implicitly both by the
    # grid-mean vertical velocity (first-order upwind, zero boundary flux) and
    # by their sedimentation terminal velocity (downward-biased, free outflow
    # bottom boundary); the EDMFX difference-form SGS mass flux
    # `-∂ᵥ(ρᵏ aᵏ (u³ᵏ - u³) (χᵏ - χ))` is added on top of this grid-mean
    # transport in `edmfx_sgs_mass_flux_tendency!`. Passive non-water tracers
    # are advected in the explicit tendency.
    sedimenting_names = sedimenting_tracer_names(Y)
    ᶜws = unrolled_map(sedimenting_names) do ρχ_name
        MatrixFields.get_field(p.precomputed, sedimentation_velocity_name(ρχ_name))
    end

    # One fused column loop for the center tendencies: the face density
    # `ᶠinterp(ρ J) / J` is a loop temporary shared by every transport term.
    DataLayouts.foreach_column(
        Yₜ.c, Y.c, ᶜJ, ᶠJ, ᶠu³, ᶜh_tot, ᶠ∂ξ³∂x³, ᶜws,
    ) do Yₜc, Yc, ᶜJ, ᶠJ, ᶠu³, ᶜh_tot, ᶠ∂ξ³∂x³, ᶜws
        ᶠρ = @. ᶠinterp(Yc.ρ * ᶜJ) / ᶠJ

        # Mass advection with zero flux through the top and bottom
        # boundaries (ᶜadvdivᵥ). The state filter in
        # set_implicit_precomputed_quantities! also sets ᶠu³ to 0 at both
        # boundaries, and the ρ row of the manual Jacobian is built from
        # ᶜadvdivᵥ_matrix(), so using ᶜadvdivᵥ here keeps the residual, the
        # boundary conditions, and the Jacobian consistent.
        @. Yₜc.ρ -= ᶜadvdivᵥ(ᶠρ * ᶠu³)

        # Central vertical advection of active scalars (ρe_tot and ρq_tot),
        # keeping the (ρe_tot, ρe_tot) and (ρq_tot, ρq_tot) advection diagonals
        # zero so dry acoustic/baroclinic solves invert via exact
        # BlockArrowheadSolve. The Courant-limited upwind dissipation is
        # applied post-Newton via T_post_imp! (see
        # `correct_implicit_advection_tendency!`).
        @. Yₜc.ρe_tot -= ᶜadvdivᵥ(ᶠρ * ᶠu³ * ᶠinterp(ᶜh_tot))
        if MatrixFields.has_field(Yc, @name(ρq_tot))
            ᶜq_tot = @. lazy(specific(Yc.ρq_tot, Yc.ρ))
            @. Yₜc.ρq_tot -= ᶜadvdivᵥ(ᶠρ * ᶠu³ * ᶠinterp(ᶜq_tot))
        end

        MatrixFields.unrolled_foreach(
            map(tuple, sedimenting_names, ᶜws),
        ) do (ρχ_name, ᶜw)
            @inline
            ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
            ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
            ᶜχ = @. lazy(specific(ᶜρχ, Yc.ρ))
            @. ᶜρχₜ -= ᶜadvdivᵥ(ᶠρ * ᶠupwind1(ᶠu³, ᶜχ))
            @. ᶜρχₜ -= ᶜprecipdivᵥ(
                CT3(ᶠ∂ξ³∂x³ * (ᶠρ * ᶠtop_bias(-(ᶜw) * ᶜχ))),
            )
        end

        if MatrixFields.has_field(Yc, @name(ρtke))
            ᶜtke = @. lazy(specific(Yc.ρtke, Yc.ρ))
            @. Yₜc.ρtke -= ᶜadvdivᵥ(ᶠρ * ᶠupwind1(ᶠu³, ᶜtke))
        end
        return nothing
    end

    vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

    # Pressure gradient, buoyancy, and Rayleigh sponge tendencies of u₃ in one
    # fused column loop; the virtual potential temperature is a loop temporary.
    # The sponge is a lazy broadcast when active and a `NullBroadcasted`
    # otherwise, which cannot be sliced, so it is passed only when active.
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
        return nothing
    end
    return nothing
end

"""
    add_implicit_u₃_tendency!(Yₜf, ᶠgradᵥ_ᶜΦ, ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice,
                              thermo_params, cp_d)

Add the Exner-form pressure gradient and buoyancy tendency
`-(∂ᵥΦ - ∂ᵥΦ_r + cp_d (θ_v - θ_vr) ∂ᵥΠ)` to `Yₜf.u₃` for one column, which is
equivalent to `grad_v(Φ) + grad_v(p) / ρ`. Called inside the fused column loop
of `implicit_vertical_advection_tendency!`, so it must stay inlined.
"""
@inline function add_implicit_u₃_tendency!(
    Yₜf,
    ᶠgradᵥ_ᶜΦ,
    ᶜp,
    ᶜT,
    ᶜq_tot_nonneg,
    ᶜq_liq,
    ᶜq_ice,
    thermo_params,
    cp_d,
)
    ᶜΦ_r = @. lazy(phi_r(thermo_params, ᶜp))
    ᶜθ_v = @. theta_v(thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)
    ᶜθ_vr = @. lazy(theta_vr(thermo_params, ᶜp))
    ᶜΠ = @. lazy(TD.exner_given_pressure(thermo_params, ᶜp))
    @. Yₜf.u₃ -= ᶠgradᵥ_ᶜΦ - ᶠgradᵥ(ᶜΦ_r) +
                 cp_d * (ᶠinterp(ᶜθ_v - ᶜθ_vr)) * ᶠgradᵥ(ᶜΠ)
    return nothing
end

"""
    _courant_limiter(u³::Geometry.Contravariant3Vector, dtγ)

Return `min(1, 1 / (|u³| dtγ))`, the weight that keeps the explicit post-Newton
diffusion step at diffusion number `≤ 1/2` where the stage Courant number
`C_γ = |u³| dtγ` exceeds one.
"""
@inline function _courant_limiter(u³::Geometry.Contravariant3Vector, dtγ)
    FT = typeof(u³.u³)
    c = abs(u³.u³) * FT(dtγ)
    return ifelse(c > one(FT), inv(c), one(FT))
end

"""
    add_upwind_vertical_transport_correction!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dtγ, upwinding)

Add the Courant-limited post-Newton upwind vertical transport correction
`-ᶜadvdivᵥ(ᶠρ * min(1, 1 / (|ᶠu³| * dtγ)) * (ᶠF_upwind - ᶠu³ * ᶠinterp(ᶜχ)))` to
`ᶜρχₜ`, where `ᶠρ = ᶠinterp(ᶜρ * ᶜJ) / ᶠJ` is the precomputed face density. A
no-op for `Val(:none)`.
"""
@inline add_upwind_vertical_transport_correction!(
    ᶜρχₜ,
    ᶠρ,
    ᶠu³,
    ᶜχ,
    dtγ,
    ::Val{:none},
) = nothing
@inline add_upwind_vertical_transport_correction!(
    ᶜρχₜ,
    ᶠρ,
    ᶠu³,
    ᶜχ,
    dtγ,
    ::Val{:first_order},
) = (
    @. ᶜρχₜ -= ᶜadvdivᵥ(
        ᶠρ *
        _courant_limiter(ᶠu³, dtγ) *
        (ᶠupwind1(ᶠu³, ᶜχ) - ᶠu³ * ᶠinterp(ᶜχ)),
    );
    nothing
)
@inline add_upwind_vertical_transport_correction!(
    ᶜρχₜ,
    ᶠρ,
    ᶠu³,
    ᶜχ,
    dtγ,
    ::Val{:vanleer_limiter},
) = (
    @. ᶜρχₜ -= ᶜadvdivᵥ(
        ᶠρ *
        _courant_limiter(ᶠu³, dtγ) *
        (ᶠlin_vanleer(ᶠu³, ᶜχ, zero(dtγ)) - ᶠu³ * ᶠinterp(ᶜχ)),
    );
    nothing
)
@inline add_upwind_vertical_transport_correction!(
    ᶜρχₜ,
    ᶠρ,
    ᶠu³,
    ᶜχ,
    dtγ,
    ::Val{:third_order},
) = (
    @. ᶜρχₜ -= ᶜadvdivᵥ(
        ᶠρ *
        _courant_limiter(ᶠu³, dtγ) *
        (ᶠupwind3(ᶠu³, ᶜχ) - ᶠu³ * ᶠinterp(ᶜχ)),
    );
    nothing
)

"""
    add_antidiffusive_upwind1_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dtγ, upwinding)

Add the Courant-limited post-Newton high-order antidiffusive correction
`-ᶜadvdivᵥ(ᶠρ * min(1, 1 / (|ᶠu³| * dtγ)) * (ᶠF_upwind - ᶠupwind1(ᶠu³, ᶜχ)))`
to `ᶜρχₜ` for tracers advected implicitly with first-order upwinding. A no-op
for `Val(:first_order)` and `Val(:none)`.
"""
@inline add_antidiffusive_upwind1_transport!(
    ᶜρχₜ,
    ᶠρ,
    ᶠu³,
    ᶜχ,
    dtγ,
    ::Val{:vanleer_limiter},
) = (
    @. ᶜρχₜ -= ᶜadvdivᵥ(
        ᶠρ *
        _courant_limiter(ᶠu³, dtγ) *
        (ᶠlin_vanleer(ᶠu³, ᶜχ, zero(dtγ)) - ᶠupwind1(ᶠu³, ᶜχ)),
    );
    nothing
)
@inline add_antidiffusive_upwind1_transport!(
    ᶜρχₜ,
    ᶠρ,
    ᶠu³,
    ᶜχ,
    dtγ,
    ::Val{:third_order},
) = (
    @. ᶜρχₜ -= ᶜadvdivᵥ(
        ᶠρ *
        _courant_limiter(ᶠu³, dtγ) *
        (ᶠupwind3(ᶠu³, ᶜχ) - ᶠupwind1(ᶠu³, ᶜχ)),
    );
    nothing
)
@inline add_antidiffusive_upwind1_transport!(
    ᶜρχₜ,
    ᶠρ,
    ᶠu³,
    ᶜχ,
    dtγ,
    ::Union{Val{:first_order}, Val{:none}},
) = nothing

"""
    correct_implicit_advection_tendency!(Yₜ, Y, p, t)

Accumulate the stage-weighted bottom-face sedimentation fluxes at the
Newton-solved stage state `Y = U*` and apply the Courant-limited post-Newton
upwind dissipation to the central implicit vertical advection of `ρe_tot` and
`ρq_tot` in `implicit_vertical_advection_tendency!`. Called by
ClimaTimeSteppers as the `T_post_imp!` hook on `ClimaODEFunction`, which
`solve_stage_implicit!` invokes once per implicit stage at `U*` and applies as
`U ← U* + dtγ · Yₜ`.

Overwrites `Yₜ` with the face-wise Courant-damped upwind correction
`-∂ᵥ(ρ min(1, 1 / C_γ) (F_upwind - u³ ᶠinterp(χ)))` for `ρe_tot` and `ρq_tot`
(using `energy_q_tot_upwinding`). Because `ᶠlin_vanleer(u³, χ, 0) - u³ ᶠinterp(χ)`
vanishes identically on smooth linear profiles, smooth acoustic and baroclinic
modes remain untouched while inversions receive Courant-bounded upwind
dissipation. When `tracer_upwinding` is `Val(:vanleer_limiter)` or
`Val(:third_order)`, the sedimenting microphysics tracers
(`sedimenting_tracer_names(Y)`) receive the Courant-limited antidiffusive
correction (`ᶠF_upwind - ᶠupwind1`), while `ρtke` stays on unconditionally
monotone `ᶠupwind1` inside `implicit_vertical_advection_tendency!` and requires
no post-Newton correction. All other fields of `Yₜ` are zero. Returns `nothing`.
"""
NVTX.@annotate function correct_implicit_advection_tendency!(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    (; energy_q_tot_upwinding, tracer_upwinding) = p.atmos.numerics
    energy_q_tot_upwinding == Val(:none) &&
        tracer_upwinding == Val(:none) &&
        return nothing
    (; dtγ) = p
    (; ᶠu³, ᶜh_tot) = p.precomputed
    ᶜJ = Fields.local_geometry_field(axes(Y.c)).J
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J
    ᶠρ = @. lazy(ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ)

    if energy_q_tot_upwinding != Val(:none)
        add_upwind_vertical_transport_correction!(
            Yₜ.c.ρe_tot,
            ᶠρ,
            ᶠu³,
            ᶜh_tot,
            dtγ,
            energy_q_tot_upwinding,
        )
        if MatrixFields.has_field(Y.c, @name(ρq_tot))
            ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
            add_upwind_vertical_transport_correction!(
                Yₜ.c.ρq_tot,
                ᶠρ,
                ᶠu³,
                ᶜq_tot,
                dtγ,
                energy_q_tot_upwinding,
            )
        end
    end

    if !(tracer_upwinding in (Val(:none), Val(:first_order)))
        MatrixFields.unrolled_foreach(
            sedimenting_tracer_names(Y),
        ) do ρχ_name
            ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
            ᶜρχₜ = MatrixFields.get_field(Yₜ.c, ρχ_name)
            ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
            add_antidiffusive_upwind1_transport!(
                ᶜρχₜ,
                ᶠρ,
                ᶠu³,
                ᶜχ,
                dtγ,
                tracer_upwinding,
            )
        end
    end
    return nothing
end
