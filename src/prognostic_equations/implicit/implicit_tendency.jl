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
[`JacobianAlgorithm`](@ref) subtypes; the residual applies the
`energy_humidity_upwinding` reconstruction for the vertical transport of
`e_tot - Φ`, `q_tot`, and the non-equilibrium microphysics tracers (keeping
`Φ` and pressure-work fluxes central), and first-order upwind for `ρtke`
(sharing its tridiagonal `(ρtke, ρtke)` block with eddy diffusion and TKE
dissipation when diffusion is implicit). The Jacobian diagonal blocks for
`ρe_tot` and `ρq_tot` remain `-I` (central approximation); Newton iterations
correct the residual mismatch from the upwind reconstruction.

Reads the precomputed quantities set by `set_implicit_precomputed_quantities!`
(`ᶠu³`, `ᶜp`, sedimentation and precipitation terminal velocities, and
the EDMFX subdomain states). Returns `nothing`.

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
@inline function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:none})
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    ᶠJ = Fields.local_geometry_field(axes(ᶠu³)).J
    return @. lazy(-(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠu³ * ᶠinterp(ᶜχ))))
end
@inline function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:first_order})
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    ᶠJ = Fields.local_geometry_field(axes(ᶠu³)).J
    return @. lazy(-(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠupwind1(ᶠu³, ᶜχ))))
end
@inline function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:vanleer_limiter})
    ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
    ᶠJ = Fields.local_geometry_field(axes(ᶠu³)).J
    # `ᶠlin_vanleer` receives `zero(dt)`, evaluating the `dt`-independent MUSCL
    # reconstruction with the `MonotoneLocalExtrema` (Lin et al. 1994, eq. 5)
    # slope limiter; without the `(1 - |u³| dt)` Lax–Wendroff factor, an
    # explicit forward-Euler step is monotone up to Courant number `1 / 2`.
    return @. lazy(
        -(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠlin_vanleer(ᶠu³, ᶜχ, zero(dt)))),
    )
end
@inline function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:third_order})
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
@inline add_vertical_transport!(
    ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dt::FT, ::Val{:vanleer_limiter},
) where {FT} =
    (@. ᶜρχₜ += -(ᶜadvdivᵥ(ᶠρ * ᶠlin_vanleer(ᶠu³, ᶜχ, $(zero(FT))))); nothing)
@inline add_vertical_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dt, ::Val{:third_order}) =
    (@. ᶜρχₜ += -(ᶜadvdivᵥ(ᶠρ * ᶠupwind3(ᶠu³, ᶜχ))); nothing)

"""
    vertical_advection(ᶠu³, ᶜχ, upwinding)

Return a lazy center-space broadcast of the advective-form vertical advection
tendency `-u³ ∂ᵥχ`, computed as the difference `-(∂ᵥ(u³ χ) - χ ∂ᵥu³)` so that
the same divergence operator (and reconstruction) is used as in
`vertical_transport`. `upwinding` is `Val(:none)` or `Val(:first_order)`, the
two reconstructions of the updraft advection.
"""
vertical_advection(ᶠu³, ᶜχ, ::Val{:none}) =
    @inline @. lazy(-(ᶜadvdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))
vertical_advection(ᶠu³, ᶜχ, ::Val{:first_order}) =
    @inline @. lazy(-(ᶜadvdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))

"""
    add_energy_transport!(ᶜρe_totₜ, ᶠρ, ᶠu³, ᶜe_tot, ᶜΦ, ᶜp, dt, upwinding)

Add the implicit vertical energy transport to `ᶜρe_totₜ`. For `Val(:none)`,
uses the central flux `-∂ᵥ(ᶠρ u³ ᶠinterp(e_tot) + u³ ᶠinterp(p))`. For all
other schemes, applies the upwinding reconstruction to `e_tot - Φ` while
keeping `Φ` and pressure work central:
`-∂ᵥ(ᶠρ ᶠupwind(u³, e_tot-Φ) + ᶠρ u³ ᶠinterp(Φ) + u³ ᶠinterp(p))`.
The upwind operators (`ᶠupwind1`, `ᶠupwind3`, `ᶠlin_vanleer`) already
multiply by `u³` and return a `Contravariant3Vector` flux, so `Φ` must be
kept as a separate `ᶠρ u³ ᶠinterp(Φ)` term rather than added inside the
operator argument. Keeping `Φ` and `p` central preserves the hydrostatic
pair `(p, Φ)` on the same stencil as continuity and gives `ρ(e_tot-Φ)` unit
Jacobian derivative at fixed `ρ`. `dt` is passed for `ᶠlin_vanleer`, which
receives `zero(dt)`. Inlined for the fused column loop.
"""
@inline function add_energy_transport!(
    ᶜρe_totₜ, ᶠρ, ᶠu³, ᶜe_tot, ᶜΦ, ᶜp, dt, ::Val{:none},
)
    @. ᶜρe_totₜ -= ᶜadvdivᵥ(ᶠρ * ᶠu³ * ᶠinterp(ᶜe_tot) + ᶠu³ * ᶠinterp(ᶜp))
    return nothing
end
@inline function add_energy_transport!(
    ᶜρe_totₜ, ᶠρ, ᶠu³, ᶜe_tot, ᶜΦ, ᶜp, dt, ::Val{:first_order},
)
    ᶜe_sans_Φ = @. lazy(ᶜe_tot - ᶜΦ)
    @. ᶜρe_totₜ -= ᶜadvdivᵥ(
        ᶠρ * ᶠupwind1(ᶠu³, ᶜe_sans_Φ) +
        ᶠρ * ᶠu³ * ᶠinterp(ᶜΦ) +
        ᶠu³ * ᶠinterp(ᶜp),
    )
    return nothing
end
@inline function add_energy_transport!(
    ᶜρe_totₜ, ᶠρ, ᶠu³, ᶜe_tot, ᶜΦ, ᶜp, dt, ::Val{:third_order},
)
    ᶜe_sans_Φ = @. lazy(ᶜe_tot - ᶜΦ)
    @. ᶜρe_totₜ -= ᶜadvdivᵥ(
        ᶠρ * ᶠupwind3(ᶠu³, ᶜe_sans_Φ) +
        ᶠρ * ᶠu³ * ᶠinterp(ᶜΦ) +
        ᶠu³ * ᶠinterp(ᶜp),
    )
    return nothing
end
@inline function add_energy_transport!(
    ᶜρe_totₜ, ᶠρ, ᶠu³, ᶜe_tot, ᶜΦ, ᶜp, dt::FT, ::Val{:vanleer_limiter},
) where {FT}
    ᶜe_sans_Φ = @. lazy(ᶜe_tot - ᶜΦ)
    @. ᶜρe_totₜ -= ᶜadvdivᵥ(
        ᶠρ * ᶠlin_vanleer(ᶠu³, ᶜe_sans_Φ, $(zero(FT))) +
        ᶠρ * ᶠu³ * ᶠinterp(ᶜΦ) +
        ᶠu³ * ᶠinterp(ᶜp),
    )
    return nothing
end

"""
    implicit_vertical_advection_tendency!(Yₜ, Y, p, t)

Add the implicit vertical-advection and vertical-momentum tendencies to `Yₜ`.

Adds, in order:

  - The mass flux divergence `-∂ᵥ(ρ u³)` to `Yₜ.c.ρ`, with zero flux through the
    top and bottom boundaries (consistent with the state filter that zeroes `ᶠu³`
    there and with the `ᶜadvdivᵥ_matrix()` used in the manual Jacobian).
  - Vertical transport of energy (and `q_tot` when present) to
    `Yₜ.c.ρe_tot` (and `Yₜ.c.ρq_tot`), applying `energy_humidity_upwinding` to
    `e_tot - Φ` while keeping `Φ` and pressure-work fluxes central
    (`add_energy_transport!`).
  - Vertical transport of the non-equilibrium microphysics tracers with the
    `energy_humidity_upwinding` scheme and the grid-mean velocity `ᶠu³`, plus
    vertical transport of the microphysics tracers with their terminal velocities
    using downward (`ᶠtop_bias`) biasing and a free-outflow bottom boundary
    (`ᶜprecipdivᵥ`), and the water sedimentation contributions to `ρ`, `ρe_tot`,
    and `ρq_tot` from `vertical_advection_of_water_tendency!`.
  - First-order upwind vertical transport of `ρtke`.
  - The Exner-form pressure gradient and buoyancy tendency
    `-(∂ᵥΦ - ∂ᵥΦ_r + cp_d (θ_v - θ_vr) ∂ᵥΠ)` and the Rayleigh sponge tendency to
    `Yₜ.f.u₃`.

`ρtke` uses first-order upwind advection directly in the residual so its
tridiagonal `(ρtke, ρtke)` block solves advection together with eddy diffusion
and stiff nonlinear TKE dissipation in one M-matrix when diffusion is implicit
(avoiding inhomogeneous diagonal scaling of central fluxes across inversion
interfaces, and omitting velocity blocks so `u³` sign reversals cannot turn a
frozen-donor upwind flux downwind). Vertical advection of passive non-water
tracers by the mean flow is treated explicitly. Returns `nothing`.
"""
function implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; rayleigh_sponge) = p.atmos
    (; params) = p
    ᶜJ = Fields.local_geometry_field(axes(Y.c)).J
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J
    (; ᶠgradᵥ_ᶜΦ, ᶠ∂ξ³∂x³, ᶜΦ) = p.core
    (; ᶠu³, ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)
    cp_d = CAP.cp_d(params)
    (; energy_humidity_upwinding) = p.atmos.numerics

    # Non-equilibrium microphysics tracers are advected implicitly both by the
    # grid-mean vertical velocity (zero boundary flux) and by their sedimentation
    # terminal velocity (downward-biased, free outflow bottom boundary); the
    # EDMFX difference-form SGS mass flux
    # `-∂ᵥ(ρᵏ aᵏ (u³ᵏ - u³) (χᵏ - χ))` is added on top of this grid-mean
    # transport in `edmfx_sgs_mass_flux_tendency!`. Passive non-water tracers
    # are advected in the explicit tendency.
    sedimenting_names = sedimenting_tracer_names(Y)
    ᶜws = unrolled_map(sedimenting_names) do ρχ_name
        MatrixFields.get_field(p.precomputed, sedimentation_velocity_name(ρχ_name))
    end

    # One fused column loop for the center tendencies: the face density
    # `ᶠinterp(ρ J) / J` is a loop temporary shared by every transport term.
    # `dt_zero` is computed here (outside the closure) so the closure captures a
    # concrete Float64 value rather than Type{Float64}, which is not isbits and
    # would prevent GPU kernel compilation.  It also ensures `add_energy_transport!`
    # and `add_vertical_transport!` receive a fully-typed `dt::FT` argument so
    # `ᶠlin_vanleer`'s `$(zero(FT))` escape resolves to a concrete literal.
    dt_zero = zero(eltype(Y))
    DataLayouts.foreach_column(
        Yₜ.c, Y.c, ᶜJ, ᶠJ, ᶠu³, ᶜp, ᶜΦ, ᶠ∂ξ³∂x³, ᶜws,
    ) do Yₜc, Yc, ᶜJ, ᶠJ, ᶠu³, ᶜp, ᶜΦ, ᶠ∂ξ³∂x³, ᶜws
        ᶠρ = @. ᶠinterp(Yc.ρ * ᶜJ) / ᶠJ

        # Mass advection with zero flux through the top and bottom
        # boundaries (ᶜadvdivᵥ). The state filter in
        # set_implicit_precomputed_quantities! also sets ᶠu³ to 0 at both
        # boundaries, and the ρ row of the manual Jacobian is built from
        # ᶜadvdivᵥ_matrix(), so using ᶜadvdivᵥ here keeps the residual, the
        # boundary conditions, and the Jacobian consistent.
        @. Yₜc.ρ -= ᶜadvdivᵥ(ᶠρ * ᶠu³)

        ᶜe_tot = @. lazy(specific(Yc.ρe_tot, Yc.ρ))
        add_energy_transport!(
            Yₜc.ρe_tot,
            ᶠρ,
            ᶠu³,
            ᶜe_tot,
            ᶜΦ,
            ᶜp,
            dt_zero,
            energy_humidity_upwinding,
        )
        if MatrixFields.has_field(Yc, @name(ρq_tot))
            ᶜq_tot = @. lazy(specific(Yc.ρq_tot, Yc.ρ))
            add_vertical_transport!(
                Yₜc.ρq_tot, ᶠρ, ᶠu³, ᶜq_tot, dt_zero, energy_humidity_upwinding,
            )
        end

        MatrixFields.unrolled_foreach(
            map(tuple, sedimenting_names, ᶜws),
        ) do (ρχ_name, ᶜw)
            @inline
            ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
            ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
            ᶜχ = @. lazy(specific(ᶜρχ, Yc.ρ))
            add_vertical_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dt_zero, energy_humidity_upwinding)
            @. ᶜρχₜ -= ᶜprecipdivᵥ(
                CT3(ᶠ∂ξ³∂x³ * (ᶠρ * ᶠtop_bias(-(ᶜw) * specific_nonneg(ᶜρχ, Yc.ρ)))),
            )
        end

        if MatrixFields.has_field(Yc, @name(ρtke))
            ᶜtke = @. lazy(specific(Yc.ρtke, Yc.ρ))
            add_vertical_transport!(
                Yₜc.ρtke, ᶠρ, ᶠu³, ᶜtke, dt_zero, Val(:first_order),
            )
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
of `implicit_vertical_advection_tendency!`.
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
    post_newton_stage_update!(Yₜ, Y, p, t)

Accumulate stage-weighted surface precipitation fluxes at the Newton-solved
stage state `Y = U*`. Registered as the `T_post_imp!` hook of
`ClimaODEFunction`; CTS calls this once per implicit stage after Newton
converges. Sets `Yₜ = 0` so no additional correction is applied to `U*`.
"""
function post_newton_stage_update!(Yₜ, Y, p, t)
    accumulate_stage_precipitation_surface_fluxes!(
        Y,
        p,
        p.atmos.microphysics_model,
    )
    Yₜ .= zero(eltype(Yₜ))
    return nothing
end
