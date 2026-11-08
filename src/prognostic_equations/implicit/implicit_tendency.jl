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
vertical transport of `ρe_tot` and `ρq_tot` with the reconstruction of
`grid_mean_base_upwinding(Y)` on `e_tot - Φ` and `q_tot` (first-order upwind in
moist configurations, central in `DryModel`) plus central geopotential and
pressure-work fluxes `ᶠρ u³ ᶠinterp(Φ) + u³ ᶠinterp(p)`, and the first-order
upwind vertical transport of the non-equilibrium microphysics tracers and
`ρtke`. The Courant-limited high-order correction of `e_tot - Φ`, `q_tot`, and
the sedimenting tracers is applied after the Newton solve by
`correct_implicit_advection_tendency!`.

Reads the precomputed quantities set by `set_implicit_precomputed_quantities!`
(`ᶠu³`, `ᶜp`, `ᶜh_tot`, sedimentation and precipitation terminal velocities, and
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
    grid_mean_base_upwinding(Y)

Return the reconstruction of the implicit grid-mean vertical advection of
`e_tot - Φ` and `q_tot` inside the Newton residual: `Val(:first_order)` when `Y`
has `c.ρq_tot` (every moist configuration: `0M`, `1M`, `2M`, and `2MP3`), and
`Val(:none)` (central) otherwise (`DryModel`). In every configuration, the
gravitational-potential flux `ᶠρ u³ ᶠinterp(Φ)` and the pressure work
`u³ ᶠinterp(p)` stay central (`add_energy_transport!`), while specific internal
plus kinetic energy `e_tot - Φ` (and, in moist configurations, `q_tot` and any
sedimenting condensates) uses `grid_mean_base_upwinding(Y)` in the implicit
stage and receives the Courant-limited `energy_q_tot_upwinding` correction after
the Newton solve. Because `e_tot - Φ = e_int(T, q_tot, q_liq, q_ice) + K` has no
static `g z` ramp, at uniform `(T, u)` it is a spatially constant affine
function of `(1, q_tot, q_liq, q_ice)`: linear reconstructions (`first_order` or
`third_order`) commute with any affine combination, and the van Leer limiter
commutes where the scalars share one spatial shape `ψ(z)` with same-sign
coefficients. In `DryModel`, the Jacobian omits the tridiagonal
`(ρe_tot, ρe_tot)` central-advection block (off-diagonals `O(C_γ)`, diagonal
`O(dtγ ∂ᵥ(ρ u³))`), keeping the `-I` diagonal that `BlockArrowheadSolve`
requires; the Newton step is forward-Euler-central in `e_tot - Φ`. Read by
`implicit_vertical_advection_tendency!`, `add_vertical_advection_corrections!`,
and `args_integrator`.
"""
grid_mean_base_upwinding(Y) =
    MatrixFields.has_field(Y, @name(c.ρq_tot)) ? Val(:first_order) : Val(:none)

"""
    add_energy_transport!(ᶜρe_totₜ, ᶠρ, ᶠu³, ᶜe_tot, ᶜΦ, ᶜp, base_upwinding)

Add the implicit vertical energy transport `-∂ᵥ(ρ u³ h_tot)` to `ᶜρe_totₜ`,
with the face density `ᶠρ = ᶠinterp(ᶜρ * ᶜJ) / ᶠJ`, splitting the face flux into
material internal-plus-kinetic energy (`ᶠρ u³ ᶠinterp(e_tot - Φ)` for
`Val(:none)` and `ᶠρ ᶠupwind1(u³, e_tot - Φ)` for `Val(:first_order)`), central
geopotential `ᶠρ u³ ᶠinterp(Φ)`, and central pressure work `u³ ᶠinterp(p)`. In
`h_tot = e_int + p / ρ + Φ + K`, `(p, Φ)` form a hydrostatic pair
(`de_int + p d(1 / ρ) = 0` along an adiabat and `dp / ρ + dΦ = 0` in hydrostatic
balance, so `d(e_int + p / ρ + Φ) = 0`), and evaluating `u³ ᶠinterp(p)`
centrally separates pressure work from density variations and avoids the
`ρ ΔR_m T / 2` error of `ᶠρ ᶠupwind1(u³, p / ρ)` across moisture jumps at
continuous `p`. Evaluating `ᶠρ u³ ᶠinterp(Φ)` centrally keeps `Φ` on the same
stencil as `p` and matches central continuity at `∂ₜΦ = 0`, where
`-∂ᵥ(ᶠρ u³ ᶠinterp(Φ)) - Φ (-∂ᵥ(ᶠρ u³))` equals
`-ᶜinterp(ᶠJ ᶠρ u³ ᶠgradᵥ(Φ)) / ᶜJ`, whereas upwinding `Φ` injects
`∂ᵥ(ᶠρ |u³| ΔΦ / 2)` into `ρ e_int` (and where upwind dissipation is active,
reconstructing `e_tot - Φ` damps gradients of `e_int + K` rather than `h_tot`).
Inlined for the fused column loop.
"""
@inline function add_energy_transport!(
    ᶜρe_totₜ,
    ᶠρ,
    ᶠu³,
    ᶜe_tot,
    ᶜΦ,
    ᶜp,
    ::Val{:none},
)
    @. ᶜρe_totₜ -= ᶜadvdivᵥ(
        ᶠρ * ᶠu³ * (ᶠinterp(ᶜe_tot - ᶜΦ) + ᶠinterp(ᶜΦ)) + ᶠu³ * ᶠinterp(ᶜp),
    )
    return nothing
end
@inline function add_energy_transport!(
    ᶜρe_totₜ,
    ᶠρ,
    ᶠu³,
    ᶜe_tot,
    ᶜΦ,
    ᶜp,
    ::Val{:first_order},
)
    @. ᶜρe_totₜ -= ᶜadvdivᵥ(
        ᶠρ * (ᶠupwind1(ᶠu³, ᶜe_tot - ᶜΦ) + ᶠu³ * ᶠinterp(ᶜΦ)) +
        ᶠu³ * ᶠinterp(ᶜp),
    )
    return nothing
end

"""
    energy_transport_scalar(Y, p)

Return the lazy specific internal-plus-kinetic energy `e_tot - Φ` reconstructed
by the post-Newton vertical transport correction. Called from
`add_vertical_advection_corrections!`.
"""
@inline energy_transport_scalar(Y, p) =
    @. lazy(specific(Y.c.ρe_tot, Y.c.ρ) - p.core.ᶜΦ)

"""
    implicit_vertical_advection_tendency!(Yₜ, Y, p, t)

Add the implicit vertical-advection and vertical-momentum tendencies to `Yₜ`.

Adds, in order:

  - The mass flux divergence `-∂ᵥ(ρ u³)` to `Yₜ.c.ρ`, with zero flux through the
    top and bottom boundaries (consistent with the state filter that zeroes `ᶠu³`
    there and with the `ᶜadvdivᵥ_matrix()` used in the manual Jacobian).
  - Vertical transport of energy (and `q_tot` when present) to `Yₜ.c.ρe_tot`
    (and `Yₜ.c.ρq_tot`), reconstructing `e_tot - Φ` (and `q_tot`) with
    `grid_mean_base_upwinding(Y)` (`Val(:first_order)` in moist configurations,
    `Val(:none)` central in `DryModel`) while evaluating the geopotential and
    pressure-work fluxes `ᶠρ u³ ᶠinterp(Φ) + u³ ᶠinterp(p)` centrally
    (`add_energy_transport!`). The Courant-limited high-order correction of the
    material transport is applied post-Newton by
    `correct_implicit_advection_tendency!`.
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
    (; ᶜΦ, ᶠgradᵥ_ᶜΦ, ᶠ∂ξ³∂x³) = p.core
    (; ᶠu³, ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
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
    base_upwinding = grid_mean_base_upwinding(Y)

    # One fused column loop for the center tendencies: the face density
    # `ᶠinterp(ρ J) / J` is a loop temporary shared by every transport term.
    DataLayouts.foreach_column(
        Yₜ.c, Y.c, ᶜJ, ᶠJ, ᶠu³, ᶜΦ, ᶜp, ᶠ∂ξ³∂x³, ᶜws,
    ) do Yₜc, Yc, ᶜJ, ᶠJ, ᶠu³, ᶜΦ, ᶜp, ᶠ∂ξ³∂x³, ᶜws
        ᶠρ = @. ᶠinterp(Yc.ρ * ᶜJ) / ᶠJ

        # Mass advection with zero flux through the top and bottom
        # boundaries (ᶜadvdivᵥ). The state filter in
        # set_implicit_precomputed_quantities! also sets ᶠu³ to 0 at both
        # boundaries, and the ρ row of the manual Jacobian is built from
        # ᶜadvdivᵥ_matrix(), so using ᶜadvdivᵥ here keeps the residual, the
        # boundary conditions, and the Jacobian consistent.
        @. Yₜc.ρ -= ᶜadvdivᵥ(ᶠρ * ᶠu³)

        # Vertical advection of energy and total water with the same mass flux
        # ᶠρ u³ as ρ and the condensates, keeping geopotential and pressure work
        # `ᶠρ u³ ᶠinterp(Φ) + u³ ᶠinterp(p)` central everywhere. `e_tot - Φ` is
        # central in dry configurations (whose central advection block is
        # omitted from the Jacobian so `(ρe_tot, ρe_tot)` stays `-I` in
        # `BlockArrowheadSolve`), and first-order upwind alongside `q_tot` in
        # moist configurations so an affine function of the transported scalars
        # stays affine under transport. The Courant-limited high-order
        # correction is applied post-Newton via T_post_imp!
        # (`correct_implicit_advection_tendency!`).
        ᶜe_tot = @. lazy(specific(Yc.ρe_tot, Yc.ρ))
        add_energy_transport!(
            Yₜc.ρe_tot, ᶠρ, ᶠu³, ᶜe_tot, ᶜΦ, ᶜp, base_upwinding,
        )
        if MatrixFields.has_field(Yc, @name(ρq_tot))
            ᶜq_tot = @. lazy(specific(Yc.ρq_tot, Yc.ρ))
            add_vertical_transport!(
                Yₜc.ρq_tot, ᶠρ, ᶠu³, ᶜq_tot, nothing, base_upwinding,
            )
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

Apply the Courant-limited post-Newton high-order correction to the implicit
vertical advection of `implicit_vertical_advection_tendency!`, evaluated at
the Newton-solved stage state `Y = U*`. Registered as `T_post_imp!` and called
by the tendency diagnostics; it leaves `p` unchanged.

Overwrites `Yₜ` with `add_vertical_advection_corrections!` at the stage step
`p.dtγ`: the face-wise Courant-limited (`min(1, 1 / C_γ)`) difference between
the `energy_q_tot_upwinding` flux and the `grid_mean_base_upwinding(Y)` base
flux (`u³ ᶠinterp(e_tot - Φ)` in `DryModel`, `ᶠupwind1` in moist
configurations) for `e_tot - Φ`, and in moist configurations also for `q_tot`
and any sedimenting condensates, leaving the central geopotential and
pressure-work fluxes `ᶠρ u³ ᶠinterp(Φ) + u³ ᶠinterp(p)` untouched. In
`DryModel` with `Val(:vanleer_limiter)`,
`ᶠlin_vanleer(u³, e_tot - Φ, 0) - u³ ᶠinterp(e_tot - Φ)` vanishes on linear
interior profiles of `e_tot - Φ`. In moist configurations, smooth `e_tot - Φ`
and `q_tot` profiles keep the fraction `1 - min(1, 1 / C_γ)` of the
first-order upwind dissipation where `C_γ > 1`. With
`energy_q_tot_upwinding = none`, the material transport stays central in
`DryModel` and first-order upwind in moist configurations with no correction.
`ρtke` keeps the monotone `ᶠupwind1` transport and receives no correction. All
other fields of `Yₜ` are zero. Returns `nothing`.
"""
NVTX.@annotate function correct_implicit_advection_tendency!(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    add_vertical_advection_corrections!(Yₜ, Y, p, p.dtγ)
    return nothing
end

"""
    add_vertical_advection_corrections!(Yₜ, Y, p, dtγ)

Add to `Yₜ` the Courant-limited high-order corrections of the implicit vertical
advection, with the limiter `min(1, 1 / (|u³| dtγ))`. With
`grid_mean_base_upwinding(Y) == Val(:first_order)` (moist configurations),
`e_tot - Φ`, `q_tot`, and any sedimenting microphysics species receive the
antidiffusive correction `F_hi - ᶠupwind1` with the `energy_q_tot_upwinding`
reconstruction, so specific internal-plus-kinetic energy and all water cross
each face with one scheme while geopotential and pressure work stay central;
`tracer_upwinding` applies to the passive tracers, which are advected
explicitly. With `Val(:none)` (`DryModel`), `e_tot - Φ` receives
`F_hi - u³ ᶠinterp(e_tot - Φ)` with `energy_q_tot_upwinding`, leaving the
central geopotential and pressure-work fluxes untouched.
`correct_implicit_advection_tendency!` passes the implicit stage step `p.dtγ`;
`fully_explicit_tendency!` passes `zero(dtγ)`, for which the limiter is 1 and
the corrected fluxes are the full `upwinding` fluxes. Returns `nothing`.
"""
function add_vertical_advection_corrections!(Yₜ, Y, p, dtγ)
    (; energy_q_tot_upwinding) = p.atmos.numerics
    base_upwinding = grid_mean_base_upwinding(Y)
    energy_active = !(
        energy_q_tot_upwinding == Val(:none) || (
            base_upwinding == Val(:first_order) &&
            energy_q_tot_upwinding == Val(:first_order)
        )
    )
    water_active =
        base_upwinding == Val(:first_order) &&
        !(energy_q_tot_upwinding in (Val(:none), Val(:first_order)))
    energy_active || water_active || return nothing
    (; ᶠu³) = p.precomputed
    ᶜJ = Fields.local_geometry_field(axes(Y.c)).J
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J
    ᶠρ = @. lazy(ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ)

    if energy_active
        ᶜenergy = energy_transport_scalar(Y, p)
        add_vertical_transport_correction!(
            Yₜ.c.ρe_tot,
            ᶠρ,
            ᶠu³,
            ᶜenergy,
            dtγ,
            energy_q_tot_upwinding,
            base_upwinding,
        )
        if MatrixFields.has_field(Y.c, @name(ρq_tot))
            ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
            add_vertical_transport_correction!(
                Yₜ.c.ρq_tot,
                ᶠρ,
                ᶠu³,
                ᶜq_tot,
                dtγ,
                energy_q_tot_upwinding,
                base_upwinding,
            )
        end
    end

    if water_active
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
                energy_q_tot_upwinding,
            )
        end
    end
    return nothing
end

"""
    add_vertical_transport_correction!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dtγ, upwinding, base_upwinding)

Add the Courant-limited post-Newton correction that takes the implicit
transport of `base_upwinding` to the `upwinding` reconstruction:
`add_upwind_vertical_transport_correction!` from central (`Val(:none)`) and
`add_antidiffusive_upwind1_transport!` from first-order upwind
(`Val(:first_order)`).
"""
@inline add_vertical_transport_correction!(
    ᶜρχₜ,
    ᶠρ,
    ᶠu³,
    ᶜχ,
    dtγ,
    upwinding,
    ::Val{:none},
) = add_upwind_vertical_transport_correction!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dtγ, upwinding)
@inline add_vertical_transport_correction!(
    ᶜρχₜ,
    ᶠρ,
    ᶠu³,
    ᶜχ,
    dtγ,
    upwinding,
    ::Val{:first_order},
) = add_antidiffusive_upwind1_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dtγ, upwinding)
