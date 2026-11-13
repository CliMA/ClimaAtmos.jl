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
central vertical transport of `ρe_tot`, `ρq_tot`, and the non-equilibrium
microphysics tracers (so that the central velocity blocks of the Newton step
advance their fluxes to the solved velocity) and the first-order upwind vertical
transport of `ρtke` (sharing its tridiagonal `(ρtke, ρtke)` block with eddy
diffusion and TKE dissipation when diffusion is implicit, and omitting velocity
blocks so `u³` sign reversals preserve M-matrix positivity). The implicit
first-order dissipation and the Courant-limited high-order correction (bounded
with Zalesak flux-corrected transport for `vanleer_limiter`) are applied to
`e_tot - Φ`, `q_tot`, and the sedimenting tracers after the Newton solve by
`correct_implicit_advection_tendency!`, at the solved velocity.

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
@inline add_vertical_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dt, ::Val{:vanleer_limiter}) =
    (@. ᶜρχₜ += -(ᶜadvdivᵥ(ᶠρ * ᶠlin_vanleer(ᶠu³, ᶜχ, zero(dt)))); nothing)
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
    add_energy_transport!(ᶜρe_totₜ, ᶠρ, ᶠu³, ᶜe_tot, ᶜp)

Add the implicit vertical energy transport
`-∂ᵥ(ᶠρ u³ ᶠinterp(e_tot) + u³ ᶠinterp(p))` to `ᶜρe_totₜ`, with the face density
`ᶠρ = ᶠinterp(ᶜρ * ᶜJ) / ᶠJ`, splitting `h_tot = e_tot + p / ρ` into central
total-energy transport and central pressure work. Because continuity
`-∂ᵥ(ᶠρ u³)` uses central `ᶠρ` and is not modified by `T_post_imp!`, evaluating
`u³ ᶠinterp(p)` centrally avoids pairing central `ᶠρ` with an upwind
`p / ρ = R_m T` across moisture or temperature jumps at continuous `p` (a
`ρ Δ(R_m T) / 2` pressure-work dipole) and gives the post-Newton diffused energy
density unit derivative with respect to prognostic `ρe_tot` at fixed `ρ`, so it
shares `I - dtγ 𝒟` with `ρq_tot` and the condensates. In
`h_tot = e_int + p / ρ + Φ + K`, `(p, Φ)` form a hydrostatic pair
(`de_int + p d(1 / ρ) = 0` along an adiabat and `dp / ρ + dΦ = 0` in hydrostatic
balance, so `-cp Γ z` cancels `+g z` in `h_tot`); once `p / ρ` is split off
centrally, `e_tot = e_int + Φ + K` lacks `+R_m T` to cancel `+g z` along an
adiabat. Leaving `Φ` out of `correct_implicit_advection_tendency!` (which
reconstructs `e_tot - Φ` alone) keeps `(p, Φ)` together on the central stencil
(matching central continuity at `∂ₜΦ = 0`, where
`-∂ᵥ(ᶠρ u³ ᶠinterp(Φ)) - Φ (-∂ᵥ(ᶠρ u³))` equals
`-ᶜinterp(ᶠJ ᶠρ u³ ᶠgradᵥ(Φ)) / ᶜJ`) and leaves
`e_tot - Φ = e_int(T, q_tot, q_liq, q_ice) + K` affine in
`(1, q_tot, q_liq, q_ice)` at uniform `(T, u)`, while active upwind dissipation
damps gradients of `e_int + K` rather than `h_tot`. Inlined for the fused column
loop.
"""
@inline function add_energy_transport!(ᶜρe_totₜ, ᶠρ, ᶠu³, ᶜe_tot, ᶜp)
    @. ᶜρe_totₜ -= ᶜadvdivᵥ(ᶠρ * ᶠu³ * ᶠinterp(ᶜe_tot) + ᶠu³ * ᶠinterp(ᶜp))
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
  - Central vertical transport of energy (and `q_tot` when present) to
    `Yₜ.c.ρe_tot` (and `Yₜ.c.ρq_tot`), with the central energy and pressure-work
    flux `ᶠρ u³ ᶠinterp(e_tot) + u³ ᶠinterp(p)` (`add_energy_transport!`).
  - Central vertical transport of the non-equilibrium microphysics tracers with
    the grid-mean velocity `ᶠu³`, plus vertical transport of the microphysics
    tracers with their terminal velocities using downward (`ᶠtop_bias`) biasing
    and a free-outflow bottom boundary (`ᶜprecipdivᵥ`), and the water
    sedimentation contributions to `ρ`, `ρe_tot`, and `ρq_tot` from
    `vertical_advection_of_water_tendency!`.
  - First-order upwind vertical transport of `ρtke`.
  - The Exner-form pressure gradient and buoyancy tendency
    `-(∂ᵥΦ - ∂ᵥΦ_r + cp_d (θ_v - θ_vr) ∂ᵥΠ)` and the Rayleigh sponge tendency to
    `Yₜ.f.u₃`.

The grid-mean transport of `e_tot - Φ`, `q_tot`, and the microphysics tracers is
central because it is evaluated at the velocity of the Newton residual, which
the central velocity Jacobian blocks advance to the solved velocity; the
implicit first-order dissipation and the Courant-limited high-order correction
for `e_tot - Φ`, `q_tot`, and the microphysics tracers are applied at the solved
velocity by `correct_implicit_advection_tendency!`. `ρtke` uses first-order
upwind advection directly in the residual so its tridiagonal `(ρtke, ρtke)`
block solves advection together with eddy diffusion and stiff nonlinear TKE
dissipation in one M-matrix when diffusion is implicit (avoiding inhomogeneous
diagonal scaling of central fluxes across inversion interfaces, and omitting
velocity blocks so `u³` sign reversals cannot turn a frozen-donor upwind flux
downwind). Vertical advection of passive non-water tracers by the mean flow is
treated explicitly. Returns `nothing`.
"""
function implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; rayleigh_sponge) = p.atmos
    (; params) = p
    ᶜJ = Fields.local_geometry_field(axes(Y.c)).J
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J
    (; ᶠgradᵥ_ᶜΦ, ᶠ∂ξ³∂x³) = p.core
    (; ᶠu³, ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)
    cp_d = CAP.cp_d(params)

    # Non-equilibrium microphysics tracers are advected implicitly both by the
    # grid-mean vertical velocity (central, zero boundary flux) and by their
    # sedimentation terminal velocity (downward-biased, free outflow bottom
    # boundary); the EDMFX difference-form SGS mass flux
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
        Yₜ.c, Y.c, ᶜJ, ᶠJ, ᶠu³, ᶜp, ᶠ∂ξ³∂x³, ᶜws,
    ) do Yₜc, Yc, ᶜJ, ᶠJ, ᶠu³, ᶜp, ᶠ∂ξ³∂x³, ᶜws
        ᶠρ = @. ᶠinterp(Yc.ρ * ᶜJ) / ᶠJ

        # Mass advection with zero flux through the top and bottom
        # boundaries (ᶜadvdivᵥ). The state filter in
        # set_implicit_precomputed_quantities! also sets ᶠu³ to 0 at both
        # boundaries, and the ρ row of the manual Jacobian is built from
        # ᶜadvdivᵥ_matrix(), so using ᶜadvdivᵥ here keeps the residual, the
        # boundary conditions, and the Jacobian consistent.
        @. Yₜc.ρ -= ᶜadvdivᵥ(ᶠρ * ᶠu³)

        # Central vertical advection of energy and total water with the same
        # mass flux ᶠρ u³ as ρ and the condensates. The implicit upwind
        # dissipation and the high-order correction are applied post-Newton via
        # T_post_imp! (`correct_implicit_advection_tendency!`).
        ᶜe_tot = @. lazy(specific(Yc.ρe_tot, Yc.ρ))
        add_energy_transport!(Yₜc.ρe_tot, ᶠρ, ᶠu³, ᶜe_tot, ᶜp)
        if MatrixFields.has_field(Yc, @name(ρq_tot))
            ᶜq_tot = @. lazy(specific(Yc.ρq_tot, Yc.ρ))
            add_vertical_transport!(
                Yₜc.ρq_tot, ᶠρ, ᶠu³, ᶜq_tot, nothing, Val(:none),
            )
        end

        MatrixFields.unrolled_foreach(
            map(tuple, sedimenting_names, ᶜws),
        ) do (ρχ_name, ᶜw)
            @inline
            ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
            ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
            ᶜχ = @. lazy(specific(ᶜρχ, Yc.ρ))
            add_vertical_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, nothing, Val(:none))
            # The residual's donor-cell sedimentation flux out of a negative
            # cell is zero (`specific_nonneg`); the single-Newton step adds the
            # Jacobian flux `w_jac (ρq* - ρq^g)` so falling precipitation
            # traverses empty or negative cells in one stage.
            @. ᶜρχₜ -= ᶜprecipdivᵥ(
                CT3(ᶠ∂ξ³∂x³ * (ᶠρ * ᶠtop_bias(-(ᶜw) * specific_nonneg(ᶜρχ, Yc.ρ)))),
            )
        end

        if MatrixFields.has_field(Yc, @name(ρtke))
            ᶜtke = @. lazy(specific(Yc.ρtke, Yc.ρ))
            add_vertical_transport!(
                Yₜc.ρtke, ᶠρ, ᶠu³, ᶜtke, nothing, Val(:first_order),
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
    _courant_limiter(u³::Geometry.Contravariant3Vector, dtγ)

Return `min(1, 1 / (|u³| dtγ))`, the weight that keeps the explicit post-Newton
antidiffusion at diffusion number `≤ 1/2` where the stage Courant number
`C_γ = |u³| dtγ` exceeds one.
"""
@inline function _courant_limiter(u³::Geometry.Contravariant3Vector, dtγ)
    FT = typeof(u³.u³)
    c = abs(u³.u³) * FT(dtγ)
    return ifelse(c > one(FT), inv(c), one(FT))
end

"""
    _dissipation_weight(u³::Geometry.Contravariant3Vector, dtγ, compensate::Bool = true)

Return `1 + 2 |u³| dtγ` when `compensate` is `true` and `1` otherwise, the
weight of the implicit first-order dissipation applied after the Newton solve.
With `ManualSparseJacobian` — or `AutoSparseJacobian` in dry explicit-diffusion
runs — without `use_krylov_method` (`compensate == true`, stored in
`p.scratch.compensate_explicit_scalar_advection`), the linear solve omits the
scalar advection derivative (its scalar Jacobian block is `-I`), so each Newton
iteration advances the central flux of a scalar explicitly and introduces the
negative Lax–Wendroff diffusivity `-(u³)² dtγ / 2 = -C_γ K₁` of forward-Euler
central advection, with `K₁ = |u³| Δz / 2` the first-order upwind diffusivity.
Weighting the implicit dissipation by `1 + 2 C_γ` keeps the first-order stage
update positive for nondivergent interior flow (uniform or decelerating Courant
numbers) at every `C_γ > 0`, gives it the leading diffusivity `K₁` of
backward-Euler upwind, matches backward-Euler central advection to `O(dtγ²)` on
smooth profiles when paired with the high-order antidiffusion, and keeps the
linear von Neumann amplification contractive at every `C_γ` (including when
line search or `max_newton_iters_ode ≥ 2` is configured). When the linear solve
includes the scalar advection derivative (`AutoSparseJacobian` in moist or
implicit-diffusion runs, `AutoDenseJacobian`, or `use_krylov_method = true`,
where `compensate == false`), `+2 C_γ` is omitted so backward-Euler central
advection receives unit-weight backward-Euler upwind dissipation without
over-damping (at the cost of `O(10⁻²)` undershoots on sharp spikes).
"""
@inline function _dissipation_weight(
    u³::Geometry.Contravariant3Vector,
    dtγ,
    compensate::Bool = true,
)
    FT = typeof(u³.u³)
    return ifelse(
        compensate,
        one(FT) + 2 * abs(u³.u³) * FT(dtγ),
        one(FT),
    )
end

@inline _c3_scalar(v::Geometry.Covariant3Vector) = v.u₃
@inline _ct3_scalar(v::Geometry.Contravariant3Vector) = v.u³

@inline function _fct_ratio(P, Q)
    FT = typeof(P)
    return ifelse(P <= zero(FT), zero(FT), ifelse(Q >= P, one(FT), Q / P))
end

"""
    add_antidiffusive_upwind1_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, ᶜρ, p, dtγ, upwinding)

Add the Courant-limited post-Newton high-order antidiffusive correction
`-ᶜadvdivᵥ(ᶠF_anti)` to `ᶜρχₜ`, relative to the first-order upwind flux. For
`Val(:vanleer_limiter)` with `dtγ > 0`, Zalesak flux-corrected transport bounds
the face transfer `dtγ ᶠJ ᶠF_raw³` against the local extrema
`[min(χ*_{i-1}, χ*_i, χ*_{i+1}), max(χ*_{i-1}, χ*_i, χ*_{i+1})]` of `ᶜχ = χ*`,
preserving the bounds and monotonicity of `χ*` per implicit stage at every
`C_γ > 0` (keeping the stage positive and monotone wherever the low-order
stage `χ*` is, such as in nondivergent interior flow) while
remaining scale- and shift-equivariant (`χ → a + b χ` for all `a, b ∈ ℝ`); at
`dtγ = 0`, the unconstrained flux difference is added. `Val(:third_order)`
applies the linear unlimited third-order upwind antidiffusion (which is not
positive or monotone on discontinuities). Uses `p.scratch.ᶠtemp_CT3`,
`p.scratch.ᶠtemp_scalar`, `p.scratch.ᶠtemp_scalar_2`, `p.scratch.ᶜtemp_scalar_3`,
and `p.scratch.ᶜtemp_scalar_4`. A no-op for `Val(:first_order)` and `Val(:none)`.
"""
@inline function add_antidiffusive_upwind1_transport!(
    ᶜρχₜ,
    ᶠρ,
    ᶠu³,
    ᶜχ,
    ᶜρ,
    p,
    dtγ,
    ::Val{:vanleer_limiter},
)
    FT = Spaces.undertype(axes(ᶜρχₜ))
    z = zero(FT)
    dtγ_ft = FT(dtγ)
    if dtγ_ft <= z
        @. ᶜρχₜ -= ᶜadvdivᵥ(
            ᶠρ * (ᶠlin_vanleer(ᶠu³, ᶜχ, z) - ᶠupwind1(ᶠu³, ᶜχ)),
        )
        return nothing
    end
    ᶜJ = Fields.local_geometry_field(axes(ᶜρχₜ)).J
    ᶠJ = Fields.local_geometry_field(axes(ᶠu³)).J
    ᶠF_raw = p.scratch.ᶠtemp_CT3
    ᶠΦ = p.scratch.ᶠtemp_scalar
    ᶠΔχ = p.scratch.ᶠtemp_scalar_2
    ᶜR_plus = p.scratch.ᶜtemp_scalar_3
    ᶜR_minus = p.scratch.ᶜtemp_scalar_4
    ᶠset_zero_ct3_bcs = Operators.SetBoundaryOperator(;
        top = Operators.SetValue(CT3(z)),
        bottom = Operators.SetValue(CT3(z)),
    )
    @. ᶠF_raw = ᶠset_zero_ct3_bcs(
        ᶠρ *
        _courant_limiter(ᶠu³, dtγ_ft) *
        (ᶠlin_vanleer(ᶠu³, ᶜχ, z) - ᶠupwind1(ᶠu³, ᶜχ)),
    )
    @. ᶠΦ = dtγ_ft * ᶠJ * _ct3_scalar(ᶠF_raw)
    @. ᶠΔχ = _c3_scalar(ᶠgradᵥ(ᶜχ))
    @. ᶜR_plus = _fct_ratio(
        max(z, ᶜbottom_bias(ᶠΦ)) - min(z, ᶜtop_bias(ᶠΦ)),
        ᶜρ * ᶜJ * max(z, -(ᶜbottom_bias(ᶠΔχ)), ᶜtop_bias(ᶠΔχ)),
    )
    @. ᶜR_minus = _fct_ratio(
        max(z, ᶜtop_bias(ᶠΦ)) - min(z, ᶜbottom_bias(ᶠΦ)),
        ᶜρ * ᶜJ * max(z, ᶜbottom_bias(ᶠΔχ), -(ᶜtop_bias(ᶠΔχ))),
    )
    ᶠb_bias = Operators.BottomBiasedC2F(; bottom = Operators.SetValue(z))
    ᶠt_bias = Operators.TopBiasedC2F(; top = Operators.SetValue(z))
    @. ᶠF_raw =
        ifelse(
            _ct3_scalar(ᶠF_raw) >= z,
            min(ᶠb_bias(ᶜR_minus), ᶠt_bias(ᶜR_plus)),
            min(ᶠb_bias(ᶜR_plus), ᶠt_bias(ᶜR_minus)),
        ) * ᶠF_raw
    @. ᶜρχₜ -= ᶜadvdivᵥ(ᶠF_raw)
    return nothing
end
@inline add_antidiffusive_upwind1_transport!(
    ᶜρχₜ,
    ᶠρ,
    ᶠu³,
    ᶜχ,
    ᶜρ,
    p,
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
    ᶜρ,
    p,
    dtγ,
    ::Union{Val{:first_order}, Val{:none}},
) = nothing

"""
    post_newton_stage_update!(Yₜ, Y, p, t)

Accumulate stage-weighted surface precipitation fluxes, write the post-Newton
vertical advection correction into `Yₜ` at `Y = U*`, and apply the linearized
surface precipitation increment to a prognostic slab ocean when active.

Registered as the `T_post_imp!` hook of `ClimaODEFunction`, which
`ClimaTimeSteppers.solve_stage_implicit!` invokes once per implicit stage and
applies as `U ← U* + dtγ · Yₜ`. Calls
`accumulate_stage_precipitation_surface_fluxes!` (advancing
`p.scratch.precip_stage_idx[]`), overwrites `Yₜ` with
`correct_implicit_advection_tendency!`, and adds the post-Newton surface
precipitation increment via `add_post_newton_surface_precipitation_tendency!` so
a `SlabOceanTemperature` surface receives the same linearized stage flux as the
atmosphere loses when the stage-weighted `ManualSparseJacobian` accumulator is
active (`has_precip_stage_accumulator(p)`). Keeping the accumulation outside
`correct_implicit_advection_tendency!` leaves the stage counter in phase when
tendency diagnostics sample the correction. Returns `nothing`.
"""
function post_newton_stage_update!(Yₜ, Y, p, t)
    accumulate_stage_precipitation_surface_fluxes!(
        Y,
        p,
        p.atmos.microphysics_model,
    )
    correct_implicit_advection_tendency!(Yₜ, Y, p, t)
    add_post_newton_surface_precipitation_tendency!(
        Yₜ,
        Y,
        p,
        p.atmos.surface.temperature,
        p.atmos.microphysics_model,
    )
    return nothing
end

"""
    correct_implicit_advection_tendency!(Yₜ, Y, p, t)

Apply the post-Newton upwind treatment of the implicit vertical advection of
`implicit_vertical_advection_tendency!`, evaluated at the Newton-solved stage
state `Y = U*` and velocity. Called by `post_newton_stage_update!` and by the
tendency diagnostics; it leaves `p` unchanged.

Overwrites `Yₜ` with `add_vertical_advection_corrections!` at the stage step
`p.dtγ`: for `e_tot - Φ`, `q_tot`, and any sedimenting condensates, the
implicit first-order dissipation at the solved velocity (weighted by
`1 + 2 C_γ`, see `_dissipation_weight`) followed by the Courant-limited
(`min(1, 1 / C_γ)`) antidiffusion of the `energy_humidity_upwinding`
reconstruction relative to first-order upwind, leaving the central geopotential
and pressure-work fluxes `ᶠρ u³ ᶠinterp(Φ) + u³ ᶠinterp(p)` untouched (`ρtke` is
already advected with first-order upwinding in the Newton residual). `e_tot - Φ`,
`q_tot`, and the condensates pass through the same linear implicit operator and
the same antidiffusion reconstruction: linear reconstructions (`first_order`,
`third_order`) preserve arbitrary affine functions of
`(1, q_tot, q_liq, q_ice)` at any stage Courant number (`third_order` is linear
and unlimited, so it is not positive on discontinuities), and `vanleer_limiter`
(bounded against the local extrema of `χ*` with Zalesak flux-corrected
transport, both scale- and shift-equivariant) preserves the bounds and
monotonicity of `χ*` per implicit stage at every `C_γ > 0` (staying positive and
monotone wherever the low-order stage `χ*` is, such as in nondivergent interior
flow with uniform or decelerating Courant numbers, whereas diverging flow, face
Courant jumps, or spikes next to a closed boundary can introduce `O(10⁻²)`
extrema in `χ*`) while
preserving uniform `(T, u)` across single-profile moisture or condensate
variations and on smooth profiles free of limiter clipping; away from
Zalesak-clamped faces (such as discrete ramp shoulders), the composite matches
the van Leer flux at the solved velocity to leading order in `dtγ` for
`C_γ ≤ 1`. With `energy_humidity_upwinding = first_order`, the antidiffusion is
zero and the transport is positive first-order with the leading diffusivity of
backward-Euler upwind; with `none`, `e_tot - Φ`, `q_tot`, and the condensates
skip both steps (leaving their vertical advection forward-Euler central under
one Newton iteration with `ManualSparseJacobian`, which amplifies oscillatory
modes without vertical diffusion). All other fields of `Yₜ` are zero. Returns
`nothing`.
"""
NVTX.@annotate function correct_implicit_advection_tendency!(Yₜ, Y, p, t)
    Yₜ .= zero(eltype(Yₜ))
    add_vertical_advection_corrections!(Yₜ, Y, p, p.dtγ)
    return nothing
end

"""
    add_vertical_advection_corrections!(Yₜ, Y, p, dtγ)

Add to `Yₜ` the post-Newton upwind treatment of `e_tot - Φ`, `q_tot`, and the
sedimenting microphysics species at the solved velocity
`u³ = p.precomputed.ᶠu³`: for each scalar `χ` with density `ρχ`,

 1. the implicit first-order dissipation step
    `(I - dtγ 𝒟) ρχ* = ρχ`, with the tridiagonal operator
    `𝒟 ρχ = -ᶜadvdivᵥ(ᶠρ (1 + 2 |u³| dtγ) (ᶠupwind1(u³, χ) - u³ ᶠinterp(χ)))`
    (`add_implicit_upwind1_dissipation!`), whose tendency `𝒟 ρχ*` is added;
 2. the Courant-limited antidiffusion `F_hi - ᶠupwind1` of the
    `energy_humidity_upwinding` reconstruction at `χ* = ρχ* / ρ`
    (`add_antidiffusive_upwind1_transport!`, Zalesak-limited for
    `vanleer_limiter`).

`𝒟` vanishes on a uniform `χ`, so the correction conserves each column
integral and is consistent with the central mass flux of the residual. The
dissipation weight `1 + 2 |u³| dtγ` compensates the forward-Euler central
advection of one direct Newton iteration with `ManualSparseJacobian`
(`_dissipation_weight`). With `dtγ = 0`, as passed by `fully_explicit_tendency!`,
step 1 is the explicit first-order dissipation and the limiter is 1, so the
corrected flux is the full `upwinding` flux. Returns immediately when
`energy_humidity_upwinding = none`, leaving the transport of `e_tot - Φ`,
`q_tot`, and the sedimenting species central (`ρtke` uses `ᶠupwind1` in the
Newton residual regardless). `tracer_upwinding` applies to the passive tracers,
which are advected explicitly. Writes `p.scratch.ᶜupwind_dissipation_matrix`,
`p.scratch.ᶜupwind_dissipation_system`,
`p.scratch.ᶜupwind_dissipation_solver_cache`, `p.scratch.ᶜtemp_scalar`,
`p.scratch.ᶜtemp_scalar_2`, `p.scratch.ᶜtemp_scalar_3`,
`p.scratch.ᶜtemp_scalar_4`, and `p.scratch.ᶠtemp_CT3`. Returns `nothing`.
"""
function add_vertical_advection_corrections!(Yₜ, Y, p, dtγ)
    (; energy_humidity_upwinding) = p.atmos.numerics
    energy_humidity_upwinding == Val(:none) && return nothing
    (; ᶠu³) = p.precomputed
    ᶜρ = Y.c.ρ
    ᶜJ = Fields.local_geometry_field(axes(Y.c)).J
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J
    ᶠρ = @. lazy(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ)
    ᶜρχ_dissipated = p.scratch.ᶜtemp_scalar
    ᶜρe_mat = p.scratch.ᶜtemp_scalar_2
    FT = Spaces.undertype(axes(Y.c))

    # The dissipation operator depends on the velocity and density only, so it
    # is assembled once and shared by every scalar.
    # TODO: Fuse the tridiagonal dissipation solve and FCT antidiffusion across
    # all scalars in one `foreach_column` kernel to share the Thomas
    # elimination and reduce per-stage kernel launches.
    # TODO: On the Krylov/autodiff branch (`compensate == false`), solving
    # `(I - dtγ 𝒰) ε = dtγ 𝒟 ρχ*` for the backward-Euler upwind increment `ε`
    # would restore strict positivity on sharp spikes without `1 + 2 C_γ`
    # over-damping.
    compensate =
        hasproperty(p.scratch, :compensate_explicit_scalar_advection) ?
        p.scratch.compensate_explicit_scalar_advection : true
    ᶜ𝒟 = p.scratch.ᶜupwind_dissipation_matrix
    ᶠset_upwind1_matrix_bcs = Operators.SetBoundaryOperator(;
        top = Operators.SetValue(zero(BidiagonalMatrixRow{CT3{FT}})),
        bottom = Operators.SetValue(zero(BidiagonalMatrixRow{CT3{FT}})),
    )
    @. ᶜ𝒟 =
        -(ᶜadvdivᵥ_matrix()) *
        (
            DiagonalMatrixRow(
                ᶠρ * _dissipation_weight(ᶠu³, dtγ, compensate),
            ) * (
                ᶠset_upwind1_matrix_bcs(ᶠupwind1_matrix(ᶠu³)) -
                DiagonalMatrixRow(ᶠu³) * ᶠinterp_matrix()
            )
        ) * DiagonalMatrixRow(1 / ᶜρ)
    ᶜA = p.scratch.ᶜupwind_dissipation_system
    @. ᶜA = (I,) - dtγ * ᶜ𝒟
    solver_cache = p.scratch.ᶜupwind_dissipation_solver_cache

    @inline add_scalar_correction!(ᶜρχₜ, ᶜρχ) = begin
        add_implicit_upwind1_dissipation!(
            ᶜρχₜ,
            ᶜρχ_dissipated,
            ᶜ𝒟,
            ᶜA,
            ᶜρχ,
            solver_cache,
        )
        ᶜχ = @. lazy(specific(ᶜρχ_dissipated, ᶜρ))
        add_antidiffusive_upwind1_transport!(
            ᶜρχₜ,
            ᶠρ,
            ᶠu³,
            ᶜχ,
            ᶜρ,
            p,
            dtγ,
            energy_humidity_upwinding,
        )
        return nothing
    end

    # `ρ (e_tot - Φ)` is the transported energy density; its dissipation step
    # applies to `ρe_tot - ρ Φ` and leaves `ρ Φ` unchanged. Materialized so that
    # every scalar enters the solve as a `Field`.
    @. ᶜρe_mat = Y.c.ρe_tot - ᶜρ * p.core.ᶜΦ
    add_scalar_correction!(Yₜ.c.ρe_tot, ᶜρe_mat)
    if MatrixFields.has_field(Y.c, @name(ρq_tot))
        add_scalar_correction!(Yₜ.c.ρq_tot, Y.c.ρq_tot)
    end
    MatrixFields.unrolled_foreach(sedimenting_tracer_names(Y)) do ρχ_name
        add_scalar_correction!(
            MatrixFields.get_field(Yₜ.c, ρχ_name),
            MatrixFields.get_field(Y.c, ρχ_name),
        )
    end
    return nothing
end

"""
    add_implicit_upwind1_dissipation!(ᶜρχₜ, ᶜρχ_dissipated, ᶜ𝒟, ᶜA, ᶜρχ, solver_cache)

Solve the implicit first-order dissipation step `ᶜA ρχ* = ρχ` for `ρχ*` into
`ᶜρχ_dissipated`, with `ᶜA = I - dtγ 𝒟` and the tridiagonal dissipation
operator `ᶜ𝒟` of `add_vertical_advection_corrections!`, and add its tendency
`𝒟 ρχ*` to `ᶜρχₜ`. `solver_cache` is the `MatrixFields.single_field_solve!`
cache of `ᶜA` (on GPUs, `single_field_solve!` dispatches to a shared-memory
parallel cyclic reduction kernel that reduces `ᶜA` and `ᶜρχ` jointly across the
column's `N_z` threads in `⌈log₂(N_z)⌉` steps without a separate factorization
pass). Returns `nothing`.
"""
function add_implicit_upwind1_dissipation!(
    ᶜρχₜ,
    ᶜρχ_dissipated,
    ᶜ𝒟,
    ᶜA,
    ᶜρχ,
    solver_cache,
)
    MatrixFields.single_field_solve!(solver_cache, ᶜρχ_dissipated, ᶜA, ᶜρχ)
    @. ᶜρχₜ += ᶜ𝒟 * ᶜρχ_dissipated
    return nothing
end
