#####
##### Implicit tendencies
#####

import ClimaCore
import ClimaCore: DataLayouts, Fields, Geometry

"""
    implicit_tendency!(Yₜ, Y, p, t)

Compute the wave-core implicit tendency of the state `Y` at time `t`,
overwriting `Yₜ`.

Contains resolved vertical continuity, central vertical transport of `ρe_tot`
(with pressure work), `ρq_tot`, and condensate mass tracers, the Exner-form
pressure gradient and buoyancy tendency of `u₃`, and Rayleigh sponge damping.
Monotone vertical scalar transport and column physics execute post-ARK.

Reads `ᶠu³`, `ᶜp`, `ᶜT`, `ᶜq_tot_nonneg`, `ᶜq_liq`, `ᶜq_ice` from
`p.precomputed`. Returns `nothing`.
"""
NVTX.@annotate function implicit_tendency!(Yₜ, Y, p, t)
    fill_with_nans!(p)
    Yₜ .= zero(eltype(Yₜ))
    maybe_record_explicit_imp_stage_energy!(Yₜ, Y, p)
    implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    zero_velocity_tendency!(Yₜ, Y, p, t)
    return nothing
end

"""
    add_vertical_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, dt, upwinding)

Add the flux-form vertical transport tendency `-∂ᵥ(ᶠρ u³ χ)` to `ᶜρχₜ`, with
zero-flux boundary conditions imposed by `ᶜadvdivᵥ`. The face density
`ᶠρ = ᶠinterp(ᶜρ * ᶜJ) / ᶠJ` is an argument, so that a fused column loop can
share it between several tracers. Inlined, since it runs inside loop bodies (a
lazy broadcast returned from a function is not inferred there).

# Arguments

  - `ᶜρχₜ`: Center-space tendency of the transported density-weighted quantity.
  - `ᶠρ`: Face-space density [kg/m³].
  - `ᶠu³`: Face-space contravariant vertical velocity [1/s].
  - `ᶜχ`: Center-space specific (per-mass) transported quantity.
  - `dt`: Timestep [s]; `ᶠlin_vanleer` receives `zero(dt)`, which evaluates the
    Courant-independent MUSCL reconstruction with the `MonotoneLocalExtrema`
    slope limiter (Lin et al. 1994, eq. 5).
  - `upwinding`: Reconstruction of `χ` on faces; one of `Val(:none)` (central
    interpolation), `Val(:first_order)`, `Val(:third_order)`, or
    `Val(:vanleer_limiter)`.
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
    add_energy_transport!(ᶜρe_totₜ, ᶠρ, ᶠu³, ᶜe_tot, ᶜp)

Add the implicit vertical energy transport
`-∂ᵥ(ᶠρ u³ ᶠinterp(e_tot) + u³ ᶠinterp(p))` to `ᶜρe_totₜ`, with the face density
`ᶠρ = ᶠinterp(ᶜρ * ᶜJ) / ᶠJ`, splitting `h_tot = e_tot + p / ρ` into central
total-energy transport and central pressure work. Because continuity
`-∂ᵥ(ᶠρ u³)` uses central `ᶠρ`, evaluating `u³ ᶠinterp(p)` centrally avoids
pairing central `ᶠρ` with an upwind `p / ρ = R_m T` across moisture or
temperature jumps at continuous `p` (a `ρ Δ(R_m T) / 2` pressure-work dipole). In
`h_tot = e_int + p / ρ + Φ + K`, `(p, Φ)` form a hydrostatic pair
(`de_int + p d(1 / ρ) = 0` along an adiabat and `dp / ρ + dΦ = 0` in hydrostatic
balance, so `-cp Γ z` cancels `+g z` in `h_tot`); once `p / ρ` is split off
centrally, `e_tot = e_int + Φ + K` lacks `+R_m T` to cancel `+g z` along an
adiabat. Inlined for the fused column loop.
"""
@inline function add_energy_transport!(ᶜρe_totₜ, ᶠρ, ᶠu³, ᶜe_tot, ᶜp)
    @. ᶜρe_totₜ -= ᶜadvdivᵥ(ᶠρ * ᶠu³ * ᶠinterp(ᶜe_tot) + ᶠu³ * ᶠinterp(ᶜp))
    return nothing
end

"""
    _implicit_wave_core_center_tendency!(Yₜ, Y, p)

Add resolved continuity, central energy plus pressure work, central `ρq_tot`,
and central condensate-mass vertical transport to `Yₜ.c`. Returns `nothing`.
"""
function _implicit_wave_core_center_tendency!(Yₜ, Y, p)
    ᶜJ = Fields.local_geometry_field(axes(Y.c)).J
    ᶠJ = Fields.local_geometry_field(axes(Y.f)).J
    (; ᶠu³, ᶜp) = p.precomputed
    mass_names = sedimenting_mass_names(Y)

    # Pure wave-core center loop: resolved continuity, central energy plus
    # pressure work, central q_tot, and central condensate-mass transport for
    # active condensate loading.
    DataLayouts.foreach_column(
        Yₜ.c, Y.c, ᶜJ, ᶠJ, ᶠu³, ᶜp,
    ) do Yₜc, Yc, ᶜJ, ᶠJ, ᶠu³, ᶜp
        ᶠρ = @. ᶠinterp(Yc.ρ * ᶜJ) / ᶠJ
        @. Yₜc.ρ -= ᶜadvdivᵥ(ᶠρ * ᶠu³)

        ᶜe_tot = @. lazy(specific(Yc.ρe_tot, Yc.ρ))
        add_energy_transport!(Yₜc.ρe_tot, ᶠρ, ᶠu³, ᶜe_tot, ᶜp)
        if MatrixFields.has_field(Yc, @name(ρq_tot))
            ᶜq_tot = @. lazy(specific(Yc.ρq_tot, Yc.ρ))
            add_vertical_transport!(
                Yₜc.ρq_tot, ᶠρ, ᶠu³, ᶜq_tot, nothing, Val(:none),
            )
        end
        MatrixFields.unrolled_foreach(mass_names) do ρχ_name
            @inline
            ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
            ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
            ᶜχ = @. lazy(specific(ᶜρχ, Yc.ρ))
            add_vertical_transport!(ᶜρχₜ, ᶠρ, ᶠu³, ᶜχ, nothing, Val(:none))
        end
        return nothing
    end
    return nothing
end

"""
    implicit_vertical_advection_tendency!(Yₜ, Y, p, t)

Add the wave-core implicit vertical-advection and vertical-momentum tendencies
to `Yₜ`.

Adds resolved continuity `-∂ᵥ(ᶠρ u³)` to `Yₜ.c.ρ`, central vertical transport
of energy plus pressure work `-(∂ᵥ(ᶠρ u³ ᶠinterp(e_tot) + u³ ᶠinterp(p)))` to
`Yₜ.c.ρe_tot`, central vertical transport of `ρq_tot` and condensate masses,
and the Exner-form pressure gradient, buoyancy, and Rayleigh sponge tendencies
to `Yₜ.f.u₃`. Returns `nothing`.
"""
function implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; rayleigh_sponge) = p.atmos
    (; params) = p
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)
    cp_d = CAP.cp_d(params)

    _implicit_wave_core_center_tendency!(Yₜ, Y, p)

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
