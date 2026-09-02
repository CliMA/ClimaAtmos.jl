#=
DG interface completions for the horizontal tendencies.

On a `Grids.DG` horizontal space the element-local horizontal operators must
be completed by interface terms (on CG spaces DSS couples the elements and
every function here is a no-op):

  - `split_divₕ(ρu, ψ)` (ρ, ρe_tot, tracers): interface flux with the
    SplitDivergence-matched central part `{ρ uv·n̂}{ψ}` plus a λ-Rusanov
    penalty on ⟦ρe⟧ / ⟦ρχ⟧. The density flux stays central — a ⟦ρ⟧ penalty
    is kinetic-energy-sign-indefinite under the vector-invariant pairing.
  - `gradₕ` PGF/K terms (uₕ): central gradient liftings of the same scalars.
  - uₕ stability: ρ-weighted λ-jump penalties on the orthonormal velocity
    components (an exactly sign-definite KE sink).
  - Vorticities (`ᶜω³` and the `wcurlₕ(u₃)` part of `ᶠω¹²`): strong curl
    plus central curl lifting replaces the CG weak curl (gated in
    `explicit_vertical_advection_tendency!`).

Face states are carried in the local orthonormal frame (`UVVector`), which is
single-valued at shared nodes including cubed-sphere panel edges, where
covariant components are not. The face velocity is the horizontal projection
of the full `ᶜu` (terrain-consistent: it carries the u₃ tilt contribution
through g¹³/g²³, and reduces bitwise to `UVVector(uₕ)` on unwarped grids).
The vorticity liftings act on the same fields as their volume curls
(`uₕ` for ω³, `u₃` for the ω¹² part), so they are terrain-consistent as
written.
=#

import ClimaCore: Grids

"""
    is_dg_horizontal(space)

Whether the horizontal part of `space` is a discontinuous (`Grids.DG`)
spectral-element space.
"""
function is_dg_horizontal(space::Spaces.AbstractSpace)
    h_space = Spaces.horizontal_space(space)
    h_space isa Spaces.SpectralElementSpace2D || return false
    return Grids.discretization(Spaces.grid(h_space)) isa Grids.DG
end

# Face functions recovered from the pre-merge DG flux library (candidates for
# upstreaming into ClimaCore.Operators alongside `central_gradient_lift`).

"""
    central_curl12_lift(normal, (w⁻,), (w⁺,))

Central lifting for the horizontal components of ``∇ × (w r̂)``:
``n̂ × r̂ (w^* − w_side)``, returned as a `UVVector`.
"""
central_curl12_lift(normal, (w⁻,), (w⁺,)) =
    ((w⁺ - w⁻) / 2) *
    Geometry.UVVector(normal.components.data.:2, -normal.components.data.:1)

"""
    rho_weighted_jump_penalty_lift(normal, (q⁻, ρ⁻, λ⁻), (q⁺, ρ⁺, λ⁺))

λ-jump penalty with own-side ρ weighting: face kinetic-energy production
``−max(λ)/2 {ρ}⟦u⟧² ≤ 0`` exactly.
"""
rho_weighted_jump_penalty_lift(normal, (q⁻, ρ⁻, λ⁻), (q⁺, ρ⁺, λ⁺)) =
    max(λ⁻, λ⁺) / 2 * ((ρ⁻ + ρ⁺) / 2 / ρ⁻) * (q⁺ - q⁻)

"""
    dg_scalars_interface(normal, argvals⁻, argvals⁺)

Interface flux for (ρ, ρe_tot): central part matched to `split_divₕ`
(`F_ρ = {ρ uv·n̂}`, `F_ρe = F_ρ {h}`) plus a λ-Rusanov penalty on ⟦ρe⟧.
State fields: `ρ`, `ρe`, `h`, `uv`, `λ`.
"""
function dg_scalars_interface(normal, (y⁻,), (y⁺,))
    Fρ = (y⁻.ρ * (y⁻.uv' * normal) + y⁺.ρ * (y⁺.uv' * normal)) / 2
    h̄ = (y⁻.h + y⁺.h) / 2
    λ = max(y⁻.λ, y⁺.λ)
    return (; ρ = Fρ, ρe_tot = Fρ * h̄ - λ / 2 * (y⁺.ρe - y⁻.ρe))
end

"""
    dg_tracer_interface(normal, argvals⁻, argvals⁺)

Interface flux for a tracer ρχ advected by the same mass flux as continuity:
`F_ρχ = {ρ uv·n̂}{χ}` plus a λ-Rusanov penalty on ⟦ρχ⟧. State fields: `ρ`,
`uv`, `λ`, `χ`.
"""
function dg_tracer_interface(normal, (y⁻,), (y⁺,))
    Fρ = (y⁻.ρ * (y⁻.uv' * normal) + y⁺.ρ * (y⁺.uv' * normal)) / 2
    χ̄ = (y⁻.χ + y⁺.χ) / 2
    λ = max(y⁻.λ, y⁺.λ)
    return (; ρχ = Fρ * χ̄ - λ / 2 * (y⁺.ρ * y⁺.χ - y⁻.ρ * y⁻.χ))
end

# Orthonormal horizontal face velocity and Rusanov speed λ = |uv| + c_snd.
# The face velocity is the horizontal physical projection of the FULL ᶜu,
# matching the contravariant contraction of the split_divₕ volume flux: over
# terrain-warped grids u₃ contributes through g¹³/g²³; on unwarped grids those
# metric entries are exact zeros, so this reduces bitwise to UVVector(uₕ).
function dg_face_velocity_and_λ(Y, p)
    (; ᶜu, ᶜp) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    γ = TD.Parameters.cp_d(thermo_params) / TD.Parameters.cv_d(thermo_params)
    ᶜuv = Geometry.project.(Ref(Geometry.UVAxis()), ᶜu)
    ᶜλ = @. sqrt(LinearAlgebra.norm_sqr(ᶜuv)) + sqrt(γ * ᶜp / Y.c.ρ)
    return (ᶜuv, ᶜλ)
end

"""
    dg_horizontal_dynamics_completion!(Yₜ, Y, p, t)

Interface completion of `horizontal_dynamics_tendency!` on DG spaces: scalar
(ρ, ρe_tot) interface fluxes, central liftings of the PGF/K `gradₕ` terms,
and ρ-weighted λ-jump penalties on uₕ. No-op on CG spaces.
"""
NVTX.@annotate function dg_horizontal_dynamics_completion!(Yₜ, Y, p, t)
    is_dg_horizontal(axes(Y.c)) || return nothing
    FT = Spaces.undertype(axes(Y.c))
    (; ᶜΦ) = p.core
    (; ᶜh_tot, ᶜp, ᶜK, ᶜT, ᶜq_liq, ᶜq_ice, ᶜq_tot_nonneg) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    cp_d = thermo_params.cp_d
    ᶜWJ = Fields.local_geometry_field(Y.c).WJ
    (ᶜuv, ᶜλ) = dg_face_velocity_and_λ(Y, p)

    # (ρ, ρe_tot) interface fluxes
    y = map(
        (ρ, ρe, h, uv, λ) -> (; ρ, ρe, h, uv, λ),
        Y.c.ρ,
        Y.c.ρe_tot,
        ᶜh_tot,
        ᶜuv,
        ᶜλ,
    )
    dy = Fields.Field(NamedTuple{(:ρ, :ρe_tot), Tuple{FT, FT}}, axes(Y.c))
    fill!(parent(dy), 0)
    Operators.add_numerical_flux_interior!(dg_scalars_interface, dy, y)
    @. Yₜ.c.ρ += dy.ρ / ᶜWJ
    @. Yₜ.c.ρe_tot += dy.ρe_tot / ᶜWJ

    # central liftings of the strong PGF/K gradients (the same scalars as the
    # gradₕ volume terms in horizontal_dynamics_tendency!)
    ᶜΠ = @. TD.exner_given_pressure(thermo_params, ᶜp)
    ᶜθ_v_diff = @. theta_v(thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) -
       theta_vr(thermo_params, ᶜp)
    ᶜKΦ = @. ᶜK + ᶜΦ - phi_r(thermo_params, ᶜp)
    ᶜθΠ = @. ᶜθ_v_diff * ᶜΠ
    lift(q) = Operators.lifting_correction(
        Operators.central_gradient_lift,
        Geometry.UVVector{FT},
        q,
    )
    L_KΦ = lift(ᶜKΦ)
    L_Π = lift(ᶜΠ)
    L_θΠ = lift(ᶜθΠ)
    L_θ = lift(ᶜθ_v_diff)
    @. Yₜ.c.uₕ -= C12(
        Geometry.transform(
            Geometry.Covariant12Axis(),
            L_KΦ + cp_d * (ᶜθ_v_diff * L_Π + L_θΠ - ᶜΠ * L_θ) / 2,
        ),
    )

    # ρ-weighted λ-jump penalties on the orthonormal velocity components
    pen_u = Operators.lifting_correction(
        rho_weighted_jump_penalty_lift,
        FT,
        ᶜuv.components.data.:1,
        Y.c.ρ,
        ᶜλ,
    )
    pen_v = Operators.lifting_correction(
        rho_weighted_jump_penalty_lift,
        FT,
        ᶜuv.components.data.:2,
        Y.c.ρ,
        ᶜλ,
    )
    @. Yₜ.c.uₕ += C12(
        Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(pen_u, pen_v),
        ),
    )
    return nothing
end

"""
    dg_horizontal_tracer_completion!(Yₜ_lim, Y, p, t)

Interface completion of `horizontal_tracer_advection_tendency!` on DG spaces:
the tracer interface flux for every grid-mean tracer. No-op on CG spaces.
"""
NVTX.@annotate function dg_horizontal_tracer_completion!(Yₜ_lim, Y, p, t)
    is_dg_horizontal(axes(Y.c)) || return nothing
    FT = Spaces.undertype(axes(Y.c))
    ᶜWJ = Fields.local_geometry_field(Y.c).WJ
    (ᶜuv, ᶜλ) = dg_face_velocity_and_λ(Y, p)
    foreach_gs_tracer(Yₜ_lim, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        y = map(
            (ρ, uv, λ, ρχ) -> (; ρ, uv, λ, χ = ρχ / ρ),
            Y.c.ρ,
            ᶜuv,
            ᶜλ,
            ᶜρχ,
        )
        dy = Fields.Field(NamedTuple{(:ρχ,), Tuple{FT}}, axes(Y.c))
        fill!(parent(dy), 0)
        Operators.add_numerical_flux_interior!(dg_tracer_interface, dy, y)
        @. ᶜρχₜ += dy.ρχ / ᶜWJ
    end
    return nothing
end

"""
    dg_ω³!(ᶜω³, Y)

DG replacement for `ᶜω³ = wcurlₕ(uₕ)`: strong horizontal curl plus the
central curl lifting.
"""
function dg_ω³!(ᶜω³, Y)
    FT = Spaces.undertype(axes(Y.c))
    @. ᶜω³ = curlₕ(Y.c.uₕ)
    ᶜuv = @. Geometry.UVVector(Y.c.uₕ)
    ω³_lift = Operators.lifting_correction(
        Operators.central_curl3_lift,
        FT,
        ᶜuv.components.data.:1,
        ᶜuv.components.data.:2,
    )
    @. ᶜω³ += CT3(Geometry.WVector(ω³_lift))
    return nothing
end

"""
    dg_ω¹²_horizontal!(ᶠω¹², Y)

DG replacement for `ᶠω¹² += CT12(wcurlₕ(u₃))`: strong horizontal curl of the
vertical velocity plus the central curl lifting. `project` (not `transform`):
on warped grids the tangential correction has an O(slope) third contravariant
component that belongs to the ω³ correction.
"""
function dg_ω¹²_horizontal!(ᶠω¹², Y)
    FT = Spaces.undertype(axes(Y.f))
    @. ᶠω¹² += CT12(curlₕ(Y.f.u₃))
    ᶠw_sc = @. Geometry.WVector(Y.f.u₃).components.data.:1
    ω¹²_lift = Operators.lifting_correction(
        central_curl12_lift,
        Geometry.UVVector{FT},
        ᶠw_sc,
    )
    ᶠω¹² .+= Geometry.project.(Ref(Geometry.Contravariant12Axis()), ω¹²_lift)
    return nothing
end
