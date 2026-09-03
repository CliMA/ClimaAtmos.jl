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
    dg_fddg_tke_advection!(Yₜ, Y, p)

Horizontal advection of prognostic TKE on DG spaces for the `:fddg` path.
`horizontal_dynamics_tendency!` returns early for fddg (its `Yₜ.c.ρtke`
split-divₕ term is skipped) and `dg_fddg_horizontal_dynamics!` only carries
(ρ, ρe_tot, uₕ). `ρtke` is excluded from `is_tracer_var`, so the grid-mean
tracer completion never sees it either — leaving TKE with NO horizontal
transport in fddg mode, which lets the eddy-diffusivity closure accumulate
TKE unboundedly over terrain. This restores it: `ρtke` is advected by the
grid-mean mass flux exactly like a tracer (element-local `split_divₕ` volume
term + `dg_tracer_interface` interface flux).
"""
function dg_fddg_tke_advection!(Yₜ, Y, p)
    FT = Spaces.undertype(axes(Y.c))
    (; ᶜu) = p.precomputed
    ᶜWJ = Fields.local_geometry_field(Y.c).WJ
    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
    @. Yₜ.c.ρtke -= split_divₕ(Y.c.ρ * ᶜu, ᶜtke)   # element-local volume divergence
    (ᶜuv, ᶜλ) = dg_face_velocity_and_λ(Y, p)
    y = map(
        (ρ, uv, λ, ρχ) -> (; ρ, uv, λ, χ = ρχ / ρ),
        Y.c.ρ, ᶜuv, ᶜλ, Y.c.ρtke,
    )
    dy = Fields.Field(NamedTuple{(:ρχ,), Tuple{FT}}, axes(Y.c))
    fill!(parent(dy), 0)
    Operators.add_numerical_flux_interior!(dg_tracer_interface, dy, y)
    @. Yₜ.c.ρtke += dy.ρχ / ᶜWJ
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

# ---------------------------------------------------------------------------
# FDDG (flux-form) horizontal dynamics — `dg_equation_form = :fddg`:
# Waruszewski entropy-conservative + well-balanced volume flux (gravity as
# the ½ρ̂⟦φ⟧ fluctuation), Roe interface dissipation, perturbation pressure
# pm = p − p_ref in the momentum slot. The Cartesian-component momentum
# tendency is built in the tangential (E, N) frame and converted to the
# velocity state: duₕ = (d(ρu) − uₕ dρ)/ρ.
# ---------------------------------------------------------------------------

"""
    dg_fddg_horizontal_dynamics!(Yₜ, Y, p, t)

Flux-form horizontal dynamics on DG spaces: replaces the split_divₕ (ρ,
ρe_tot) terms, the Exner-split PGF, and the ω³ horizontal momentum advection
of `horizontal_dynamics_tendency!` with Waruszewski flux-differencing volume
terms + Roe interface fluxes over (ρ, ρe, ρu⃗) in global Cartesian momentum
components. Coriolis, all vertical terms, and tracer advection are unchanged.
"""
NVTX.@annotate function dg_fddg_horizontal_dynamics!(Yₜ, Y, p, t)
    FT = Spaces.undertype(axes(Y.c))
    (; ᶜΦ) = p.core
    (; ᶜp) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    coords = Fields.coordinate_field(axes(Y.c))
    ᶜWJ = Fields.local_geometry_field(Y.c).WJ

    # Geographic components of the Cartesian unit vectors (position-dependent,
    # state-independent) and their tangential projections as UVVectors.
    # TODO: memoize (static per grid), together with ᶜp_ref below.
    eE1 = @. -sind(coords.long)
    eE2 = @. cosd(coords.long)
    eE3 = @. zero(eE1)
    eN1 = @. -sind(coords.lat) * cosd(coords.long)
    eN2 = @. -sind(coords.lat) * sind(coords.long)
    eN3 = @. cosd(coords.lat)
    E1 = @. Geometry.UVVector(eE1, eN1)
    E2 = @. Geometry.UVVector(eE2, eN2)
    E3 = @. Geometry.UVVector(eE3, eN3)

    (ᶜuv, ᶜλ) = dg_face_velocity_and_λ(Y, p)
    uE = ᶜuv.components.data.:1
    uN = ᶜuv.components.data.:2
    u1 = @. uE * eE1 + uN * eN1
    u2 = @. uE * eE2 + uN * eN2
    u3 = @. uE * eE3 + uN * eN3
    e = @. Y.c.ρe_tot / Y.c.ρ
    ᶜp_ref = @. pref_from_phi(thermo_params, ᶜΦ)
    pm = @. ᶜp - ᶜp_ref
    # Reference density ρ_ref = p_ref/(R_d T_r) and perturbation density
    # ρm = ρ − ρ_ref. The interface dissipation's contact/entropy amplitude
    # uses ⟦ρm⟧ (not ⟦ρ⟧) so it vanishes at the hydrostatic reference over
    # terrain — without this, the O(⟦ρ_ref⟧) contact residual injects a
    # spurious mass flux at rest, unbalancing the column (spurious vertical
    # velocity that destabilizes boundary-layer closures).
    R_d = TD.Parameters.R_d(thermo_params)
    ᶜρ_ref = @. ᶜp_ref / (R_d * air_temperature_reference(thermo_params, ᶜp_ref))
    ρm = @. Y.c.ρ - ᶜρ_ref

    y = map(
        (ρ, ρe, e, p_, pm_, ρm_, uv, u1_, u2_, u3_, E1_, E2_, E3_, λ, φ) -> (;
            ρ, ρe, e, p = p_, pm = pm_, ρm = ρm_, uv,
            u1 = u1_, u2 = u2_, u3 = u3_,
            E1 = E1_, E2 = E2_, E3 = E3_, λ, φ,
        ),
        Y.c.ρ, Y.c.ρe_tot, e, ᶜp, pm, ρm, ᶜuv, u1, u2, u3, E1, E2, E3, ᶜλ, ᶜΦ,
    )
    dy = Fields.Field(
        NamedTuple{(:ρ, :ρe, :ρu1, :ρu2, :ρu3), NTuple{5, FT}},
        axes(Y.c),
    )
    fill!(parent(dy), 0)
    if p.atmos.numerics.dg_volume_flux === Val(:kg_pert)
        Operators.add_flux_differencing_divergence!(
            kennedy_gruber_cartesian_flux,
            dy,
            y,
        )
        # rusanov = full |u|+c interface dissipation (stronger grid-scale noise
        # control than roe's ~5% contact floor — needed when no hyperdiffusion
        # is present, e.g. to keep EDMFX TKE production bounded over terrain).
        kg_interface = p.atmos.numerics.dg_interface_flux === Val(:rusanov) ?
            kennedy_gruber_rusanov_cartesian : kennedy_gruber_roe_cartesian
        Operators.add_numerical_flux_interior!(kg_interface, dy, y)
    else
        Operators.add_flux_differencing_divergence!(
            waruszewski_cartesian_flux,
            dy,
            y,
        )
        if p.atmos.numerics.dg_interface_flux === Val(:es)
            Operators.add_numerical_flux_interior!(
                waruszewski_es_cartesian,
                dy,
                y,
            )
        else
            Operators.add_numerical_flux_interior!(
                waruszewski_roe_cartesian,
                dy,
                y,
            )
        end
    end

    @. Yₜ.c.ρ += dy.ρ / ᶜWJ
    @. Yₜ.c.ρe_tot += dy.ρe / ᶜWJ
    # Geographic (E, N) projections of the Cartesian momentum tendency are
    # tangential by construction (ê_E, ê_N ⊥ r̂); the velocity-form conversion
    # accounts for the flux-form density scaling.
    dρuE = @. (dy.ρu1 * eE1 + dy.ρu2 * eE2 + dy.ρu3 * eE3) / ᶜWJ
    dρuN = @. (dy.ρu1 * eN1 + dy.ρu2 * eN2 + dy.ρu3 * eN3) / ᶜWJ
    duE = @. (dρuE - uE * dy.ρ / ᶜWJ) / Y.c.ρ
    duN = @. (dρuN - uN * dy.ρ / ᶜWJ) / Y.c.ρ
    @. Yₜ.c.uₕ += C12(
        Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(duE, duN),
        ),
    )
    return nothing
end
