#=
DG interface completions for the horizontal tendencies.

On a `Grids.DG` horizontal space the element-local horizontal operators must
be completed by interface terms (on CG spaces DSS couples the elements and
every function here is a no-op):

  - `split_divₕ(ρu, ψ)` (ρ, ρe_tot, tracers): interface flux with the
    SplitDivergence-matched central part `{ρ uv·n̂}{ψ}` plus a λ-Rusanov
    penalty on the ρ-weighted jump of the specific quantity, `{ρ}⟦e_tot⟧` /
    `{ρ}⟦χ⟧`, so a uniform ψ stays uniform across a density jump
    (`F*_ρψ = ψ F*_ρ` exactly). The density flux stays central — a ⟦ρ⟧ penalty
    is kinetic-energy-sign-indefinite under the vector-invariant pairing.
  - `gradₕ` PGF/K terms (uₕ): central gradient liftings of the same scalars.
  - uₕ stability: ρ-weighted λ-jump penalties on the orthonormal velocity
    components (an exactly sign-definite KE sink). With
    `dg_vi_velocity_penalty = wave_selective`, the normal jump is damped at
    the acoustic speed and the tangential and vertical-velocity jumps at the
    shear speed (still a sign-definite KE sink).
  - With `dg_vi_interface_flux = rusanov_moist`, the energy penalty acts on
    the energy jump at fixed composition, `{ρ}({c_v,m}⟦T⟧ + ⟦K⟧)`, which is
    `{ρ}⟦e_tot⟧` for dry air and vanishes across a humidity front at uniform
    temperature and velocity; the total-water penalty moves the water's mass
    and the energy of its phases as under `es`, so such a front is damped
    without temperature jumps. The velocity penalties stay ρ-weighted (an
    exact kinetic-energy sink). The Rusanov rate uses the moist sound speed.
  - With `dg_vi_interface_flux = es`, the energy penalty acts on the jump of
    the entropy variable ∂(ρs)/∂(ρe) = 1/T of the physical entropy ρs, and the
    velocity penalties carry the own-side weight {β}/β with β = 1/T, so every
    face penalty is a sink of −ρs at fixed ρ and composition (the mass flux
    stays central) and vanishes at uniform T whatever the humidity. The
    velocity penalties stay Galilean invariant; they conserve momentum up to
    O(⟦u⟧⟦1/β⟧).
    The total-water penalty also moves the mass and energy the water carries.
    The energy is that of the water's actual phases, from the diagnostic
    vapor/liquid/ice partition (saturation adjustment for 0M, the prognostic
    condensates for 1M), so an exchange of water across a face leaves T and u
    unchanged with either microphysics scheme. The central part is not
    entropy-conservative, so this is entropy-dissipative face dissipation, not
    a discrete entropy inequality for the whole scheme.
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
    rho_weighted_jump_penalty_lift(normal, (q⁻, ρ⁻, λ⁻, β⁻), (q⁺, ρ⁺, λ⁺, β⁺))

λ-jump penalty ``max(λ)/2 ({ρ}{β}/ρ⁻β⁻) ⟦q⟧`` with own-side ρβ weighting.
With ``β ≡ 1`` the face kinetic-energy production is ``−max(λ)/2 {ρ}⟦u⟧² ≤ 0``
and momentum is conserved exactly; with ``β = ρ/p`` (``ρβu`` is the entropy
variable of ``u`` at fixed ρ, ρe_tot) the face entropy production is
``−max(λ)/2 {ρ}{β}⟦u⟧² ≤ 0`` exactly.
"""
rho_weighted_jump_penalty_lift(normal, (q⁻, ρ⁻, λ⁻, β⁻), (q⁺, ρ⁺, λ⁺, β⁺)) =
    max(λ⁻, λ⁺) / 2 * ((ρ⁻ + ρ⁺) / 2 / ρ⁻) * ((β⁻ + β⁺) / 2 / β⁻) * (q⁺ - q⁻)

# Contact/shear speed of the wave-selective penalty: |{u_n}| with a Harten-type
# floor ε c, so jumps at near-stagnant faces still decay. Symmetric under
# (n̂, −, +) → (−n̂, +, −), so both sides of a face see the same speed.
@inline dg_shear_speed(un⁻, un⁺, c⁻, c⁺, ε) =
    max(abs(un⁻ + un⁺) / 2, ε * max(c⁻, c⁺))

"""
    wave_selective_jump_penalty_lift(normal, (uv⁻, ρ⁻, c⁻, β⁻, ε), (uv⁺, ρ⁺, c⁺, β⁺, _))

ρ-weighted jump penalty on the orthonormal horizontal velocity with separate
normal and tangential rates, ``({ρ}{β}/2ρ⁻β⁻) Λ ⟦u⟧`` with
``Λ = λ_n n̂n̂ᵀ + λ_t (I − n̂n̂ᵀ)``, ``λ_n = max(|u_n| + c)`` (acoustic) and
``λ_t = max(|{u_n}|, ε max(c))`` (shear). With ``β ≡ 1`` the face
kinetic-energy production is ``−{ρ}(λ_n⟦u_n⟧² + λ_t⟦u_t⟧²)/2 ≤ 0`` and
``ρ⁻ℓ⁻ + ρ⁺ℓ⁺ = 0``; with ``β = ρ/p`` the face entropy production is
``−({ρ}{β}/2)⟦u⟧·Λ⟦u⟧ ≤ 0``. Returns a `UVVector`.
"""
function wave_selective_jump_penalty_lift(
    normal,
    (uv⁻, ρ⁻, c⁻, β⁻, ε),
    (uv⁺, ρ⁺, c⁺, β⁺, _),
)
    n1 = normal.components.data.:1
    n2 = normal.components.data.:2
    un⁻ = uv⁻' * normal
    un⁺ = uv⁺' * normal
    λn = max(abs(un⁻) + c⁻, abs(un⁺) + c⁺)
    λt = dg_shear_speed(un⁻, un⁺, c⁻, c⁺, oftype(ρ⁻, ε))
    s = (ρ⁻ + ρ⁺) / 4 / ρ⁻ * ((β⁻ + β⁺) / 2 / β⁻)
    ΔE = uv⁺.components.data.:1 - uv⁻.components.data.:1
    ΔN = uv⁺.components.data.:2 - uv⁻.components.data.:2
    a = (λn - λt) * n1 * n2
    return s * Geometry.UVVector(
        (λn * n1^2 + λt * n2^2) * ΔE + a * ΔN,
        a * ΔE + (λn * n2^2 + λt * n1^2) * ΔN,
    )
end

"""
    shear_jump_penalty_lift(normal, (w⁻, ρ⁻, uv⁻, c⁻, β⁻, ε), (w⁺, ρ⁺, uv⁺, c⁺, β⁺, _))

ρ-weighted jump penalty on the vertical velocity `w` across a horizontal
interface, where `w` is a tangential component and is damped at the shear
speed ``λ_t = max(|{u_n}|, ε max(c))``: ``λ_t/2 ({ρ}{β}/ρ⁻β⁻) ⟦w⟧``.
"""
function shear_jump_penalty_lift(
    normal,
    (w⁻, ρ⁻, uv⁻, c⁻, β⁻, ε),
    (w⁺, ρ⁺, uv⁺, c⁺, β⁺, _),
)
    λt = dg_shear_speed(uv⁻' * normal, uv⁺' * normal, c⁻, c⁺, oftype(ρ⁻, ε))
    return λt / 2 * ((ρ⁻ + ρ⁺) / 2 / ρ⁻) * ((β⁻ + β⁺) / 2 / β⁻) * (w⁺ - w⁻)
end

"""
    dg_scalars_interface(normal, argvals⁻, argvals⁺)

Interface flux for (ρ, ρe_tot): central part matched to `split_divₕ`
(`F_ρ = {ρ uv·n̂}`, `F_ρe = F_ρ {h}`) plus a λ-Rusanov penalty on the
ρ-weighted jump of the specific total energy, `−(λ/2) {ρ} ⟦e⟧`; a density
jump at uniform `e` moves no energy. State fields: `ρ`, `e`, `h`, `uv`, `λ`.
"""
function dg_scalars_interface(normal, (y⁻,), (y⁺,))
    Fρ = (y⁻.ρ * (y⁻.uv' * normal) + y⁺.ρ * (y⁺.uv' * normal)) / 2
    h̄ = (y⁻.h + y⁺.h) / 2
    λ = max(y⁻.λ, y⁺.λ)
    ρ̄ = (y⁻.ρ + y⁺.ρ) / 2
    return (; ρ = Fρ, ρe_tot = Fρ * h̄ - λ / 2 * ρ̄ * (y⁺.e - y⁻.e))
end

"""
    dg_scalars_es_interface(normal, (y⁻,), (y⁺,))

Interface flux for (ρ, ρe_tot) with the central part of
[`dg_scalars_interface`](@ref) and a λ-penalty on the jump of the entropy
variable ``v = ∂(ρs)/∂(ρe)|_{ρ,u,q} = 1/T`` of the physical entropy ``ρs``:
``F_ρe = F_ρ {h} − (λ/2) ŵ (1/T⁻ − 1/T⁺)``, ``ŵ = {ρ}{c_v}{T}²``, so that
``ŵ (1/T⁻ − 1/T⁺) ≈ {ρ}{c_v}⟦T⟧ ≈ ⟦ρe⟧`` at fixed ρ, u and composition. The
face production of ``ρs`` is ``+(λ/2) ŵ ⟦1/T⟧² ≥ 0``, and the penalty
vanishes at uniform ``T`` whatever the humidity. State fields: `ρ`, `T`, `cv`
(moist heat capacity ``c_{vm}``), `h`, `uv`, `λ`.
"""
function dg_scalars_es_interface(normal, (y⁻,), (y⁺,))
    Fρ = (y⁻.ρ * (y⁻.uv' * normal) + y⁺.ρ * (y⁺.uv' * normal)) / 2
    h̄ = (y⁻.h + y⁺.h) / 2
    λ = max(y⁻.λ, y⁺.λ)
    T̄ = (y⁻.T + y⁺.T) / 2
    ŵ = (y⁻.ρ + y⁺.ρ) / 2 * ((y⁻.cv + y⁺.cv) / 2) * T̄^2
    Δv = 1 / y⁻.T - 1 / y⁺.T
    return (; ρ = Fρ, ρe_tot = Fρ * h̄ - λ / 2 * ŵ * Δv)
end

"""
    dg_scalars_moist_interface(normal, argvals⁻, argvals⁺)

Interface flux for (ρ, ρe_tot) of `dg_vi_interface_flux = rusanov_moist`:
the central part of [`dg_scalars_interface`](@ref) and a λ-penalty on the
ρ-weighted jump of the specific total energy at fixed composition,
``{ρ}({c_v,m}⟦T⟧ + ⟦K⟧)`` (Φ is single-valued at the face). For dry air
``c_v,d ⟦T⟧ + ⟦K⟧ = ⟦e_tot⟧`` exactly, so this is the `rusanov` penalty;
across a humidity front at uniform temperature and velocity it vanishes,
whereas ``⟦e_tot⟧`` carries the latent and heat-capacity difference of the
water, which the water flux moves together with the water's mass
([`dg_water_es_interface`](@ref)). State fields: `ρ`, `T`, `cv`, `K`, `h`,
`uv`, `λ`.
"""
function dg_scalars_moist_interface(normal, (y⁻,), (y⁺,))
    Fρ = (y⁻.ρ * (y⁻.uv' * normal) + y⁺.ρ * (y⁺.uv' * normal)) / 2
    h̄ = (y⁻.h + y⁺.h) / 2
    λ = max(y⁻.λ, y⁺.λ)
    ρ̄ = (y⁻.ρ + y⁺.ρ) / 2
    c̄v = (y⁻.cv + y⁺.cv) / 2
    Δe = c̄v * (y⁺.T - y⁻.T) + (y⁺.K - y⁻.K)
    return (; ρ = Fρ, ρe_tot = Fρ * h̄ - λ / 2 * ρ̄ * Δe)
end

"""
    dg_tracer_interface(normal, argvals⁻, argvals⁺)

Interface flux for a tracer ρχ advected by the same mass flux as continuity:
`F_ρχ = {ρ uv·n̂}{χ}` plus a λ-Rusanov penalty on the ρ-weighted jump of the
specific quantity, `−(λ/2) {ρ} ⟦χ⟧`. With χ uniform the penalty vanishes and
`F_ρχ = χ F_ρ` exactly, so a uniform tracer stays uniform across a density
jump. The flux is monotone in (χ⁻, χ⁺) for `λ ≥ |uv·n̂|` on both sides, as the
Zhang–Shu positivity argument requires. State fields: `ρ`, `uv`, `λ`, `χ`.
"""
function dg_tracer_interface(normal, (y⁻,), (y⁺,))
    Fρ = (y⁻.ρ * (y⁻.uv' * normal) + y⁺.ρ * (y⁺.uv' * normal)) / 2
    χ̄ = (y⁻.χ + y⁺.χ) / 2
    λ = max(y⁻.λ, y⁺.λ)
    ρ̄ = (y⁻.ρ + y⁺.ρ) / 2
    return (; ρχ = Fρ * χ̄ - λ / 2 * ρ̄ * (y⁺.χ - y⁻.χ))
end

"""
    dg_water_es_interface(normal, (y⁻,), (y⁺,))

[`dg_tracer_interface`](@ref) for total water under
`dg_vi_interface_flux = es`: the penalty `d = −(λ/2){ρ}⟦q_tot⟧` also moves
the mass of the water, `F_ρ = d`, and the energy of the water's actual
phases. With the diagnostic condensate partition `q_liq`, `q_ice` (so the
vapor is `q_tot − q_liq − q_ice`) and the per-phase penalties
`d_s = −(λ/2){ρ}⟦q_s⟧`, the energy flux is
`F_ρe = d {e_vap} + d_liq {e_liq} + d_ice {e_ice}` with `e_vap = I_v + K + Φ`,
`e_liq = I_l − I_v` and `e_ice = I_i − I_v`, which equals
`Σ_s d_s {I_s} + d {K + Φ}` over vapor, liquid and ice: the moved water carries
the energy of its phases. State fields: `ρ`, `uv`, `λ`, `χ` (= q_tot),
`q_liq`, `q_ice`, `e_vap`, `e_liq`, `e_ice`.
"""
function dg_water_es_interface(normal, (y⁻,), (y⁺,))
    Fρ = (y⁻.ρ * (y⁻.uv' * normal) + y⁺.ρ * (y⁺.uv' * normal)) / 2
    χ̄ = (y⁻.χ + y⁺.χ) / 2
    s = -max(y⁻.λ, y⁺.λ) / 2 * (y⁻.ρ + y⁺.ρ) / 2
    d = s * (y⁺.χ - y⁻.χ)
    d_liq = s * (y⁺.q_liq - y⁻.q_liq)
    d_ice = s * (y⁺.q_ice - y⁻.q_ice)
    ρe_tot =
        d * (y⁻.e_vap + y⁺.e_vap) / 2 +
        d_liq * (y⁻.e_liq + y⁺.e_liq) / 2 +
        d_ice * (y⁻.e_ice + y⁺.e_ice) / 2
    return (; ρχ = Fρ * χ̄ + d, ρ = d, ρe_tot)
end

"""
    moist_sound_speed(thermo_params, ρ, p, q_tot, q_liq, q_ice)

Sound speed ``sqrt(γ_m p / ρ)`` with ``γ_m = c_p,m / c_v,m`` at the given
composition (the frozen-composition value, equal to
`Thermodynamics.soundspeed_air` since ``p = ρ R_m T``); reduces to
``sqrt(γ_d p / ρ)`` for dry air.
"""
@inline function moist_sound_speed(thermo_params, ρ, p, q_tot, q_liq, q_ice)
    γ =
        TD.cp_m(thermo_params, q_tot, q_liq, q_ice) /
        TD.cv_m(thermo_params, q_tot, q_liq, q_ice)
    return sqrt(γ * p / ρ)
end

# Sound speed at the diagnosed composition, c_snd = sqrt(γ_m p / ρ).
function dg_sound_speed(Y, p)
    (; ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    return @. moist_sound_speed(
        thermo_params,
        Y.c.ρ,
        ᶜp,
        ᶜq_tot_nonneg,
        ᶜq_liq,
        ᶜq_ice,
    )
end

# Orthonormal horizontal face velocity, Rusanov speed λ = |uv| + c_snd, and the
# tracer penalty rate (λ, or max(|uv|, ε c_snd) with `dg_tracer_penalty =
# advective`). The face velocity is the horizontal physical projection of the FULL ᶜu,
# matching the contravariant contraction of the split_divₕ volume flux: over
# terrain-warped grids u₃ contributes through g¹³/g²³; on unwarped grids those
# metric entries are exact zeros, so this reduces bitwise to UVVector(uₕ).
function dg_face_velocity_and_λ(Y, p)
    (; ᶜu) = p.precomputed
    ᶜuv = Geometry.project.(Ref(Geometry.UVAxis()), ᶜu)
    ᶜc = dg_sound_speed(Y, p)
    ᶜλ = @. sqrt(LinearAlgebra.norm_sqr(ᶜuv)) + ᶜc
    # tracer penalty rate: acoustic |u| + c, or advective max(|u|, ε c)
    (; dg_tracer_penalty, dg_vi_shear_floor) = p.atmos.numerics
    ᶜλ_tracer = if dg_tracer_penalty === Val(:advective)
        ε = Spaces.undertype(axes(Y.c))(dg_vi_shear_floor)
        @. max(sqrt(LinearAlgebra.norm_sqr(ᶜuv)), ε * ᶜc)
    else
        ᶜλ
    end
    return (ᶜuv, ᶜλ, ᶜλ_tracer)
end

"""
    dg_wave_selective_velocity_penalty!(Yₜ, Y, p, ᶜuv, ᶜβ)

Wave-selective velocity jump penalties (`dg_vi_velocity_penalty = wave_selective`): [`wave_selective_jump_penalty_lift`](@ref) on uₕ and
[`shear_jump_penalty_lift`](@ref) on the face vertical velocity, with the
shear floor `p.atmos.numerics.dg_vi_shear_floor` and the own-side weight `ᶜβ`
(a center field, or `1`).
"""
function dg_wave_selective_velocity_penalty!(Yₜ, Y, p, ᶜuv, ᶜβ)
    FT = Spaces.undertype(axes(Y.c))
    ε = FT(p.atmos.numerics.dg_vi_shear_floor)
    ᶜc = dg_sound_speed(Y, p)
    pen_uv = Operators.lifting_correction(
        wave_selective_jump_penalty_lift,
        Geometry.UVVector{FT},
        ᶜuv,
        Y.c.ρ,
        ᶜc,
        ᶜβ,
        ε,
    )
    @. Yₜ.c.uₕ += C12(Geometry.transform(Geometry.Covariant12Axis(), pen_uv))

    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶠw = @. Geometry.WVector(Y.f.u₃).components.data.:1
    ᶠρ = @. ᶠinterp(Y.c.ρ)
    ᶠuv = @. ᶠinterp(ᶜuv)
    ᶠc = @. ᶠinterp(ᶜc)
    ᶠβ = ᶜβ isa Number ? ᶜβ : (@. ᶠinterp(ᶜβ))
    pen_w = Operators.lifting_correction(
        shear_jump_penalty_lift,
        FT,
        ᶠw,
        ᶠρ,
        ᶠuv,
        ᶠc,
        ᶠβ,
        ε,
    )
    # the surface and top-face w are set by the boundary conditions
    Fields.level(pen_w, half) .= 0
    Fields.level(pen_w, Spaces.nlevels(axes(Y.f)) - half) .= 0
    @. Yₜ.f.u₃ += C3(Geometry.WVector(pen_w), ᶠlg)
    return nothing
end

"""
    dg_horizontal_dynamics_completion!(Yₜ, Y, p, t)

Interface completion of `horizontal_dynamics_tendency!` on DG spaces: scalar
(ρ, ρe_tot) interface fluxes, central liftings of the PGF/K `gradₕ` terms,
and ρ-weighted velocity jump penalties — uniform λ on uₕ (`rusanov`), or
wave-selective on uₕ and u₃ (`wave_selective`, see
[`dg_wave_selective_velocity_penalty!`](@ref)). With
`dg_vi_interface_flux = es` the ρe_tot penalty acts on the entropy-variable
jump ⟦1/T⟧ (see [`dg_scalars_es_interface`](@ref)) and the velocity penalties
carry the own-side weight {β}/β, β = 1/T; with `rusanov_moist` it acts on
the energy jump at fixed composition (see
[`dg_scalars_moist_interface`](@ref)). No-op on CG spaces.
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
    es = p.atmos.numerics.dg_vi_interface_flux === Val(:es)
    rusanov_moist =
        p.atmos.numerics.dg_vi_interface_flux === Val(:rusanov_moist)
    # own-side velocity penalty weight β for es: 1/T, since the entropy
    # variable of ρu at fixed ρ, ρe and composition is −ρu/T; 1 otherwise
    ᶜβ = es ? (@. 1 / ᶜT) : one(FT)

    # (ρ, ρe_tot) interface fluxes
    dy = Fields.Field(NamedTuple{(:ρ, :ρe_tot), Tuple{FT, FT}}, axes(Y.c))
    fill!(parent(dy), 0)
    if rusanov_moist
        ᶜcv = @. TD.cv_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)
        y = map(
            (ρ, T, cv, K, h, uv, λ) -> (; ρ, T, cv, K, h, uv, λ),
            Y.c.ρ,
            ᶜT,
            ᶜcv,
            ᶜK,
            ᶜh_tot,
            ᶜuv,
            ᶜλ,
        )
        Operators.add_numerical_flux_interior!(
            dg_scalars_moist_interface,
            dy,
            y,
        )
    elseif es
        ᶜcv = @. TD.cv_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)
        y = map(
            (ρ, T, cv, h, uv, λ) -> (; ρ, T, cv, h, uv, λ),
            Y.c.ρ,
            ᶜT,
            ᶜcv,
            ᶜh_tot,
            ᶜuv,
            ᶜλ,
        )
        Operators.add_numerical_flux_interior!(dg_scalars_es_interface, dy, y)
    else
        ᶜe_tot = @. specific(Y.c.ρe_tot, Y.c.ρ)
        y = map(
            (ρ, e, h, uv, λ) -> (; ρ, e, h, uv, λ),
            Y.c.ρ,
            ᶜe_tot,
            ᶜh_tot,
            ᶜuv,
            ᶜλ,
        )
        Operators.add_numerical_flux_interior!(dg_scalars_interface, dy, y)
    end
    @. Yₜ.c.ρ += dy.ρ / ᶜWJ
    @. Yₜ.c.ρe_tot += dy.ρe_tot / ᶜWJ

    # central liftings of the strong PGF/K gradients (the same scalars as the
    # gradₕ volume terms in horizontal_dynamics_tendency!)
    ᶜKΦ = @. ᶜK + ᶜΦ - phi_r(thermo_params, ᶜp)
    dg_exner_split_pgf_lift!(Yₜ, Y, p, ᶜKΦ)

    if p.atmos.numerics.dg_vi_velocity_penalty === Val(:wave_selective)
        dg_wave_selective_velocity_penalty!(Yₜ, Y, p, ᶜuv, ᶜβ)
        return nothing
    end
    # ρ-weighted λ-jump penalties on the orthonormal velocity components
    pen_u = Operators.lifting_correction(
        rho_weighted_jump_penalty_lift,
        FT,
        ᶜuv.components.data.:1,
        Y.c.ρ,
        ᶜλ,
        ᶜβ,
    )
    pen_v = Operators.lifting_correction(
        rho_weighted_jump_penalty_lift,
        FT,
        ᶜuv.components.data.:2,
        Y.c.ρ,
        ᶜλ,
        ᶜβ,
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
    dg_exner_split_pgf_lift!(Yₜ, Y, p, ᶜKΦ)

Central interface liftings of the Exner-split horizontal pressure-gradient
force, `gradₕ(ᶜKΦ) + cp_d (θ_v′ gradₕ Π + gradₕ(θ_v′ Π) − Π gradₕ θ_v′) / 2`
with `θ_v′ = θ_v − θ_vr(p)`: the same scalars as the `gradₕ` volume terms,
with `ᶜKΦ` the scalar of the plain gradient (`K + Φ − Φ_r` for the
vector-invariant form, `Φ − Φ_r` when the flux-form advection carries ∇K).
"""
function dg_exner_split_pgf_lift!(Yₜ, Y, p, ᶜKΦ)
    FT = Spaces.undertype(axes(Y.c))
    (; ᶜp, ᶜT, ᶜq_liq, ᶜq_ice, ᶜq_tot_nonneg) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    cp_d = thermo_params.cp_d
    ᶜΠ = @. TD.exner_given_pressure(thermo_params, ᶜp)
    ᶜθ_v_diff = @. theta_v(thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) -
       theta_vr(thermo_params, ᶜp)
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
    return nothing
end

# Add the `dg_tracer_interface` flux of the mass-weighted scalar `ᶜρχ`,
# advected by the grid-mean mass flux, to `ᶜρχₜ`.
function add_dg_tracer_interface_flux!(ᶜρχₜ, ᶜρχ, Y, ᶜuv, ᶜλ)
    FT = Spaces.undertype(axes(Y.c))
    ᶜWJ = Fields.local_geometry_field(Y.c).WJ
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
    return nothing
end

# Specific energies carried by the total-water penalty of
# `dg_water_es_interface`: vapor added with ρ (`vap`), and liquid or ice in
# place of vapor (`liq`, `ice`).
function dg_water_specific_energies(p)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜT, ᶜK) = p.precomputed
    (; ᶜΦ) = p.core
    ᶜe_vap = @. TD.internal_energy_vapor(thermo_params, ᶜT)
    return (;
        vap = @.(ᶜe_vap + ᶜK + ᶜΦ),
        liq = @.(TD.internal_energy_liquid(thermo_params, ᶜT) - ᶜe_vap),
        ice = @.(TD.internal_energy_ice(thermo_params, ᶜT) - ᶜe_vap),
    )
end

# Add the `dg_water_es_interface` flux of total water `ᶜρq_tot` to `ᶜρq_totₜ`,
# `Yₜ.c.ρ` and `Yₜ.c.ρe_tot`. The diagnostic condensate partition (`ᶜq_liq`,
# `ᶜq_ice`, from saturation adjustment for 0M and the prognostic condensates
# for 1M) sets the phases, and so the energy, of the water moved.
function add_dg_water_es_flux!(Yₜ, ᶜρq_totₜ, ᶜρq_tot, Y, p, ᶜuv, ᶜλ)
    FT = Spaces.undertype(axes(Y.c))
    ᶜWJ = Fields.local_geometry_field(Y.c).WJ
    (; ᶜq_liq, ᶜq_ice) = p.precomputed
    (; vap, liq, ice) = dg_water_specific_energies(p)
    y = map(
        (ρ, uv, λ, ρq_tot, q_liq, q_ice, e_vap, e_liq, e_ice) ->
            (; ρ, uv, λ, χ = ρq_tot / ρ, q_liq, q_ice, e_vap, e_liq, e_ice),
        Y.c.ρ,
        ᶜuv,
        ᶜλ,
        ᶜρq_tot,
        ᶜq_liq,
        ᶜq_ice,
        vap,
        liq,
        ice,
    )
    dy = Fields.Field(NamedTuple{(:ρχ, :ρ, :ρe_tot), NTuple{3, FT}}, axes(Y.c))
    fill!(parent(dy), 0)
    Operators.add_numerical_flux_interior!(dg_water_es_interface, dy, y)
    @. ᶜρq_totₜ += dy.ρχ / ᶜWJ
    @. Yₜ.c.ρ += dy.ρ / ᶜWJ
    @. Yₜ.c.ρe_tot += dy.ρe_tot / ᶜWJ
    return nothing
end

"""
    dg_horizontal_tracer_completion!(Yₜ_lim, Y, p, t)

Interface completion of `horizontal_tracer_advection_tendency!` on DG spaces:
the tracer interface flux for every grid-mean tracer. With the VI form and
`dg_vi_interface_flux = es`, total water uses [`dg_water_es_interface`](@ref):
its penalty also moves ρ and the energy of the water's phases, taken from the
diagnostic partition `ᶜq_liq`, `ᶜq_ice` (saturation adjustment for 0M; cloud
liquid + rain and cloud ice + snow for 1M), so a water exchange leaves T and u
unchanged to first order in the jumps with either scheme. The 1M condensate
species are plain tracers; their energy enters through that partition. No-op
on CG spaces.
"""
NVTX.@annotate function dg_horizontal_tracer_completion!(Yₜ_lim, Y, p, t)
    is_dg_horizontal(axes(Y.c)) || return nothing
    (ᶜuv, _, ᶜλ) = dg_face_velocity_and_λ(Y, p)
    # vi es / rusanov_moist: total water moves its mass and phase energies
    es =
        p.atmos.numerics.dg_equation_form === Val(:vi) && (
            p.atmos.numerics.dg_vi_interface_flux === Val(:es) ||
            p.atmos.numerics.dg_vi_interface_flux === Val(:rusanov_moist)
        )
    # fddg moist_es: total water is dissipated with the (ρ, ρu, ρe, ρq_tot)
    # entropy system in dg_fddg_horizontal_dynamics!; keep only the central
    # flux here
    moist_es =
        p.atmos.numerics.dg_equation_form === Val(:fddg) &&
        p.atmos.numerics.dg_interface_flux === Val(:moist_es)
    ᶜλ_zero = @. zero(ᶜλ)
    foreach_gs_tracer(Yₜ_lim, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        if es && ρχ_name == @name(ρq_tot)
            add_dg_water_es_flux!(Yₜ_lim, ᶜρχₜ, ᶜρχ, Y, p, ᶜuv, ᶜλ)
        elseif moist_es && ρχ_name == @name(ρq_tot)
            add_dg_tracer_interface_flux!(ᶜρχₜ, ᶜρχ, Y, ᶜuv, ᶜλ_zero)
        else
            add_dg_tracer_interface_flux!(ᶜρχₜ, ᶜρχ, Y, ᶜuv, ᶜλ)
        end
    end
    return nothing
end

"""
    dg_ρtke_interface_completion!(Yₜ, Y, p)

Interface completion of the element-local `split_divₕ` horizontal advection of
prognostic TKE on DG spaces. `ρtke` is excluded from `is_tracer_var`, so the
grid-mean tracer completion does not cover it; it is advected by the grid-mean
mass flux with the same interface flux as the grid-mean tracers. No-op on CG
spaces.
"""
function dg_ρtke_interface_completion!(Yₜ, Y, p)
    is_dg_horizontal(axes(Y.c)) || return nothing
    (ᶜuv, _, ᶜλ) = dg_face_velocity_and_λ(Y, p)
    add_dg_tracer_interface_flux!(Yₜ.c.ρtke, Y.c.ρtke, Y, ᶜuv, ᶜλ)
    return nothing
end

"""
    dg_fddg_tke_advection!(Yₜ, Y, p)

Horizontal advection of prognostic TKE on DG spaces for the `:fddg` path.
`horizontal_dynamics_tendency!` returns early for fddg (its `Yₜ.c.ρtke`
split-divₕ term is skipped) and `dg_fddg_horizontal_dynamics!` only carries
(ρ, ρe_tot, uₕ), so this adds the element-local `split_divₕ` volume term and
its interface completion ([`dg_ρtke_interface_completion!`](@ref)).
"""
function dg_fddg_tke_advection!(Yₜ, Y, p)
    (; ᶜu) = p.precomputed
    ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
    @. Yₜ.c.ρtke -= split_divₕ(Y.c.ρ * ᶜu, ᶜtke)
    dg_ρtke_interface_completion!(Yₜ, Y, p)
    return nothing
end

# Level-wise horizontal DG divergence (weak volume term + central interface
# lifting), the linear operator `hdiv` used by the GCL cross-term commutator.
function dg_central_weak_divergence(X)
    lg = Fields.local_geometry_field(axes(X))
    wd = Operators.Divergence{Operators.WeakForm}()
    r = @. wd(X) * (-lg.WJ)
    Operators.add_numerical_flux_interior!(
        Operators.CentralNumericalFlux(identity),
        r,
        X,
    )
    return @. -r / lg.WJ
end

"""
    set_dg_gcl_ᶠuₕ³!(ᶠuₕ³, ᶜuₕ, ᶜρ)

GCL-consistent (free-stream-preserving) replacement for `compute_ᶠuₕ³` on DG
horizontal spaces. The chain-form cross-term `ᶠwinterp(ρJ, CT3(uₕ))` applies
`u·∇Z` with the center `∇Z` and a wide vertical mean, which does not telescope
against the horizontal operator's terrain metric (`J∇ξ^{1,2} = J_v ⊙ flat`
with `J_v_c[v] = z_f[v+1] − z_f[v]` exactly): a divergence-free flow over
terrain then produces a spurious mass source. The commutator form

    F³[face] = −( T̂(z_f ⊙ ŝ) − z_f ⊙ T̂(ŝ) ),   T̂(X) = ᶠJ · hdiv(X ⊘ ᶠJv)

with `ᶠJv = GradientC2F(ᶜz)` (bitwise the stored face `J_v`) telescopes
exactly: `δ_v(F³)` cancels the horizontal terrain defect, reducing the 3D
divergence over terrain to the flat-form horizontal divergence (verified to
the flat baseline; see docs/dg_fd_gcl_crossterm.md and dg_campaign jobs
37–41). On flat grids the commutator vanishes identically.

The surface and top face levels keep the chain-form value: the vertical mass
divergence never reads them (`ᶜadvdivᵥ` has zero-flux boundaries), they only
feed the `u₃` impenetrability filter — where the pointwise `u·∇Z` is the
consistent tangency condition — and the boundary-face `WJ_v = J_v/2` special
case makes the commutator inconsistent by 2× there.
"""
function set_dg_gcl_ᶠuₕ³!(ᶠuₕ³, ᶜuₕ, ᶜρ)
    FT = Spaces.undertype(axes(ᶠuₕ³))
    face_space = axes(ᶠuₕ³)
    center_space = axes(ᶜρ)
    ᶜz = Fields.coordinate_field(center_space).z
    ᶠz = Fields.coordinate_field(face_space).z
    ᶜJ = Fields.local_geometry_field(center_space).J
    ᶠJ = Fields.local_geometry_field(face_space).J
    gradᵥ_c2f = Operators.GradientC2F(;
        bottom = Operators.SetGradient(Geometry.WVector(FT(1))),
        top = Operators.SetGradient(Geometry.WVector(FT(1))),
    )
    ᶠ∇z = @. gradᵥ_c2f(ᶜz)
    ᶠJv = ᶠ∇z.components.data.:1    # z_c[v] − z_c[v−1] at interior faces
    ᶜuv = @. Geometry.UVVector(ᶜuₕ)
    ᶠs = @. ᶠinterp(ᶜρ * ᶜuv)       # face horizontal mass flux
    h1 = dg_central_weak_divergence(@. ᶠz * ᶠs / ᶠJv)
    h2 = dg_central_weak_divergence(@. ᶠs / ᶠJv)
    ᶠρJ = @. ᶠinterp(ᶜρ * ᶜJ)
    ᶠuₕ³_gcl = @. CT3(ᶠJ * (ᶠz * h2 - h1) / ᶠρJ)
    # chain form everywhere (supplies the boundary-face values), then the
    # commutator form at interior faces
    @. ᶠuₕ³ = ᶠwinterp(ᶜρ * ᶜJ, CT3(ᶜuₕ))
    for i in 1:(Spaces.nlevels(center_space) - 1)
        Fields.level(ᶠuₕ³, i + half) .= Fields.level(ᶠuₕ³_gcl, i + half)
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

# ---------------------------------------------------------------------------
# FDDG (flux-form) horizontal dynamics — `dg_equation_form = :fddg`:
# Waruszewski entropy-conservative + well-balanced volume flux (gravity as
# the ½ρ̂⟦φ⟧ fluctuation), Roe interface dissipation, perturbation pressure
# pm = p − p_ref in the momentum slot. The Cartesian-component momentum
# tendency is built in the tangential (E, N) frame and converted to the
# velocity state: duₕ = (d(ρu) − uₕ dρ)/ρ. With `dg_volume_flux = kg_exner`
# the two-point momentum flux is advective only and the pressure force is
# the Exner-split gradient of the vector-invariant form (volume gradₕ terms
# plus central liftings), added to uₕ directly.
# ---------------------------------------------------------------------------

"""
    dg_fddg_horizontal_dynamics!(Yₜ, Y, p, t)

Flux-form horizontal dynamics on DG spaces: replaces the split_divₕ (ρ,
ρe_tot) terms, the Exner-split PGF, and the ω³ horizontal momentum advection
of `horizontal_dynamics_tendency!` with Waruszewski flux-differencing volume
terms + Roe interface fluxes over (ρ, ρe, ρu⃗) in global Cartesian momentum
components. Coriolis, all vertical terms, and tracer advection are unchanged.

With `dg_fddg_vertical = flux` the vertical transport of the horizontal
momentum is also in flux form ([`dg_fddg_vertical_flux!`](@ref)): the face
mass flux of the continuity equation times the interpolated Cartesian
velocity, which is kinetic-energy preserving on the staggered grid for the
same reason the horizontal two-point flux is, plus, for the flux-form
pressure families, the vertical two-point pressure flux
`∂_ξ3({p_m}{ê_c ⋅ Ja³})` through the horizontal part of `Ja³`. That term is
the terrain tilt of the horizontal pressure force; the horizontal flux
differencing alone carries only `Σ_{k=1,2} ∂_ξk(p_m Ja^k)`, which over sloped
surfaces is the along-surface gradient and not the horizontal one.

With `dg_volume_flux = div_exner` the two-point flux is the divergence form
[`divergence_cartesian_flux`](@ref): the mass flux is the average of the
nodal mass fluxes, as in the vector-invariant form, so the nodal gradients of
`κ` and `Φ` exchange kinetic with potential energy exactly. With
`dg_fddg_vertical = vi` this makes the scheme kinetic- and potential-energy
preserving (KPEP); `flux` breaks the deep-atmosphere metric pair of the
vertical exchange.

With `dg_volume_flux = kg_exner` only the advection is in flux form
([`kennedy_gruber_advective_cartesian_flux`](@ref) with Roe or Rusanov
interface dissipation); the horizontal pressure force is the Exner-split
gradient of the vector-invariant form, `gradₕ(Φ − Φ_r) + cp_d [θ_v′ ∇Π]`
with its central liftings ([`dg_exner_split_pgf_lift!`](@ref)). Gradients
of scalars are pointwise exact on constants, so over terrain this form
does not depend on the discrete metric identity that the two-point
pressure flux `{p_m}{ê_c ⋅ Ja^k}` fails to satisfy across the horizontal
collocation and vertical FD operators.
"""
NVTX.@annotate function dg_fddg_horizontal_dynamics!(Yₜ, Y, p, t)
    FT = Spaces.undertype(axes(Y.c))
    (; ᶜΦ) = p.core
    (; ᶜp, ᶜK) = p.precomputed
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

    (ᶜuv, ᶜλ, ᶜλ_tracer) = dg_face_velocity_and_λ(Y, p)
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
    # equation-of-state inputs of the `roe_eos` dissipation: specific
    # internal energy, the water species at the diagnosed partition, and the
    # rates at which the tracer penalties move the species (q_tot for any
    # moist model, the condensates only when they are prognostic)
    (; ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; microphysics_model) = p.atmos
    ᶜe_int = @. e - ᶜK - ᶜΦ
    ᶜλq = microphysics_model isa DryModel ? (@. zero(ᶜλ)) : ᶜλ_tracer
    ᶜλqc =
        microphysics_model isa NonEquilibriumMicrophysics1M ? ᶜλ_tracer :
        (@. zero(ᶜλ))

    # potential of the two-point gravity of kg_kpep: Φ − Φ_r(p), the same
    # combination as the nodal gradient of the Exner split (so ψ ≡ 0 at the
    # reference state and the rest state stays balanced), plus the
    # non-horizontal kinetic energy when the vertical exchange is VI (its
    # gradient pairs with the mass flux like Φ). The pairing is linear in ψ,
    # so the work of Φ alone is exchanged with the potential energy exactly.
    kpep = p.atmos.numerics.dg_volume_flux === Val(:kg_kpep)
    cp_d = thermo_params.cp_d
    ᶜθ_v_diff = @. theta_v(
        thermo_params,
        p.precomputed.ᶜT,
        ᶜp,
        ᶜq_tot_nonneg,
        ᶜq_liq,
        ᶜq_ice,
    ) - theta_vr(thermo_params, ᶜp)
    ᶜΠ = @. TD.exner_given_pressure(thermo_params, ᶜp)
    vertical_vi = p.atmos.numerics.dg_fddg_vertical === Val(:vi)
    ᶜψ_kpep =
        vertical_vi ?
        (@. ᶜΦ - phi_r(thermo_params, ᶜp) + ᶜK -
            FT(1 // 2) * LinearAlgebra.norm_sqr(ᶜuv)) :
        (@. ᶜΦ - phi_r(thermo_params, ᶜp))
    y = map(
        (
            ρ, ρe, e, p_, pm_, ρm_, uv, u1_, u2_, u3_, E1_, E2_, E3_, λ, φ,
            e_int, q_tot, q_liq, q_ice, λq, λqc, ψ, θ, Π,
        ) -> (;
            ρ, ρe, e, p = p_, pm = pm_, ρm = ρm_, uv,
            u1 = u1_, u2 = u2_, u3 = u3_,
            E1 = E1_, E2 = E2_, E3 = E3_, λ, φ,
            e_int, q_tot, q_liq, q_ice, λq, λqc, ψ, θ, Π,
        ),
        Y.c.ρ, Y.c.ρe_tot, e, ᶜp, pm, ρm, ᶜuv, u1, u2, u3, E1, E2, E3, ᶜλ, ᶜΦ,
        ᶜe_int, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜλq, ᶜλqc, ᶜψ_kpep,
        ᶜθ_v_diff, ᶜΠ,
    )
    # volume flux family and interface dissipation
    (; dg_volume_flux, dg_interface_flux) = p.atmos.numerics
    # the Exner-gradient families carry no pressure in the two-point flux
    exner_pgf =
        dg_volume_flux === Val(:kg_exner) ||
        dg_volume_flux === Val(:div_exner) || kpep
    kg_advective = dg_volume_flux === Val(:kg_exner) || kpep
    volume_flux =
        dg_volume_flux === Val(:div_exner) ? divergence_cartesian_flux :
        kpep ? KennedyGruberKPEPFlux(cp_d) :
        dg_volume_flux === Val(:kg_exner) ?
        kennedy_gruber_advective_cartesian_flux :
        dg_volume_flux === Val(:kg_pert) ? kennedy_gruber_cartesian_flux :
        waruszewski_cartesian_flux
    # the gravity fluctuation of kg_kpep is not an antisymmetric face flux;
    # its face part is the lifting KPEPFaceLift below
    iface_central = kpep ? kennedy_gruber_advective_cartesian_flux : volume_flux
    moist_es = dg_interface_flux === Val(:moist_es)
    # moist_es dissipates total water with the system (6 slots) unless dry
    Nslots = moist_es && !(microphysics_model isa DryModel) ? 6 : 5
    interface = if dg_interface_flux === Val(:rusanov)
        # rusanov = full |u|+c interface dissipation (stronger grid-scale noise
        # control than roe's ~5% contact floor — needed when no hyperdiffusion
        # is present, e.g. to keep EDMFX TKE production bounded over terrain).
        dg_volume_flux === Val(:div_exner) ?
        CentralMinusDissipation(iface_central, rusanov_dissipation_cartesian) :
        kg_advective ?
        kennedy_gruber_advective_rusanov_cartesian :
        kennedy_gruber_rusanov_cartesian
    elseif dg_interface_flux === Val(:roe_eos)
        RoeEOSInterface(iface_central, thermo_params)
    elseif moist_es
        MoistESInterface{Nslots}(iface_central, thermo_params)
    elseif dg_interface_flux === Val(:es)
        waruszewski_es_cartesian
    else
        dg_volume_flux === Val(:div_exner) ?
        CentralMinusDissipation(iface_central, roe_dissipation_cartesian) :
        kg_advective ?
        kennedy_gruber_advective_roe_cartesian :
        dg_volume_flux === Val(:kg_pert) ? kennedy_gruber_roe_cartesian :
        waruszewski_roe_cartesian
    end
    dy = if Nslots == 6
        Fields.Field(
            NamedTuple{(:ρ, :ρe, :ρu1, :ρu2, :ρu3, :ρq_tot), NTuple{6, FT}},
            axes(Y.c),
        )
    else
        Fields.Field(
            NamedTuple{(:ρ, :ρe, :ρu1, :ρu2, :ρu3), NTuple{5, FT}},
            axes(Y.c),
        )
    end
    fill!(parent(dy), 0)
    Operators.add_flux_differencing_divergence!(
        Nslots == 6 ? with_water_slot(volume_flux) : volume_flux,
        dy,
        y,
    )
    Operators.add_numerical_flux_interior!(interface, dy, y)
    if kpep
        L_kpep = Operators.lifting_correction(
            KPEPFaceLift(cp_d),
            Geometry.UVVector{FT},
            ᶜψ_kpep,
            ᶜθ_v_diff,
            ᶜΠ,
            Y.c.ρ,
        )
        @. Yₜ.c.uₕ -= C12(
            Geometry.transform(Geometry.Covariant12Axis(), L_kpep / Y.c.ρ),
        )
    end
    if exner_pgf && !kpep
        # Exner-split horizontal PGF of the vector-invariant form (as in
        # horizontal_dynamics_tendency!) without the ∇K part, which the
        # flux-form advection above carries. kg_kpep carries the whole Exner
        # force in its two-point form instead.
        ᶜΦ_diff = @. ᶜΦ - phi_r(thermo_params, ᶜp)
        @. Yₜ.c.uₕ -= C12(
            gradₕ(ᶜΦ_diff) +
            cp_d * (
                ᶜθ_v_diff * gradₕ(ᶜΠ) + gradₕ(ᶜθ_v_diff * ᶜΠ) -
                ᶜΠ * gradₕ(ᶜθ_v_diff)
            ) / 2,
        )
        dg_exner_split_pgf_lift!(Yₜ, Y, p, ᶜΦ_diff)
    end
    if Nslots == 6
        # water dissipation of the moist entropy system; its central flux and
        # volume term are in the tracer completion, whose penalty is off
        @. Yₜ.c.ρq_tot += dy.ρq_tot / ᶜWJ
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
    if p.atmos.numerics.dg_fddg_vertical === Val(:full_flux)
        dg_fddg_w_flux!(Yₜ, Y, p, y, volume_flux, interface, dy.ρ, ᶜuv)
    end
    if p.atmos.numerics.dg_fddg_vertical !== Val(:vi)
        dg_fddg_vertical_flux!(
            Yₜ,
            Y,
            p,
            pm,
            (u1, u2, u3),
            (eE1, eE2, eE3),
            (eN1, eN2, eN3),
        )
        return nothing
    end
    # Metric-exact KE-gradient split. The flux-form advects the physical
    # horizontal velocity ᶜuv, so it carries only gradₕ(½|uᵥ|²); the vertical
    # ᶠgradᵥ(ᶜK) uses the full ᶜK. Add the missing horizontal gradient of the
    # vertical kinetic energy, gradₕ(ᶜK − ½|uᵥ|²) = gradₕ(½w²), so uₕ and u₃ see
    # the two components of one ∇ᶜK (as in CG). Zero at rest and negligible where
    # w is small (flat); nonzero only for vertical motion over terrain. (With
    # the vector-invariant vertical term for uₕ this cancels its w ∇ₕw part;
    # the flux-form vertical transport has no such part.)
    # kg_kpep carries this gradient in its two-point gravity
    kpep && return nothing
    # The DG strong gradient is completed by its central lift, so that this
    # term pairs with the mass flux like the other gradients (exactly for
    # the divergence-form mass flux of `div_exner`).
    ᶜψ = @. ᶜK - FT(1 // 2) * LinearAlgebra.norm_sqr(ᶜuv)
    L_ψ = Operators.lifting_correction(
        Operators.central_gradient_lift,
        Geometry.UVVector{FT},
        ᶜψ,
    )
    @. Yₜ.c.uₕ -=
        C12(gradₕ(ᶜψ)) +
        C12(Geometry.transform(Geometry.Covariant12Axis(), L_ψ))
    return nothing
end

# Center field holding, at level k, the face value above (k + 1/2) or below
# (k − 1/2).
function dg_face_to_center_shift(ᶠx, ᶜspace, upper::Bool)
    ᶜy = Fields.Field(eltype(ᶠx), ᶜspace)
    for k in 1:Spaces.nlevels(ᶜspace)
        parent(Fields.level(ᶜy, k)) .=
            parent(Fields.level(ᶠx, upper ? k + half : k - half))
    end
    return ᶜy
end

# Face field equal to (ᶜa[k] + ᶜb[k + 1]) / 2 on the interior faces and zero on
# the bottom and top faces.
function dg_centers_to_interior_faces(ᶜa, ᶜb, ᶠspace)
    ᶠy = Fields.Field(eltype(ᶜa), ᶠspace)
    fill!(parent(ᶠy), 0)
    for k in 1:(Spaces.nlevels(axes(ᶜa)) - 1)
        parent(Fields.level(ᶠy, k + half)) .=
            (parent(Fields.level(ᶜa, k)) .+ parent(Fields.level(ᶜb, k + 1))) ./ 2
    end
    return ᶠy
end

# Zero a face field on the bottom and top faces.
function dg_zero_boundary_faces!(ᶠx)
    nf = Spaces.nlevels(axes(ᶠx))
    fill!(parent(Fields.level(ᶠx, half)), 0)
    fill!(parent(Fields.level(ᶠx, nf - half)), 0)
    return ᶠx
end

"""
    dg_carried_scalar_residual(volume_flux, interface, y, ᶜχa, ᶜχb)

Mass-weighted center residuals of two scalars `χa`, `χb` carried by the mass
flux of the flux-form DG family: the volume flux `F_ρ(i, i'){χ}` with `F_ρ`
the density slot of `volume_flux`, and the interface flux `F*_ρ {χ}` with
`F*_ρ` the density slot of `interface`, including its mass dissipation. This
is the transport that makes `χ` kinetic-energy consistent with the density
(the horizontal kinetic-energy lemma with `F_{ρχ} = {χ} F_ρ`).
"""
function dg_carried_scalar_residual(volume_flux, interface, y, ᶜχa, ᶜχb)
    FT = Spaces.undertype(axes(ᶜχa))
    yχ = map((yy, ca, cb) -> merge(yy, (; χa = ca, χb = cb)), y, ᶜχa, ᶜχb)
    vol(na, nb, ya, yb) =
        let F = volume_flux(na, nb, ya, yb).ρ
            (; a = F * (ya.χa + yb.χa) / 2, b = F * (ya.χb + yb.χb) / 2)
        end
    ifc(n, (ym,), (yp,)) =
        let F = interface(n, (ym,), (yp,)).ρ
            (; a = F * (ym.χa + yp.χa) / 2, b = F * (ym.χb + yp.χb) / 2)
        end
    dy = Fields.Field(NamedTuple{(:a, :b), NTuple{2, FT}}, axes(ᶜχa))
    fill!(parent(dy), 0)
    Operators.add_flux_differencing_divergence!(vol, dy, yχ)
    Operators.add_numerical_flux_interior!(ifc, dy, yχ)
    return dy
end

"""
    dg_w_horizontal_tendency(volume_flux, interface, y, ᶜdyρ, ᶠw, ᶜρWJ)

Horizontal flux-form advection of the face vertical velocity `w` on the dual
grid: the face two-point mass flux is the average of the two adjacent
center-level ones, so the face residual of `ρ_f w` is
`(R_k(w_{k+1/2}) + R_{k+1}(w_{k+1/2})) / 2` with `R_k(χ)` the carried-scalar
residual of center level `k`, and the face mass residual is the average of
the center ones, `ᶜdyρ`. Returns `ẇ` on the interior faces (zero on the
bottom and top faces) in velocity form, `(R_{ρw} − w R_ρ) / I_f(ρ WJ)`.
"""
function dg_w_horizontal_tendency(volume_flux, interface, y, ᶜdyρ, ᶠw, ᶜρWJ)
    ᶜspace = axes(ᶜdyρ)
    ᶠspace = axes(ᶠw)
    ᶜw_above = dg_face_to_center_shift(ᶠw, ᶜspace, true)
    ᶜw_below = dg_face_to_center_shift(ᶠw, ᶜspace, false)
    R = dg_carried_scalar_residual(volume_flux, interface, y, ᶜw_above, ᶜw_below)
    ᶠRρw = dg_centers_to_interior_faces(R.a, R.b, ᶠspace)
    ᶠRρ = dg_centers_to_interior_faces(ᶜdyρ, ᶜdyρ, ᶠspace)
    ᶠρWJ = dg_centers_to_interior_faces(ᶜρWJ, ᶜρWJ, ᶠspace)
    ẇ = @. (ᶠRρw - ᶠw * ᶠRρ) / ifelse(ᶠρWJ == 0, one(ᶠρWJ), ᶠρWJ)
    return dg_zero_boundary_faces!(ẇ)
end

"""
    dg_w_vertical_tendency(ᶠw, ᶠG, ᶠρJ)

Vertical flux-form advection of `w` on the dual grid, with
`G = I_f(ρJ) u³` the face mass flux of the continuity equation (zero on the
bottom and top faces): the dual-cell flux at center `k` is
`I_c(G)_k I_c(w)_k`, and
`I_f(ρJ) ẇ = −δ(I_c(G) I_c(w)) + w δ(I_c(G))`. Because
`I_f(ρJ)` evolves by `−δ(I_c(G))` under the continuity equation, this is the
two-point flux `{w} I_c(G)` on the dual grid and conserves `½ I_f(ρJ) w²`.
Returns `ẇ` on the interior faces.
"""
function dg_w_vertical_tendency(ᶠw, ᶠG, ᶠρJ)
    δ = Operators.GradientC2F(;
        bottom = Operators.SetGradient(Geometry.Covariant3Vector(0)),
        top = Operators.SetGradient(Geometry.Covariant3Vector(0)),
    )
    ᶜG = @. ᶜinterp(ᶠG)
    ᶜGw = @. ᶜG * ᶜinterp(ᶠw)
    ᶠδGw = (@. δ(ᶜGw)).components.data.:1
    ᶠδG = (@. δ(ᶜG)).components.data.:1
    ẇ = @. (ᶠw * ᶠδG - ᶠδGw) / ᶠρJ
    return dg_zero_boundary_faces!(ẇ)
end

"""
    dg_fddg_w_flux!(Yₜ, Y, p, y, volume_flux, interface, ᶜdyρ, ᶜuv)

Flux-form momentum for the vertical velocity with
`dg_fddg_vertical = full_flux`, replacing the vector-invariant advection
`ω¹² × uₕ + ∇ᵥκ` of the `u₃` equation (the Coriolis term stays):

  - the physical vertical velocity `w = u₃ / J_v` on faces is advected in
    flux form on the dual grid, horizontally by
    [`dg_w_horizontal_tendency`](@ref) and vertically by
    [`dg_w_vertical_tendency`](@ref), both with mass fluxes whose dual-cell
    divergence is the face average of the center mass tendency, so the
    advection conserves `½ ρ_f w²`;

  - on the deep-atmosphere grid, the metric pair of the momentum equation
    in a radial frame, `−w uₕ / r` on centers and `+|uₕ|² / r` on faces, with
    the face-to-center weights the adjoint of the center-to-face average, so
    the pair exchanges kinetic energy exactly.
  - a face dissipation of `w`, the `ρ_f`-weighted jump penalty
    [`shear_jump_penalty_lift`](@ref) at the shear speed `max(|{u_n}|, ε c)`
    with `ρ_f` the dual density: an exact kinetic-energy sink that conserves
    momentum and moves no mass, so the KPEP gravity pairing is unchanged and
    the removed kinetic energy is returned as internal energy. Without it
    `w` has no face dissipation, since its advection no longer exchanges
    energy with the Roe-damped `uₕ` through the vector-invariant terms.

The physical vertical acceleration `a` is applied as `u₃ += J_v a` with the
covariant horizontal compensation `uₕ += (I_c(J_v a) / J_v) e_z ⋅ e_i`, which
cancels the horizontal velocity the `u₃` increment induces through the
terrain tilt (zero on unwarped grids). The bottom and top faces are left to
the impenetrability condition.
"""
function dg_fddg_w_flux!(Yₜ, Y, p, y, volume_flux, interface, ᶜdyρ, ᶜuv)
    FT = Spaces.undertype(axes(Y.c))
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜJ = ᶜlg.J
    ᶜWJ = ᶜlg.WJ
    (; ᶠu³) = p.precomputed
    ᶜz = Fields.coordinate_field(axes(Y.c)).z
    ᶠz = Fields.coordinate_field(axes(Y.f)).z
    gradᵥ_c2f = Operators.GradientC2F(;
        bottom = Operators.SetGradient(Geometry.WVector(FT(1))),
        top = Operators.SetGradient(Geometry.WVector(FT(1))),
    )
    gradᵥ_f2c = Operators.GradientF2C()
    ᶠJv = (@. gradᵥ_c2f(ᶜz)).components.data.:1
    ᶜJv = (@. gradᵥ_f2c(ᶠz)).components.data.:1
    ᶠw = @. Y.f.u₃.components.data.:1 / ᶠJv
    ᶠρJ = @. ᶠinterp(Y.c.ρ * ᶜJ)
    ᶠG = dg_zero_boundary_faces!(@. ᶠρJ * ᶠu³.components.data.:1)

    ẇ_h = dg_w_horizontal_tendency(
        volume_flux,
        interface,
        y,
        ᶜdyρ,
        ᶠw,
        @. Y.c.ρ * ᶜWJ
    )
    ẇ = @. ẇ_h + $(dg_w_vertical_tendency(ᶠw, ᶠG, ᶠρJ))

    if !isnothing(p.core.ᶠf¹²)
        # deep-atmosphere metric pair −w uₕ / r ↔ +|uₕ|² / r
        R = FT(CAP.planet_radius(p.params))
        ᶜr = @. R + ᶜz
        ᶠρJw = dg_zero_boundary_faces!(@. ᶠρJ * ᶠw)
        ᶜs = @. ᶜinterp(ᶠρJw) / (ᶜr * Y.c.ρ * ᶜJ)
        @. Yₜ.c.uₕ -= C12(
            Geometry.transform(Geometry.Covariant12Axis(), ᶜs * ᶜuv),
        )
        ᶠcentripetal =
            dg_zero_boundary_faces!(@. ᶠinterp(LinearAlgebra.norm_sqr(ᶜuv) / ᶜr))
        @. ẇ += ᶠcentripetal
    end

    # face dissipation of w: ρ_f-weighted jump penalty at the shear speed
    # max(|{u_n}|, ε c), with ρ_f the dual density of the kinetic energy
    # ½ ρ_f w², so it is an exact kinetic-energy sink that conserves
    # momentum and moves no mass (the KPEP gravity pairing is unchanged); the
    # removed kinetic energy reappears as internal energy since E is fixed
    ε = FT(p.atmos.numerics.dg_vi_shear_floor)
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶠρ_dual = @. ᶠρJ / ᶠJ
    pen_w = Operators.lifting_correction(
        shear_jump_penalty_lift,
        FT,
        ᶠw,
        ᶠρ_dual,
        (@. ᶠinterp(ᶜuv)),
        (@. ᶠinterp($(dg_sound_speed(Y, p)))),
        one(FT),
        ε,
    )
    @. ẇ += $(dg_zero_boundary_faces!(pen_w))

    @. Yₜ.f.u₃ += Geometry.Covariant3Vector(ᶠJv * ẇ)
    # cancel the horizontal velocity the u₃ increment induces through the
    # terrain tilt
    ᶜa = @. ᶜinterp(ᶠJv * ẇ) / ᶜJv
    @. Yₜ.c.uₕ += Geometry.project(
        Geometry.Covariant12Axis(),
        Geometry.WVector(ᶜa),
        ᶜlg,
    )
    return nothing
end

"""
    dg_fddg_vertical_flux!(Yₜ, Y, p, pm, (u1, u2, u3), (eE1, eE2, eE3), (eN1, eN2, eN3))

Staggered flux-form vertical terms of the Cartesian horizontal momentum for
`dg_fddg_vertical = flux` (replacing the `ω¹² × u³` term of
`explicit_vertical_advection_tendency!`):

  - advection by the face mass flux of the continuity equation,
    `m = ᶠinterp(ρJ) u³` (zero through the surface and the top), in velocity
    form `du_c = −(δ(m {u_c}) − u_c δ(m)) / (ρJ)`, with `{u_c} = ᶠinterp(u_c)`.
    This is the two-point Kennedy-Gruber momentum flux `{ρ}{u_c}{u³}` on the
    two-point vertical stencil, so the vertical advection produces no
    spurious kinetic energy (the summand `½ u_k·u_{k+1} m_{k+½}` telescopes);
  - for `dg_volume_flux = kg_pert`, the vertical two-point pressure flux
    `du_c = −δ({J p_m ê_c³}) / (ρJ)` with `ê_c³ = ∇ξ³ ⋅ ê_c` the
    contravariant vertical component of the Cartesian unit vector. On
    unwarped grids `J ê_c³ ∝ r̂ ⋅ ê_c`, so the tangential projection of the
    term vanishes pointwise; over terrain `∇ξ³` tilts and the term carries
    the slope part of the horizontal pressure force. The horizontal two-point
    flux keeps its `{p_m}{ê_c ⋅ Ja^k}` part; the two stencils satisfy the
    metric identity only to truncation, so a terrain rest residual of that
    order remains (the vector-invariant Exner gradient has none).

The Cartesian tendencies are projected onto the tangential (E, N) frame and
converted to the covariant `uₕ` state like the horizontal flux-form terms.
"""
function dg_fddg_vertical_flux!(
    Yₜ,
    Y,
    p,
    pm,
    (u1, u2, u3),
    (eE1, eE2, eE3),
    (eN1, eN2, eN3),
)
    (; ᶠu³) = p.precomputed
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜJ = ᶜlg.J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    coords = Fields.coordinate_field(axes(Y.c))
    # face mass flux of the continuity equation (J-weighted contravariant)
    ᶠm = @. ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠu³
    ᶜdivm = @. ᶜadvdivᵥ(ᶠm)
    # advection of u_c in velocity form; the own-side mass-flux correction
    # makes this the flux-form transport (d(ρu) − u dρ)/ρ
    adv(uc) = @. (ᶜadvdivᵥ(ᶠm * ᶠinterp(uc)) - uc * ᶜdivm) / Y.c.ρ
    du1 = adv(u1)
    du2 = adv(u2)
    du3 = adv(u3)
    if p.atmos.numerics.dg_volume_flux === Val(:kg_pert)
        # Perturbation-pressure family only: with p_m = p − p_ref(Φ) the
        # tangential part of −∇p − ρ∇Φ is exactly −(∇p_m)_t, so the 3D
        # divergence of p_m ê_c needs no gravity term. The Waruszewski flux
        # carries full p with the ½ρ̂⟦φ⟧ gravity fluctuation in its horizontal
        # momentum slot; its vertical counterpart (p* plus the own-side
        # fluctuation, which cancels at hydrostatic rest) is not implemented,
        # so that family gets the flux-form advection only.
        # Boundary faces: the surface pressure force on the sloped ground is
        # the first-order (linear) extrapolation of the flux, since the
        # zero-order value misses the half-cell hydrostatic increment.
        ᶠinterp_lin = Operators.InterpolateC2F(;
            bottom = Operators.Extrapolate{1}(),
            top = Operators.Extrapolate{1}(),
        )
        # radial components of the Cartesian unit vectors
        eR1 = @. cosd(coords.lat) * cosd(coords.long)
        eR2 = @. cosd(coords.lat) * sind(coords.long)
        eR3 = @. sind(coords.lat)
        tilt(eE, eN, eR) = begin
            ê³ = @. Geometry.contravariant3(Geometry.UVWVector(eE, eN, eR), ᶜlg)
            X = @. ᶜJ * pm * ê³
            @. ᶜdivᵥ(CT3(ᶠinterp_lin(X) / ᶠJ)) / Y.c.ρ
        end
        du1 .+= tilt(eE1, eN1, eR1)
        du2 .+= tilt(eE2, eN2, eR2)
        du3 .+= tilt(eE3, eN3, eR3)
    end
    duE = @. du1 * eE1 + du2 * eE2 + du3 * eE3
    duN = @. du1 * eN1 + du2 * eN2 + du3 * eN3
    @. Yₜ.c.uₕ -= C12(
        Geometry.transform(
            Geometry.Covariant12Axis(),
            Geometry.UVVector(duE, duN),
        ),
    )
    return nothing
end
