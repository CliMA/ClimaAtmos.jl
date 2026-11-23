#=
DG interface completions for the horizontal tendencies.

Face states are carried in the local orthonormal frame (`UVVector`), which is
single-valued at shared nodes including cubed-sphere panel edges, where
covariant components are not. The face velocity is the horizontal projection
of the full `ᶜu` (terrain-consistent: it carries the u₃ tilt contribution
through g¹³/g²³, and reduces bitwise to `UVVector(uₕ)` on unwarped grids).
The vorticity liftings act on the same fields as their volume curls
(`uₕ` for ω³, `u₃` for the ω¹² part), so they are terrain-consistent as
written.
=#

import ClimaCore: Grids, Quadratures

"""
    horizontal_scalar_laplacian!(out, χ; weight)

Write the horizontal Laplacian `∇ₕ⋅(weight ∇ₕχ)` into `out`. On DG spaces, the
SIPG Laplacian with penalty `τ = weight Nq² / h_elem`, `h_elem` the smaller
element width; `Operators.scalar_laplacian!` otherwise.
"""
function horizontal_scalar_laplacian!(out, χ; weight)
    is_dg_horizontal(axes(χ)) ||
        return Operators.scalar_laplacian!(out, χ; weight)
    space = axes(χ)
    FT = Spaces.undertype(space)
    q = Operators._dg_scalar_argument(space, χ)
    q === out && (q = Operators._aliased_argument_copy(out))
    T = eltype(q)
    G_uv = Operators._laplacian_scratch_field(
        space,
        Geometry.UVVector{T},
        :scalar_laplacian_gradient,
    )
    R_uv = Operators._laplacian_scratch_field(
        space,
        Geometry.UVVector{T},
        :scalar_laplacian_lifting,
    )
    τ = Operators._laplacian_scratch_field(space, FT, :scalar_laplacian_penalty)
    hspace = Spaces.horizontal_space(space)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(hspace))
    C = FT(Nq^2)
    lg = Fields.local_geometry_field(space)
    @. τ = weight * C / element_width(lg)
    return Operators.sipg_laplacian_tendency!(
        out,
        G_uv,
        R_uv,
        q,
        weight,
        one(FT),
        τ,
    )
end

"""
    element_width(lg)

Return the smaller horizontal width [m] of the element containing the node
with local geometry `lg`.
"""
@inline function element_width(lg)
    ∂x∂ξ = lg.∂x∂ξ
    h₁ = hypot(∂x∂ξ[1, 1], ∂x∂ξ[2, 1])
    h₂ = hypot(∂x∂ξ[1, 2], ∂x∂ξ[2, 2])
    return 2 * min(h₁, h₂)
end

"""
    is_dg_horizontal(space)

Whether the horizontal part of `space` is a DG spectral-element space.
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

Central face lifting of the horizontal curl of `w r̂`, as a `UVVector`.
"""
central_curl12_lift(normal, (w⁻,), (w⁺,)) =
    ((w⁺ - w⁻) / 2) *
    Geometry.UVVector(normal.components.data.:2, -normal.components.data.:1)

"""
    rho_weighted_jump_penalty_lift(normal, (q⁻, ρ⁻, λ⁻, β⁻), (q⁺, ρ⁺, λ⁺, β⁺))

Face jump penalty ``max(λ)/2 ({ρ}{β}/ρ⁻β⁻) ⟦q⟧`` on the scalar `q`.
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

ρ-weighted face jump penalty on the horizontal velocity, at the acoustic speed
`|u_n| + c` for the normal component and the shear speed `max(|{u_n}|, ε c)` for
the tangential one. Returns a `UVVector`.
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

ρ-weighted face jump penalty on the vertical velocity `w` at the shear speed
`max(|{u_n}|, ε c)`.
"""
function shear_jump_penalty_lift(
    normal,
    (w⁻, ρ⁻, uv⁻, c⁻, β⁻, ε),
    (w⁺, ρ⁺, uv⁺, c⁺, β⁺, _),
)
    λt = dg_shear_speed(uv⁻' * normal, uv⁺' * normal, c⁻, c⁺, oftype(ρ⁻, ε))
    return λt / 2 * ((ρ⁻ + ρ⁺) / 2 / ρ⁻) * ((β⁻ + β⁺) / 2 / β⁻) * (w⁺ - w⁻)
end

face_max_lift(normal, (s⁻,), (s⁺,)) = max(s⁻, s⁺)
face_unit_lift(normal, (s⁻,), _) = one(s⁻)

"""
    dg_face_neighbor_max(ᶜs)

Return `ᶜs` with each face node raised to the larger of its value and the
value across the face (averaged over faces at element corners).
"""
function dg_face_neighbor_max(ᶜs)
    FT = Spaces.undertype(axes(ᶜs))
    ᶜs_max = Operators.lifting_correction(face_max_lift, FT, ᶜs)
    ᶜface_weight = Operators.lifting_correction(face_unit_lift, FT, ᶜs)
    return @. ifelse(ᶜface_weight > 0, ᶜs_max / ᶜface_weight, ᶜs)
end

"""
    dg_scalars_interface(normal, argvals⁻, argvals⁺)

Interface flux of (ρ, ρe_tot): central mass and enthalpy flux plus a Rusanov
penalty `−(λ/2) {ρ} ⟦e_tot⟧`.
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

Interface flux of (ρ, ρe_tot) with a penalty on the jump of the entropy
variable `1/T`.
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

Interface flux of (ρ, ρe_tot) with a penalty on the energy jump at fixed
composition, `{ρ}({c_v,m}⟦T⟧ + ⟦K⟧)`.
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

Interface flux of a tracer `ρχ`: `{ρ uv⋅n̂}{χ}` plus a Rusanov penalty
`−(λ/2) {ρ} ⟦χ⟧`.
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

Interface flux of total water whose penalty also moves the mass of the water
and the energy of its vapor, liquid, and ice phases.
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

Moist-air sound speed `sqrt(γ_m p / ρ)` [m/s], with `γ_m = c_p,m / c_v,m`.
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

Add the wave-selective face jump penalties on `uₕ` and `u₃` to `Yₜ`.
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
    dg_updraft_w_penalty!(Yₜ, Y, p)

Add a ρ-weighted face jump penalty on each updraft `w` at the shear speed
`max(|{u_n}|, ε c)`, on interior faces of DG spaces.
"""
function dg_updraft_w_penalty!(Yₜ, Y, p)
    n = n_prognostic_mass_flux_subdomains(p.atmos.turbconv_model)
    (n == 0 || !is_dg_horizontal(axes(Y.c))) && return nothing
    FT = Spaces.undertype(axes(Y.c))
    ε = FT(p.atmos.numerics.dg_vi_shear_floor)
    (ᶜuv, _) = dg_face_velocity_and_λ(Y, p)
    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶠρ = @. ᶠinterp(Y.c.ρ)
    ᶠuv = @. ᶠinterp(ᶜuv)
    ᶠc = @. ᶠinterp($(dg_sound_speed(Y, p)))
    for j in 1:n
        ᶠwʲ = @. Geometry.WVector(Y.f.sgsʲs.:($$j).u₃).components.data.:1
        pen_w = Operators.lifting_correction(
            shear_jump_penalty_lift,
            FT,
            ᶠwʲ,
            ᶠρ,
            ᶠuv,
            ᶠc,
            one(FT),
            ε,
        )
        @. Yₜ.f.sgsʲs.:($$j).u₃ +=
            C3(Geometry.WVector($(dg_zero_boundary_faces!(pen_w))), ᶠlg)
    end
    return nothing
end

"""
    dg_horizontal_dynamics_completion!(Yₜ, Y, p, t)

Add the DG interface terms of `horizontal_dynamics_tendency!`: (ρ, ρe_tot)
interface fluxes, pressure-gradient liftings, and velocity jump penalties.
No-op on CG spaces.
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
    dg_updraft_w_penalty!(Yₜ, Y, p)

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

Subtract the central face liftings of the Exner-split horizontal pressure
gradient, with `ᶜKΦ` the scalar of its plain gradient term, from `Yₜ.c.uₕ`.
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

Add the DG interface fluxes of the grid-mean tracers to `Yₜ_lim`. No-op on CG
spaces.
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
    foreach_gs_tracer(Yₜ_lim, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        if es && ρχ_name == @name(ρq_tot)
            add_dg_water_es_flux!(Yₜ_lim, ᶜρχₜ, ᶜρχ, Y, p, ᶜuv, ᶜλ)
        else
            add_dg_tracer_interface_flux!(ᶜρχₜ, ᶜρχ, Y, ᶜuv, ᶜλ)
        end
    end
    return nothing
end

"""
    dg_ρtke_interface_completion!(Yₜ, Y, p)

Add the DG interface flux of the horizontal `ρtke` advection to `Yₜ`. No-op on
CG spaces.
"""
function dg_ρtke_interface_completion!(Yₜ, Y, p)
    is_dg_horizontal(axes(Y.c)) || return nothing
    (ᶜuv, _, ᶜλ) = dg_face_velocity_and_λ(Y, p)
    add_dg_tracer_interface_flux!(Yₜ.c.ρtke, Y.c.ρtke, Y, ᶜuv, ᶜλ)
    return nothing
end

"""
    dg_fddg_tke_advection!(Yₜ, Y, p)

Add the horizontal advection of `ρtke` by the flux-form DG density flux to
`Yₜ`.
"""
function dg_fddg_tke_advection!(Yₜ, Y, p)
    ᶜWJ = Fields.local_geometry_field(Y.c).WJ
    (; y, volume_flux, interface, ᶜλ_tracer) = dg_fddg_flux_setup(Y, p)
    r = dg_carried_tracer_residual(
        volume_flux,
        interface,
        y,
        (@. Y.c.ρtke / Y.c.ρ),
        ᶜλ_tracer,
    )
    @. Yₜ.c.ρtke += r / ᶜWJ
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
    central_velocity_gradient_lift(normal, (u⁻, lg⁻), (u⁺, lg⁺))

Central face lifting `n̂ ⊗ (u⁺ − u⁻)/2` of the DG gradient of the velocity `u`,
in `UVW` components.
"""
function central_velocity_gradient_lift(normal, (u⁻, lg⁻), (u⁺, lg⁺))
    n₁ = normal.components.data.:1
    n̂ = Geometry.UVWVector(n₁, normal.components.data.:2, zero(n₁))
    return Geometry.outer(n̂, (UVW(u⁺, lg⁺) - UVW(u⁻, lg⁻)) / 2)
end

# The velocity `u` in global Cartesian components, held in a `UVWVector`.
cartesian_uvw(global_geometry, u, lg) =
    Geometry.local_to_cartesian(global_geometry, lg.coordinates) * UVW(u, lg)

# The gradient `∇u_c` of the Cartesian components, with its component axis
# rotated back to the local frame: `∇u = ∇u_c G`, `G` the local-to-Cartesian
# rotation.
local_frame_gradient(global_geometry, ∇u_c, lg) =
    ∇u_c * Geometry.local_to_cartesian(global_geometry, lg.coordinates)

# The horizontal gradient of `u` with the frame held fixed, as in
# `compute_strain_rate_center_full!`, is exact where the local frame is the
# global Cartesian one. On the sphere the frame turns with position, and the
# gradient of the Cartesian components carries that turning (the connection
# terms of the local frame, O(|u|/R)), so solid-body rotation has no strain
# and the gradient is the adjoint of `Operators.cartesian_tensor_divergence`.
add_dg_frame_turning_strain!(S, u, ::Geometry.CartesianGlobalGeometry) = S
function add_dg_frame_turning_strain!(S, u, global_geometry)
    lg = Fields.local_geometry_field(axes(S))
    axis_uvw = (Geometry.UVWAxis(),)
    @. S += symmetric_part(
        local_frame_gradient(
            global_geometry,
            Geometry.project(
                axis_uvw,
                gradₕ(cartesian_uvw(global_geometry, u, lg)),
            ),
            lg,
        ) - Geometry.project(axis_uvw, gradₕ(UVW(u))),
    )
    return S
end

"""
    complete_dg_strain_rate!(S, u)

Add the face-jump liftings of `u` (and, on the sphere, the frame turning) to
the strain rate `S`, and return `S`. No-op on CG spaces.
"""
function complete_dg_strain_rate!(S, u)
    space = axes(S)
    is_dg_horizontal(space) || return S
    lg = Fields.local_geometry_field(space)
    add_dg_frame_turning_strain!(S, u, Spaces.global_geometry(space))
    # the face kernels add the lifting to a WJ-weighted residual; the lifting of
    # the Cartesian jumps, rotated back, is the lifting of the local ones, the
    # rotation being single-valued at shared nodes
    @. S *= lg.WJ
    Operators.add_lifting_flux_interior!(
        central_velocity_gradient_lift,
        S,
        u,
        lg,
    )
    @. S = symmetric_part(S / lg.WJ)
    return S
end

"""
    dg_horizontal_stress_divergence!(Yₜ, Y, p, ᶜτ, ᶠτ, ᶠρ)

Subtract the horizontal divergence `∇ₕ⋅(ρτ)/ρ` of the center stress `ᶜτ` and
face stress `ᶠτ` from `uₕ` and `u₃` on a DG space; `ᶜτ` and `ᶠτ` are
overwritten with `ρτ`.
"""
function dg_horizontal_stress_divergence!(Yₜ, Y, p, ᶜτ, ᶠτ, ᶠρ)
    (; ᶜtemp_strain, ᶠtemp_strain, ᶜtemp_UVW, ᶠtemp_UVW) = p.scratch
    ᶜρ = Y.c.ρ
    numflux = Operators.CentralNumericalFlux(identity)
    @. ᶜτ = ᶜρ * ᶜτ
    ᶜdivρτ = Operators.cartesian_tensor_divergence!(
        ᶜtemp_UVW,
        ᶜtemp_strain,
        ᶜτ,
        Operators.tendency_completion(ᶜtemp_UVW; numflux),
    )
    @. Yₜ.c.uₕ -= C12(ᶜdivρτ / ᶜρ)
    @. ᶠτ = ᶠρ * ᶠτ
    ᶠdivρτ = Operators.cartesian_tensor_divergence!(
        ᶠtemp_UVW,
        ᶠtemp_strain,
        ᶠτ,
        Operators.tendency_completion(ᶠtemp_UVW; numflux),
    )
    # the surface and top-face w are set by the boundary conditions
    fill!(Fields.level(ᶠdivρτ, half), zero(eltype(ᶠdivρτ)))
    fill!(
        Fields.level(ᶠdivρτ, Spaces.nlevels(axes(Y.f)) - half),
        zero(eltype(ᶠdivρτ)),
    )
    @. Yₜ.f.u₃ -= C3(ᶠdivρτ / ᶠρ)
    return nothing
end

"""
    set_dg_gcl_ᶠuₕ³!(ᶠuₕ³, ᶜuₕ, ᶜρ)

Set `ᶠuₕ³`, the horizontal-velocity part of the vertical mass flux, in a
free-stream-preserving form for DG horizontal spaces over terrain.
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

Set `ᶜω³` to the DG horizontal curl of `uₕ`.
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
    dg_ω¹²_horizontal!(ᶠω¹², ᶠu₃)

Add the DG horizontal curl of `ᶠu₃` to `ᶠω¹²`.
"""
function dg_ω¹²_horizontal!(ᶠω¹², ᶠu₃)
    FT = Spaces.undertype(axes(ᶠu₃))
    @. ᶠω¹² += CT12(curlₕ(ᶠu₃))
    ᶠw_sc = @. Geometry.WVector(ᶠu₃).components.data.:1
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
# pressure-free two-point advective fluxes for (ρ, ρe, ρu⃗) with momentum in
# global Cartesian components, the Exner-split pressure force of the
# vector-invariant form (as a nodal gradient, or in the two-point form of
# `kg_kpep`), and an equation-of-state or Rusanov interface dissipation. The
# Cartesian-component momentum tendency is built in the tangential (E, N)
# frame and converted to the velocity state: duₕ = (d(ρu) − uₕ dρ)/ρ.
# ---------------------------------------------------------------------------

"""
    dg_fddg_flux_setup(Y, p)

Return the state tuple and the volume and interface flux functions of the
flux-form DG dynamics.
"""
function dg_fddg_flux_setup(Y, p)
    FT = Spaces.undertype(axes(Y.c))
    (; ᶜΦ) = p.core
    (; ᶜp, ᶜK) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    coords = Fields.coordinate_field(axes(Y.c))

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
    ᶜc = dg_sound_speed(Y, p)
    y = map(
        (
            ρ, ρe, e, p_, pm_, ρm_, uv, u1_, u2_, u3_, E1_, E2_, E3_, λ, c, φ,
            e_int, q_tot, q_liq, q_ice, λq, λqc, ψ, θ, Π,
        ) -> (;
            ρ, ρe, e, p = p_, pm = pm_, ρm = ρm_, uv,
            u1 = u1_, u2 = u2_, u3 = u3_,
            E1 = E1_, E2 = E2_, E3 = E3_, λ, c, φ,
            e_int, q_tot, q_liq, q_ice, λq, λqc, ψ, θ, Π,
        ),
        Y.c.ρ, Y.c.ρe_tot, e, ᶜp, pm, ρm, ᶜuv, u1, u2, u3, E1, E2, E3, ᶜλ, ᶜc,
        ᶜΦ, ᶜe_int, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜλq, ᶜλqc, ᶜψ_kpep,
        ᶜθ_v_diff, ᶜΠ,
    )
    # volume flux family and interface dissipation
    (; dg_volume_flux, dg_interface_flux) = p.atmos.numerics
    volume_flux =
        dg_volume_flux === Val(:div_exner) ? divergence_cartesian_flux :
        kpep ? KennedyGruberKPEPFlux(cp_d) :
        kennedy_gruber_advective_cartesian_flux
    # the gravity fluctuation of kg_kpep is not an antisymmetric face flux;
    # its face part is the lifting KPEPFaceLift below
    iface_central = kpep ? kennedy_gruber_advective_cartesian_flux : volume_flux
    moist_es = dg_interface_flux === Val(:moist_es)
    # moist_es dissipates total water with the system (6 slots) unless dry
    Nslots = moist_es && !(microphysics_model isa DryModel) ? 6 : 5
    interface = if dg_interface_flux === Val(:rusanov)
        dg_volume_flux === Val(:div_exner) ?
        CentralMinusDissipation(iface_central, rusanov_dissipation_cartesian) :
        kennedy_gruber_advective_rusanov_cartesian
    elseif moist_es
        MoistESInterface{Nslots}(iface_central, thermo_params)
    elseif dg_interface_flux === Val(:lmars)
        CentralMinusDissipation(
            iface_central,
            LMARSDissipation(FT(p.atmos.numerics.dg_lmars_floor)),
        )
    else
        RoeEOSInterface(iface_central, thermo_params)
    end
    # LMARS upwinds the carried quantities by its own interface mass flux, so
    # they get no separate tracer penalty
    ᶜλ_tracer =
        dg_interface_flux === Val(:lmars) ? (@. zero(ᶜλ_tracer)) : ᶜλ_tracer
    return (;
        eE1,
        eE2,
        eE3,
        eN1,
        eN2,
        eN3,
        E1,
        E2,
        E3,
        ᶜuv,
        ᶜλ,
        ᶜλ_tracer,
        uE,
        uN,
        u1,
        u2,
        u3,
        e,
        pm,
        y,
        kpep,
        cp_d,
        ᶜθ_v_diff,
        ᶜΠ,
        vertical_vi,
        ᶜψ_kpep,
        volume_flux,
        iface_central,
        moist_es,
        Nslots,
        interface,
    )
end

"""
    dg_carried_tracer_residual(volume_flux, mass_interface, y, ᶜχ, ᶜλχ)

Return the mass-weighted residual of `ρχ` for `χ` carried by the flux-form DG
mass flux, with a tracer penalty at the rate `ᶜλχ`.
"""
function dg_carried_tracer_residual(volume_flux, mass_interface, y, ᶜχ, ᶜλχ)
    FT = Spaces.undertype(axes(ᶜχ))
    yχ = map((yy, c, l) -> merge(yy, (; χ = c, λχ = l)), y, ᶜχ, ᶜλχ)
    vol(na, nb, ya, yb) =
        (; ρχ = volume_flux(na, nb, ya, yb).ρ * (ya.χ + yb.χ) / 2)
    ifc(n, (ym,), (yp,)) =
        let F = mass_interface(n, (ym,), (yp,)).ρ
            r =
                carried_upwind_rate(mass_interface, F, ym, yp) +
                max(ym.λχ, yp.λχ) * (ym.ρ + yp.ρ) / 2
            (; ρχ = F * (ym.χ + yp.χ) / 2 - r / 2 * (yp.χ - ym.χ))
        end
    dy = Fields.Field(NamedTuple{(:ρχ,), Tuple{FT}}, axes(ᶜχ))
    fill!(parent(dy), 0)
    Operators.add_flux_differencing_divergence!(vol, dy, yχ)
    Operators.add_numerical_flux_interior!(ifc, dy, yχ)
    return dy.ρχ
end

"""
    dg_fddg_tracer_transport!(Yₜ_lim, Y, p)

Add the horizontal transport of the grid-mean tracers by the flux-form DG
density flux to `Yₜ_lim`.
"""
function dg_fddg_tracer_transport!(Yₜ_lim, Y, p)
    ᶜWJ = Fields.local_geometry_field(Y.c).WJ
    (; y, volume_flux, interface, iface_central, Nslots, ᶜλ_tracer) =
        dg_fddg_flux_setup(Y, p)
    central(n, (ym,), (yp,)) = iface_central(n, n, ym, yp)
    ᶜλ_zero = @. zero(ᶜλ_tracer)
    foreach_gs_tracer(Yₜ_lim, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        water_in_system = Nslots == 6 && ρχ_name == @name(ρq_tot)
        r = dg_carried_tracer_residual(
            volume_flux,
            water_in_system ? central : interface,
            y,
            (@. ᶜρχ / Y.c.ρ),
            water_in_system ? ᶜλ_zero : ᶜλ_tracer,
        )
        @. ᶜρχₜ += r / ᶜWJ
    end
    return nothing
end

"""
    dg_fddg_mass_carrier(Y, p)

Return the carried-residual map `χ ↦ R(χ)` of the flux-form DG mass flux and
the density residual `R(1)`.
"""
function dg_fddg_mass_carrier(Y, p)
    (; y, volume_flux, interface, ᶜλ_tracer) = dg_fddg_flux_setup(Y, p)
    carried(ᶜχ) =
        dg_carried_tracer_residual(volume_flux, interface, y, ᶜχ, ᶜλ_tracer)
    ᶜR₁ = carried(@. one(Y.c.ρ))
    return carried, ᶜR₁
end

# Advective-form transport `-uₕ⋅∇ₕχ` of the updraft specific quantity `ᶜχ`
# into `ᶜχₜ`: `(R(χ) − χ R(1)) / ρ` with the grid-mean mass flux of the
# dynamics (the updrafts share the grid-mean horizontal velocity). It vanishes
# for uniform `χ`, and its weight is the grid-mean ρ, not the updraft ρa,
# which may be arbitrarily small.
function add_dg_fddg_advective_tendency!(ᶜχₜ, ᶜχ, carried, ᶜR₁, Y)
    ᶜWJ = Fields.local_geometry_field(Y.c).WJ
    ᶜR = carried(ᶜχ)
    @. ᶜχₜ += (ᶜR - ᶜχ * ᶜR₁) / (Y.c.ρ * ᶜWJ)
    return nothing
end

"""
    dg_fddg_sgs_dynamics!(Yₜ, Y, p)

Add the horizontal transport of the updraft `ρa` and `mse` by the flux-form DG
density flux to `Yₜ`.
"""
function dg_fddg_sgs_dynamics!(Yₜ, Y, p)
    n = n_prognostic_mass_flux_subdomains(p.atmos.turbconv_model)
    ᶜWJ = Fields.local_geometry_field(Y.c).WJ
    carried, ᶜR₁ = dg_fddg_mass_carrier(Y, p)
    for j in 1:n
        ᶜR_ρa = carried(@. Y.c.sgsʲs.:($$j).ρa / Y.c.ρ)
        @. Yₜ.c.sgsʲs.:($$j).ρa += ᶜR_ρa / ᶜWJ
        add_dg_fddg_advective_tendency!(
            Yₜ.c.sgsʲs.:($j).mse,
            Y.c.sgsʲs.:($j).mse,
            carried,
            ᶜR₁,
            Y,
        )
    end
    return nothing
end

"""
    dg_fddg_sgs_tracer_transport!(Yₜ_lim, Y, p)

Add the horizontal advection of the updraft `q_tot` and SGS tracers by the
flux-form DG density flux to `Yₜ_lim`.
"""
function dg_fddg_sgs_tracer_transport!(Yₜ_lim, Y, p)
    n = n_prognostic_mass_flux_subdomains(p.atmos.turbconv_model)
    carried, ᶜR₁ = dg_fddg_mass_carrier(Y, p)
    for j in 1:n
        add_dg_fddg_advective_tendency!(
            Yₜ_lim.c.sgsʲs.:($j).q_tot,
            Y.c.sgsʲs.:($j).q_tot,
            carried,
            ᶜR₁,
            Y,
        )
        for χ_name in sgs_tracer_names(Y)
            add_dg_fddg_advective_tendency!(
                MatrixFields.get_field(Yₜ_lim.c.sgsʲs.:($j), χ_name),
                MatrixFields.get_field(Y.c.sgsʲs.:($j), χ_name),
                carried,
                ᶜR₁,
                Y,
            )
        end
    end
    return nothing
end

"""
    dg_fddg_horizontal_dynamics!(Yₜ, Y, p, t)

Add the flux-form DG horizontal dynamics tendencies of (ρ, ρe_tot, uₕ), and of
`u₃` with `dg_fddg_vertical = full_flux`, to `Yₜ`.
"""
NVTX.@annotate function dg_fddg_horizontal_dynamics!(Yₜ, Y, p, t)
    FT = Spaces.undertype(axes(Y.c))
    (; ᶜΦ) = p.core
    (; ᶜp, ᶜK) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    coords = Fields.coordinate_field(axes(Y.c))
    ᶜWJ = Fields.local_geometry_field(Y.c).WJ

    (;
        eE1,
        eE2,
        eE3,
        eN1,
        eN2,
        eN3,
        E1,
        E2,
        E3,
        ᶜuv,
        ᶜλ,
        ᶜλ_tracer,
        uE,
        uN,
        u1,
        u2,
        u3,
        e,
        pm,
        y,
        kpep,
        cp_d,
        ᶜθ_v_diff,
        ᶜΠ,
        vertical_vi,
        ᶜψ_kpep,
        volume_flux,
        iface_central,
        moist_es,
        Nslots,
        interface,
    ) =
        dg_fddg_flux_setup(Y, p)
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
    if !kpep
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

Return the mass-weighted residuals of two scalars carried by the flux-form DG
mass flux.
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
            r = carried_upwind_rate(interface, F, ym, yp)
            (;
                a = F * (ym.χa + yp.χa) / 2 - r / 2 * (yp.χa - ym.χa),
                b = F * (ym.χb + yp.χb) / 2 - r / 2 * (yp.χb - ym.χb),
            )
        end
    dy = Fields.Field(NamedTuple{(:a, :b), NTuple{2, FT}}, axes(ᶜχa))
    fill!(parent(dy), 0)
    Operators.add_flux_differencing_divergence!(vol, dy, yχ)
    Operators.add_numerical_flux_interior!(ifc, dy, yχ)
    return dy
end

"""
    dg_w_horizontal_tendency(volume_flux, interface, y, ᶜdyρ, ᶠw, ᶜρWJ)

Return the horizontal flux-form advection tendency `ẇ` [m/s²] of the face
vertical velocity, zero on the bottom and top faces.
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

Return the vertical flux-form advection tendency `ẇ` [m/s²] of the face
vertical velocity, zero on the bottom and top faces.
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

Add the flux-form advection, deep-atmosphere metric terms, and face jump
penalty of the vertical velocity to `Yₜ`, for `dg_fddg_vertical = full_flux`.
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
    # (LMARS upwinds w in its horizontal advection instead)
    if p.atmos.numerics.dg_interface_flux !== Val(:lmars)
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
    end

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
    dg_fddg_vertical_flux!(Yₜ, Y, p, (u1, u2, u3), (eE1, eE2, eE3), (eN1, eN2, eN3))

Add the flux-form vertical transport of the horizontal momentum to `Yₜ.c.uₕ`,
for `dg_fddg_vertical = full_flux`.
"""
function dg_fddg_vertical_flux!(
    Yₜ,
    Y,
    p,
    (u1, u2, u3),
    (eE1, eE2, eE3),
    (eN1, eN2, eN3),
)
    (; ᶠu³) = p.precomputed
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    # face mass flux of the continuity equation (J-weighted contravariant)
    ᶠm = @. ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠu³
    ᶜdivm = @. ᶜadvdivᵥ(ᶠm)
    # advection of u_c in velocity form; the own-side mass-flux correction
    # makes this the flux-form transport (d(ρu) − u dρ)/ρ
    adv(uc) = @. (ᶜadvdivᵥ(ᶠm * ᶠinterp(uc)) - uc * ᶜdivm) / Y.c.ρ
    du1 = adv(u1)
    du2 = adv(u2)
    du3 = adv(u3)
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
