#####
##### Split column-physics step
#####
##### GPU-compatible (`DataLayouts.foreach_column`), `isbits`, zero-allocation
##### column-local implicit solvers for vertical diffusion, EDMF two-stream
##### transport, prognostic TKE, column water repair, and slab-ocean coupling.
#####

import StaticArrays as SA

# ============================================================================
# 1. Column-local tridiagonal Thomas solvers
# ============================================================================

"""
    column_thomas_solve!(x, rhs, diag, upper, lower, nz)

Solve `A x = rhs` in place for an `nz × nz` tridiagonal M-matrix given by
`diag` (`1:nz`), `upper` (`1:nz-1`), and `lower` (`1:nz-1`). Overwrites
`upper` and `rhs`; writes the solution to `x`.
"""
@inline function column_thomas_solve!(x, rhs, diag, upper, lower, nz)
    if nz == 1
        @inbounds x[1] = rhs[1] / diag[1]
        return nothing
    end
    @inbounds begin
        w = inv(diag[1])
        upper[1] = upper[1] * w
        rhs[1] = rhs[1] * w

        for k in 2:(nz - 1)
            denom = diag[k] - lower[k - 1] * upper[k - 1]
            w = inv(denom)
            upper[k] = upper[k] * w
            rhs[k] = (rhs[k] - lower[k - 1] * rhs[k - 1]) * w
        end

        denom = diag[nz] - lower[nz - 1] * upper[nz - 1]
        x[nz] = (rhs[nz] - lower[nz - 1] * rhs[nz - 1]) / denom
    end

    @inbounds for k in (nz - 1):-1:1
        x[k] = rhs[k] - upper[k] * x[k + 1]
    end
    return nothing
end

"""
    column_thomas_solve_2!(x1, x2, rhs1, rhs2, diag, upper, lower, nz)

Solve two tridiagonal systems sharing the same matrix `(diag, upper, lower)`
with a single forward elimination of `upper`. Overwrites `upper`, `rhs1`, and
`rhs2`; writes solutions to `x1` and `x2`.
"""
@inline function column_thomas_solve_2!(
    x1,
    x2,
    rhs1,
    rhs2,
    diag,
    upper,
    lower,
    nz,
)
    if nz == 1
        @inbounds begin
            w = inv(diag[1])
            x1[1] = rhs1[1] * w
            x2[1] = rhs2[1] * w
        end
        return nothing
    end
    @inbounds begin
        w = inv(diag[1])
        upper[1] = upper[1] * w
        rhs1[1] = rhs1[1] * w
        rhs2[1] = rhs2[1] * w

        for k in 2:(nz - 1)
            denom = diag[k] - lower[k - 1] * upper[k - 1]
            w = inv(denom)
            upper[k] = upper[k] * w
            l = lower[k - 1]
            rhs1[k] = (rhs1[k] - l * rhs1[k - 1]) * w
            rhs2[k] = (rhs2[k] - l * rhs2[k - 1]) * w
        end

        denom = diag[nz] - lower[nz - 1] * upper[nz - 1]
        w_nz = inv(denom)
        l_nz = lower[nz - 1]
        x1[nz] = (rhs1[nz] - l_nz * rhs1[nz - 1]) * w_nz
        x2[nz] = (rhs2[nz] - l_nz * rhs2[nz - 1]) * w_nz
    end

    @inbounds for k in (nz - 1):-1:1
        u_k = upper[k]
        x1[k] = rhs1[k] - u_k * x1[k + 1]
        x2[k] = rhs2[k] - u_k * x2[k + 1]
    end
    return nothing
end

# ============================================================================
# 2. Metric, conductance, and FCT helpers
# ============================================================================

"""
    face_conductance(ᶠlg_f, ᶠw_f)

Symmetric face conductance `G_f = ᶠJ[f] * ᶠg³³[f] * ᶠw_f` for face weight
`ᶠw_f` (such as `ᶠinterp(ρ) * K_h`).
"""
@inline function face_conductance(ᶠlg_f, ᶠw_f)
    return ᶠlg_f.J * parent(ᶠlg_f.gⁱʲ)[end] * ᶠw_f
end

"""
    surface_c3_flux_factor(ᶠlg_bot)

Factor `ᶠJ[1] * ᶠg³³[1]` converting a surface `C3` flux vector component
`ρ_flux.u₃` to cell-1 extensive flux, matching `ᶜdivᵥ` on flat and terrain
grids.
"""
@inline function surface_c3_flux_factor(ᶠlg_bot)
    return ᶠlg_bot.J * parent(ᶠlg_bot.gⁱʲ)[end]
end

# Extract scalar component of a Covariant3Vector or extensive face mass flux.
@inline _c3_scalar(v::Geometry.Covariant3Vector) = v.u₃
@inline _extensive_face_flux(val::Real, ᶠlg_f) = val
@inline _extensive_face_flux(val::Geometry.Contravariant3Vector, ᶠlg_f) =
    ᶠlg_f.J * val.u³
@inline _extensive_face_flux(val::Geometry.Covariant3Vector, ᶠlg_f) =
    ᶠlg_f.J * parent(ᶠlg_f.gⁱʲ)[end] * val.u₃

# Zalesak FCT ratio returning 1 when incoming/outgoing raw flux P ≤ 0.
@inline function _zalesak_ratio(P, Q)
    FT = typeof(P)
    return ifelse(P <= zero(FT), one(FT), ifelse(Q >= P, one(FT), max(zero(FT), Q / P)))
end

# 3rd-order finite-volume weights from cell thicknesses `(h1, h2, h3)` evaluated
# at the interface between cell 2 (`h2`) and cell 3 (`h3`).
@inline function _fv3_weights(h1, h2, h3)
    x0 = -(h1 + h2)
    x1 = -h2
    x3 = h3
    c0 = (x1 * x3) / ((x0 - x1) * x0 * (x0 - x3))
    c1 = (x0 * x3) / ((x1 - x0) * x1 * (x1 - x3))
    c3 = (x0 * x1) / ((x3 - x0) * (x3 - x1) * x3)
    return -c0 * h1, -(c0 + c1) * h2, c3 * h3
end

# Metric-aware 3rd-order finite-volume upwind face reconstruction (2nd-order
# thickness-weighted at boundary-adjacent faces); strictly linear in `χ`.
@inline function _face_ho_upwind3(χ, f::Int, F, ᶠz, nz::Int)
    FT = typeof(F)
    @inbounds if F >= zero(FT)
        if f >= 3
            h1 = ᶠz[f - 1] - ᶠz[f - 2]
            h2 = ᶠz[f] - ᶠz[f - 1]
            h3 = ᶠz[f + 1] - ᶠz[f]
            w1, w2, w3 = _fv3_weights(h1, h2, h3)
            return w1 * χ[f - 2] + w2 * χ[f - 1] + w3 * χ[f]
        else
            h1 = ᶠz[2] - ᶠz[1]
            h2 = ᶠz[3] - ᶠz[2]
            return (h2 * χ[1] + h1 * χ[2]) / (h1 + h2)
        end
    else
        if f <= nz - 1
            h1 = ᶠz[f + 2] - ᶠz[f + 1]
            h2 = ᶠz[f + 1] - ᶠz[f]
            h3 = ᶠz[f] - ᶠz[f - 1]
            w1, w2, w3 = _fv3_weights(h1, h2, h3)
            return w1 * χ[f + 1] + w2 * χ[f] + w3 * χ[f - 1]
        else
            hm = ᶠz[nz] - ᶠz[nz - 1]
            ht = ᶠz[nz + 1] - ᶠz[nz]
            return (ht * χ[nz - 1] + hm * χ[nz]) / (hm + ht)
        end
    end
end

# Temperature diagnosed directly from conserved cell state.
@inline function _cell_temperature(
    thermo_params,
    ::Val{HasNonEquil},
    ::Val{HasQ},
    ρ::FT,
    ρe_tot::FT,
    Φ::FT,
    K::FT,
    q_tot::FT,
    q_liq::FT,
    q_ice::FT,
) where {HasNonEquil, HasQ, FT}
    e_int = ρe_tot / max(ρ, eps(FT)) - K - Φ
    if !HasQ
        return TD.air_temperature(
            thermo_params, e_int, zero(FT), zero(FT), zero(FT),
        )
    elseif HasNonEquil
        return TD.air_temperature(thermo_params, e_int, q_tot, q_liq, q_ice)
    else
        sa = TD.saturation_adjustment(
            thermo_params, TD.ρe(), ρ, e_int, max(zero(FT), q_tot); maxiter = 4,
        )
        return sa.T
    end
end

# Thermodynamic phase partition diagnosed from cell state.
@inline function _cell_thermo_state(
    thermo_params,
    ::Val{HasNonEquil},
    ::Val{HasQ},
    ρ::FT,
    ρe_tot::FT,
    Φ::FT,
    K::FT,
    q_tot::FT,
    q_lcl::FT,
    q_icl::FT,
    q_rai::FT,
    q_sno::FT,
) where {HasNonEquil, HasQ, FT}
    e_int = ρe_tot / max(ρ, eps(FT)) - K - Φ
    if !HasQ
        T = TD.air_temperature(
            thermo_params, e_int, zero(FT), zero(FT), zero(FT),
        )
        return (; T, q_liq = zero(FT), q_ice = zero(FT))
    elseif HasNonEquil
        q_liq = max(zero(FT), q_lcl) + max(zero(FT), q_rai)
        q_ice = max(zero(FT), q_icl) + max(zero(FT), q_sno)
        T = TD.air_temperature(
            thermo_params, e_int, max(zero(FT), q_tot), q_liq, q_ice,
        )
        return (; T, q_liq, q_ice)
    else
        sa = TD.saturation_adjustment(
            thermo_params, TD.ρe(), ρ, e_int, max(zero(FT), q_tot); maxiter = 4,
        )
        return (; T = sa.T, q_liq = sa.q_liq, q_ice = sa.q_ice)
    end
end

# Donor-limit a conservative face flux `JF` against available donor mass
# `J_k * max(0, mass_density_k)` so explicit flux divergence preserves ≥ 0.
@inline function _donor_limit_face_flux!(
    JF,
    mass_density,
    ᶜJ_fv,
    Δt,
    nz,
    ::Type{FT},
) where {FT}
    ε = floatmin(FT)
    @inbounds for k in 1:nz
        out_k = zero(FT)
        if k < nz && JF[k + 1] > zero(FT)
            out_k += Δt * JF[k + 1]
        end
        if k > 1 && JF[k] < zero(FT)
            out_k += Δt * (-JF[k])
        end
        avail_k = ᶜJ_fv[k] * max(zero(FT), mass_density[k])
        if out_k > avail_k && out_k > ε
            ratio = avail_k / out_k
            if k < nz && JF[k + 1] > zero(FT)
                JF[k + 1] *= ratio
            end
            if k > 1 && JF[k] < zero(FT)
                JF[k] *= ratio
            end
        end
    end
    return nothing
end

# ============================================================================
# 3. Vertical diffusion column functors & step_vertical_diffusion_and_surface!
# ============================================================================

struct MomentumVDiffColumnSweep{DiffuseU₃, FT} <: ColumnSweep
    Δt::FT
    has_sfc::Bool
    is_standalone::Bool
end

# Physical horizontal velocity at cell `k`: the covariant basis tilts with the
# terrain, so the `UV` components combine `uₕ` with the center-interpolated
# `u₃`, as in `ᶜu = C123(ᶜuₕ) + ᶜinterp(C123(ᶠu₃))`.
@inline function _cell_physical_uv(ᶜuₕ_fv, ᶠu₃_fv, ᶜlg_fv, k)
    lg = ᶜlg_fv[k]
    ᶜu₃ = (ᶠu₃_fv[k] + ᶠu₃_fv[k + 1]) / 2
    return Geometry.UVVector(ᶜuₕ_fv[k], lg) + Geometry.UVVector(ᶜu₃, lg)
end

# Physical vertical velocity `w` from a face `u₃`.
@inline _face_physical_w(u₃, lg) = Geometry.WVector(u₃, lg).components.data[1]

"""
    (sw::MomentumVDiffColumnSweep{DiffuseU₃})(Yc_col, Yf_col, geom_col,
        weights_col, scratch_col, sfc_col)

Backward-Euler vertical diffusion of the physical horizontal velocity with face
weights `weights_col.ᶠw_u` (`ρK` at faces) and the surface stress
`sfc_col.ρ_flux_uₕ` as a Neumann flux, or as an implicit Robin drag when
`sw.is_standalone`. The increment of the physical `UV` components converts back
to `Covariant12` with the cell geometry, so `w` stays fixed on sloped
coordinates. With `DiffuseU₃`, the vertical stress `τ₃₃ = -2νₜ ∂w/∂z` then
diffuses `w` on interior faces with center weights `weights_col.ᶜw_w = ρνₜ`,
keeping `w` fixed at the top and bottom faces; the matching `uₕ` increment
keeps the physical horizontal velocity fixed.
"""
@inline function (sw::MomentumVDiffColumnSweep{DiffuseU₃})(
    Yc_col,
    Yf_col,
    geom_col,
    weights_col,
    scratch_col,
    sfc_col,
) where {DiffuseU₃}
    FT = typeof(sw.Δt)
    Δt = sw.Δt
    ε = eps(FT)

    ᶜρ_fv = Fields.field_values(Yc_col.ρ)
    ᶜuₕ_fv = Fields.field_values(Yc_col.uₕ)
    ᶠu₃_fv = Fields.field_values(Yf_col.u₃)
    ᶜlg_fv = Fields.field_values(geom_col.ᶜlg)
    ᶠlg_fv = Fields.field_values(geom_col.ᶠlg)
    ᶠw_u_fv = Fields.field_values(weights_col.ᶠw_u)

    diag_fv = Fields.field_values(scratch_col.diag)
    upper_fv = Fields.field_values(scratch_col.upper)
    lower_fv = Fields.field_values(scratch_col.lower)
    rhs1_fv = Fields.field_values(scratch_col.rhs1)
    rhs2_fv = Fields.field_values(scratch_col.rhs2)
    sol1_fv = Fields.field_values(scratch_col.sol1)
    sol2_fv = Fields.field_values(scratch_col.sol2)

    nz = Spaces.nlevels(axes(Yc_col.ρ))

    @inbounds for k in 1:nz
        J_k = ᶜlg_fv[k].J
        ρ_k = ᶜρ_fv[k]
        G_top = k < nz ? face_conductance(ᶠlg_fv[k + 1], ᶠw_u_fv[k + 1]) : zero(FT)
        G_bot = k > 1 ? face_conductance(ᶠlg_fv[k], ᶠw_u_fv[k]) : zero(FT)

        diag_fv[k] = J_k * ρ_k + Δt * (G_top + G_bot)
        if k < nz
            upper_fv[k] = -Δt * G_top
        end
        if k > 1
            lower_fv[k - 1] = -Δt * G_bot
        end
        uv_k = _cell_physical_uv(ᶜuₕ_fv, ᶠu₃_fv, ᶜlg_fv, k)
        rhs1_fv[k] = J_k * ρ_k * uv_k.components.data[1]
        rhs2_fv[k] = J_k * ρ_k * uv_k.components.data[2]
    end

    γ_drag = zero(FT)
    f_rem_1 = zero(FT)
    f_rem_2 = zero(FT)
    sfc_factor = zero(FT)

    if sw.has_sfc
        sfc_factor = surface_c3_flux_factor(ᶠlg_fv[1])
        # Physical components of the horizontal stress in cell 1.
        τ_old = Fields.field_values(sfc_col.ρ_flux_uₕ)[]
        f_old = Geometry.UVVector(
            C12(τ_old.components.data[1], τ_old.components.data[2]),
            ᶜlg_fv[1],
        )
        f_old_1 = f_old.components.data[1]
        f_old_2 = f_old.components.data[2]

        if sw.is_standalone
            uv_1 = _cell_physical_uv(ᶜuₕ_fv, ᶠu₃_fv, ᶜlg_fv, 1)
            u1_old = uv_1.components.data[1]
            u2_old = uv_1.components.data[2]
            u_sq = u1_old * u1_old + u2_old * u2_old
            dot_fu = f_old_1 * u1_old + f_old_2 * u2_old
            if u_sq > ε && dot_fu < zero(FT)
                # Implicit Robin drag linearization: f_new = f_rem - γ * u_new[1]
                γ_drag = -dot_fu / u_sq
                f_rem_1 = f_old_1 + γ_drag * u1_old
                f_rem_2 = f_old_2 + γ_drag * u2_old
                diag_fv[1] += Δt * sfc_factor * γ_drag
                rhs1_fv[1] += Δt * sfc_factor * f_rem_1
                rhs2_fv[1] += Δt * sfc_factor * f_rem_2
            else
                rhs1_fv[1] += Δt * sfc_factor * f_old_1
                rhs2_fv[1] += Δt * sfc_factor * f_old_2
            end
        else
            rhs1_fv[1] += Δt * sfc_factor * f_old_1
            rhs2_fv[1] += Δt * sfc_factor * f_old_2
        end
    end

    column_thomas_solve_2!(
        sol1_fv, sol2_fv, rhs1_fv, rhs2_fv, diag_fv, upper_fv, lower_fv, nz,
    )

    @inbounds for k in 1:nz
        uv_k = _cell_physical_uv(ᶜuₕ_fv, ᶠu₃_fv, ᶜlg_fv, k)
        Δuv = Geometry.UVVector(
            sol1_fv[k] - uv_k.components.data[1],
            sol2_fv[k] - uv_k.components.data[2],
        )
        ᶜuₕ_fv[k] = ᶜuₕ_fv[k] + C12(Δuv, ᶜlg_fv[k])
    end

    if sw.has_sfc && sw.is_standalone && γ_drag > zero(FT)
        τ_old = Fields.field_values(sfc_col.ρ_flux_uₕ)[]
        f_new = C12(
            Geometry.UVVector(
                f_rem_1 - γ_drag * sol1_fv[1],
                f_rem_2 - γ_drag * sol2_fv[1],
            ),
            ᶜlg_fv[1],
        )
        Fields.field_values(sfc_col.ρ_flux_uₕ)[] = typeof(τ_old)(
            SA.SMatrix{1, 2, FT, 2}(
                f_new.components.data[1],
                f_new.components.data[2],
            ),
            axes(τ_old),
        )
    end

    if DiffuseU₃ && nz > 1
        # Unknowns are `w` on the interior faces `f = j + 1`, `j = 1:(nz - 1)`;
        # the flux `2ρνₜ g³³ ∂w/∂ξ³` lives at centers, as in `ᶠdiffdivᵥ_u₃`.
        ᶜw_w_fv = Fields.field_values(weights_col.ᶜw_w)
        nf = nz - 1
        w_bot = _face_physical_w(ᶠu₃_fv[1], ᶠlg_fv[1])
        w_top = _face_physical_w(ᶠu₃_fv[nz + 1], ᶠlg_fv[nz + 1])
        @inbounds for j in 1:nf
            f = j + 1
            G_bot = face_conductance(ᶜlg_fv[f - 1], 2 * ᶜw_w_fv[f - 1])
            G_top = face_conductance(ᶜlg_fv[f], 2 * ᶜw_w_fv[f])
            Jρ_f = ᶠlg_fv[f].J * (ᶜρ_fv[f - 1] + ᶜρ_fv[f]) / 2
            diag_fv[j] = Jρ_f + Δt * (G_top + G_bot)
            if j < nf
                upper_fv[j] = -Δt * G_top
            end
            if j > 1
                lower_fv[j - 1] = -Δt * G_bot
            end
            rhs = Jρ_f * _face_physical_w(ᶠu₃_fv[f], ᶠlg_fv[f])
            if j == 1
                rhs += Δt * G_bot * w_bot
            end
            if j == nf
                rhs += Δt * G_top * w_top
            end
            rhs1_fv[j] = rhs
        end
        column_thomas_solve!(sol1_fv, rhs1_fv, diag_fv, upper_fv, lower_fv, nf)

        # `Δu₃` per interior face, kept in `sol2` for the `uₕ` update below.
        @inbounds for j in 1:nf
            f = j + 1
            Δw = sol1_fv[j] - _face_physical_w(ᶠu₃_fv[f], ᶠlg_fv[f])
            Δu₃ = C3(Geometry.WVector(Δw), ᶠlg_fv[f])
            ᶠu₃_fv[f] = ᶠu₃_fv[f] + Δu₃
            sol2_fv[j] = Δu₃.components.data[1]
        end
        # `C12` of the `w` increment cancels the `UV` part of `C3(Δu₃)`, so
        # the physical horizontal velocity stays fixed; it is zero on flat
        # coordinates.
        @inbounds for k in 1:nz
            Δu₃_bot = k > 1 ? sol2_fv[k - 1] : zero(FT)
            Δu₃_top = k < nz ? sol2_fv[k] : zero(FT)
            lg = ᶜlg_fv[k]
            Δw_c = Geometry.WVector(C3((Δu₃_bot + Δu₃_top) / 2), lg)
            ᶜuₕ_fv[k] = ᶜuₕ_fv[k] + C12(Δw_c, lg)
        end
    end

    return nothing
end

struct QtEffVDiffColumnSweep{HasQ, HasNonEquil, TP, FT} <: ColumnSweep
    thermo_params::TP
    Δt::FT
    has_sfc::Bool
    precipitation::Bool
end

@inline function (sw::QtEffVDiffColumnSweep{HasQ, HasNonEquil})(
    Yc_col,
    geom_col,
    weights_col,
    scratch_col,
    sfc_col,
) where {HasQ, HasNonEquil}
    FT = typeof(sw.Δt)
    Δt = sw.Δt
    tps = sw.thermo_params
    ε = eps(FT)

    ᶜρ_fv = Fields.field_values(Yc_col.ρ)
    ᶜρe_fv = Fields.field_values(Yc_col.ρe_tot)
    ᶜK_fv = Fields.field_values(geom_col.ᶜK)
    ᶜJ_fv = Fields.field_values(geom_col.ᶜJ)
    ᶠlg_fv = Fields.field_values(geom_col.ᶠlg)
    ᶜΦ_fv = Fields.field_values(geom_col.ᶜΦ)
    ᶠw_h_fv = Fields.field_values(weights_col.ᶠw_h)

    diag_fv = Fields.field_values(scratch_col.diag)
    upper_fv = Fields.field_values(scratch_col.upper)
    lower_fv = Fields.field_values(scratch_col.lower)
    rhs_fv = Fields.field_values(scratch_col.rhs)
    qt_new_fv = Fields.field_values(scratch_col.qt_new)
    jf_qt_fv = Fields.field_values(scratch_col.jf_qt)
    uv_c = scratch_col.uv_c.components.data
    cp_fv = Fields.field_values(uv_c.:1)
    hvp_fv = Fields.field_values(uv_c.:2)
    hlv_fv = Fields.field_values(uv_c.:3)
    hiv_fv = Fields.field_values(uv_c.:4)
    ql_diff_fv = Fields.field_values(uv_c.:5)
    qi_diff_fv = Fields.field_values(uv_c.:6)

    nz = Spaces.nlevels(axes(Yc_col.ρ))

    # Diagnose pre-step temperature and phase partition before ρq_tot updates.
    @inbounds for k in 1:nz
        ρ_k = ᶜρ_fv[k]
        inv_ρ = inv(max(ρ_k, ε))
        qt_k = HasQ ? Fields.field_values(Yc_col.ρq_tot)[k] * inv_ρ : zero(FT)
        ql_k = HasNonEquil ? Fields.field_values(Yc_col.ρq_lcl)[k] * inv_ρ : zero(FT)
        qi_k = HasNonEquil ? Fields.field_values(Yc_col.ρq_icl)[k] * inv_ρ : zero(FT)
        qr_k = HasNonEquil ? Fields.field_values(Yc_col.ρq_rai)[k] * inv_ρ : zero(FT)
        qs_k = HasNonEquil ? Fields.field_values(Yc_col.ρq_sno)[k] * inv_ρ : zero(FT)
        ts = _cell_thermo_state(
            tps,
            Val(HasNonEquil),
            Val(HasQ),
            ρ_k,
            ᶜρe_fv[k],
            ᶜΦ_fv[k],
            ᶜK_fv[k],
            qt_k,
            ql_k,
            qi_k,
            qr_k,
            qs_k,
        )
        T_k = ts.T

        q_t_diff = if HasQ && HasNonEquil && !sw.precipitation
            qt_k - qr_k - qs_k
        else
            qt_k
        end
        q_l_diff = if !HasQ
            zero(FT)
        elseif HasNonEquil
            sw.precipitation ? (ql_k + qr_k) : ql_k
        else
            ts.q_liq
        end
        q_i_diff = if !HasQ
            zero(FT)
        elseif HasNonEquil
            sw.precipitation ? (qi_k + qs_k) : qi_k
        else
            ts.q_ice
        end
        ql_diff_fv[k] = q_l_diff
        qi_diff_fv[k] = q_i_diff
        cp_fv[k] =
            HasQ ? TD.cp_m(tps, q_t_diff, q_l_diff, q_i_diff) : TD.Parameters.cp_d(tps)
        hvp_fv[k] = TD.enthalpy_vapor(tps, T_k) + ᶜΦ_fv[k]
        hlv_fv[k] = TD.enthalpy_liquid(tps, T_k) - TD.enthalpy_vapor(tps, T_k)
        hiv_fv[k] = TD.enthalpy_ice(tps, T_k) - TD.enthalpy_vapor(tps, T_k)
    end

    # Solve implicit tridiagonal system for q_tot_eff and update ρq_tot and ρ
    # in flux-divergence form (conserving dry-air mass ρ_d cell-wise).
    if HasQ
        ᶜρqt_fv = Fields.field_values(Yc_col.ρq_tot)
        @inbounds for k in 1:nz
            J_k = ᶜJ_fv[k]
            ρ_k = ᶜρ_fv[k]
            G_top = k < nz ? face_conductance(ᶠlg_fv[k + 1], ᶠw_h_fv[k + 1]) : zero(FT)
            G_bot = k > 1 ? face_conductance(ᶠlg_fv[k], ᶠw_h_fv[k]) : zero(FT)

            diag_fv[k] = J_k * ρ_k + Δt * (G_top + G_bot)
            if k < nz
                upper_fv[k] = -Δt * G_top
            end
            if k > 1
                lower_fv[k - 1] = -Δt * G_bot
            end
            ρq_nd = if HasNonEquil && !sw.precipitation
                Fields.field_values(Yc_col.ρq_rai)[k] +
                Fields.field_values(Yc_col.ρq_sno)[k]
            else
                zero(FT)
            end
            rhs_fv[k] = J_k * (ᶜρqt_fv[k] - ρq_nd)
        end

        JF_sfc_q = zero(FT)
        if sw.has_sfc
            sfc_factor = surface_c3_flux_factor(ᶠlg_fv[1])
            ρ_flux_q = _c3_scalar(Fields.field_values(sfc_col.ρ_flux_q_tot)[])
            JF_sfc_q = sfc_factor * ρ_flux_q
            # Strong prescribed Neumann dew can make rhs_fv[1] negative before
            # repair_column_water_conservatively! restores column positivity.
            rhs_fv[1] += Δt * JF_sfc_q
        end

        column_thomas_solve!(qt_new_fv, rhs_fv, diag_fv, upper_fv, lower_fv, nz)

        @inbounds begin
            jf_qt_fv[1] = JF_sfc_q
            for f in 2:nz
                G_f = face_conductance(ᶠlg_fv[f], ᶠw_h_fv[f])
                jf_qt_fv[f] = -G_f * (qt_new_fv[f] - qt_new_fv[f - 1])
            end
            jf_qt_fv[nz + 1] = zero(FT)

            for k in 1:nz
                Δρq_k = -(Δt / ᶜJ_fv[k]) * (jf_qt_fv[k + 1] - jf_qt_fv[k])
                ᶜρqt_fv[k] += Δρq_k
                ᶜρ_fv[k] += Δρq_k
            end
        end
    else
        @inbounds for f in 1:(nz + 1)
            jf_qt_fv[f] = zero(FT)
        end
    end

    return nothing
end

struct EnthalpyTVDiffColumnSweep{HasQ, HasNonEquil, TP, FT} <: ColumnSweep
    thermo_params::TP
    Δt::FT
    has_sfc::Bool
    blend::Bool
end

@inline function (sw::EnthalpyTVDiffColumnSweep{HasQ, HasNonEquil})(
    Yc_col,
    geom_col,
    weights_col,
    scratch_col,
    sfc_col,
) where {HasQ, HasNonEquil}
    FT = typeof(sw.Δt)
    Δt = sw.Δt
    tps = sw.thermo_params
    ε = eps(FT)

    ᶜρ_fv = Fields.field_values(Yc_col.ρ)
    ᶜρe_fv = Fields.field_values(Yc_col.ρe_tot)
    ᶜK_fv = Fields.field_values(geom_col.ᶜK)
    ᶜJ_fv = Fields.field_values(geom_col.ᶜJ)
    ᶠlg_fv = Fields.field_values(geom_col.ᶠlg)
    ᶜΦ_fv = Fields.field_values(geom_col.ᶜΦ)
    ᶠw_h_fv = Fields.field_values(weights_col.ᶠw_h)

    diag_fv = Fields.field_values(scratch_col.diag)
    upper_fv = Fields.field_values(scratch_col.upper)
    lower_fv = Fields.field_values(scratch_col.lower)
    rhs_fv = Fields.field_values(scratch_col.rhs)
    sol_fv = Fields.field_values(scratch_col.sol)
    jf_qt_fv = Fields.field_values(scratch_col.jf_qt)
    uv_c = scratch_col.uv_c.components.data
    cp_fv = Fields.field_values(uv_c.:1)
    hvp_fv = Fields.field_values(uv_c.:2)
    hlv_fv = Fields.field_values(uv_c.:3)
    hiv_fv = Fields.field_values(uv_c.:4)
    ql_diff_fv = Fields.field_values(uv_c.:5)
    qi_diff_fv = Fields.field_values(uv_c.:6)
    uv_f = scratch_col.uv_f.components.data
    jf_sl_fv = Fields.field_values(uv_f.:1)
    jf_ql_fv = Fields.field_values(uv_f.:2)
    jf_qi_fv = Fields.field_values(uv_f.:3)
    jf_w_fv = Fields.field_values(uv_f.:4)

    nz = Spaces.nlevels(axes(Yc_col.ρ))

    # Build unique water-enthalpy face fluxes once per face for bitwise
    # telescoping. Unblended non-equilibrium microphysics uses the realized
    # implicit condensate face fluxes jf_ql/jf_qi. Equilibrium microphysics and
    # the blended cloud-species flux use the gradients of ql_diff/qi_diff, so
    # that the energy flux is `F_sl + (h_v + Φ) F_qt` for every blend weight:
    # a parcel conserves `(s_l, q_t)` in transit whatever its condensate does.
    @inbounds begin
        jf_w_fv[1] = zero(FT)
        for f in 2:nz
            if HasQ
                hvp_f = (hvp_fv[f - 1] + hvp_fv[f]) / 2
                hlv_f = (hlv_fv[f - 1] + hlv_fv[f]) / 2
                hiv_f = (hiv_fv[f - 1] + hiv_fv[f]) / 2
                jf_cond = if HasNonEquil && !sw.blend
                    hlv_f * jf_ql_fv[f] + hiv_f * jf_qi_fv[f]
                else
                    G_f = face_conductance(ᶠlg_fv[f], ᶠw_h_fv[f])
                    -G_f * (
                        hlv_f * (ql_diff_fv[f] - ql_diff_fv[f - 1]) +
                        hiv_f * (qi_diff_fv[f] - qi_diff_fv[f - 1])
                    )
                end
                jf_w_fv[f] = hvp_f * jf_qt_fv[f] + jf_cond
            else
                jf_w_fv[f] = zero(FT)
            end
        end
        jf_w_fv[nz + 1] = zero(FT)
    end

    @inbounds for k in 1:nz
        J_k = ᶜJ_fv[k]
        ρ_new_k = ᶜρ_fv[k]
        inv_ρ_new = inv(max(ρ_new_k, ε))
        qt_new_full = HasQ ? Fields.field_values(Yc_col.ρq_tot)[k] * inv_ρ_new : zero(FT)
        ql_k = HasNonEquil ? Fields.field_values(Yc_col.ρq_lcl)[k] * inv_ρ_new : zero(FT)
        qi_k = HasNonEquil ? Fields.field_values(Yc_col.ρq_icl)[k] * inv_ρ_new : zero(FT)
        qr_k = HasNonEquil ? Fields.field_values(Yc_col.ρq_rai)[k] * inv_ρ_new : zero(FT)
        qs_k = HasNonEquil ? Fields.field_values(Yc_col.ρq_sno)[k] * inv_ρ_new : zero(FT)

        ρe_mid_k = ᶜρe_fv[k] - (Δt / J_k) * (jf_w_fv[k + 1] - jf_w_fv[k])
        ᶜρe_fv[k] = ρe_mid_k

        ts_mid = _cell_thermo_state(
            tps,
            Val(HasNonEquil),
            Val(HasQ),
            ρ_new_k,
            ρe_mid_k,
            ᶜΦ_fv[k],
            ᶜK_fv[k],
            qt_new_full,
            ql_k,
            qi_k,
            qr_k,
            qs_k,
        )
        T_star_k = ts_mid.T
        cv_k =
            HasQ ? TD.cv_m(tps, qt_new_full, ts_mid.q_liq, ts_mid.q_ice) :
            TD.Parameters.cv_d(tps)

        G_top = k < nz ? face_conductance(ᶠlg_fv[k + 1], ᶠw_h_fv[k + 1]) : zero(FT)
        G_bot = k > 1 ? face_conductance(ᶠlg_fv[k], ᶠw_h_fv[k]) : zero(FT)
        cp_top = k < nz ? (cp_fv[k] + cp_fv[k + 1]) / 2 : zero(FT)
        cp_bot = k > 1 ? (cp_fv[k - 1] + cp_fv[k]) / 2 : zero(FT)

        diag_fv[k] = J_k * ρ_new_k * cv_k + Δt * (G_top * cp_top + G_bot * cp_bot)
        if k < nz
            upper_fv[k] = -Δt * G_top * cp_top
        end
        if k > 1
            lower_fv[k - 1] = -Δt * G_bot * cp_bot
        end
        ΔΦ_top = k < nz ? (ᶜΦ_fv[k + 1] - ᶜΦ_fv[k]) : zero(FT)
        ΔΦ_bot = k > 1 ? (ᶜΦ_fv[k] - ᶜΦ_fv[k - 1]) : zero(FT)
        rhs_fv[k] =
            J_k * ρ_new_k * cv_k * T_star_k +
            Δt * (G_top * ΔΦ_top - G_bot * ΔΦ_bot)
    end

    JF_sfc_h = zero(FT)
    if sw.has_sfc
        sfc_factor = surface_c3_flux_factor(ᶠlg_fv[1])
        ρ_flux_h = _c3_scalar(Fields.field_values(sfc_col.ρ_flux_h_tot)[])
        JF_sfc_h = sfc_factor * ρ_flux_h
        rhs_fv[1] += Δt * JF_sfc_h
    end

    column_thomas_solve!(sol_fv, rhs_fv, diag_fv, upper_fv, lower_fv, nz)

    @inbounds begin
        jf_w_fv[1] = JF_sfc_h
        for f in 2:nz
            G_f = face_conductance(ᶠlg_fv[f], ᶠw_h_fv[f])
            cp_f = (cp_fv[f - 1] + cp_fv[f]) / 2
            jf_w_fv[f] =
                -G_f * (cp_f * (sol_fv[f] - sol_fv[f - 1]) + (ᶜΦ_fv[f] - ᶜΦ_fv[f - 1]))
        end
        jf_w_fv[nz + 1] = zero(FT)

        if HasNonEquil && sw.blend
            jf_sl_fv[1] = zero(FT)
            for f in 2:nz
                G_f = face_conductance(ᶠlg_fv[f], ᶠw_h_fv[f])
                jf_sl_fv[f] =
                    jf_w_fv[f] -
                    G_f * (
                        (hlv_fv[f - 1] + hlv_fv[f]) / 2 *
                        (ql_diff_fv[f] - ql_diff_fv[f - 1]) +
                        (hiv_fv[f - 1] + hiv_fv[f]) / 2 *
                        (qi_diff_fv[f] - qi_diff_fv[f - 1])
                    )
            end
            jf_sl_fv[nz + 1] = zero(FT)
        end

        for k in 1:nz
            ᶜρe_fv[k] -= (Δt / ᶜJ_fv[k]) * (jf_w_fv[k + 1] - jf_w_fv[k])
        end
    end

    return nothing
end

struct PassiveScalarVDiffColumnSweep{AccumFlux, FT} <: ColumnSweep
    Δt::FT
end

@inline function (sw::PassiveScalarVDiffColumnSweep{AccumFlux})(
    ᶜρχ_col,
    Yc_col,
    geom_col,
    weights_col,
    scratch_col,
    jf_acc_col,
) where {AccumFlux}
    FT = typeof(sw.Δt)
    Δt = sw.Δt
    ε = eps(FT)

    ᶜρχ_fv = Fields.field_values(ᶜρχ_col)
    ᶜρ_fv = Fields.field_values(Yc_col.ρ)
    ᶜJ_fv = Fields.field_values(geom_col.ᶜJ)
    ᶠlg_fv = Fields.field_values(geom_col.ᶠlg)
    ᶠw_h_fv = Fields.field_values(weights_col.ᶠw_h)

    diag_fv = Fields.field_values(scratch_col.diag)
    upper_fv = Fields.field_values(scratch_col.upper)
    lower_fv = Fields.field_values(scratch_col.lower)
    rhs_fv = Fields.field_values(scratch_col.rhs)
    sol_fv = Fields.field_values(scratch_col.sol)
    jf_acc_fv = Fields.field_values(jf_acc_col)

    nz = Spaces.nlevels(axes(Yc_col.ρ))

    @inbounds for k in 1:nz
        J_k = ᶜJ_fv[k]
        ρ_k = ᶜρ_fv[k]
        G_top = k < nz ? face_conductance(ᶠlg_fv[k + 1], ᶠw_h_fv[k + 1]) : zero(FT)
        G_bot = k > 1 ? face_conductance(ᶠlg_fv[k], ᶠw_h_fv[k]) : zero(FT)
        diag_fv[k] = J_k * ρ_k + Δt * (G_top + G_bot)
        if k < nz
            upper_fv[k] = -Δt * G_top
        end
        if k > 1
            lower_fv[k - 1] = -Δt * G_bot
        end
        rhs_fv[k] = J_k * ᶜρχ_fv[k]
    end

    column_thomas_solve!(sol_fv, rhs_fv, diag_fv, upper_fv, lower_fv, nz)

    @inbounds for k in 1:nz
        JF_top = if k < nz
            G_top = face_conductance(ᶠlg_fv[k + 1], ᶠw_h_fv[k + 1])
            flux = -G_top * (sol_fv[k + 1] - sol_fv[k])
            if AccumFlux
                jf_acc_fv[k + 1] += flux
            end
            flux
        else
            zero(FT)
        end
        JF_bot = if k > 1
            G_bot = face_conductance(ᶠlg_fv[k], ᶠw_h_fv[k])
            -G_bot * (sol_fv[k] - sol_fv[k - 1])
        else
            zero(FT)
        end
        ᶜρχ_fv[k] -= (Δt / ᶜJ_fv[k]) * (JF_top - JF_bot)
    end
    return nothing
end

struct BlendedCloudDiffColumnSweep{IsLiq, TP, CMP, FT} <: ColumnSweep
    thermo_params::TP
    cmp::CMP
    c_d::FT
    τ_const::FT
    Δt::FT
    use_const_τ::Bool
end

@inline function (sw::BlendedCloudDiffColumnSweep{IsLiq})(
    ᶜρχ_col,
    Yc_col,
    geom_col,
    pre_col,
    weights_col,
    scratch_col,
) where {IsLiq}
    FT = typeof(sw.Δt)
    Δt = sw.Δt
    ε = eps(FT)
    phase = IsLiq ? TD.Liquid() : TD.Ice()

    ᶜρχ_fv = Fields.field_values(ᶜρχ_col)
    ᶜρ_fv = Fields.field_values(Yc_col.ρ)
    ᶜρtke_fv = Fields.field_values(Yc_col.ρtke)
    ᶜρqt_fv = Fields.field_values(Yc_col.ρq_tot)
    ᶜρql_fv = Fields.field_values(Yc_col.ρq_lcl)
    ᶜρqi_fv = Fields.field_values(Yc_col.ρq_icl)

    ᶜJ_fv = Fields.field_values(geom_col.ᶜJ)
    ᶠlg_fv = Fields.field_values(geom_col.ᶠlg)
    ᶜT_fv = Fields.field_values(pre_col.ᶜT)
    ᶜp_fv = Fields.field_values(pre_col.ᶜp)
    ᶜl_mix_fv = Fields.field_values(pre_col.ᶜl_mix)
    ᶜmom_fv = Fields.field_values(pre_col.ᶜsgs_moments)
    ᶠw_h_fv = Fields.field_values(weights_col.ᶠw_h)

    diag_fv = Fields.field_values(scratch_col.diag)
    upper_fv = Fields.field_values(scratch_col.upper)
    lower_fv = Fields.field_values(scratch_col.lower)
    rhs_fv = Fields.field_values(scratch_col.rhs)
    sol_fv = Fields.field_values(scratch_col.sol)
    λ_fv = Fields.field_values(scratch_col.λ_col)
    cqt_fv = Fields.field_values(scratch_col.cqt_col)
    jf_qt_fv = Fields.field_values(scratch_col.jf_qt)
    uv_c = scratch_col.uv_c.components.data
    csl_fv = Fields.field_values(uv_c.:7)
    uv_f = scratch_col.uv_f.components.data
    jf_sl_fv = Fields.field_values(uv_f.:1)
    jf_cg_fv = Fields.field_values(uv_f.:5)

    nz = Spaces.nlevels(axes(Yc_col.ρ))

    @inbounds for k in 1:nz
        ρ_k = ᶜρ_fv[k]
        inv_ρ = inv(max(ρ_k, ε))
        qt_k = max(zero(FT), ᶜρqt_fv[k] * inv_ρ)
        ql_k = max(zero(FT), ᶜρql_fv[k] * inv_ρ)
        qi_k = max(zero(FT), ᶜρqi_fv[k] * inv_ρ)
        T_k = ᶜT_fv[k]
        p_k = ᶜp_fv[k]
        ρ_air = TD.air_density(sw.thermo_params, T_k, p_k, qt_k, ql_k, qi_k)
        tke_k = max(zero(FT), ᶜρtke_fv[k] * inv_ρ)
        l_k = ᶜl_mix_fv[k]
        τ_μ = if sw.use_const_τ
            sw.τ_const
        else
            cloud_species_relaxation_time(
                sw.cmp, sw.thermo_params, T_k, ρ_air, qt_k, ql_k, qi_k, phase,
            )
        end
        λ_fv[k] = cloud_species_flux_blend(l_k, tke_k, sw.c_d, τ_μ)
        mom_k = ᶜmom_fv[k]
        c_qt_k, c_sl_k = equilibrium_condensate_sensitivities(
            sw.thermo_params,
            T_k,
            ρ_air,
            qt_k,
            ql_k,
            qi_k,
            mom_k.sigma_S,
            mom_k.λ_eq,
            phase,
        )
        cqt_fv[k] = c_qt_k
        csl_fv[k] = c_sl_k
    end

    # Raw counter-gradient face flux JF_cg[f], then donor-limited so RHS ≥ 0.
    @inbounds begin
        jf_cg_fv[1] = zero(FT)
        for f in 2:nz
            λ_f = (λ_fv[f - 1] + λ_fv[f]) / 2
            c_qt_f = (cqt_fv[f - 1] + cqt_fv[f]) / 2
            c_sl_f = (csl_fv[f - 1] + csl_fv[f]) / 2
            jf_cg_fv[f] = λ_f * (c_qt_f * jf_qt_fv[f] + c_sl_f * jf_sl_fv[f])
        end
        jf_cg_fv[nz + 1] = zero(FT)
    end
    _donor_limit_face_flux!(jf_cg_fv, ᶜρχ_fv, ᶜJ_fv, Δt, nz, FT)

    @inbounds for k in 1:nz
        J_k = ᶜJ_fv[k]
        ρ_k = ᶜρ_fv[k]
        G_top = if k < nz
            λ_top = (λ_fv[k] + λ_fv[k + 1]) / 2
            (one(FT) - λ_top) * face_conductance(ᶠlg_fv[k + 1], ᶠw_h_fv[k + 1])
        else
            zero(FT)
        end
        G_bot = if k > 1
            λ_bot = (λ_fv[k - 1] + λ_fv[k]) / 2
            (one(FT) - λ_bot) * face_conductance(ᶠlg_fv[k], ᶠw_h_fv[k])
        else
            zero(FT)
        end
        diag_fv[k] = J_k * ρ_k + Δt * (G_top + G_bot)
        if k < nz
            upper_fv[k] = -Δt * G_top
        end
        if k > 1
            lower_fv[k - 1] = -Δt * G_bot
        end
        rhs_fv[k] = J_k * ᶜρχ_fv[k] - Δt * (jf_cg_fv[k + 1] - jf_cg_fv[k])
    end

    column_thomas_solve!(sol_fv, rhs_fv, diag_fv, upper_fv, lower_fv, nz)

    # Rebuild realized face fluxes `(1 - λ) (-G Δsol) + jf_cg` and commit the
    # species mass in exact flux-divergence form. The energy flux
    # `F_sl + (h_v + Φ) F_qt` of `EnthalpyTVDiffColumnSweep` is independent of
    # the blend, so `ρe_tot` stays unchanged here.
    @inbounds begin
        for f in 2:nz
            λ_f = (λ_fv[f - 1] + λ_fv[f]) / 2
            G_eff_f = (one(FT) - λ_f) * face_conductance(ᶠlg_fv[f], ᶠw_h_fv[f])
            jf_cg_fv[f] = -G_eff_f * (sol_fv[f] - sol_fv[f - 1]) + jf_cg_fv[f]
        end
        for k in 1:nz
            ᶜρχ_fv[k] -= (Δt / ᶜJ_fv[k]) * (jf_cg_fv[k + 1] - jf_cg_fv[k])
        end
    end
    return nothing
end

# Apply vertical diffusion to each entry in `species` (`diffused_water_species(precipitation)`).
function _diffuse_water_species!(
    Y,
    p,
    species,
    blend::Bool,
    thermo_params,
    Δt_FT,
    passive_geom,
    passive_scratch,
    weights,
    jf_ql_tot,
    jf_qi_tot,
    jf_dummy,
)
    FT = typeof(Δt_FT)
    sw_passive_accum = PassiveScalarVDiffColumnSweep{true, FT}(Δt_FT)
    sw_passive_noacc = PassiveScalarVDiffColumnSweep{false, FT}(Δt_FT)

    if blend
        cmp = CAP.cloud_formation_1m_params(p.params)
        c_d = tke_dissipation_coefficient(CAP.turbconv_params(p.params))
        use_const_τ = CAP.sgs_cloud_species_constant_timescales(p.params) > 0
        blend_pre = (;
            p.precomputed.ᶜT,
            p.precomputed.ᶜp,
            p.precomputed.ᶜl_mix,
            p.precomputed.ᶜsgs_moments,
        )
        blend_scratch = (;
            diag = p.scratch.ᶜtemp_scalar_3,
            upper = p.scratch.ᶜtemp_scalar_4,
            lower = p.scratch.ᶜtemp_scalar_5,
            rhs = p.scratch.ᶜtemp_scalar_6,
            sol = p.scratch.ᶜtemp_scalar_7,
            λ_col = p.scratch.ᶜtemp_scalar,
            cqt_col = p.scratch.ᶜtemp_scalar_2,
            jf_qt = p.scratch.ᶠtemp_UVW.components.data.:1,
            uv_c = p.scratch.ᶜtemp_UVWxUVW,
            uv_f = p.scratch.ᶠtemp_UVWxUVW,
        )
        MatrixFields.unrolled_foreach(species) do (χ_name, ρχ_name)
            MatrixFields.has_field(Y.c, ρχ_name) || return nothing
            ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
            phase = cloud_species_phase(χ_name)
            if !isnothing(phase)
                is_liq = phase isa TD.Liquid
                τ_const =
                    is_liq ?
                    CAP.condensation_evaporation_timescale(p.params) :
                    CAP.sublimation_deposition_timescale(p.params)
                sw_blend = BlendedCloudDiffColumnSweep{
                    is_liq,
                    typeof(thermo_params),
                    typeof(cmp),
                    FT,
                }(
                    thermo_params,
                    cmp,
                    c_d,
                    τ_const,
                    Δt_FT,
                    use_const_τ,
                )
                DataLayouts.foreach_column(
                    sw_blend,
                    ᶜρχ,
                    Y.c,
                    passive_geom,
                    blend_pre,
                    weights,
                    blend_scratch;
                    shmem = Val(:none),
                )
            else
                DataLayouts.foreach_column(
                    sw_passive_noacc,
                    ᶜρχ,
                    Y.c,
                    passive_geom,
                    weights,
                    passive_scratch,
                    jf_dummy;
                    shmem = Val(:none),
                )
            end
            return nothing
        end
    else
        MatrixFields.unrolled_foreach(species) do (χ_name, ρχ_name)
            MatrixFields.has_field(Y.c, ρχ_name) || return nothing
            ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
            if χ_name == @name(q_lcl) || χ_name == @name(q_rai)
                DataLayouts.foreach_column(
                    sw_passive_accum,
                    ᶜρχ,
                    Y.c,
                    passive_geom,
                    weights,
                    passive_scratch,
                    jf_ql_tot;
                    shmem = Val(:none),
                )
            elseif χ_name == @name(q_icl) || χ_name == @name(q_sno)
                DataLayouts.foreach_column(
                    sw_passive_accum,
                    ᶜρχ,
                    Y.c,
                    passive_geom,
                    weights,
                    passive_scratch,
                    jf_qi_tot;
                    shmem = Val(:none),
                )
            else
                DataLayouts.foreach_column(
                    sw_passive_noacc,
                    ᶜρχ,
                    Y.c,
                    passive_geom,
                    weights,
                    passive_scratch,
                    jf_dummy;
                    shmem = Val(:none),
                )
            end
            return nothing
        end
    end
    return nothing
end

"""
    step_vertical_diffusion_and_surface!(Y, p, t, Δt)

Implicit column-local vertical diffusion of momentum, moisture, enthalpy,
suspended cloud species, and passive tracers, with surface boundary conditions.
Handles `VerticalDiffusion`, `DecayWithHeightDiffusion`, vertical
`SmagorinskyLilly`, `EDOnlyEDMFX`, and surface-only flux configurations when
no diffusion closure is active. Skips `PrognosticEDMFX`.
"""
function step_vertical_diffusion_and_surface!(Y, p, t, Δt)
    (; turbconv_model, vertical_diffusion, smagorinsky_lilly) = p.atmos
    turbconv_model isa PrognosticEDMFX && return nothing

    has_edonly =
        turbconv_model isa EDOnlyEDMFX && p.atmos.edmfx_model.sgs_diffusive_flux
    has_vdiff =
        vertical_diffusion isa Union{VerticalDiffusion, DecayWithHeightDiffusion}
    has_smag =
        smagorinsky_lilly isa SmagorinskyLilly &&
        is_smagorinsky_vertical(smagorinsky_lilly)
    has_diff = has_edonly || has_vdiff || has_smag
    has_sfc = !p.atmos.disable_surface_flux_tendency
    (has_diff || has_sfc) || return nothing

    FT = Spaces.undertype(axes(Y.c))
    ε_FT = eps(FT)
    Δt_FT = FT(Δt)

    ᶠw_h = p.scratch.ᶠtemp_scalar
    ᶠw_u = p.scratch.ᶠtemp_scalar_2
    fill!(ᶠw_h, zero(FT))
    fill!(ᶠw_u, zero(FT))
    precipitation = false

    if has_edonly
        @. ᶠw_h += ᶠinterp(Y.c.ρ) * p.precomputed.ᶠK_h
        @. ᶠw_u += ᶠinterp(Y.c.ρ) * p.precomputed.ᶠK_u
    elseif has_vdiff
        ᶜK_h = p.scratch.ᶜtemp_scalar
        if vertical_diffusion isa DecayWithHeightDiffusion
            ᶜK_h .= ᶜcompute_eddy_diffusivity_coefficient(
                Y.c.ρ, vertical_diffusion,
            )
        else
            ᶜK_h .= ᶜcompute_eddy_diffusivity_coefficient(
                Y.c.uₕ, p.precomputed.ᶜp, vertical_diffusion,
            )
        end
        @. ᶠw_h += ᶠinterp(Y.c.ρ) / ᶠinterp(1 / max(ᶜK_h, ε_FT))
        if !disable_momentum_vertical_diffusion(vertical_diffusion)
            @. ᶠw_u += ᶠw_h
        end
    end

    if has_smag
        ᶜνₜ_v = p.precomputed.ᶜνₜ_v
        Pr_t = CAP.Prandtl_number_0(CAP.turbconv_params(p.params))
        @. ᶠw_u += ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜνₜ_v)
        @. ᶠw_h += ᶠinterp(Y.c.ρ) * ᶠinterp(ᶜνₜ_v) / Pr_t
        precipitation = true
    end

    is_standalone = p.atmos.surface.flux_scheme !== nothing
    sfc_conditions = p.precomputed.sfc_conditions

    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠlg = Fields.local_geometry_field(Y.f)
    weights = (; ᶠw_h, ᶠw_u)

    # ── Momentum diffusion (column-local tridiagonal + Robin/Neumann BC) ──
    # The vertical Smagorinsky-Lilly stress also diffuses `w` on faces, and it
    # acts on momentum also when the vertical diffusion model diffuses scalars
    # only.
    if has_smag || !disable_momentum_vertical_diffusion(vertical_diffusion)
        ᶜlg = Fields.local_geometry_field(Y.c)
        mom_geom = (; ᶜlg, ᶠlg)
        mom_scratch = (;
            diag = p.scratch.ᶜtemp_scalar_3,
            upper = p.scratch.ᶜtemp_scalar_4,
            lower = p.scratch.ᶜtemp_scalar_5,
            rhs1 = p.scratch.ᶜtemp_scalar_6,
            rhs2 = p.scratch.ᶜtemp_scalar_7,
            sol1 = p.scratch.ᶜtemp_UVWxUVW.components.data.:1,
            sol2 = p.scratch.ᶜtemp_UVWxUVW.components.data.:2,
        )
        if has_smag
            ᶜw_w = p.scratch.ᶜtemp_scalar_2
            @. ᶜw_w = Y.c.ρ * p.precomputed.ᶜνₜ_v
            mom_weights = (; ᶠw_u, ᶜw_w)
        else
            mom_weights = (; ᶠw_u)
        end
        # The surface stress acts only where momentum vertical diffusion is on;
        # the Smagorinsky-Lilly stress alone has a zero-flux bottom boundary.
        has_sfc_mom =
            has_sfc && !disable_momentum_vertical_diffusion(vertical_diffusion)
        sw_mom = MomentumVDiffColumnSweep{has_smag, FT}(
            Δt_FT, has_sfc_mom, is_standalone,
        )
        DataLayouts.foreach_column(
            sw_mom,
            Y.c,
            Y.f,
            mom_geom,
            mom_weights,
            mom_scratch,
            sfc_conditions;
            shmem = Val(:none),
        )
        p.precomputed.ᶜK .= compute_kinetic(Y.c.uₕ, Y.f.u₃)
    end

    # ── Moisture + enthalpy sequential solve ──
    thermo_params = CAP.thermodynamics_params(p.params)
    has_q_tot = MatrixFields.has_field(Y.c, @name(ρq_tot))
    has_nonequil =
        p.atmos.microphysics_model isa NonEquilibriumMicrophysics &&
        MatrixFields.has_field(Y.c, @name(ρq_lcl))
    blend = has_diff && blended_cloud_species_flux(Y, p)

    scal_geom = (; ᶜJ, ᶠlg, p.core.ᶜΦ, p.precomputed.ᶜK)
    scal_scratch = (;
        diag = p.scratch.ᶜtemp_scalar_3,
        upper = p.scratch.ᶜtemp_scalar_4,
        lower = p.scratch.ᶜtemp_scalar_5,
        rhs = p.scratch.ᶜtemp_scalar_6,
        sol = p.scratch.ᶜtemp_scalar_7,
        qt_new = p.scratch.ᶜtemp_scalar,
        jf_qt = p.scratch.ᶠtemp_UVW.components.data.:1,
        uv_c = p.scratch.ᶜtemp_UVWxUVW,
        uv_f = p.scratch.ᶠtemp_UVWxUVW,
    )
    sw_q = QtEffVDiffColumnSweep{has_q_tot, has_nonequil, typeof(thermo_params), FT}(
        thermo_params, Δt_FT, has_sfc, precipitation,
    )
    DataLayouts.foreach_column(
        sw_q,
        Y.c,
        scal_geom,
        weights,
        scal_scratch,
        sfc_conditions;
        shmem = Val(:none),
    )

    passive_geom = (; ᶜJ, ᶠlg)
    passive_scratch = (;
        diag = p.scratch.ᶜtemp_scalar_3,
        upper = p.scratch.ᶜtemp_scalar_4,
        lower = p.scratch.ᶜtemp_scalar_5,
        rhs = p.scratch.ᶜtemp_scalar_6,
        sol = p.scratch.ᶜtemp_scalar_7,
    )
    jf_ql_tot = p.scratch.ᶠtemp_UVWxUVW.components.data.:2
    jf_qi_tot = p.scratch.ᶠtemp_UVWxUVW.components.data.:3
    jf_dummy = p.scratch.ᶠtemp_UVWxUVW.components.data.:9
    if has_nonequil
        fill!(jf_ql_tot, zero(FT))
        fill!(jf_qi_tot, zero(FT))
    end
    # Without the blend, the water species diffuse before the enthalpy solve
    # so that their realized fluxes enter jf_ql_tot/jf_qi_tot.
    if has_diff && has_q_tot && has_nonequil && !blend
        if precipitation
            _diffuse_water_species!(
                Y, p, diffused_water_species(true), false, thermo_params, Δt_FT,
                passive_geom, passive_scratch, weights, jf_ql_tot, jf_qi_tot, jf_dummy,
            )
        else
            _diffuse_water_species!(
                Y, p, diffused_water_species(false), false, thermo_params, Δt_FT,
                passive_geom, passive_scratch, weights, jf_ql_tot, jf_qi_tot, jf_dummy,
            )
        end
    end

    sw_h = EnthalpyTVDiffColumnSweep{has_q_tot, has_nonequil, typeof(thermo_params), FT}(
        thermo_params, Δt_FT, has_sfc, blend,
    )
    DataLayouts.foreach_column(
        sw_h,
        Y.c,
        scal_geom,
        weights,
        scal_scratch,
        sfc_conditions;
        shmem = Val(:none),
    )

    # The blended cloud-species flux uses the face fluxes `F_qt` and `F_sl` of
    # the moisture and enthalpy solves.
    if has_diff && has_q_tot && has_nonequil && blend
        if precipitation
            _diffuse_water_species!(
                Y, p, diffused_water_species(true), true, thermo_params, Δt_FT,
                passive_geom, passive_scratch, weights, jf_ql_tot, jf_qi_tot, jf_dummy,
            )
        else
            _diffuse_water_species!(
                Y, p, diffused_water_species(false), true, thermo_params, Δt_FT,
                passive_geom, passive_scratch, weights, jf_ql_tot, jf_qi_tot, jf_dummy,
            )
        end
    end

    # Passive grid-scale tracers (no surface flux)
    if has_diff
        sw_passive_noacc = PassiveScalarVDiffColumnSweep{false, FT}(Δt_FT)
        mp_names = microphysics_tracer_names(Y)
        MatrixFields.unrolled_foreach(gs_tracer_names(Y)) do ρχ_name
            ρχ_name in mp_names && return nothing
            ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
            DataLayouts.foreach_column(
                sw_passive_noacc,
                ᶜρχ,
                Y.c,
                passive_geom,
                weights,
                passive_scratch,
                jf_dummy;
                shmem = Val(:none),
            )
            return nothing
        end
    end

    return nothing
end

# ============================================================================
# 4. EDMF updraft sweep and advective-draft / conservative grid-mean solver
# ============================================================================

@inline _sgsflux_upw_mode(::Val{:first_order}) = Int8(0)
@inline _sgsflux_upw_mode(::Val{:none}) = Int8(1)
@inline _sgsflux_upw_mode(::Val{:third_order}) = Int8(2)
@inline _sgsflux_upw_mode(::Val{:vanleer_limiter}) = Int8(2)
@inline _sgsflux_upw_mode(_) = Int8(1)

@inline _updraft_upw_mode(::Val{:none}) = Int8(0)
@inline _updraft_upw_mode(_) = Int8(1)

struct EDMFTwoStreamColumnSweep{
    HasQ,
    HasNonEquil,
    FtildeMode,
    HasBlend,
    HasTailEntr,
    TurbParams,
    ThermoParams,
    CMP,
    Sponge,
    FT,
} <: ColumnSweep
    turbconv_params::TurbParams
    thermo_params::ThermoParams
    cmp::CMP
    rayleigh_sponge::Sponge
    a_max::FT
    a_min::FT
    α_b::FT
    α_d::FT
    c_R::FT
    R_max::FT
    f_tail::FT
    scale_height::FT
    detr_buoy_coeff::FT
    entr_detr_buoy_inv_tau_max::FT
    detr_massflux_vertdiv_coeff::FT
    updraft_kh_factor::FT
    c_d::FT
    τ_lcl_const::FT
    τ_icl_const::FT
    T_min_sgs::FT
    Δt::FT
    z_max::FT
    has_sfc::Bool
    has_nh_drag::Bool
    has_sgs_mass_flux::Bool
    has_env_diff::Bool
    has_updraft_diff::Bool
    use_const_τ::Bool
    couple_updraft_cond::Bool
    sgsflux_upw_mode::Int8
    updraft_upw_mode::Int8
    n_picard::Int8
end

# Difference-form SGS mass-flux correction `A_out[1:(nz + 1)]` matching
# `edmfx_sgsflux_upwinding` (`0`: first-order upwind, `1`: centered, `2`:
# third-order upwind). The net SGS advective flux vanishes when `x¹ = x⁰ = x̄`,
# with per-scalar positivity limiting on `ρχ_L` when `limit_pos` is `true`.
@inline function _compute_edmf_sgs_flux_corr!(
    A_out_fv,
    ρχ_L_fv,
    x1_fv,
    x0_fv,
    JF1_rel_fv,
    W1_rel_fv,
    ρa1_new_fv,
    ρa0_new_fv,
    ᶜJ_fv,
    ᶠz_fv,
    A_raw_fv,
    Δx_buf_fv,
    R_pos_fv,
    Δt,
    nz::Int,
    limit_pos::Bool,
    sgsflux_upw_mode::Int8,
)
    FT = typeof(Δt)
    if nz < 2
        @inbounds for f in 1:(nz + 1)
            A_out_fv[f] = zero(FT)
        end
        return nothing
    end
    ε = floatmin(FT)
    @inbounds begin
        for k in 1:nz
            ρ_new_k = max(ρa1_new_fv[k] + ρa0_new_fv[k], ε)
            a1_k = clamp(ρa1_new_fv[k] / ρ_new_k, zero(FT), one(FT))
            a0_k = one(FT) - a1_k
            Δx_buf_fv[k] = a1_k * a0_k * (x1_fv[k] - x0_fv[k])
        end

        A_raw_fv[1] = zero(FT)
        A_raw_fv[nz + 1] = zero(FT)
        for f in 2:nz
            F1_f = JF1_rel_fv[f]
            W1_f = W1_rel_fv[f]
            if F1_f == zero(FT) && W1_f == zero(FT)
                A_raw_fv[f] = zero(FT)
            else
                f_adv_low =
                    max(zero(FT), F1_f) * x1_fv[f - 1] +
                    min(zero(FT), F1_f) * x1_fv[f] +
                    max(zero(FT), -F1_f) * x0_fv[f - 1] +
                    min(zero(FT), -F1_f) * x0_fv[f]
                ρa1_f = FT(0.5) * (ρa1_new_fv[f - 1] + ρa1_new_fv[f])
                ρa0_f = max(FT(0.5) * (ρa0_new_fv[f - 1] + ρa0_new_fv[f]), ε)
                W0_f = -W1_f * (ρa1_f / ρa0_f)
                q1_lo = Δx_buf_fv[f - 1]
                q1_hi = Δx_buf_fv[f]
                Φ_sgs_target = if sgsflux_upw_mode == Int8(0)
                    (
                        max(zero(FT), W1_f) * q1_lo +
                        min(zero(FT), W1_f) * q1_hi
                    ) + (
                        max(zero(FT), W0_f) * (-q1_lo) +
                        min(zero(FT), W0_f) * (-q1_hi)
                    )
                elseif sgsflux_upw_mode == Int8(1)
                    (W1_f - W0_f) * FT(0.5) * (q1_lo + q1_hi)
                else
                    W1_f * _face_ho_upwind3(Δx_buf_fv, f, W1_f, ᶠz_fv, nz) -
                    W0_f * _face_ho_upwind3(Δx_buf_fv, f, W0_f, ᶠz_fv, nz)
                end
                d0_idx = (-F1_f) >= zero(FT) ? (f - 1) : f
                C0_f = abs(F1_f) * Δt / max(ᶜJ_fv[d0_idx] * ρa0_new_fv[d0_idx], ε)
                w_C = min(one(FT), one(FT) / max(C0_f, ε))
                A_raw_fv[f] = w_C * (Φ_sgs_target - f_adv_low)
            end
        end

        A_out_fv[1] = zero(FT)
        A_out_fv[nz + 1] = zero(FT)
        if !limit_pos
            for f in 2:nz
                A_out_fv[f] = A_raw_fv[f]
            end
        else
            for k in 1:nz
                P_out_k =
                    Δt * (max(zero(FT), A_raw_fv[k + 1]) + max(zero(FT), -A_raw_fv[k]))
                Q_avail_k = max(zero(FT), ᶜJ_fv[k] * ρχ_L_fv[k])
                R_pos_fv[k] = _zalesak_ratio(P_out_k, Q_avail_k)
            end
            for f in 2:nz
                A_f = A_raw_fv[f]
                if A_f == zero(FT)
                    A_out_fv[f] = zero(FT)
                else
                    d_idx = A_f >= zero(FT) ? (f - 1) : f
                    A_out_fv[f] = R_pos_fv[d_idx] * A_f
                end
            end
        end
    end
    return nothing
end

# Redistribute trace negative density `ρχ < 0` conservatively across positive
# cells while bounding `ρa¹ χ¹ ≤ ρχ`; no-op when `ρχ ≥ 0` everywhere.
@inline function _repair_nonneg_col!(ρχ_fv, χ1_fv, ρa1_fv, ᶜJ_fv, nz::Int)
    FT = eltype(ρχ_fv)
    ε = floatmin(FT)
    ε_num = ϵ_numerics(FT)
    deficit = zero(FT)
    pos_mass = zero(FT)
    @inbounds for k in 1:nz
        m_k = ᶜJ_fv[k] * ρχ_fv[k]
        if m_k < zero(FT)
            deficit -= m_k
        else
            pos_mass += m_k
        end
    end
    if deficit > zero(FT) && pos_mass > deficit
        scale = (pos_mass - deficit) / max(pos_mass, ε)
        @inbounds for k in 1:nz
            ρχ_k = ρχ_fv[k]
            ρχ_new_k = ρχ_k < zero(FT) ? zero(FT) : ρχ_k * scale
            ρχ_fv[k] = ρχ_new_k
            if ρa1_fv[k] > ε_num
                χ1_fv[k] = min(max(zero(FT), χ1_fv[k]), ρχ_new_k / ρa1_fv[k])
            end
        end
    end
    return nothing
end

# Surface terms of an updraft scalar solve without a surface source: zero flux
# `JF_sfc`, zero relaxation rate `r_sfc_1` toward `χ_buoyant`, and a unit
# cell-1 updraft density `ρ1_1`.
@inline _no_surface_source(::Type{FT}) where {FT} = (;
    JF_sfc = zero(FT),
    r_sfc_1 = zero(FT),
    ρ1_1 = one(FT),
    χ_buoyant = zero(FT),
)

# Intensive advective-diffusive updraft solve for `χ¹_new`:
# `(χ¹_new - χ¹_old)/Δt + u³¹ ∂ᵥχ¹_new = ε_tot (χ⁰_old - χ¹_new)`
# `+ δ_{k,1} [r_sfc (χ_b - χ¹_new) + btt/ρ¹]`
# `+ (1/ρa¹) ᶜdiffdivᵥ(G¹ ∇χ¹_new) + extra_src1`.
@inline function _solve_edmf_updraft_var!(
    x1_out_fv,
    χ1_in_fv,
    χ0_in_fv,
    extra_src1_fv,
    ρa1_new_fv,
    V1_fv,
    ε_tot_fv,
    G1_fv,
    ᶜJ_fv,
    diag1_fv,
    upper1_fv,
    lower1_fv,
    rhs1_fv,
    Δt,
    nz::Int,
    opts,
    sfc,
    updraft_upw_mode::Int8,
    has_relax::Bool = false,
    r_relax_fv = extra_src1_fv,
    q_eq_fv = extra_src1_fv,
)
    FT = typeof(Δt)
    (; apply_diff, apply_sfc, has_extra_src1, limit_pos) = opts
    (; JF_sfc, r_sfc_1, ρ1_1, χ_buoyant) = sfc
    ε = eps(FT)
    ε_num = ϵ_numerics(FT)
    @inbounds for k in 1:nz
        J_k = ᶜJ_fv[k]
        ρa1_k = ρa1_new_fv[k]
        inv_m1_k = ρa1_k > ε_num ? inv(J_k * ρa1_k) : zero(FT)
        g1_top = (apply_diff && k < nz) ? G1_fv[k + 1] * inv_m1_k : zero(FT)
        g1_bot = (apply_diff && k > 1) ? G1_fv[k] * inv_m1_k : zero(FT)
        r_sfc_k = (k == 1 && apply_sfc) ? r_sfc_1 : zero(FT)
        r_rel_k = has_relax ? r_relax_fv[k] : zero(FT)
        ε_k = ε_tot_fv[k]

        if updraft_upw_mode == Int8(0)
            v_top = k < nz ? FT(0.5) * V1_fv[k + 1] / J_k : zero(FT)
            v_bot = k > 1 ? FT(0.5) * V1_fv[k] / J_k : zero(FT)
            diag1_fv[k] = max(
                FT(0.1),
                one(FT) +
                Δt * ((v_bot - v_top) + ε_k + r_sfc_k + r_rel_k + g1_top + g1_bot),
            )
            if k < nz
                upper1_fv[k] = Δt * (v_top - g1_top)
            end
            if k > 1
                lower1_fv[k - 1] = -Δt * (v_bot + g1_bot)
            end
        else
            adv_top = k < nz ? -min(zero(FT), V1_fv[k + 1]) / J_k : zero(FT)
            adv_bot = k > 1 ? max(zero(FT), V1_fv[k]) / J_k : zero(FT)
            diag1_fv[k] =
                one(FT) +
                Δt * (adv_top + adv_bot + ε_k + r_sfc_k + r_rel_k + g1_top + g1_bot)
            if k < nz
                upper1_fv[k] = -Δt * (adv_top + g1_top)
            end
            if k > 1
                lower1_fv[k - 1] = -Δt * (adv_bot + g1_bot)
            end
        end

        χ1_o = limit_pos ? max(zero(FT), χ1_in_fv[k]) : χ1_in_fv[k]
        χ0_o = limit_pos ? max(zero(FT), χ0_in_fv[k]) : χ0_in_fv[k]
        rhs_k =
            χ1_o +
            Δt * (
                ε_k * χ0_o +
                (has_extra_src1 ? extra_src1_fv[k] : zero(FT)) +
                (has_relax ? r_rel_k * q_eq_fv[k] : zero(FT))
            )
        if k == 1 && apply_sfc
            rhs_k += Δt * (r_sfc_1 * χ_buoyant + JF_sfc / (J_k * max(ρ1_1, ε)))
        end
        rhs1_fv[k] = limit_pos ? max(zero(FT), rhs_k) : rhs_k
    end

    column_thomas_solve!(x1_out_fv, rhs1_fv, diag1_fv, upper1_fv, lower1_fv, nz)
    return nothing
end

# 1-moment updraft cloud phase-relaxation coefficients `(r_relax, q_eq)` at
# the transported updraft state, following the relaxation model of
# `CloudMicrophysics.MicrophysicsNonEq`: the condensate relaxes at rate `1/τ`
# toward `q* = q + S τ = q + δ / Γ`, with `S` the condensation/evaporation
# (deposition/sublimation) tendency, `δ` the vapor excess, and `Γ` the factor
# of `CMNe.gamma_helper`. At the transported state, `r_relax * (q_eq - q)`
# equals `S`, including its evaporation bound `S ≥ -q/τ`, which `q_eq ≥ 0`
# reproduces. Clamping `q_eq` to `[0, q_avail]` keeps `0 ≤ q_new ≤ q_avail` by
# M-matrix diagonal dominance.
@inline function _compute_updraft_1m_relax!(
    ::Val{IsLiq},
    r_relax_fv,
    q_eq_fv,
    mp,
    tps,
    T_min_sgs::FT,
    ᶜρa1_fv,
    mse1_fv,
    q_tot_arg_fv,
    ql_ref_fv,
    qi_ref_fv,
    qr1_fv,
    qs1_fv,
    ᶜΦ_fv,
    ᶜp_fv,
    is_qte::Bool,
    nz::Int,
) where {IsLiq, FT}
    ε = eps(FT)
    ε_num = ϵ_numerics(FT)
    R_v = FT(TD.Parameters.R_v(tps))
    procs = mp.processes
    T_hom = IsLiq ? _liq_hom_temp(procs.cloud_liquid_formation, mp, FT) : FT(-Inf)

    @inbounds for k in 1:nz
        if ᶜρa1_fv[k] <= ε_num
            r_relax_fv[k] = zero(FT)
            q_eq_fv[k] = zero(FT)
        else
            qr_k = max(zero(FT), qr1_fv[k])
            qs_k = max(zero(FT), qs1_fv[k])
            qte_k =
                is_qte ? max(zero(FT), q_tot_arg_fv[k]) :
                max(zero(FT), q_tot_arg_fv[k] - qr_k - qs_k)
            ql_tr = clamp(ql_ref_fv[k], zero(FT), qte_k)
            qi_tr = clamp(qi_ref_fv[k], zero(FT), max(zero(FT), qte_k - ql_tr))
            qv_tr = max(zero(FT), qte_k - ql_tr - qi_tr)
            qt_tot_k = qte_k + qr_k + qs_k
            h1_k = mse1_fv[k] - ᶜΦ_fv[k]
            p_k = ᶜp_fv[k]
            q_liq_tot = ql_tr + qr_k
            q_ice_tot = qi_tr + qs_k
            T1_tr = max(
                T_min_sgs,
                TD.air_temperature(
                    tps, TD.ph(), h1_k, qt_tot_k, q_liq_tot, q_ice_tot,
                ),
            )
            ρ1_tr = max(
                ε,
                TD.air_density(
                    tps, T1_tr, p_k, qt_tot_k, q_liq_tot, q_ice_tot,
                ),
            )
            micro = (;
                q_tot = qt_tot_k,
                q_lcl = ql_tr,
                q_icl = qi_tr,
                q_rai = qr_k,
                q_sno = qs_k,
            )
            thermo = (; ρ = ρ1_tr, T = T1_tr, w = zero(FT))

            if IsLiq
                q_sat = TD.q_vap_saturation(tps, T1_tr, ρ1_tr, TD.Liquid())
                if (ql_tr <= zero(FT)) & (qv_tr <= q_sat)
                    r_relax_fv[k] = zero(FT)
                    q_eq_fv[k] = zero(FT)
                    continue
                end
                S_cm = CMNe.conv_q_vap_to_q_lcl(
                    procs.cloud_liquid_formation, mp, tps, micro, thermo,
                )
                τ_cm = CMNe.τ_vap_to_q_lcl(
                    procs.cloud_liquid_formation, mp, tps, micro, thermo,
                )
                L_lat = TD.latent_heat_vapor(tps, T1_tr)
                block_src = T1_tr < T_hom
                q_self = ql_tr
                q_avail_max = max(zero(FT), qte_k - qi_tr)
            else
                block_src = CMNe.INP_limiter(
                    one(FT),
                    tps,
                    T1_tr;
                    q_lcl = ql_tr,
                    _ice_dep_limits(procs.cloud_ice_formation, mp)...,
                )
                if (qi_tr <= zero(FT)) & block_src
                    r_relax_fv[k] = zero(FT)
                    q_eq_fv[k] = zero(FT)
                    continue
                end
                q_sat = TD.q_vap_saturation(tps, T1_tr, ρ1_tr, TD.Ice())
                if (qi_tr <= zero(FT)) & (qv_tr <= q_sat)
                    r_relax_fv[k] = zero(FT)
                    q_eq_fv[k] = zero(FT)
                    continue
                end
                S_cm = CMNe.conv_q_vap_to_q_icl(
                    procs.cloud_ice_formation, mp, tps, micro, thermo,
                )
                τ_cm = CMNe.τ_vap_to_q_icl(
                    procs.cloud_ice_formation, mp, tps, micro, thermo,
                )
                L_lat = TD.latent_heat_sublim(tps, T1_tr)
                q_self = qi_tr
                q_avail_max = max(zero(FT), qte_k - ql_tr)
            end

            cp_m = TD.cp_m(tps, qt_tot_k, q_liq_tot, q_ice_tot)
            dqsat_dT_cm = CMNe.dqcld_dT(q_sat, L_lat, R_v, T1_tr)
            Γ_cm = CMNe.gamma_helper(L_lat, cp_m, dqsat_dT_cm)
            δ_tr = qv_tr - q_sat
            q_tol = ε * max(q_sat, floatmin(FT))
            c_raw = _cloud_relax_coeff(S_cm, τ_cm, Γ_cm, δ_tr, q_tol)
            if block_src && δ_tr > zero(FT)
                c_raw = zero(FT)
            end

            if c_raw <= zero(FT)
                r_relax_fv[k] = zero(FT)
                q_eq_fv[k] = q_self
            else
                q_eq_fv[k] = clamp(q_self + δ_tr / Γ_cm, zero(FT), q_avail_max)
                r_relax_fv[k] = c_raw * Γ_cm
            end
        end
    end
    return nothing
end

# Conservative grid-mean scalar solve for diagnostic environment `χ⁰_new`:
# `J_k (ρa¹_new χ¹_gm + ρa⁰_new χ⁰_new + s_k) + Δt ΔΦ = J_k ρχ_in +
# δ_{k,1} Δt JF_sfc`, accumulating environment FCT `A0` and committing `ρχ_out`
# in flux form. The storage offset `s_k` (zero unless `has_storage_offset`)
# separates the transported `χ⁰` from the residual `(ρχ - ρa¹ χ¹) / ρa⁰`.
@inline function _solve_and_commit_edmf_gm_var!(
    ρχ_out_fv,
    x0_save_fv,
    A0_out_fv,
    ρχ_in_fv,
    x1_gm_fv,
    x1_diff_fv,
    W1_rel_fv,
    ρa1_new_fv,
    ρa0_new_fv,
    JF1_rel_fv,
    JF0_rel_fv,
    G1_fv,
    G0_eff_fv,
    jf_extra_fv,
    ᶜJ_fv,
    ᶠz_fv,
    Φ1_fv,
    diag0_fv,
    upper0_fv,
    lower0_fv,
    rhs0_fv,
    x0_fv,
    Δt,
    nz::Int,
    opts,
    JF_sfc,
    sgsflux_upw_mode::Int8,
    has_storage_offset::Bool = false,
    storage_offset_fv = ρχ_in_fv,
)
    FT = typeof(Δt)
    (; limit_pos, apply_diff, apply_sfc, has_jf_extra, save_x0) = opts
    @inbounds begin
        Φ1_fv[1] = zero(FT)
        Φ1_fv[nz + 1] = zero(FT)
        for f in 2:nz
            F1_f = JF1_rel_fv[f]
            f_adv1 =
                max(zero(FT), F1_f) * x1_gm_fv[f - 1] +
                min(zero(FT), F1_f) * x1_gm_fv[f]
            f_diff1 =
                apply_diff ? -G1_fv[f] * (x1_diff_fv[f] - x1_diff_fv[f - 1]) : zero(FT)
            Φ1_fv[f] = f_adv1 + f_diff1
        end

        for k in 1:nz
            J_k = ᶜJ_fv[k]
            F0_top = JF0_rel_fv[k + 1]
            F0_bot = JF0_rel_fv[k]
            G0_top = (apply_diff && k < nz) ? G0_eff_fv[k + 1] : zero(FT)
            G0_bot = (apply_diff && k > 1) ? G0_eff_fv[k] : zero(FT)

            diag0_fv[k] =
                J_k * ρa0_new_fv[k] +
                Δt * (max(zero(FT), F0_top) - min(zero(FT), F0_bot) + G0_top + G0_bot)
            if k < nz
                upper0_fv[k] = Δt * min(zero(FT), F0_top) - Δt * G0_top
            end
            if k > 1
                lower0_fv[k - 1] = -Δt * max(zero(FT), F0_bot) - Δt * G0_bot
            end

            rhs0_k =
                J_k * (ρχ_in_fv[k] - ρa1_new_fv[k] * x1_gm_fv[k]) -
                Δt * (Φ1_fv[k + 1] - Φ1_fv[k])
            if has_storage_offset
                rhs0_k -= J_k * storage_offset_fv[k]
            end
            if has_jf_extra
                rhs0_k -= Δt * (jf_extra_fv[k + 1] - jf_extra_fv[k])
            end
            if k == 1 && apply_sfc
                rhs0_k += Δt * JF_sfc
            end
            rhs0_fv[k] = rhs0_k
        end
    end

    column_thomas_solve!(x0_fv, rhs0_fv, diag0_fv, upper0_fv, lower0_fv, nz)

    @inbounds begin
        Φ_bot = apply_sfc ? JF_sfc : zero(FT)
        for k in 1:nz
            Φ_top = if k < nz
                F0_f = JF0_rel_fv[k + 1]
                f_adv0 =
                    max(zero(FT), F0_f) * x0_fv[k] +
                    min(zero(FT), F0_f) * x0_fv[k + 1]
                f_diff0 =
                    apply_diff ? -G0_eff_fv[k + 1] * (x0_fv[k + 1] - x0_fv[k]) :
                    zero(FT)
                Φ1_fv[k + 1] +
                f_adv0 +
                f_diff0 +
                (has_jf_extra ? jf_extra_fv[k + 1] : zero(FT))
            else
                zero(FT)
            end
            if save_x0
                x0_save_fv[k] = x0_fv[k]
            end
            ρχ_out_fv[k] = ρχ_in_fv[k] - (Δt / ᶜJ_fv[k]) * (Φ_top - Φ_bot)
            Φ_bot = Φ_top
        end
    end

    _compute_edmf_sgs_flux_corr!(
        A0_out_fv,
        ρχ_out_fv,
        x1_gm_fv,
        x0_fv,
        JF1_rel_fv,
        W1_rel_fv,
        ρa1_new_fv,
        ρa0_new_fv,
        ᶜJ_fv,
        ᶠz_fv,
        Φ1_fv,
        diag0_fv,
        upper0_fv,
        Δt,
        nz,
        limit_pos,
        sgsflux_upw_mode,
    )
    return nothing
end

# Solve updraft `χ¹` and conservative grid-mean `ρχ` for one scalar variable.
@inline function _solve_and_commit_edmf_var!(
    χ1_out_fv,
    ρχ_out_fv,
    x0_save_fv,
    A0_out_fv,
    W1_rel_fv,
    ρχ_in_fv,
    χ1_in_fv,
    χ0_in_fv,
    ρa1_new_fv,
    ρa0_new_fv,
    V1_fv,
    ε_tot_fv,
    JF1_rel_fv,
    JF0_rel_fv,
    G1_fv,
    G0_eff_fv,
    jf_extra_fv,
    ᶜJ_fv,
    ᶠz_fv,
    wk,
    Δt,
    nz::Int,
    opts,
    sfc,
    sgsflux_upw_mode::Int8,
    updraft_upw_mode::Int8,
    has_extra_src1::Bool = false,
    extra_src1_fv = χ1_in_fv,
)
    (; apply_diff, apply_sfc, has_jf_extra, save_x0, limit_pos) = opts
    (; JF_sfc) = sfc
    _solve_edmf_updraft_var!(
        wk.x1,
        χ1_in_fv,
        χ0_in_fv,
        extra_src1_fv,
        ρa1_new_fv,
        V1_fv,
        ε_tot_fv,
        G1_fv,
        ᶜJ_fv,
        wk.diag1,
        wk.upper1,
        wk.lower1,
        wk.r1,
        Δt,
        nz,
        (; apply_diff, apply_sfc, has_extra_src1, limit_pos),
        sfc,
        updraft_upw_mode,
    )
    @inbounds for k in 1:nz
        χ1_out_fv[k] = wk.x1[k]
    end
    _solve_and_commit_edmf_gm_var!(
        ρχ_out_fv,
        x0_save_fv,
        A0_out_fv,
        ρχ_in_fv,
        χ1_out_fv,
        χ1_out_fv,
        W1_rel_fv,
        ρa1_new_fv,
        ρa0_new_fv,
        JF1_rel_fv,
        JF0_rel_fv,
        G1_fv,
        G0_eff_fv,
        jf_extra_fv,
        ᶜJ_fv,
        ᶠz_fv,
        wk.Φ1,
        wk.diag0,
        wk.upper0,
        wk.lower0,
        wk.r0,
        wk.x0,
        Δt,
        nz,
        (; limit_pos, apply_diff, apply_sfc, has_jf_extra, save_x0),
        JF_sfc,
        sgsflux_upw_mode,
    )
    return nothing
end

# Face mean of the enthalpy of a water phase in excess of dry air,
# `h(T) - h_d(T)`, between the cells at `T_lo` and `T_hi`; `enthalpy` is
# `TD.enthalpy_vapor`, `TD.enthalpy_liquid`, or `TD.enthalpy_ice`.
@inline _face_mean_enthalpy_excess(enthalpy::F, tps, T_lo, T_hi) where {F} =
    (
        (enthalpy(tps, T_lo) - TD.enthalpy_dry(tps, T_lo)) +
        (enthalpy(tps, T_hi) - TD.enthalpy_dry(tps, T_hi))
    ) / 2

# Environment specific value `(ρχ - ρa¹ χ¹⁺) / ρa⁰` as the residual of the grid
# mean and the non-negative updraft value, with `ρa⁰` floored at `ε`.
@inline _env_residual_value(ρχ, ρa1, χ1, ρa0, ε) =
    (ρχ - ρa1 * max(zero(χ1), χ1)) / max(ρa0, ε)

# Contravariant draft velocity `u³¹ = uₕ³ + g³³ u₃¹` at face `f`, built from
# the same `uₕ³` as the grid-mean `u³`, so that `u³¹ - u³ = g³³ (u₃¹ - u₃)`
# holds over sloped terrain.
@inline _draft_face_u3_contravariant(ᶠuh3_f, ᶠlg_f, u3_f) =
    ᶠuh3_f.u³ + parent(ᶠlg_f.gⁱʲ)[end] * u3_f

# Extensive face conductance `c = ⟨J ρ¹⟩_f u³¹ / ρ¹_donor` of the draft mass
# flux `c ρa¹_donor` through face `f`, with the `J`-weighted face density
# `⟨J ρ¹⟩_f = (J ρ¹)_{f-1/2} / 2 + (J ρ¹)_{f+1/2} / 2`.
@inline function _draft_face_conductance(
    u3_cont_f,
    ρ1_lo,
    ρ1_hi,
    J_lo,
    J_hi,
)
    FT = typeof(u3_cont_f)
    Jρ1_f = (J_lo * ρ1_lo + J_hi * ρ1_hi) / 2
    ρ1_don = u3_cont_f >= zero(FT) ? ρ1_lo : ρ1_hi
    return Jρ1_f * u3_cont_f / ρ1_don
end

# Draft relative mass flux `F_rel = J ρ̂¹ a¹ u³¹ - J ρ̄ a¹ ū³` (donor-cell
# area fractions), contravariant draft volume flux `V1 = J_f u³¹`, and SGS
# flux weight `W1 = ⟨J ρ⟩_f (u³¹ - ū³)` at face `f`, where `ū³` is the
# grid-mean contravariant velocity `u3_cont_gm`. The weight uses the
# grid-mean face density because `Σₖ ρ̂ᵏ (u³ᵏ - ū³)(χᵏ - χ̄) =
# ρ̂¹ (u³¹ - ū³)(χ¹ - χ⁰)` with `ρ̂¹ = ρ̄ a¹`; the low-order flux `F_rel (χ¹ -
# χ⁰)` satisfies the same identity.
@inline function _draft_relative_face_fluxes(
    c1_f,
    u3_cont_1,
    u3_cont_gm,
    ρa1_lo,
    ρa1_hi,
    ρ_lo,
    ρ_hi,
    J_lo,
    J_hi,
    ᶠJ_f,
)
    FT = typeof(c1_f)
    ε_num = ϵ_numerics(FT)
    JF1_full_f = c1_f >= zero(FT) ? c1_f * ρa1_lo : c1_f * ρa1_hi
    Jρ_gm_f = (ρ_lo * J_lo + ρ_hi * J_hi) / 2
    JF_gm = Jρ_gm_f * u3_cont_gm
    a1_gm_don =
        JF_gm >= zero(FT) ? ρa1_lo / max(ρ_lo, ε_num) :
        ρa1_hi / max(ρ_hi, ε_num)
    F_rel = JF1_full_f - JF_gm * a1_gm_don
    V1 = ᶠJ_f * u3_cont_1
    W1 = Jρ_gm_f * (u3_cont_1 - u3_cont_gm)
    return (F_rel, V1, W1)
end

# Upward `u₃¹ ≥ 0` quadratic sweep from `(u3_in, ρa1_in)` followed by the
# 2-pass `ρa¹` M-matrix solve under the current `(ᶜρ1, ᶜρ_diff1)` iterate.
# On entry, `ᶜρa1_fv` holds the latest `ρa¹` iterate. The area-bounding and
# turbulent entrainment rates follow that iterate on every pass, so the
# `a < a_min` entrainment stops once the draft has grown past `a_min`. The
# draft mass flux uses the full contravariant `u³¹` of
# `_draft_face_u3_contravariant`, which carries `ρa¹` through sloped
# coordinate surfaces with the grid-mean `uₕ³`. The solve bounds
# `ρa¹ ≤ min(a_max ρ¹, (1 - 10⁻⁴) ρ)`, so that `ρa⁰ = ρ - ρa¹ > 0` also when
# `ρ¹ > ρ`.
@inline function _step_edmf_u3_and_rhoa!(
    sw::EDMFTwoStreamColumnSweep{
        HasQ,
        HasNonEquil,
        FtildeMode,
        HasBlend,
        HasTailEntr,
    },
    u3_new,
    ᶜρa1_fv,
    JF1_fv,
    ε_tot_fv,
    b_work_col,
    tail_qt_fv,
    u3_in,
    ρa1_in,
    ᶜρ_fv,
    ᶜρ1_fv,
    ᶜρ_diff1_fv,
    ᶜρ_ref_excess_fv,
    ᶜevs_fv,
    ᶜenr_fv,
    ᶜρtke_fv,
    ᶜT0_fv,
    tail_fvs,
    ᶜlg_fv,
    ᶜJ_fv,
    ᶜz_fv,
    z_sfc_val,
    ᶠlg_fv,
    ᶠdz_fv,
    ᶠz_fv,
    ᶠgradΦ_fv,
    ᶜgradΦ_fv,
    ᶠuh3_fv,
    d11,
    u1_blk,
    l1_blk,
    r1_blk,
    x1_blk,
    E_sfc,
    Δt,
    nz::Int,
) where {HasQ, HasNonEquil, FtildeMode, HasBlend, HasTailEntr}
    FT = typeof(E_sfc)
    ε = eps(FT)
    tcp = sw.turbconv_params
    a0_floor = one(FT) - sw.a_max
    ab_zero = area_bounding_entr_detr(tcp, zero(FT))
    L_zero = lower_area_limiter_factor(zero(FT), tcp)
    U_zero = upper_area_limiter_factor(zero(FT), tcp)

    @inbounds begin
        u3_new[1] = zero(FT)
        u3_new[nz + 1] = zero(FT)
    end

    @inbounds begin
        ρ_lo = max(ᶜρ1_fv[1], ε)
        a1_cur_lo = draft_area(ᶜρa1_fv[1], ρ_lo)
        a0_lo = clamp(one(FT) - a1_cur_lo, a0_floor, one(FT))
        a0_inv_lo = inv(a0_lo)
        z_lo = ᶜz_fv[1]
        tke_lo = max(zero(FT), ᶜρtke_fv[1] / max(ᶜρ_fv[1], ε))
        R_lo = plume_radius(
            sw.scale_height,
            a1_cur_lo,
            sw.a_min,
            sw.a_max,
            z_lo,
            z_sfc_val,
            sw.c_R,
            sw.R_max,
        )
        dc_lo = if sw.has_nh_drag
            pressure_drag_coefficient(
                sw.α_d,
                sw.scale_height,
                a1_cur_lo,
                a0_lo,
                sw.a_min,
                sw.a_max,
                z_lo,
                z_sfc_val,
                sw.c_R,
                sw.R_max,
            )
        else
            zero(FT)
        end
        ab_lo = a1_cur_lo == zero(FT) ? ab_zero : area_bounding_entr_detr(tcp, a1_cur_lo)
        te_lo = turbulent_entrainment(tcp, a1_cur_lo, z_lo, z_sfc_val, tke_lo, R_lo)
        b_lo = (max(zero(FT), ab_lo) + ᶜenr_fv[1] + te_lo) * a0_inv_lo

        for f in 2:nz
            dz_f = ᶠdz_fv[f]
            ρ_hi = max(ᶜρ1_fv[f], ε)
            a1_cur_hi = draft_area(ᶜρa1_fv[f], ρ_hi)
            a0_hi = clamp(one(FT) - a1_cur_hi, a0_floor, one(FT))
            a0_inv_hi = inv(a0_hi)
            z_hi = ᶜz_fv[f]
            tke_hi = max(zero(FT), ᶜρtke_fv[f] / max(ᶜρ_fv[f], ε))
            R_hi = plume_radius(
                sw.scale_height,
                a1_cur_hi,
                sw.a_min,
                sw.a_max,
                z_hi,
                z_sfc_val,
                sw.c_R,
                sw.R_max,
            )

            a_f =
                (
                    ᶜevs_fv[f - 1] * a0_inv_lo * a0_inv_lo +
                    ᶜevs_fv[f] * a0_inv_hi * a0_inv_hi
                ) / (2 * dz_f) + inv(dz_f * dz_f) / 2
            dc_hi = if sw.has_nh_drag
                pressure_drag_coefficient(
                    sw.α_d,
                    sw.scale_height,
                    a1_cur_hi,
                    a0_hi,
                    sw.a_min,
                    sw.a_max,
                    z_hi,
                    z_sfc_val,
                    sw.c_R,
                    sw.R_max,
                )
            else
                zero(FT)
            end
            if sw.has_nh_drag
                a_f += (dc_lo * a0_inv_lo + dc_hi * a0_inv_hi) / (2 * dz_f)
            end

            ab_hi =
                a1_cur_hi == zero(FT) ? ab_zero :
                area_bounding_entr_detr(tcp, a1_cur_hi)
            te_hi = turbulent_entrainment(
                tcp, a1_cur_hi, z_hi, z_sfc_val, tke_hi, R_hi,
            )
            b_hi = (max(zero(FT), ab_hi) + ᶜenr_fv[f] + te_hi) * a0_inv_hi
            b_f = inv(Δt) + (b_lo + b_hi) / 2
            if !isnothing(sw.rayleigh_sponge)
                b_f += β_rayleigh_u₃(sw.rayleigh_sponge, ᶠz_fv[f], sw.z_max)
            end

            gradΦ_f = _c3_scalar(ᶠgradΦ_fv[f])
            ρ_diff_f = (ᶜρ_diff1_fv[f - 1] + ᶜρ_diff1_fv[f]) / 2
            # Physical `w` at the face below, with that face's own spacing.
            w_prev = u3_new[f - 1] / ᶠdz_fv[f - 1]
            c_f =
                -u3_in[f] / Δt +
                (one(FT) - sw.α_b) * ρ_diff_f * gradΦ_f -
                (w_prev * w_prev) / 2
            # Positive root of `a u² + b u + c = 0` for `c ≤ 0`, avoiding
            # cancellation in `-b + √disc` when `4a|c| ≪ b²`.
            c_neg = min(zero(FT), c_f)
            disc = max(zero(FT), b_f * b_f - 4 * a_f * c_neg)
            u3_new[f] = max(zero(FT), -2 * c_neg / (b_f + sqrt(disc)))

            a0_inv_lo = a0_inv_hi
            dc_lo = dc_hi
            b_lo = b_hi
        end
    end

    # Face conductances of the draft mass flux, fixed during the passes.
    @inbounds begin
        JF1_fv[1] = zero(FT)
        JF1_fv[nz + 1] = zero(FT)
        for f in 2:nz
            u3_cont_f =
                _draft_face_u3_contravariant(ᶠuh3_fv[f], ᶠlg_fv[f], u3_new[f])
            JF1_fv[f] = _draft_face_conductance(
                u3_cont_f,
                max(ᶜρ1_fv[f - 1], ε),
                max(ᶜρ1_fv[f], ε),
                ᶜJ_fv[f - 1],
                ᶜJ_fv[f],
            )
        end
    end

    for _iter in 1:2
        @inbounds begin
            buoy_bot = zero(FT)
            for k in 1:nz
                J_k = ᶜJ_fv[k]
                ρ1_k = max(ᶜρ1_fv[k], ε)
                ρa_eff_k = max(ρa1_in[k], ᶜρa1_fv[k])
                a_eff_k = draft_area(ρa_eff_k, ρ1_k)
                L_k =
                    a_eff_k == zero(FT) ? L_zero :
                    lower_area_limiter_factor(a_eff_k, tcp)
                U_k =
                    a_eff_k == zero(FT) ? U_zero :
                    upper_area_limiter_factor(a_eff_k, tcp)

                c_top = JF1_fv[k + 1]
                c_bot = JF1_fv[k]
                flux_top_eff = if k < nz
                    ρa_eff_kp1 = max(ρa1_in[k + 1], ᶜρa1_fv[k + 1])
                    c_top >= zero(FT) ? c_top * ρa_eff_k : c_top * ρa_eff_kp1
                else
                    zero(FT)
                end
                flux_bot_eff = if k > 1
                    ρa_eff_km1 = max(ρa1_in[k - 1], ᶜρa1_fv[k - 1])
                    c_bot >= zero(FT) ? c_bot * ρa_eff_km1 : c_bot * ρa_eff_k
                else
                    zero(FT)
                end
                div_eff = (flux_top_eff - flux_bot_eff) / J_k
                one_minus_detr = if k == nz
                    zero(FT)
                elseif div_eff < zero(FT)
                    U_k * (one(FT) - L_k * sw.detr_massflux_vertdiv_coeff)
                else
                    one(FT)
                end

                buoy_top = if k < nz
                    w_top = u3_new[k + 1] / ᶠdz_fv[k + 1]
                    gradΦ_top = _c3_scalar(ᶠgradΦ_fv[k + 1])
                    vb_top = vertical_buoyancy_acceleration(
                        (ᶜρ_diff1_fv[k] + ᶜρ_diff1_fv[k + 1]) / 2,
                        gradΦ_top,
                        ᶠlg_fv[k + 1],
                    )
                    detr_buoy_inv_time_scale(
                        w_top, vb_top, sw.entr_detr_buoy_inv_tau_max,
                    )
                else
                    zero(FT)
                end
                buoy_inv_tau = (buoy_top + buoy_bot) / 2
                buoy_bot = buoy_top
                w_c = (u3_new[k] / ᶠdz_fv[k] + u3_new[k + 1] / ᶠdz_fv[k + 1]) / 2

                a_cur_k = draft_area(ᶜρa1_fv[k], ρ1_k)
                ab_cur_k =
                    a_cur_k == zero(FT) ? ab_zero :
                    area_bounding_entr_detr(tcp, a_cur_k)
                entr_k =
                    max(zero(FT), ab_cur_k) +
                    ᶜevs_fv[k] * abs(w_c) +
                    ᶜenr_fv[k]
                detr_k =
                    max(zero(FT), -ab_cur_k) +
                    L_k * sw.detr_buoy_coeff * buoy_inv_tau

                out_top = one_minus_detr * max(zero(FT), c_top)
                out_bot = max(zero(FT), -c_bot)
                if k < nz
                    u1_blk[k] = -max(zero(FT), -c_top) / J_k
                end
                if k > 1
                    l1_blk[k - 1] = -(one_minus_detr * max(zero(FT), c_bot)) / J_k
                end
                d11[k] = max(
                    FT(0.1) / Δt,
                    inv(Δt) - (entr_k - detr_k) + (out_top + out_bot) / J_k,
                )
                r1_blk[k] = ρa1_in[k] / Δt + (k == 1 ? E_sfc : zero(FT))
            end

            column_thomas_solve!(x1_blk, r1_blk, d11, u1_blk, l1_blk, nz)

            for k in 1:nz
                ρa1_max_k =
                    min(sw.a_max * max(ᶜρ1_fv[k], ε), (1 - FT(1e-4)) * ᶜρ_fv[k])
                ᶜρa1_fv[k] = clamp(x1_blk[k], zero(FT), max(zero(FT), ρa1_max_k))
            end
        end
    end

    @inbounds for k in 1:nz
        ρ_k = ᶜρ_fv[k]
        ρ1_k = max(ᶜρ1_fv[k], ε)
        a1_k = draft_area(ᶜρa1_fv[k], ρ1_k)
        w_c = (u3_new[k] / ᶠdz_fv[k] + u3_new[k + 1] / ᶠdz_fv[k + 1]) / 2
        ab_k = a1_k == zero(FT) ? ab_zero : area_bounding_entr_detr(tcp, a1_k)
        z_k = ᶜz_fv[k]
        tke_k = max(zero(FT), ᶜρtke_fv[k] / max(ρ_k, ε))
        R_k = plume_radius(
            sw.scale_height,
            a1_k,
            sw.a_min,
            sw.a_max,
            z_k,
            z_sfc_val,
            sw.c_R,
            sw.R_max,
        )
        te_k = turbulent_entrainment(tcp, a1_k, z_k, z_sfc_val, tke_k, R_k)
        entr_dyn_k = max(zero(FT), ab_k) + ᶜevs_fv[k] * abs(w_c) + ᶜenr_fv[k]
        ε_tot_fv[k] = entr_dyn_k + te_k
        u3_c = FT(0.5) * (u3_new[k] + u3_new[k + 1])
        b_work =
            (u3_c * parent(ᶜlg_fv[k].gⁱʲ)[end]) *
            (ᶜρ_diff1_fv[k] + ᶜρ_ref_excess_fv[k] / ρ1_k) *
            _c3_scalar(ᶜgradΦ_fv[k])
        if HasTailEntr
            (ᶜqt0_fv, ᶜql0_fv, ᶜqi0_fv, ᶜTT_fv, ᶜqq_fv, ᶜcorr_fv) = tail_fvs
            ρa0_k = max(ρ_k - ᶜρa1_fv[k], zero(FT))
            tail_k = tail_entrainment_increments(
                sw.thermo_params,
                sw.f_tail,
                ᶜT0_fv[k],
                ᶜqt0_fv[k],
                ᶜql0_fv[k],
                ᶜqi0_fv[k],
                ᶜTT_fv[k],
                ᶜqq_fv[k],
                ᶜcorr_fv[k],
                ρa0_k,
                ᶜρa1_fv[k],
                entr_dyn_k,
                Δt,
            )
            b_work += entr_dyn_k * tail_k.mse
            tail_qt_fv[k] = entr_dyn_k * tail_k.q_tot
        end
        b_work_col[k] = b_work
    end
    return nothing
end

@inline function _solve_edmf_cloud_species!(
    ::Val{IsLiq},
    ::Val{HasBlend},
    sw,
    ᶜχ1_f,
    ᶜρχ_f,
    A0_f,
    Yc_col,
    blend_pre_col,
    edmf_col,
    pre_col,
    ᶜΦ_fv,
    ᶜqt1_fv,
    ᶜρqt_fv,
    ᶜql1_fv,
    ᶜqi1_fv,
    ᶜρql_fv,
    ᶜρqi_fv,
    χ1_in,
    χ0_in,
    ρa1_in,
    ρa0_in,
    ᶜρa1_fv,
    ρa0_new,
    ᶜρ_fv,
    V1_fv,
    ε_tot_fv,
    JF1_rel_fv,
    JF0_rel_fv,
    G1_fv,
    G0_fv,
    G0_b_fv,
    ᶜJ_fv,
    ᶠz_fv,
    α_face_fv,
    A0_qv_fv,
    A1_qv_fv,
    jf_extra_fv,
    A1_qr_fv,
    A0_qr_fv,
    A1_qs_fv,
    A0_qs_fv,
    wk,
    Δt::FT,
    nz::Int,
) where {IsLiq, HasBlend, FT}
    ε = eps(FT)
    ε_num = ϵ_numerics(FT)
    tps = sw.thermo_params
    ᶜp_fv = Fields.field_values(pre_col.ᶜp)

    has_in = false
    @inbounds for k in 1:nz
        ρχ_k = ᶜρχ_f[k]
        χ1_raw = max(zero(FT), ᶜχ1_f[k])
        χ1_p = if ρχ_k >= zero(FT) && ρa1_in[k] > ε_num
            min(χ1_raw, ρχ_k / ρa1_in[k])
        else
            χ1_raw
        end
        χ1_in[k] = χ1_p
        χ0_in[k] = (ρχ_k - ρa1_in[k] * χ1_p) / max(ρa0_in[k], ε)
        has_in |= (ρχ_k != zero(FT)) | (χ1_raw > zero(FT))
    end

    couple_cond = sw.couple_updraft_cond && !(sw.cmp === nothing)
    precomputed_relax = false
    if !has_in
        if !couple_cond
            @inbounds for k in 1:nz
                ᶜχ1_f[k] = zero(FT)
            end
            @inbounds for f in 1:(nz + 1)
                A0_f[f] = zero(FT)
            end
            return nothing
        end
        @inbounds for k in 1:nz
            wk.x1[k] = zero(FT)
        end
        sgs1_chk = Yc_col.sgsʲs.:(1)
        _compute_updraft_1m_relax!(
            Val(IsLiq),
            wk.r0,
            wk.x0,
            _unwrapped(sw.cmp),
            tps,
            sw.T_min_sgs,
            ᶜρa1_fv,
            Fields.field_values(sgs1_chk.mse),
            ᶜqt1_fv,
            IsLiq ? wk.x1 : ᶜql1_fv,
            IsLiq ? ᶜqi1_fv : wk.x1,
            Fields.field_values(sgs1_chk.q_rai),
            Fields.field_values(sgs1_chk.q_sno),
            ᶜΦ_fv,
            ᶜp_fv,
            true,
            nz,
        )
        has_cond_src = false
        @inbounds for k in 1:nz
            has_cond_src |= (wk.r0[k] > zero(FT)) & (wk.x0[k] > zero(FT))
        end
        if !has_cond_src
            @inbounds for k in 1:nz
                ᶜχ1_f[k] = zero(FT)
            end
            @inbounds for f in 1:(nz + 1)
                A0_f[f] = zero(FT)
            end
            return nothing
        end
        precomputed_relax = true
    end

    if HasBlend
        ᶜT0_fv = Fields.field_values(edmf_col.ᶜT0)
        ᶜl_mix_fv = Fields.field_values(blend_pre_col.ᶜl_mix)
        ᶜmom_fv = Fields.field_values(blend_pre_col.ᶜsgs_moments)
        ᶜρtke_fv = Fields.field_values(Yc_col.ρtke)
        phase = IsLiq ? TD.Liquid() : TD.Ice()
        τ_const = IsLiq ? sw.τ_lcl_const : sw.τ_icl_const
        const_liq_tau =
            IsLiq &&
            !(sw.cmp === nothing) &&
            (
                _unwrapped(sw.cmp).processes.cloud_liquid_formation isa
                Union{Nothing, CMP.CloudLiquidFormation}
            )

        @inbounds for k in 1:nz
            inv_ρ = inv(max(ᶜρ_fv[k], ε))
            T0_k = ᶜT0_fv[k]
            tke_k = max(zero(FT), ᶜρtke_fv[k] * inv_ρ)
            mom_k = ᶜmom_fv[k]
            σ_S_k = max(FT(mom_k.sigma_S), ε_num)
            C_k = ifelse(
                σ_S_k > ε_num,
                normal_cdf(FT(mom_k.λ_eq) / σ_S_k),
                FT(mom_k.λ_eq > zero(mom_k.λ_eq)),
            )
            need_thermo = (C_k > zero(FT)) | (!sw.use_const_τ & !const_liq_tau)
            qt0_k = zero(FT)
            ql0_k = zero(FT)
            qi0_k = zero(FT)
            ρ_air0 = one(FT)
            if need_thermo
                qt0_k = max(
                    zero(FT),
                    (ᶜρqt_fv[k] - ᶜρa1_fv[k] * ᶜqt1_fv[k]) / max(ρa0_new[k], ε),
                )
                ρa1_ql_k = IsLiq ? ρa1_in[k] : ᶜρa1_fv[k]
                ρa0_ql_k = IsLiq ? ρa0_in[k] : ρa0_new[k]
                ql0_k = max(
                    zero(FT),
                    (ᶜρql_fv[k] - ρa1_ql_k * max(zero(FT), ᶜql1_fv[k])) /
                    max(ρa0_ql_k, ε),
                )
                qi0_k = max(
                    zero(FT),
                    (ᶜρqi_fv[k] - ρa1_in[k] * max(zero(FT), ᶜqi1_fv[k])) /
                    max(ρa0_in[k], ε),
                )
                ρ_air0 = TD.air_density(tps, T0_k, ᶜp_fv[k], qt0_k, ql0_k, qi0_k)
            end
            τ_μ = if sw.use_const_τ
                τ_const
            else
                cloud_species_relaxation_time(
                    _unwrapped(sw.cmp), tps, T0_k, ρ_air0, qt0_k, ql0_k, qi0_k, phase,
                )
            end
            λ_k = cloud_species_flux_blend(ᶜl_mix_fv[k], tke_k, sw.c_d, τ_μ)
            c_qt_k, c_sl_k = if C_k <= zero(FT)
                (zero(FT), zero(FT))
            else
                equilibrium_condensate_sensitivities(
                    tps,
                    T0_k,
                    ρ_air0,
                    qt0_k,
                    ql0_k,
                    qi0_k,
                    mom_k.sigma_S,
                    mom_k.λ_eq,
                    phase,
                )
            end
            A1_qr_fv[k] = λ_k
            A0_qr_fv[k] = c_qt_k
            A1_qs_fv[k] = c_sl_k
            A0_qs_fv[k] = ρa0_in[k] * χ0_in[k]
        end

        @inbounds begin
            jf_extra_fv[1] = zero(FT)
            G0_b_fv[1] = zero(FT)
            for f in 2:nz
                λ_f = (A1_qr_fv[f - 1] + A1_qr_fv[f]) / 2
                c_qt_f = (A0_qr_fv[f - 1] + A0_qr_fv[f]) / 2
                c_sl_f = (A1_qs_fv[f - 1] + A1_qs_fv[f]) / 2
                jf_extra_fv[f] = λ_f * (c_qt_f * A0_qv_fv[f] + c_sl_f * A1_qv_fv[f])
                G0_b_fv[f] = (one(FT) - λ_f) * G0_fv[f]
            end
            jf_extra_fv[nz + 1] = zero(FT)
            G0_b_fv[nz + 1] = zero(FT)
        end
        _donor_limit_face_flux!(jf_extra_fv, A0_qs_fv, ᶜJ_fv, Δt, nz, FT)
    end

    G0_eff_fv = HasBlend ? G0_b_fv : G0_fv
    if couple_cond
        # Arguments shared by the transport-only solve and the solve with phase
        # relaxation.
        updraft_solve_args = (
            wk.x1,
            χ1_in,
            χ0_in,
            χ1_in,
            ᶜρa1_fv,
            V1_fv,
            ε_tot_fv,
            G1_fv,
            ᶜJ_fv,
            wk.diag1,
            wk.upper1,
            wk.lower1,
            wk.r1,
            Δt,
            nz,
            (;
                apply_diff = true,
                apply_sfc = false,
                has_extra_src1 = false,
                limit_pos = true,
            ),
            _no_surface_source(FT),
            sw.updraft_upw_mode,
        )
        if !precomputed_relax
            _solve_edmf_updraft_var!(updraft_solve_args...)
            sgs1 = Yc_col.sgsʲs.:(1)
            _compute_updraft_1m_relax!(
                Val(IsLiq),
                wk.r0,
                wk.x0,
                _unwrapped(sw.cmp),
                tps,
                sw.T_min_sgs,
                ᶜρa1_fv,
                Fields.field_values(sgs1.mse),
                ᶜqt1_fv,
                IsLiq ? wk.x1 : ᶜql1_fv,
                IsLiq ? ᶜqi1_fv : wk.x1,
                Fields.field_values(sgs1.q_rai),
                Fields.field_values(sgs1.q_sno),
                ᶜΦ_fv,
                ᶜp_fv,
                true,
                nz,
            )
        end
        _solve_edmf_updraft_var!(updraft_solve_args..., true, wk.r0, wk.x0)
        @inbounds for k in 1:nz
            q_sol_k = wk.x1[k]
            ᶜχ1_f[k] = q_sol_k
            Δρχ_src_k = Δt * ᶜρa1_fv[k] * wk.r0[k] * (wk.x0[k] - q_sol_k)
            ρq_other_k =
                IsLiq ? max(zero(FT), ᶜρqi_fv[k]) : max(zero(FT), ᶜρql_fv[k])
            ρqv_avail_k = max(
                zero(FT),
                ᶜρqt_fv[k] - max(zero(FT), ᶜρχ_f[k]) - ρq_other_k,
            )
            ᶜρχ_f[k] +=
                clamp(Δρχ_src_k, -max(zero(FT), ᶜρχ_f[k]), ρqv_avail_k)
        end
        _solve_and_commit_edmf_gm_var!(
            ᶜρχ_f,
            χ1_in,
            A0_f,
            ᶜρχ_f,
            ᶜχ1_f,
            ᶜχ1_f,
            α_face_fv,
            ᶜρa1_fv,
            ρa0_new,
            JF1_rel_fv,
            JF0_rel_fv,
            G1_fv,
            G0_eff_fv,
            jf_extra_fv,
            ᶜJ_fv,
            ᶠz_fv,
            wk.Φ1,
            wk.diag0,
            wk.upper0,
            wk.lower0,
            wk.r0,
            wk.x0,
            Δt,
            nz,
            (;
                limit_pos = true,
                apply_diff = true,
                apply_sfc = false,
                has_jf_extra = HasBlend,
                save_x0 = false,
            ),
            zero(FT),
            sw.sgsflux_upw_mode,
        )
    else
        _solve_and_commit_edmf_var!(
            ᶜχ1_f,
            ᶜρχ_f,
            χ1_in,
            A0_f,
            α_face_fv,
            ᶜρχ_f,
            χ1_in,
            χ0_in,
            ᶜρa1_fv,
            ρa0_new,
            V1_fv,
            ε_tot_fv,
            JF1_rel_fv,
            JF0_rel_fv,
            G1_fv,
            G0_eff_fv,
            jf_extra_fv,
            ᶜJ_fv,
            ᶠz_fv,
            wk,
            Δt,
            nz,
            (;
                apply_diff = true,
                apply_sfc = false,
                has_jf_extra = HasBlend,
                save_x0 = false,
                limit_pos = true,
            ),
            _no_surface_source(FT),
            sw.sgsflux_upw_mode,
            sw.updraft_upw_mode,
        )
    end
    return nothing
end

@inline function _picard_edmf_species_step!(
    out_p,
    ᶜχ1_sgs,
    ᶜρχ_gs,
    do_diff::Bool,
    ρa1_in,
    ρa0_in,
    χ1_in,
    χ0_in,
    ᶜρa1_fv,
    V1_fv,
    ε_tot_fv,
    G1_fv,
    ᶜJ_fv,
    diag1_fv,
    upper1_fv,
    lower1_fv,
    r1_blk,
    Δt,
    nz::Int,
    updraft_upw_mode::Int8,
)
    FT = typeof(Δt)
    ε = eps(FT)
    ε_num = ϵ_numerics(FT)
    has_sp_in = false
    @inbounds for k in 1:nz
        ρχ_k = ᶜρχ_gs[k]
        χ1_raw = max(zero(FT), ᶜχ1_sgs[k])
        χ1_in_k =
            (ρχ_k >= zero(FT) && ρa1_in[k] > ε_num) ?
            min(χ1_raw, ρχ_k / ρa1_in[k]) : χ1_raw
        χ1_in[k] = χ1_in_k
        χ0_in[k] = (ρχ_k - ρa1_in[k] * χ1_in_k) / max(ρa0_in[k], ε)
        has_sp_in |= (ρχ_k != zero(FT)) | (χ1_raw > zero(FT))
    end
    if has_sp_in
        _solve_edmf_updraft_var!(
            out_p,
            χ1_in,
            χ0_in,
            χ1_in,
            ᶜρa1_fv,
            V1_fv,
            ε_tot_fv,
            G1_fv,
            ᶜJ_fv,
            diag1_fv,
            upper1_fv,
            lower1_fv,
            r1_blk,
            Δt,
            nz,
            (;
                apply_diff = do_diff,
                apply_sfc = false,
                has_extra_src1 = false,
                limit_pos = true,
            ),
            _no_surface_source(FT),
            updraft_upw_mode,
        )
    else
        @inbounds for k in 1:nz
            out_p[k] = zero(FT)
        end
    end
    return nothing
end

@inline function _step_edmf_precip_species!(
    ᶜχ1_f,
    ᶜρχ_f,
    A0_f,
    ᶜqt1_fv,
    ᶜρqt_fv,
    A0_qt_fv,
    α_face_fv,
    χ1_in,
    χ0_in,
    ρa1_in,
    ρa0_in,
    ᶜρa1_fv,
    ρa0_new,
    V1_fv,
    ε_tot_fv,
    JF1_rel_fv,
    JF0_rel_fv,
    G1_fv,
    G0_fv,
    jf_extra_fv,
    ᶜJ_fv,
    ᶠz_fv,
    wk,
    Δt,
    nz::Int,
    sgsflux_upw_mode::Int8,
    updraft_upw_mode::Int8,
)
    FT = typeof(Δt)
    ε = eps(FT)
    ε_num = ϵ_numerics(FT)
    has_nz_p = false
    @inbounds for k in 1:nz
        ρχ_k = ᶜρχ_f[k]
        χ1_raw = max(zero(FT), ᶜχ1_f[k])
        χ1_p = if ρχ_k >= zero(FT) && ρa1_in[k] > ε_num
            min(χ1_raw, ρχ_k / ρa1_in[k])
        else
            χ1_raw
        end
        χ1_in[k] = χ1_p
        χ0_in[k] = (ρχ_k - ρa1_in[k] * χ1_p) / max(ρa0_in[k], ε)
        has_nz_p |= (ρχ_k != zero(FT)) | (χ1_raw > zero(FT))
    end
    if has_nz_p
        _solve_and_commit_edmf_var!(
            ᶜχ1_f,
            ᶜρχ_f,
            χ1_in,
            A0_f,
            α_face_fv,
            ᶜρχ_f,
            χ1_in,
            χ0_in,
            ᶜρa1_fv,
            ρa0_new,
            V1_fv,
            ε_tot_fv,
            JF1_rel_fv,
            JF0_rel_fv,
            G1_fv,
            G0_fv,
            jf_extra_fv,
            ᶜJ_fv,
            ᶠz_fv,
            wk,
            Δt,
            nz,
            (;
                apply_diff = false,
                apply_sfc = false,
                has_jf_extra = false,
                save_x0 = false,
                limit_pos = true,
            ),
            _no_surface_source(FT),
            sgsflux_upw_mode,
            updraft_upw_mode,
        )
        @inbounds for k in 1:nz
            ᶜqt1_fv[k] += ᶜχ1_f[k]
            ᶜρqt_fv[k] += ᶜρχ_f[k]
        end
        @inbounds for f in 1:(nz + 1)
            A0_qt_fv[f] += A0_f[f]
        end
    else
        @inbounds for k in 1:nz
            ᶜχ1_f[k] = zero(FT)
        end
        @inbounds for f in 1:(nz + 1)
            A0_f[f] = zero(FT)
        end
    end
    return nothing
end

@inline function (sw::EDMFTwoStreamColumnSweep{
    HasQ,
    HasNonEquil,
    FtildeMode,
    HasBlend,
    HasTailEntr,
})(
    Yc_col,
    Yf_col,
    geom_col,
    pre_col,
    sfc_col,
    edmf_col,
    ftilde_col,
    blend_pre_col,
    tail_col,
    scratch_s_col,
    scratch_v_col,
) where {HasQ, HasNonEquil, FtildeMode, HasBlend, HasTailEntr}
    FT = typeof(sw.Δt)
    Δt = sw.Δt
    ε = eps(FT)
    ε_num = ϵ_numerics(FT)
    tps = sw.thermo_params
    tcp = sw.turbconv_params

    ᶜρ_fv = Fields.field_values(Yc_col.ρ)
    ᶜρe_fv = Fields.field_values(Yc_col.ρe_tot)
    ᶜρtke_fv = Fields.field_values(Yc_col.ρtke)
    sgs1 = Yc_col.sgsʲs.:(1)
    ᶜρa1_fv = Fields.field_values(sgs1.ρa)
    ᶜmse1_fv = Fields.field_values(sgs1.mse)
    ᶠu31_fv = Fields.field_values(Yf_col.sgsʲs.:(1).u₃)

    ᶜlg_fv = Fields.field_values(geom_col.ᶜlg)
    ᶜJ_fv = Fields.field_values(geom_col.ᶜJ)
    ᶜz_fv = Fields.field_values(geom_col.ᶜz)
    z_sfc_val = Fields.field_values(geom_col.z_sfc)[]
    ᶠlg_fv = Fields.field_values(geom_col.ᶠlg)
    ᶠdz_fv = Fields.field_values(geom_col.ᶠdz)
    ᶜΦ_fv = Fields.field_values(geom_col.ᶜΦ)
    ᶠgradΦ_fv = Fields.field_values(geom_col.ᶠgradᵥ_ᶜΦ)
    ᶜgradΦ_fv = Fields.field_values(geom_col.ᶜgradᵥ_ᶠΦ)
    ᶠuh3_fv = Fields.field_values(geom_col.ᶠuₕ³)
    ᶜK_fv = Fields.field_values(pre_col.ᶜK)
    ᶜK1_fv = Fields.field_values(edmf_col.ᶜK1)
    ᶜK0_fv = Fields.field_values(edmf_col.ᶜK0)
    ᶜT0_fv = Fields.field_values(edmf_col.ᶜT0)
    ᶜρ1_fv = Fields.field_values(edmf_col.ᶜρ1)
    ᶜρ_diff1_fv = Fields.field_values(edmf_col.ᶜρ_diff1)
    ᶜρ_ref_excess_fv = Fields.field_values(edmf_col.ᶜρ_ref_excess)
    ᶜevs_fv = Fields.field_values(edmf_col.ᶜevs1)
    ᶜenr_fv = Fields.field_values(edmf_col.ᶜenr1)
    ᶠK_h_fv = Fields.field_values(pre_col.ᶠK_h)
    ᶠz_fv = Fields.field_values(geom_col.ᶠz)
    tail_fvs = if HasTailEntr
        (
            Fields.field_values(tail_col.ᶜqt0),
            Fields.field_values(tail_col.ᶜql0),
            Fields.field_values(tail_col.ᶜqi0),
            Fields.field_values(tail_col.ᶜTT),
            Fields.field_values(tail_col.ᶜqq),
            Fields.field_values(tail_col.ᶜcorr),
        )
    else
        ()
    end

    nz = Spaces.nlevels(axes(Yc_col.ρ))

    # Scratch aliases (`ρa1_prev` holds `ε_tot`, `JF1` holds `V1` after pass 1)
    ρa1_in = Fields.field_values(scratch_s_col.ρa1_in)
    ρa0_in = Fields.field_values(scratch_s_col.ρa0_in)
    ρa1_prev = Fields.field_values(scratch_s_col.ρa1_prev)
    ρa0_new = Fields.field_values(scratch_s_col.ρa0_new)
    b_work_col = Fields.field_values(scratch_s_col.E_col)
    extra_src1_col = Fields.field_values(scratch_s_col.D_col)
    tail_qt_fv = Fields.field_values(scratch_s_col.empty_reg)

    JF1_fv = Fields.field_values(scratch_s_col.JF1)
    JF1_rel_fv = Fields.field_values(scratch_s_col.JF1_rel)
    JF0_rel_fv = Fields.field_values(scratch_s_col.JF0_rel)

    uv_c = scratch_v_col.uv_c.components.data
    st_c = scratch_v_col.st_c.components.data
    c12_c = scratch_v_col.c12_c.components.data
    c3_c = scratch_v_col.c3_c.components.data
    ct123_c = scratch_v_col.ct123_c.components.data
    uvw_c = scratch_v_col.uvw_c.components.data
    uv_f = scratch_v_col.uv_f.components.data
    st_f = scratch_v_col.st_f.components.data
    c123_f = scratch_v_col.c123_f.components.data

    α_face_fv = Fields.field_values(uv_f.:(1))
    u3_in = Fields.field_values(uv_f.:(2))
    u3_new = Fields.field_values(uv_f.:(3))
    G0_fv = Fields.field_values(uv_f.:(5))
    G1_fv = Fields.field_values(uv_f.:(6))

    diag1_fv = Fields.field_values(uv_c.:(1))
    upper1_fv = Fields.field_values(uv_c.:(2))
    lower1_fv = Fields.field_values(uv_c.:(3))
    diag0_fv = Fields.field_values(uv_c.:(5))
    upper0_fv = Fields.field_values(uv_c.:(6))
    lower0_fv = Fields.field_values(uv_c.:(7))
    d11 = Fields.field_values(uv_c.:(9))
    d00 = Fields.field_values(st_c.:(3))
    u1_blk = Fields.field_values(st_c.:(4))
    l1_blk = Fields.field_values(st_c.:(6))
    r1_blk = Fields.field_values(st_c.:(8))
    r0_blk = Fields.field_values(st_c.:(9))
    x1_blk = Fields.field_values(c12_c.:(1))
    x0_blk = Fields.field_values(c12_c.:(2))
    χ1_in = Fields.field_values(c3_c.:(1))
    χ0_in = Fields.field_values(ct123_c.:(1))
    η1_L_fv = Fields.field_values(ct123_c.:(2))
    ρη_L_fv = Fields.field_values(ct123_c.:(3))
    Δρ_col_fv = Fields.field_values(uvw_c.:(1))
    dummy1_fv = Fields.field_values(uvw_c.:(2))
    dummy0_fv = Fields.field_values(uvw_c.:(3))
    A1_qv_fv = Fields.field_values(uv_f.:(7))
    A0_qv_fv = Fields.field_values(uv_f.:(8))
    A0_ql_fv = Fields.field_values(st_f.:(1))
    A0_qi_fv = Fields.field_values(st_f.:(3))
    A1_qr_fv = Fields.field_values(st_f.:(4))
    A0_qr_fv = Fields.field_values(st_f.:(5))
    A1_qs_fv = Fields.field_values(st_f.:(6))
    A0_qs_fv = Fields.field_values(st_f.:(7))
    A0_qt_fv = Fields.field_values(st_f.:(9))
    A0_η_fv = Fields.field_values(c123_f.:(2))

    # Surface scalar BC values (single-level horizontal fields)
    E_sfc = max(zero(FT), Fields.field_values(edmf_col.sfc_mf_src)[])
    mse_buoy = Fields.field_values(edmf_col.sfc_mse_buoy)[]
    qt_buoy = HasQ ? Fields.field_values(edmf_col.sfc_qt_buoy)[] : zero(FT)
    JF_sfc_h =
        sw.has_sfc ?
        surface_c3_flux_factor(ᶠlg_fv[1]) *
        _c3_scalar(Fields.field_values(sfc_col.ρ_flux_h_tot)[]) : zero(FT)
    JF_sfc_q =
        (sw.has_sfc && HasQ) ?
        surface_c3_flux_factor(ᶠlg_fv[1]) *
        _c3_scalar(Fields.field_values(sfc_col.ρ_flux_q_tot)[]) : zero(FT)

    # Save clamped entry states, with `ρa⁰ ≥ 10⁻⁴ ρ` as in the `ρa¹` solve.
    @inbounds for k in 1:nz
        ρa1_k = clamp(ᶜρa1_fv[k], zero(FT), (1 - FT(1e-4)) * ᶜρ_fv[k])
        ᶜρa1_fv[k] = ρa1_k
        ρa1_in[k] = ρa1_k
        ρa0_in[k] = ᶜρ_fv[k] - ρa1_k
    end
    @inbounds begin
        u3_in[1] = zero(FT)
        u3_new[1] = zero(FT)
        for f in 2:nz
            u3_in[f] = _c3_scalar(ᶠu31_fv[f])
            u3_new[f] = u3_in[f]
        end
        u3_in[nz + 1] = zero(FT)
        u3_new[nz + 1] = zero(FT)
    end

    # Step 1: Picard predictor rounds (`u₃¹ → ρa¹ → updraft scalars and 1M
    # phase relaxation → ρ_diff¹`), then final `u₃¹` and 2-pass `ρa¹` solves.
    V1_fv = JF1_fv
    ε_tot_fv = ρa1_prev
    ᶜp_fv = Fields.field_values(pre_col.ᶜp)
    ᶜT1_fv = Fields.field_values(pre_col.ᶜT1)

    # Without a draft or a surface mass-flux source, every `ρa¹` solve returns
    # `ρa¹ = 0`, and the predictor rounds change only the scalars of the empty
    # draft, which enter the grid-mean solves multiplied by `ρa¹`.
    has_draft = E_sfc > zero(FT)
    @inbounds for k in 1:nz
        has_draft |= ρa1_in[k] != zero(FT)
    end
    n_picard = has_draft ? Int(sw.n_picard) : 0

    # Arguments shared by the `u₃¹` and `ρa¹` solves of the Picard rounds and of
    # the final sweep.
    u3_ρa_args = (
        u3_new,
        ᶜρa1_fv,
        JF1_fv,
        ε_tot_fv,
        b_work_col,
        tail_qt_fv,
        u3_in,
        ρa1_in,
        ᶜρ_fv,
        ᶜρ1_fv,
        ᶜρ_diff1_fv,
        ᶜρ_ref_excess_fv,
        ᶜevs_fv,
        ᶜenr_fv,
        ᶜρtke_fv,
        ᶜT0_fv,
        tail_fvs,
        ᶜlg_fv,
        ᶜJ_fv,
        ᶜz_fv,
        z_sfc_val,
        ᶠlg_fv,
        ᶠdz_fv,
        ᶠz_fv,
        ᶠgradΦ_fv,
        ᶜgradΦ_fv,
        ᶠuh3_fv,
        d11,
        u1_blk,
        l1_blk,
        r1_blk,
        x1_blk,
        E_sfc,
        Δt,
        nz,
    )
    for _picard in 1:n_picard
        _step_edmf_u3_and_rhoa!(sw, u3_ρa_args...)

        @inbounds begin
            V1_fv[1] = zero(FT)
            V1_fv[nz + 1] = zero(FT)
            G1_fv[1] = zero(FT)
            G1_fv[nz + 1] = zero(FT)
            for f in 2:nz
                V1_fv[f] =
                    ᶠlg_fv[f].J * _draft_face_u3_contravariant(
                        ᶠuh3_fv[f], ᶠlg_fv[f], u3_new[f],
                    )
                G1_fv[f] = if sw.has_updraft_diff
                    ρa1_lo = ᶜρa1_fv[f - 1]
                    ρa1_hi = ᶜρa1_fv[f]
                    ρa1_harm =
                        (ρa1_lo > ε_num && ρa1_hi > ε_num) ?
                        FT(2) * ρa1_lo * (ρa1_hi / (ρa1_lo + ρa1_hi)) : zero(FT)
                    face_conductance(
                        ᶠlg_fv[f], sw.updraft_kh_factor * ρa1_harm * ᶠK_h_fv[f],
                    )
                else
                    zero(FT)
                end
            end
        end

        ρ1_1_p = max(ᶜρ1_fv[1], ε)
        r_sfc_p =
            sw.has_sfc ?
            edmfx_sfc_relaxation_rate(E_sfc, ρa1_in[1], ρ1_1_p, sw.a_min) :
            zero(FT)

        mse1_p = diag0_fv
        qt1_p = upper0_fv
        ql1_p = lower0_fv
        qi1_p = d00
        qr1_p = r0_blk
        qs1_p = x0_blk

        @inbounds for k in 1:nz
            mse1_old_k = ᶜmse1_fv[k]
            ρh_tot_k = ᶜρe_fv[k] + ᶜp_fv[k]
            χ1_in[k] = mse1_old_k
            χ0_in[k] =
                ((ρh_tot_k - ᶜρ_fv[k] * ᶜK_fv[k]) - ρa1_in[k] * mse1_old_k) /
                max(ρa0_in[k], ε)
        end
        _solve_edmf_updraft_var!(
            mse1_p,
            χ1_in,
            χ0_in,
            b_work_col,
            ᶜρa1_fv,
            V1_fv,
            ε_tot_fv,
            G1_fv,
            ᶜJ_fv,
            diag1_fv,
            upper1_fv,
            lower1_fv,
            r1_blk,
            Δt,
            nz,
            (;
                apply_diff = sw.has_updraft_diff,
                apply_sfc = sw.has_sfc,
                has_extra_src1 = true,
                limit_pos = false,
            ),
            (; JF_sfc = JF_sfc_h, r_sfc_1 = r_sfc_p, ρ1_1 = ρ1_1_p, χ_buoyant = mse_buoy),
            sw.updraft_upw_mode,
        )

        if HasQ
            ᶜρqt_p_fv = Fields.field_values(Yc_col.ρq_tot)
            ᶜqt1_p_fv = Fields.field_values(sgs1.q_tot)
            @inbounds for k in 1:nz
                ρqt_k = ᶜρqt_p_fv[k]
                qt1_raw = max(zero(FT), ᶜqt1_p_fv[k])
                qt1_in_k =
                    (ρqt_k >= zero(FT) && ρa1_in[k] > ε_num) ?
                    min(qt1_raw, ρqt_k / ρa1_in[k]) : qt1_raw
                χ1_in[k] = qt1_in_k
                χ0_in[k] = (ρqt_k - ρa1_in[k] * qt1_in_k) / max(ρa0_in[k], ε)
            end
            _solve_edmf_updraft_var!(
                qt1_p,
                χ1_in,
                χ0_in,
                tail_qt_fv,
                ᶜρa1_fv,
                V1_fv,
                ε_tot_fv,
                G1_fv,
                ᶜJ_fv,
                diag1_fv,
                upper1_fv,
                lower1_fv,
                r1_blk,
                Δt,
                nz,
                (;
                    apply_diff = sw.has_updraft_diff,
                    apply_sfc = sw.has_sfc,
                    has_extra_src1 = HasTailEntr,
                    limit_pos = true,
                ),
                (;
                    JF_sfc = JF_sfc_q,
                    r_sfc_1 = r_sfc_p,
                    ρ1_1 = ρ1_1_p,
                    χ_buoyant = qt_buoy,
                ),
                sw.updraft_upw_mode,
            )

            if HasNonEquil
                pic_sp_tail = (
                    ρa1_in,
                    ρa0_in,
                    χ1_in,
                    χ0_in,
                    ᶜρa1_fv,
                    V1_fv,
                    ε_tot_fv,
                    G1_fv,
                    ᶜJ_fv,
                    diag1_fv,
                    upper1_fv,
                    lower1_fv,
                    r1_blk,
                    Δt,
                    nz,
                    sw.updraft_upw_mode,
                )
                _picard_edmf_species_step!(
                    ql1_p,
                    Fields.field_values(sgs1.q_lcl),
                    Fields.field_values(Yc_col.ρq_lcl),
                    sw.has_updraft_diff,
                    pic_sp_tail...,
                )
                _picard_edmf_species_step!(
                    qi1_p,
                    Fields.field_values(sgs1.q_icl),
                    Fields.field_values(Yc_col.ρq_icl),
                    sw.has_updraft_diff,
                    pic_sp_tail...,
                )
                _picard_edmf_species_step!(
                    qr1_p,
                    Fields.field_values(sgs1.q_rai),
                    Fields.field_values(Yc_col.ρq_rai),
                    false,
                    pic_sp_tail...,
                )
                _picard_edmf_species_step!(
                    qs1_p,
                    Fields.field_values(sgs1.q_sno),
                    Fields.field_values(Yc_col.ρq_sno),
                    false,
                    pic_sp_tail...,
                )
                if HasNonEquil && !(sw.cmp === nothing)
                    mp_1m_p = _unwrapped(sw.cmp)
                    _compute_updraft_1m_relax!(
                        Val(true),
                        η1_L_fv,
                        ρη_L_fv,
                        mp_1m_p,
                        tps,
                        sw.T_min_sgs,
                        ᶜρa1_fv,
                        mse1_p,
                        qt1_p,
                        ql1_p,
                        qi1_p,
                        qr1_p,
                        qs1_p,
                        ᶜΦ_fv,
                        ᶜp_fv,
                        false,
                        nz,
                    )
                    ᶜql1_sgs = Fields.field_values(sgs1.q_lcl)
                    ᶜρql_gs = Fields.field_values(Yc_col.ρq_lcl)
                    has_ql_p = false
                    @inbounds for k in 1:nz
                        ρχ_k = ᶜρql_gs[k]
                        χ1_raw = max(zero(FT), ᶜql1_sgs[k])
                        χ1_in_k =
                            (ρχ_k >= zero(FT) && ρa1_in[k] > ε_num) ?
                            min(χ1_raw, ρχ_k / ρa1_in[k]) : χ1_raw
                        χ1_in[k] = χ1_in_k
                        χ0_in[k] =
                            (ρχ_k - ρa1_in[k] * χ1_in_k) /
                            max(ρa0_in[k], ε)
                        has_ql_p |=
                            (ρχ_k != zero(FT)) |
                            (χ1_raw > zero(FT)) |
                            ((η1_L_fv[k] > zero(FT)) & (ρη_L_fv[k] > zero(FT)))
                    end
                    # Arguments shared by the cloud liquid and cloud ice
                    # predictor solves.
                    species_solve_args = (
                        χ1_in,
                        χ0_in,
                        χ1_in,
                        ᶜρa1_fv,
                        V1_fv,
                        ε_tot_fv,
                        G1_fv,
                        ᶜJ_fv,
                        diag1_fv,
                        upper1_fv,
                        lower1_fv,
                        r1_blk,
                        Δt,
                        nz,
                        (;
                            apply_diff = sw.has_updraft_diff,
                            apply_sfc = false,
                            has_extra_src1 = false,
                            limit_pos = true,
                        ),
                        _no_surface_source(FT),
                        sw.updraft_upw_mode,
                        true,
                        η1_L_fv,
                        ρη_L_fv,
                    )
                    if has_ql_p
                        _solve_edmf_updraft_var!(ql1_p, species_solve_args...)
                    end

                    _compute_updraft_1m_relax!(
                        Val(false),
                        η1_L_fv,
                        ρη_L_fv,
                        mp_1m_p,
                        tps,
                        sw.T_min_sgs,
                        ᶜρa1_fv,
                        mse1_p,
                        qt1_p,
                        ql1_p,
                        qi1_p,
                        qr1_p,
                        qs1_p,
                        ᶜΦ_fv,
                        ᶜp_fv,
                        false,
                        nz,
                    )
                    ᶜqi1_sgs = Fields.field_values(sgs1.q_icl)
                    ᶜρqi_gs = Fields.field_values(Yc_col.ρq_icl)
                    has_qi_p = false
                    @inbounds for k in 1:nz
                        ρχ_k = ᶜρqi_gs[k]
                        χ1_raw = max(zero(FT), ᶜqi1_sgs[k])
                        χ1_in_k =
                            (ρχ_k >= zero(FT) && ρa1_in[k] > ε_num) ?
                            min(χ1_raw, ρχ_k / ρa1_in[k]) : χ1_raw
                        χ1_in[k] = χ1_in_k
                        χ0_in[k] =
                            (ρχ_k - ρa1_in[k] * χ1_in_k) /
                            max(ρa0_in[k], ε)
                        has_qi_p |=
                            (ρχ_k != zero(FT)) |
                            (χ1_raw > zero(FT)) |
                            ((η1_L_fv[k] > zero(FT)) & (ρη_L_fv[k] > zero(FT)))
                    end
                    if has_qi_p
                        _solve_edmf_updraft_var!(qi1_p, species_solve_args...)
                    end
                end
            end
        end

        @inbounds for k in 1:nz
            h1_k = mse1_p[k] - ᶜΦ_fv[k]
            p_k = ᶜp_fv[k]
            if !HasQ
                T1_k = max(sw.T_min_sgs, TD.air_temperature(tps, TD.ph(), h1_k))
                ρ1_k = max(ε, TD.air_density(tps, T1_k, p_k))
                ᶜT1_fv[k] = T1_k
                ᶜρ1_fv[k] = ρ1_k
                ᶜρ_diff1_fv[k] = (ρ1_k - ᶜρ_fv[k] - ᶜρ_ref_excess_fv[k]) / ρ1_k
            elseif !HasNonEquil
                qt_nn = max(zero(FT), qt1_p[k])
                sa1 = TD.saturation_adjustment(tps, TD.ph(), p_k, h1_k, qt_nn)
                T1_k = max(sw.T_min_sgs, sa1.T)
                ρ1_k = max(
                    ε,
                    TD.air_density(tps, T1_k, p_k, qt_nn, sa1.q_liq, sa1.q_ice),
                )
                ᶜT1_fv[k] = T1_k
                ᶜρ1_fv[k] = ρ1_k
                ᶜρ_diff1_fv[k] = (ρ1_k - ᶜρ_fv[k] - ᶜρ_ref_excess_fv[k]) / ρ1_k
            else
                ql_tot_p = max(zero(FT), ql1_p[k]) + max(zero(FT), qr1_p[k])
                qi_tot_p = max(zero(FT), qi1_p[k]) + max(zero(FT), qs1_p[k])
                qt_nn_p = max(ql_tot_p + qi_tot_p, max(zero(FT), qt1_p[k]))
                T1_k = max(
                    sw.T_min_sgs,
                    TD.air_temperature(
                        tps, TD.ph(), h1_k, qt_nn_p, ql_tot_p, qi_tot_p,
                    ),
                )
                ρ1_k = max(
                    ε,
                    TD.air_density(
                        tps, T1_k, p_k, qt_nn_p, ql_tot_p, qi_tot_p,
                    ),
                )
                ᶜT1_fv[k] = T1_k
                ᶜρ1_fv[k] = ρ1_k
                ᶜρ_diff1_fv[k] = (ρ1_k - ᶜρ_fv[k] - ᶜρ_ref_excess_fv[k]) / ρ1_k
            end
        end
    end

    _step_edmf_u3_and_rhoa!(sw, u3_ρa_args...)

    # Write final u₃¹, relative SGS mass fluxes `JF1_rel = -JF0_rel`,
    # contravariant updraft volume flux `V1_fv`, and SGS flux weight
    # `α_face_fv` (`_draft_relative_face_fluxes`).
    @inbounds begin
        JF1_rel_fv[1] = zero(FT)
        JF0_rel_fv[1] = zero(FT)
        V1_fv[1] = zero(FT)
        α_face_fv[1] = zero(FT)
        for f in 2:nz
            u3_f = u3_new[f]
            ᶠu31_fv[f] = C3(u3_f)
            u3_cont_gm = if FtildeMode === :gm_flux
                Jρ_gm_f =
                    (ᶜρ_fv[f - 1] * ᶜJ_fv[f - 1] + ᶜρ_fv[f] * ᶜJ_fv[f]) / 2
                _extensive_face_flux(
                    Fields.field_values(ftilde_col)[f],
                    ᶠlg_fv[f],
                ) / max(Jρ_gm_f, ε)
            else
                Fields.field_values(pre_col.ᶠu3_gm)[f].u³
            end
            u3_cont_1 =
                _draft_face_u3_contravariant(ᶠuh3_fv[f], ᶠlg_fv[f], u3_f)
            F_rel_f, V1_f, W1_f = _draft_relative_face_fluxes(
                JF1_fv[f],
                u3_cont_1,
                u3_cont_gm,
                ᶜρa1_fv[f - 1],
                ᶜρa1_fv[f],
                ᶜρ_fv[f - 1],
                ᶜρ_fv[f],
                ᶜJ_fv[f - 1],
                ᶜJ_fv[f],
                ᶠlg_fv[f].J,
            )
            JF1_rel_fv[f] = sw.has_sgs_mass_flux ? F_rel_f : zero(FT)
            JF0_rel_fv[f] = -JF1_rel_fv[f]
            V1_fv[f] = V1_f
            α_face_fv[f] = sw.has_sgs_mass_flux ? W1_f : zero(FT)
        end
        JF1_rel_fv[nz + 1] = zero(FT)
        JF0_rel_fv[nz + 1] = zero(FT)
        V1_fv[nz + 1] = zero(FT)
        α_face_fv[nz + 1] = zero(FT)

        for k in 1:nz
            ρa0_new[k] = ᶜρ_fv[k] - ᶜρa1_fv[k]
        end
    end

    ρ1_1 = max(ᶜρ1_fv[1], ε)
    r_sfc_1 =
        sw.has_sfc ?
        edmfx_sfc_relaxation_rate(E_sfc, ρa1_in[1], ρ1_1, sw.a_min) : zero(FT)

    # ── Step 2: Face diffusive conductances + decoupled scalar solves ──
    @inbounds for f in 1:(nz + 1)
        if 1 < f <= nz
            G0_fv[f] = if sw.has_env_diff
                ρa0_f = (ρa0_new[f - 1] + ρa0_new[f]) / 2
                face_conductance(ᶠlg_fv[f], ρa0_f * ᶠK_h_fv[f])
            else
                zero(FT)
            end
            G1_fv[f] = if sw.has_updraft_diff
                ρa1_lo = ᶜρa1_fv[f - 1]
                ρa1_hi = ᶜρa1_fv[f]
                ρa1_harm =
                    (ρa1_lo > ε_num && ρa1_hi > ε_num) ?
                    FT(2) * ρa1_lo * (ρa1_hi / (ρa1_lo + ρa1_hi)) : zero(FT)
                face_conductance(
                    ᶠlg_fv[f], sw.updraft_kh_factor * ρa1_harm * ᶠK_h_fv[f],
                )
            else
                zero(FT)
            end
        else
            G0_fv[f] = zero(FT)
            G1_fv[f] = zero(FT)
        end
    end

    # `u3_in` (`uv_f.:(2)`) and `u3_new` (`uv_f.:(3)`) are free after Step 1.
    Φ1_fv = u3_in
    G0_b_fv = u3_new
    wk = (;
        diag1 = diag1_fv,
        upper1 = upper1_fv,
        lower1 = lower1_fv,
        diag0 = diag0_fv,
        upper0 = upper0_fv,
        lower0 = lower0_fv,
        Φ1 = Φ1_fv,
        r1 = r1_blk,
        r0 = r0_blk,
        x1 = x1_blk,
        x0 = x0_blk,
    )

    ᶜp_fv = Fields.field_values(pre_col.ᶜp)
    ᶜT0_fv = Fields.field_values(edmf_col.ᶜT0)
    ᶜT1_fv = Fields.field_values(pre_col.ᶜT1)
    jf_extra_fv = Fields.field_values(c123_f.:(3))

    if HasQ
        ᶜρqt_fv = Fields.field_values(Yc_col.ρq_tot)
        ᶜqt1_fv = Fields.field_values(sgs1.q_tot)
        @inbounds for k in 1:nz
            Δρ_col_fv[k] = ᶜρqt_fv[k]
        end
        qt_buoy_eff = qt_buoy

        if HasNonEquil
            ᶜρql_fv = Fields.field_values(Yc_col.ρq_lcl)
            ᶜρqi_fv = Fields.field_values(Yc_col.ρq_icl)
            ᶜρqr_fv = Fields.field_values(Yc_col.ρq_rai)
            ᶜρqs_fv = Fields.field_values(Yc_col.ρq_sno)
            ᶜql1_fv = Fields.field_values(sgs1.q_lcl)
            ᶜqi1_fv = Fields.field_values(sgs1.q_icl)
            ᶜqr1_fv = Fields.field_values(sgs1.q_rai)
            ᶜqs1_fv = Fields.field_values(sgs1.q_sno)

            # Save entry q_vap (with conservative entry projection matching
            # projected q_tot_eff - q_lcl - q_icl) for shared-FCT bounds.
            @inbounds for k in 1:nz
                ρqv_gs = max(
                    zero(FT),
                    ᶜρqt_fv[k] - (
                        max(zero(FT), ᶜρql_fv[k]) +
                        max(zero(FT), ᶜρqi_fv[k]) +
                        max(zero(FT), ᶜρqr_fv[k]) +
                        max(zero(FT), ᶜρqs_fv[k])
                    ),
                )
                qv1_raw = max(
                    zero(FT),
                    ᶜqt1_fv[k] - (
                        max(zero(FT), ᶜql1_fv[k]) +
                        max(zero(FT), ᶜqi1_fv[k]) +
                        max(zero(FT), ᶜqr1_fv[k]) +
                        max(zero(FT), ᶜqs1_fv[k])
                    ),
                )
                qv1_p =
                    ρa1_in[k] > ε_num ? min(qv1_raw, ρqv_gs / ρa1_in[k]) : qv1_raw
                dummy1_fv[k] = qv1_p
                dummy0_fv[k] = (ρqv_gs - ρa1_in[k] * qv1_p) / max(ρa0_in[k], ε)

                # Reduce grid-mean ρq_tot to ρq_tot_eff before the flux-divergence solve.
                ρqte_k = ᶜρqt_fv[k] - (ᶜρqr_fv[k] + ᶜρqs_fv[k])
                ᶜρqt_fv[k] = ρqte_k
                ql1_p = if ᶜρql_fv[k] >= zero(FT) && ρa1_in[k] > ε_num
                    min(max(zero(FT), ᶜql1_fv[k]), ᶜρql_fv[k] / ρa1_in[k])
                else
                    max(zero(FT), ᶜql1_fv[k])
                end
                qi1_p = if ᶜρqi_fv[k] >= zero(FT) && ρa1_in[k] > ε_num
                    min(max(zero(FT), ᶜqi1_fv[k]), ᶜρqi_fv[k] / ρa1_in[k])
                else
                    max(zero(FT), ᶜqi1_fv[k])
                end
                qte1_sum = qv1_p + ql1_p + qi1_p
                qte1_p = if ρqte_k >= zero(FT) && ρa1_in[k] > ε_num
                    min(qte1_sum, ρqte_k / ρa1_in[k])
                else
                    qte1_sum
                end
                χ1_in[k] = qte1_p
                χ0_in[k] = (ρqte_k - ρa1_in[k] * qte1_p) / max(ρa0_in[k], ε)
            end

            # Surface relaxation acts on the full `q_tot¹`, so the cell-1 rain
            # and snow shift the target of `q_tot¹ - q_rai¹ - q_sno¹`.
            @inbounds begin
                qr1_1_raw = max(zero(FT), ᶜqr1_fv[1])
                qr1_1_p =
                    (ᶜρqr_fv[1] >= zero(FT) && ρa1_in[1] > ε_num) ?
                    min(qr1_1_raw, ᶜρqr_fv[1] / ρa1_in[1]) : qr1_1_raw
                qs1_1_raw = max(zero(FT), ᶜqs1_fv[1])
                qs1_1_p =
                    (ᶜρqs_fv[1] >= zero(FT) && ρa1_in[1] > ε_num) ?
                    min(qs1_1_raw, ᶜρqs_fv[1] / ρa1_in[1]) : qs1_1_raw
                qt_buoy_eff = qt_buoy - (qr1_1_p + qs1_1_p)
            end
        else
            @inbounds for k in 1:nz
                ρqt_k = ᶜρqt_fv[k]
                qt1_raw = max(zero(FT), ᶜqt1_fv[k])
                qt1_p = if ρqt_k >= zero(FT) && ρa1_in[k] > ε_num
                    min(qt1_raw, ρqt_k / ρa1_in[k])
                else
                    qt1_raw
                end
                χ1_in[k] = qt1_p
                χ0_in[k] = (ρqt_k - ρa1_in[k] * qt1_p) / max(ρa0_in[k], ε)
            end
        end

        # 1) q_tot_eff = q_tot - q_rai - q_sno (x0 saved into u1_blk for η)
        _solve_and_commit_edmf_var!(
            ᶜqt1_fv,
            ᶜρqt_fv,
            u1_blk,
            A0_qt_fv,
            α_face_fv,
            ᶜρqt_fv,
            χ1_in,
            χ0_in,
            ᶜρa1_fv,
            ρa0_new,
            V1_fv,
            ε_tot_fv,
            JF1_rel_fv,
            JF0_rel_fv,
            G1_fv,
            G0_fv,
            jf_extra_fv,
            ᶜJ_fv,
            ᶠz_fv,
            wk,
            Δt,
            nz,
            (;
                apply_diff = true,
                apply_sfc = sw.has_sfc,
                has_jf_extra = false,
                save_x0 = true,
                limit_pos = true,
            ),
            (; JF_sfc = JF_sfc_q, r_sfc_1, ρ1_1, χ_buoyant = qt_buoy_eff),
            sw.sgsflux_upw_mode,
            sw.updraft_upw_mode,
            HasTailEntr,
            tail_qt_fv,
        )

        if HasBlend
            @inbounds for f in 2:nz
                A0_qv_fv[f] = -G0_fv[f] * (u1_blk[f] - u1_blk[f - 1])
            end
        end
    end

    # 2) Energy: solve intensive updraft `mse¹` directly with buoyancy work and
    # `mse_eff¹` diffusion, then solve conservative grid-mean `ρh_tot` with
    # `η¹ = mse¹ + K¹` and full subdomain enthalpy diffusion.
    @inbounds begin
        A1_qr_fv[1] = zero(FT)
        A1_qr_fv[nz + 1] = zero(FT)
        jf_extra_fv[1] = zero(FT)
        jf_extra_fv[nz + 1] = zero(FT)
        for f in 2:nz
            jf_c0_f = zero(FT)
            jf_p1_f = zero(FT)
            jf_c1_dq_f = zero(FT)
            if sw.has_env_diff
                hd0_f =
                    (
                        TD.enthalpy_dry(tps, ᶜT0_fv[f - 1]) + ᶜΦ_fv[f - 1] +
                        TD.enthalpy_dry(tps, ᶜT0_fv[f]) + ᶜΦ_fv[f]
                    ) / 2
                dq0_f = HasQ ? (u1_blk[f] - u1_blk[f - 1]) : zero(FT)
                corr0 = -(ᶜK0_fv[f] - ᶜK0_fv[f - 1]) + hd0_f * dq0_f
                if HasQ && HasNonEquil
                    ᶜρqr_fv = Fields.field_values(Yc_col.ρq_rai)
                    ᶜρqs_fv = Fields.field_values(Yc_col.ρq_sno)
                    ᶜqr1_fv = Fields.field_values(sgs1.q_rai)
                    ᶜqs1_fv = Fields.field_values(sgs1.q_sno)
                    qr0_lo =
                        _env_residual_value(
                            ᶜρqr_fv[f - 1], ρa1_in[f - 1], ᶜqr1_fv[f - 1], ρa0_in[f - 1], ε,
                        )
                    qr0_hi =
                        _env_residual_value(
                            ᶜρqr_fv[f], ρa1_in[f], ᶜqr1_fv[f], ρa0_in[f], ε,
                        )
                    qs0_lo =
                        _env_residual_value(
                            ᶜρqs_fv[f - 1], ρa1_in[f - 1], ᶜqs1_fv[f - 1], ρa0_in[f - 1], ε,
                        )
                    qs0_hi =
                        _env_residual_value(
                            ᶜρqs_fv[f], ρa1_in[f], ᶜqs1_fv[f], ρa0_in[f], ε,
                        )
                    hl_d0_f =
                        _face_mean_enthalpy_excess(
                            TD.enthalpy_liquid, tps, ᶜT0_fv[f - 1], ᶜT0_fv[f],
                        )
                    hi_d0_f =
                        _face_mean_enthalpy_excess(
                            TD.enthalpy_ice, tps, ᶜT0_fv[f - 1], ᶜT0_fv[f],
                        )
                    corr0 -= hl_d0_f * (qr0_hi - qr0_lo) + hi_d0_f * (qs0_hi - qs0_lo)
                end
                jf_c0_f = -G0_fv[f] * corr0
            end
            if sw.has_updraft_diff
                hd1_f =
                    (
                        TD.enthalpy_dry(tps, ᶜT1_fv[f - 1]) + ᶜΦ_fv[f - 1] +
                        TD.enthalpy_dry(tps, ᶜT1_fv[f]) + ᶜΦ_fv[f]
                    ) / 2
                dq1_f = HasQ ? (ᶜqt1_fv[f] - ᶜqt1_fv[f - 1]) : zero(FT)
                if HasQ && HasNonEquil
                    ᶜqr1_fv = Fields.field_values(sgs1.q_rai)
                    ᶜqs1_fv = Fields.field_values(sgs1.q_sno)
                    hl_d1_f =
                        _face_mean_enthalpy_excess(
                            TD.enthalpy_liquid, tps, ᶜT1_fv[f - 1], ᶜT1_fv[f],
                        )
                    hi_d1_f =
                        _face_mean_enthalpy_excess(
                            TD.enthalpy_ice, tps, ᶜT1_fv[f - 1], ᶜT1_fv[f],
                        )
                    jf_p1_f =
                        G1_fv[f] * (
                            hl_d1_f * (ᶜqr1_fv[f] - ᶜqr1_fv[f - 1]) +
                            hi_d1_f * (ᶜqs1_fv[f] - ᶜqs1_fv[f - 1])
                        )
                end
                jf_c1_dq_f = -G1_fv[f] * hd1_f * dq1_f
            end
            A1_qr_fv[f] = jf_p1_f
            jf_extra_fv[f] = jf_c0_f + jf_p1_f + jf_c1_dq_f
        end
    end

    @inbounds for k in 1:nz
        J_k = ᶜJ_fv[k]
        mse1_old_k = ᶜmse1_fv[k]
        ρh_tot_k = ᶜρe_fv[k] + ᶜp_fv[k]
        d00[k] = ρh_tot_k
        χ1_in[k] = mse1_old_k
        χ0_in[k] =
            ((ρh_tot_k - ᶜρ_fv[k] * ᶜK_fv[k]) - ρa1_in[k] * mse1_old_k) /
            max(ρa0_in[k], ε)
        inv_m1_k = ᶜρa1_fv[k] > ε_num ? inv(J_k * ᶜρa1_fv[k]) : zero(FT)
        extra_src1_col[k] =
            b_work_col[k] - inv_m1_k * (A1_qr_fv[k + 1] - A1_qr_fv[k])
    end

    _solve_edmf_updraft_var!(
        ᶜmse1_fv,
        χ1_in,
        χ0_in,
        extra_src1_col,
        ᶜρa1_fv,
        V1_fv,
        ε_tot_fv,
        G1_fv,
        ᶜJ_fv,
        wk.diag1,
        wk.upper1,
        wk.lower1,
        wk.r1,
        Δt,
        nz,
        (;
            apply_diff = true,
            apply_sfc = sw.has_sfc,
            has_extra_src1 = true,
            limit_pos = false,
        ),
        (; JF_sfc = JF_sfc_h, r_sfc_1, ρ1_1, χ_buoyant = mse_buoy),
        sw.updraft_upw_mode,
    )

    # The environment carries `mse⁰ + K⁰` with `K⁰` from `u⁰`. It differs from
    # the residual `(ρ h_tot - ρa¹ (mse¹ + K¹)) / ρa⁰` by the storage offset
    # `(ρ K - ρa¹ K¹ - ρa⁰ K⁰) / ρa⁰`.
    @inbounds for k in 1:nz
        η1_L_fv[k] = ᶜmse1_fv[k] + ᶜK1_fv[k]
        χ0_in[k] =
            ᶜρ_fv[k] * ᶜK_fv[k] - ᶜρa1_fv[k] * ᶜK1_fv[k] -
            ρa0_new[k] * ᶜK0_fv[k]
    end

    _solve_and_commit_edmf_gm_var!(
        ρη_L_fv,
        χ1_in,
        A0_η_fv,
        d00,
        η1_L_fv,
        ᶜmse1_fv,
        α_face_fv,
        ᶜρa1_fv,
        ρa0_new,
        JF1_rel_fv,
        JF0_rel_fv,
        G1_fv,
        G0_fv,
        jf_extra_fv,
        ᶜJ_fv,
        ᶠz_fv,
        wk.Φ1,
        wk.diag0,
        wk.upper0,
        wk.lower0,
        wk.r0,
        wk.x0,
        Δt,
        nz,
        (;
            limit_pos = false,
            apply_diff = true,
            apply_sfc = sw.has_sfc,
            has_jf_extra = true,
            save_x0 = false,
        ),
        JF_sfc_h,
        sw.sgsflux_upw_mode,
        true,
        χ0_in,
    )

    @inbounds if HasBlend
        ᶜqr1_fv = Fields.field_values(sgs1.q_rai)
        ᶜqs1_fv = Fields.field_values(sgs1.q_sno)
        ᶜρqr_fv = Fields.field_values(Yc_col.ρq_rai)
        ᶜρqs_fv = Fields.field_values(Yc_col.ρq_sno)
        A1_qv_fv[1] = zero(FT)
        for f in 2:nz
            hv_d0_f =
                _face_mean_enthalpy_excess(
                    TD.enthalpy_vapor, tps, ᶜT0_fv[f - 1], ᶜT0_fv[f],
                )
            hl_d0_f =
                _face_mean_enthalpy_excess(
                    TD.enthalpy_liquid, tps, ᶜT0_fv[f - 1], ᶜT0_fv[f],
                )
            hi_d0_f =
                _face_mean_enthalpy_excess(
                    TD.enthalpy_ice, tps, ᶜT0_fv[f - 1], ᶜT0_fv[f],
                )
            qr0_lo =
                _env_residual_value(
                    ᶜρqr_fv[f - 1], ρa1_in[f - 1], ᶜqr1_fv[f - 1], ρa0_in[f - 1], ε,
                )
            qr0_hi =
                _env_residual_value(
                    ᶜρqr_fv[f], ρa1_in[f], ᶜqr1_fv[f], ρa0_in[f], ε,
                )
            qs0_lo =
                _env_residual_value(
                    ᶜρqs_fv[f - 1], ρa1_in[f - 1], ᶜqs1_fv[f - 1], ρa0_in[f - 1], ε,
                )
            qs0_hi =
                _env_residual_value(
                    ᶜρqs_fv[f], ρa1_in[f], ᶜqs1_fv[f], ρa0_in[f], ε,
                )
            A1_qv_fv[f] =
                -G0_fv[f] * (
                    (wk.x0[f] - wk.x0[f - 1]) -
                    (ᶜK0_fv[f] - ᶜK0_fv[f - 1]) -
                    hv_d0_f * (u1_blk[f] - u1_blk[f - 1]) -
                    hl_d0_f * (qr0_hi - qr0_lo) -
                    hi_d0_f * (qs0_hi - qs0_lo)
                )
        end
        A1_qv_fv[nz + 1] = zero(FT)
    end

    if HasQ && HasNonEquil
        ᶜρql_fv = Fields.field_values(Yc_col.ρq_lcl)
        ᶜρqi_fv = Fields.field_values(Yc_col.ρq_icl)
        ᶜρqr_fv = Fields.field_values(Yc_col.ρq_rai)
        ᶜρqs_fv = Fields.field_values(Yc_col.ρq_sno)
        ᶜql1_fv = Fields.field_values(sgs1.q_lcl)
        ᶜqi1_fv = Fields.field_values(sgs1.q_icl)
        ᶜqr1_fv = Fields.field_values(sgs1.q_rai)
        ᶜqs1_fv = Fields.field_values(sgs1.q_sno)

        # 3) q_lcl, q_icl (with optional donor-limited blended flux when 1M + ρtke)
        _solve_edmf_cloud_species!(
            Val(true), Val(HasBlend), sw, ᶜql1_fv, ᶜρql_fv, A0_ql_fv,
            Yc_col, blend_pre_col, edmf_col, pre_col, ᶜΦ_fv,
            ᶜqt1_fv, ᶜρqt_fv, ᶜql1_fv, ᶜqi1_fv, ᶜρql_fv, ᶜρqi_fv,
            χ1_in, χ0_in, ρa1_in, ρa0_in, ᶜρa1_fv, ρa0_new, ᶜρ_fv,
            V1_fv, ε_tot_fv, JF1_rel_fv, JF0_rel_fv, G1_fv, G0_fv, G0_b_fv,
            ᶜJ_fv, ᶠz_fv, α_face_fv, A0_qv_fv, A1_qv_fv, jf_extra_fv,
            A1_qr_fv, A0_qr_fv, A1_qs_fv, A0_qs_fv, wk, Δt, nz,
        )
        _solve_edmf_cloud_species!(
            Val(false), Val(HasBlend), sw, ᶜqi1_fv, ᶜρqi_fv, A0_qi_fv,
            Yc_col, blend_pre_col, edmf_col, pre_col, ᶜΦ_fv,
            ᶜqt1_fv, ᶜρqt_fv, ᶜql1_fv, ᶜqi1_fv, ᶜρql_fv, ᶜρqi_fv,
            χ1_in, χ0_in, ρa1_in, ρa0_in, ᶜρa1_fv, ρa0_new, ᶜρ_fv,
            V1_fv, ε_tot_fv, JF1_rel_fv, JF0_rel_fv, G1_fv, G0_fv, G0_b_fv,
            ᶜJ_fv, ᶠz_fv, α_face_fv, A0_qv_fv, A1_qv_fv, jf_extra_fv,
            A1_qr_fv, A0_qr_fv, A1_qs_fv, A0_qs_fv, wk, Δt, nz,
        )
        if sw.couple_updraft_cond
            @inbounds for k in 1:nz
                q1_sum = max(zero(FT), ᶜql1_fv[k]) + max(zero(FT), ᶜqi1_fv[k])
                qte1_pos = max(zero(FT), ᶜqt1_fv[k])
                if q1_sum > qte1_pos
                    sc1 = qte1_pos / max(q1_sum, floatmin(FT))
                    ᶜql1_fv[k] = max(zero(FT), ᶜql1_fv[k]) * sc1
                    ᶜqi1_fv[k] = max(zero(FT), ᶜqi1_fv[k]) * sc1
                end
                ρq_sum = max(zero(FT), ᶜρql_fv[k]) + max(zero(FT), ᶜρqi_fv[k])
                ρqte_pos = max(zero(FT), ᶜρqt_fv[k])
                if ρq_sum > ρqte_pos
                    scg = ρqte_pos / max(ρq_sum, floatmin(FT))
                    ᶜρql_fv[k] = max(zero(FT), ᶜρql_fv[k]) * scg
                    ᶜρqi_fv[k] = max(zero(FT), ᶜρqi_fv[k]) * scg
                end
            end
        end

        # 4) Vapor nonnegativity guard on `A_qv = A0_qt_eff - A0_ql - A0_qi`
        @inbounds if nz >= 2
            for f in 1:(nz + 1)
                A0_qv_fv[f] = A0_qt_fv[f] - A0_ql_fv[f] - A0_qi_fv[f]
            end
            for k in 1:nz
                ρqv_L_k = max(zero(FT), ᶜρqt_fv[k] - ᶜρql_fv[k] - ᶜρqi_fv[k])
                P_qv_k =
                    Δt *
                    (max(zero(FT), A0_qv_fv[k + 1]) + max(zero(FT), -A0_qv_fv[k]))
                wk.upper0[k] = _zalesak_ratio(P_qv_k, ᶜJ_fv[k] * ρqv_L_k)
            end
            for f in 2:nz
                A_qv_f = A0_qv_fv[f]
                if A_qv_f != zero(FT)
                    d_idx = A_qv_f >= zero(FT) ? (f - 1) : f
                    r_qv = wk.upper0[d_idx]
                    if r_qv < one(FT)
                        A0_qt_fv[f] *= r_qv
                        A0_ql_fv[f] *= r_qv
                        A0_qi_fv[f] *= r_qv
                    end
                end
            end
        end

        # 5) Precipitation species (`q_rai`, `q_sno`), which do not diffuse
        precip_sp_tail = (
            ᶜqt1_fv,
            ᶜρqt_fv,
            A0_qt_fv,
            α_face_fv,
            χ1_in,
            χ0_in,
            ρa1_in,
            ρa0_in,
            ᶜρa1_fv,
            ρa0_new,
            V1_fv,
            ε_tot_fv,
            JF1_rel_fv,
            JF0_rel_fv,
            G1_fv,
            G0_fv,
            jf_extra_fv,
            ᶜJ_fv,
            ᶠz_fv,
            wk,
            Δt,
            nz,
            sw.sgsflux_upw_mode,
            sw.updraft_upw_mode,
        )
        _step_edmf_precip_species!(ᶜqr1_fv, ᶜρqr_fv, A0_qr_fv, precip_sp_tail...)
        _step_edmf_precip_species!(ᶜqs1_fv, ᶜρqs_fv, A0_qs_fv, precip_sp_tail...)
    end

    # ── Step 3: Commit SGS flux corrections and update ρe_tot & ρ ──
    @inbounds for k in 1:nz
        J_k = ᶜJ_fv[k]
        dt_J = Δt / J_k
        ᶜρe_fv[k] =
            ρη_L_fv[k] - ᶜp_fv[k] - dt_J * (A0_η_fv[k + 1] - A0_η_fv[k])

        if HasQ
            ᶜρqt_fv = Fields.field_values(Yc_col.ρq_tot)
            ᶜρqt_fv[k] -= dt_J * (A0_qt_fv[k + 1] - A0_qt_fv[k])

            if HasNonEquil
                Fields.field_values(Yc_col.ρq_lcl)[k] -=
                    dt_J * (A0_ql_fv[k + 1] - A0_ql_fv[k])
                Fields.field_values(Yc_col.ρq_icl)[k] -=
                    dt_J * (A0_qi_fv[k + 1] - A0_qi_fv[k])
                Fields.field_values(Yc_col.ρq_rai)[k] -=
                    dt_J * (A0_qr_fv[k + 1] - A0_qr_fv[k])
                Fields.field_values(Yc_col.ρq_sno)[k] -=
                    dt_J * (A0_qs_fv[k + 1] - A0_qs_fv[k])
            end
        end
    end

    if HasQ
        ᶜρqt_fv = Fields.field_values(Yc_col.ρq_tot)
        ᶜqt1_fv = Fields.field_values(sgs1.q_tot)
        if HasNonEquil
            _repair_nonneg_col!(
                Fields.field_values(Yc_col.ρq_lcl),
                Fields.field_values(sgs1.q_lcl),
                ᶜρa1_fv,
                ᶜJ_fv,
                nz,
            )
            _repair_nonneg_col!(
                Fields.field_values(Yc_col.ρq_icl),
                Fields.field_values(sgs1.q_icl),
                ᶜρa1_fv,
                ᶜJ_fv,
                nz,
            )
            _repair_nonneg_col!(
                Fields.field_values(Yc_col.ρq_rai),
                Fields.field_values(sgs1.q_rai),
                ᶜρa1_fv,
                ᶜJ_fv,
                nz,
            )
            _repair_nonneg_col!(
                Fields.field_values(Yc_col.ρq_sno),
                Fields.field_values(sgs1.q_sno),
                ᶜρa1_fv,
                ᶜJ_fv,
                nz,
            )
        end
        _repair_nonneg_col!(ᶜρqt_fv, ᶜqt1_fv, ᶜρa1_fv, ᶜJ_fv, nz)

        # Update total density by realized Δ(ρq_tot) so dry-air density
        # ρ_d = ρ - ρq_tot is conserved cell-wise.
        @inbounds for k in 1:nz
            ᶜρ_fv[k] += ᶜρqt_fv[k] - Δρ_col_fv[k]
        end
    end

    return nothing
end

struct EDMFExtraScalarTwoStreamSweep{HasDiff, FT} <: ColumnSweep
    updraft_kh_factor::FT
    Δt::FT
    has_env_diff::Bool
    has_updraft_diff::Bool
    sgsflux_upw_mode::Int8
    updraft_upw_mode::Int8
end

@inline function (sw::EDMFExtraScalarTwoStreamSweep{HasDiff})(
    ᶜχ1_col,
    ᶜρχ_col,
    Yc_col,
    geom_col,
    pre_col,
    scratch_s_col,
    scratch_v_col,
) where {HasDiff}
    FT = typeof(sw.Δt)
    Δt = sw.Δt
    ε = eps(FT)
    ε_num = ϵ_numerics(FT)

    ᶜχ1_fv = Fields.field_values(ᶜχ1_col)
    ᶜρχ_fv = Fields.field_values(ᶜρχ_col)
    ᶜρa1_fv = Fields.field_values(Yc_col.sgsʲs.:(1).ρa)
    ᶜJ_fv = Fields.field_values(geom_col.ᶜJ)
    ᶠlg_fv = Fields.field_values(geom_col.ᶠlg)
    ᶠz_fv = Fields.field_values(geom_col.ᶠz)
    ᶠK_h_fv = Fields.field_values(pre_col.ᶠK_h)

    ρa1_in = Fields.field_values(scratch_s_col.ρa1_in)
    ρa0_in = Fields.field_values(scratch_s_col.ρa0_in)
    ε_tot_fv = Fields.field_values(scratch_s_col.ρa1_prev)
    ρa0_new = Fields.field_values(scratch_s_col.ρa0_new)
    V1_fv = Fields.field_values(scratch_s_col.JF1)
    JF1_rel_fv = Fields.field_values(scratch_s_col.JF1_rel)
    JF0_rel_fv = Fields.field_values(scratch_s_col.JF0_rel)

    uv_c = scratch_v_col.uv_c.components.data
    st_c = scratch_v_col.st_c.components.data
    c12_c = scratch_v_col.c12_c.components.data
    c3_c = scratch_v_col.c3_c.components.data
    ct123_c = scratch_v_col.ct123_c.components.data
    uv_f = scratch_v_col.uv_f.components.data
    c123_f = scratch_v_col.c123_f.components.data

    # `uv_f.:(1)` preserves `W1_rel_fv` written by `EDMFTwoStreamColumnSweep`.
    α_face_fv = Fields.field_values(uv_f.:(1))
    Φ1_fv = Fields.field_values(uv_f.:(2))
    G0_fv = Fields.field_values(uv_f.:(5))
    G1_fv = Fields.field_values(uv_f.:(6))
    A0_fv = Fields.field_values(c123_f.:(2))
    jf_extra_fv = Fields.field_values(c123_f.:(3))

    diag1_fv = Fields.field_values(uv_c.:(1))
    upper1_fv = Fields.field_values(uv_c.:(2))
    lower1_fv = Fields.field_values(uv_c.:(3))
    diag0_fv = Fields.field_values(uv_c.:(5))
    upper0_fv = Fields.field_values(uv_c.:(6))
    lower0_fv = Fields.field_values(uv_c.:(7))
    r1_blk = Fields.field_values(st_c.:(8))
    r0_blk = Fields.field_values(st_c.:(9))
    x1_blk = Fields.field_values(c12_c.:(1))
    x0_blk = Fields.field_values(c12_c.:(2))
    χ1_in = Fields.field_values(c3_c.:(1))
    χ0_in = Fields.field_values(ct123_c.:(1))

    nz = Spaces.nlevels(axes(Yc_col.ρ))

    @inbounds for f in 1:(nz + 1)
        if HasDiff && 1 < f <= nz
            G0_fv[f] = if sw.has_env_diff
                ρa0_f = (ρa0_new[f - 1] + ρa0_new[f]) / 2
                face_conductance(ᶠlg_fv[f], ρa0_f * ᶠK_h_fv[f])
            else
                zero(FT)
            end
            G1_fv[f] = if sw.has_updraft_diff
                ρa1_lo = ᶜρa1_fv[f - 1]
                ρa1_hi = ᶜρa1_fv[f]
                ρa1_harm =
                    (ρa1_lo > ε_num && ρa1_hi > ε_num) ?
                    FT(2) * ρa1_lo * (ρa1_hi / (ρa1_lo + ρa1_hi)) : zero(FT)
                face_conductance(
                    ᶠlg_fv[f], sw.updraft_kh_factor * ρa1_harm * ᶠK_h_fv[f],
                )
            else
                zero(FT)
            end
        else
            G0_fv[f] = zero(FT)
            G1_fv[f] = zero(FT)
        end
    end

    @inbounds for k in 1:nz
        ρχ_k = ᶜρχ_fv[k]
        χ1_raw = max(zero(FT), ᶜχ1_fv[k])
        χ1_p = if ρχ_k >= zero(FT) && ρa1_in[k] > ε_num
            min(χ1_raw, ρχ_k / ρa1_in[k])
        else
            χ1_raw
        end
        χ1_in[k] = χ1_p
        χ0_in[k] = (ρχ_k - ρa1_in[k] * χ1_p) / max(ρa0_in[k], ε)
    end

    wk = (;
        diag1 = diag1_fv,
        upper1 = upper1_fv,
        lower1 = lower1_fv,
        diag0 = diag0_fv,
        upper0 = upper0_fv,
        lower0 = lower0_fv,
        Φ1 = Φ1_fv,
        r1 = r1_blk,
        r0 = r0_blk,
        x1 = x1_blk,
        x0 = x0_blk,
    )
    _solve_and_commit_edmf_var!(
        ᶜχ1_fv,
        ᶜρχ_fv,
        χ1_in,
        A0_fv,
        α_face_fv,
        ᶜρχ_fv,
        χ1_in,
        χ0_in,
        ᶜρa1_fv,
        ρa0_new,
        V1_fv,
        ε_tot_fv,
        JF1_rel_fv,
        JF0_rel_fv,
        G1_fv,
        G0_fv,
        jf_extra_fv,
        ᶜJ_fv,
        ᶠz_fv,
        wk,
        Δt,
        nz,
        (;
            apply_diff = HasDiff,
            apply_sfc = false,
            has_jf_extra = false,
            save_x0 = false,
            limit_pos = true,
        ),
        _no_surface_source(FT),
        sw.sgsflux_upw_mode,
        sw.updraft_upw_mode,
    )

    @inbounds for k in 1:nz
        J_k = ᶜJ_fv[k]
        ᶜρχ_fv[k] -= (Δt / J_k) * (A0_fv[k + 1] - A0_fv[k])
    end
    _repair_nonneg_col!(ᶜρχ_fv, ᶜχ1_fv, ᶜρa1_fv, ᶜJ_fv, nz)
    return nothing
end

@inline function _run_edmf_twostream_column!(
    ::Val{HasQ},
    ::Val{HasNonEquil},
    ::Val{FtildeMode},
    ::Val{HasBlend},
    ::Val{HasTailEntr},
    Y,
    geom,
    pre,
    sfc_conditions,
    edmf,
    ftilde_arg,
    blend_pre,
    tail_arg,
    scratch_s,
    scratch_v,
    opts,
    cmp,
) where {HasQ, HasNonEquil, FtildeMode, HasBlend, HasTailEntr}
    FT = typeof(opts.Δt_FT)
    (; turbconv_params, thermo_params, rayleigh_sponge, params) = opts
    sw = EDMFTwoStreamColumnSweep{
        HasQ,
        HasNonEquil,
        FtildeMode,
        HasBlend,
        HasTailEntr,
        typeof(turbconv_params),
        typeof(thermo_params),
        typeof(cmp),
        typeof(rayleigh_sponge),
        FT,
    }(
        turbconv_params,
        thermo_params,
        cmp,
        rayleigh_sponge,
        FT(CAP.max_area(turbconv_params)),
        FT(CAP.min_area(turbconv_params)),
        FT(CAP.pressure_normalmode_buoy_coeff1(turbconv_params)),
        FT(CAP.pressure_normalmode_drag_coeff(turbconv_params)),
        FT(CAP.pressure_drag_plume_radius_slope(turbconv_params)),
        FT(CAP.pressure_drag_plume_radius_max(turbconv_params)),
        FT(opts.f_tail),
        FT(CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)),
        FT(CAP.detr_buoy_coeff(turbconv_params)),
        FT(CAP.entr_detr_buoy_inv_tau_max(turbconv_params)),
        FT(CAP.detr_massflux_vertdiv_coeff(turbconv_params)),
        FT(CAP.edmfx_updraft_kh_diffusion_factor(turbconv_params)),
        FT(tke_dissipation_coefficient(turbconv_params)),
        FT(CAP.condensation_evaporation_timescale(params)),
        FT(CAP.sublimation_deposition_timescale(params)),
        FT(CAP.T_min_sgs(params)),
        opts.Δt_FT,
        opts.z_max,
        opts.has_sfc,
        opts.has_nh_drag,
        opts.has_sgs_mass_flux,
        opts.has_env_diff,
        opts.has_updraft_diff,
        opts.use_const_tau,
        opts.couple_updraft_cond,
        opts.sgsflux_upw_mode,
        opts.updraft_upw_mode,
        opts.n_picard,
    )
    DataLayouts.foreach_column(
        sw,
        Y.c,
        Y.f,
        geom,
        pre,
        sfc_conditions,
        edmf,
        ftilde_arg,
        blend_pre,
        tail_arg,
        scratch_s,
        scratch_v;
        shmem = Val(:none),
    )
    return nothing
end

@inline function _dispatch_edmf_tail!(
    vq::Val{HasQ},
    vne::Val{HasNonEquil},
    vft::Val{FtildeMode},
    vb::Val{HasBlend},
    Y,
    p,
    geom,
    pre,
    edmf,
    ftilde_arg,
    blend_pre,
    scratch_s,
    scratch_v,
    opts,
    cmp_cf,
) where {HasQ, HasNonEquil, FtildeMode, HasBlend}
    tail_fields = tail_entrainment_fields(p, opts.f_tail)
    if !isempty(tail_fields)
        tail_arg = (;
            ᶜqt0 = tail_fields[2],
            ᶜql0 = tail_fields[3],
            ᶜqi0 = tail_fields[4],
            ᶜTT = tail_fields[5],
            ᶜqq = tail_fields[6],
            ᶜcorr = tail_fields[7],
        )
        _run_edmf_twostream_column!(
            vq,
            vne,
            vft,
            vb,
            Val(true),
            Y,
            geom,
            pre,
            p.precomputed.sfc_conditions,
            edmf,
            ftilde_arg,
            blend_pre,
            tail_arg,
            scratch_s,
            scratch_v,
            opts,
            cmp_cf,
        )
    else
        _run_edmf_twostream_column!(
            vq,
            vne,
            vft,
            vb,
            Val(false),
            Y,
            geom,
            pre,
            p.precomputed.sfc_conditions,
            edmf,
            ftilde_arg,
            blend_pre,
            (),
            scratch_s,
            scratch_v,
            opts,
            cmp_cf,
        )
    end
    return nothing
end

@inline function _dispatch_edmf_blend!(
    vq::Val{HasQ},
    vne::Val{HasNonEquil},
    vft::Val{FtildeMode},
    has_blend::Bool,
    Y,
    p,
    geom,
    pre,
    edmf,
    ftilde_arg,
    scratch_s,
    scratch_v,
    opts,
) where {HasQ, HasNonEquil, FtildeMode}
    dev = ClimaComms.device(Y.c)
    cmp_cf =
        p.atmos.microphysics_model isa NonEquilibriumMicrophysics1M ?
        _fold(dev, CAP.cloud_formation_1m_params(opts.params)) : nothing
    if has_blend
        _dispatch_edmf_tail!(
            vq,
            vne,
            vft,
            Val(true),
            Y,
            p,
            geom,
            pre,
            edmf,
            ftilde_arg,
            (; p.precomputed.ᶜl_mix, p.precomputed.ᶜsgs_moments),
            scratch_s,
            scratch_v,
            opts,
            cmp_cf,
        )
    else
        _dispatch_edmf_tail!(
            vq,
            vne,
            vft,
            Val(false),
            Y,
            p,
            geom,
            pre,
            edmf,
            ftilde_arg,
            (),
            scratch_s,
            scratch_v,
            opts,
            cmp_cf,
        )
    end
    return nothing
end

@inline function _dispatch_edmf_ftilde!(
    vq::Val{HasQ},
    vne::Val{HasNonEquil},
    ᶠF̃,
    has_blend::Bool,
    Y,
    p,
    geom,
    pre,
    edmf,
    scratch_s,
    scratch_v,
    opts,
) where {HasQ, HasNonEquil}
    if isnothing(ᶠF̃)
        _dispatch_edmf_blend!(
            vq,
            vne,
            Val(:none),
            has_blend,
            Y,
            p,
            geom,
            pre,
            edmf,
            (),
            scratch_s,
            scratch_v,
            opts,
        )
    else
        _dispatch_edmf_blend!(
            vq,
            vne,
            Val(:gm_flux),
            has_blend,
            Y,
            p,
            geom,
            pre,
            edmf,
            ᶠF̃,
            scratch_s,
            scratch_v,
            opts,
        )
    end
    return nothing
end

"""
    step_edmf_and_env_turbulence!(
        Y,
        p,
        t,
        Δt;
        ᶠF̃ = nothing,
        couple_updraft_condensation = false,
    )

Advance `PrognosticEDMFX` updraft momentum and area (`u₃ʲ ≥ 0`,
`ρaʲ ∈ [0, a_max ρʲ]`), intensive updraft scalars (`mseʲ`, `q_totʲ`,
condensates, and extra SGS tracers) in advective form, and grid-mean scalars in
conservative flux-divergence form with shared-limiter Zalesak FCT. When
`couple_updraft_condensation` is `true` under `NonEquilibriumMicrophysics1M`,
updraft cloud liquid and ice phase relaxation is coupled into the implicit
updraft scalar solve. Remaining updraft microphysics and sedimentation run in
`step_coupled_microphysics_sedimentation!`.

No-op when `turbconv_model` is not `PrognosticEDMFX`.
"""
function step_edmf_and_env_turbulence!(
    Y,
    p,
    t,
    Δt;
    ᶠF̃ = nothing,
    couple_updraft_condensation::Bool = false,
)
    p.atmos.turbconv_model isa PrognosticEDMFX || return nothing

    (; turbconv_model, edmfx_model) = p.atmos
    n = n_mass_flux_subdomains(turbconv_model)
    n == 0 && return nothing
    n == 1 || error("step_edmf_and_env_turbulence! supports a single updraft")

    FT = Spaces.undertype(axes(Y.c))
    Δt_FT = FT(Δt)
    turbconv_params = CAP.turbconv_params(p.params)
    thermo_params = CAP.thermodynamics_params(p.params)

    has_q_tot = MatrixFields.has_field(Y.c, @name(ρq_tot))
    has_nonequil =
        p.atmos.microphysics_model isa NonEquilibriumMicrophysics &&
        MatrixFields.has_field(Y.c.sgsʲs.:(1), @name(q_lcl))
    has_sfc = !p.atmos.disable_surface_flux_tendency
    has_nh_drag = edmfx_model.nh_pressure
    has_sgs_mass_flux = edmfx_model.sgs_mass_flux
    rayleigh_sponge = p.atmos.rayleigh_sponge
    has_env_diff = edmfx_model.sgs_diffusive_flux
    has_updraft_diff = edmfx_model.sgs_diffusive_flux && edmfx_model.vertical_diffusion
    has_blend = has_env_diff && has_nonequil && blended_cloud_species_flux(Y, p)
    use_const_tau = CAP.sgs_cloud_species_constant_timescales(p.params) > 0

    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜJ = ᶜlg.J
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶠdz = Fields.Δz_field(axes(Y.f))
    ᶠz = Fields.coordinate_field(Y.f).z
    z_sfc = Fields.level(ᶠz, Fields.half)
    z_max = !isnothing(rayleigh_sponge) ? FT(Spaces.z_max(axes(ᶠz))) : zero(FT)
    ᶠuₕ³ = p.scratch.ᶠtemp_CT3
    @. ᶠuₕ³ = $compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ)

    geom = (;
        ᶜlg,
        ᶜJ,
        ᶜz,
        z_sfc,
        ᶠlg,
        ᶠdz,
        ᶠz,
        ᶠuₕ³,
        p.core.ᶜΦ,
        p.core.ᶠgradᵥ_ᶜΦ,
        p.core.ᶜgradᵥ_ᶠΦ,
    )
    pre = (;
        p.precomputed.ᶜK,
        p.precomputed.ᶜp,
        p.precomputed.ᶠK_h,
        ᶜT1 = p.precomputed.ᶜTʲs.:(1),
        ᶠu3_gm = p.precomputed.ᶠu³,
    )
    edmf = (;
        ᶜK1 = p.precomputed.ᶜKʲs.:(1),
        ᶜK0 = p.precomputed.ᶜK⁰,
        ᶜT0 = p.precomputed.ᶜT⁰,
        ᶜρ1 = p.precomputed.ᶜρʲs.:(1),
        ᶜρ_diff1 = p.precomputed.ᶜρ_diffʲs.:(1),
        p.precomputed.ᶜρ_ref_excess,
        ᶜevs1 = p.precomputed.ᶜentr_vel_scaleʲs.:(1),
        ᶜenr1 = p.precomputed.ᶜentr_nonvel_rateʲs.:(1),
        sfc_mf_src = Fields.level(p.precomputed.sfc_mass_flux_sourceʲs.:(1), 1),
        sfc_mse_buoy = Fields.level(p.precomputed.sfc_mse_buoyantʲs.:(1), 1),
        sfc_qt_buoy = Fields.level(p.precomputed.sfc_q_tot_buoyantʲs.:(1), 1),
    )

    scratch_s = (;
        ρa1_in = p.scratch.ᶜtemp_scalar,
        ρa0_in = p.scratch.ᶜtemp_scalar_2,
        ρa1_prev = p.scratch.ᶜtemp_scalar_3,
        ρa0_new = p.scratch.ᶜtemp_scalar_4,
        E_col = p.scratch.ᶜtemp_scalar_5,
        D_col = p.scratch.ᶜtemp_scalar_6,
        empty_reg = p.scratch.ᶜtemp_scalar_7,
        JF1 = p.scratch.ᶠtemp_scalar,
        JF1_rel = p.scratch.ᶠtemp_scalar_2,
        JF0_rel = p.scratch.ᶠtemp_UVW.components.data.:1,
    )
    scratch_v = (;
        uv_c = p.scratch.ᶜtemp_UVWxUVW,
        st_c = p.scratch.ᶜtemp_strain,
        c12_c = p.scratch.ᶜtemp_C12,
        c3_c = p.scratch.ᶜtemp_C3,
        ct123_c = p.scratch.ᶜtemp_CT123,
        uvw_c = p.scratch.ᶜtemp_UVW,
        uv_f = p.scratch.ᶠtemp_UVWxUVW,
        st_f = p.scratch.ᶠtemp_strain,
        c123_f = p.scratch.ᶠtemp_C123,
    )
    sgsflux_upw_mode =
        _sgsflux_upw_mode(p.atmos.numerics.edmfx_sgsflux_upwinding)
    updraft_upw_mode =
        _updraft_upw_mode(p.atmos.numerics.edmfx_mse_humidity_upwinding)
    updraft_tr_upw_mode =
        _updraft_upw_mode(p.atmos.numerics.edmfx_tracer_upwinding)
    n_picard = Int8(p.atmos.edmfx_model.updraft_picard_rounds)
    couple_updraft_cond =
        couple_updraft_condensation &&
        p.atmos.microphysics_model isa NonEquilibriumMicrophysics1M
    f_tail = FT(CAP.edmf_tail_entr_fraction(p.params))
    opts = (;
        turbconv_params,
        thermo_params,
        rayleigh_sponge,
        params = p.params,
        Δt_FT,
        f_tail,
        z_max,
        has_sfc,
        has_nh_drag,
        has_sgs_mass_flux,
        has_env_diff,
        has_updraft_diff,
        use_const_tau,
        couple_updraft_cond,
        sgsflux_upw_mode,
        updraft_upw_mode,
        n_picard,
    )

    if has_q_tot
        if has_nonequil
            _dispatch_edmf_ftilde!(
                Val(true), Val(true), ᶠF̃, has_blend,
                Y, p, geom, pre, edmf, scratch_s, scratch_v, opts,
            )
        else
            _dispatch_edmf_ftilde!(
                Val(true), Val(false), ᶠF̃, false,
                Y, p, geom, pre, edmf, scratch_s, scratch_v, opts,
            )
        end
    else
        _dispatch_edmf_ftilde!(
            Val(false), Val(false), ᶠF̃, false,
            Y, p, geom, pre, edmf, scratch_s, scratch_v, opts,
        )
    end

    # Extra SGS tracers carried by sgsʲs.:(1) (2M/P3 number/rime and passive
    # chemistry tracers, before MomentumVDiff overwrites `ᶠtemp_scalar`).
    extra_geom = (; ᶜJ, ᶠlg, ᶠz)
    extra_pre = (; p.precomputed.ᶠK_h)
    kh_fac = FT(CAP.edmfx_updraft_kh_diffusion_factor(turbconv_params))
    sw_extra_water_diff = EDMFExtraScalarTwoStreamSweep{true, FT}(
        kh_fac, Δt_FT, has_env_diff, has_updraft_diff, sgsflux_upw_mode,
        updraft_upw_mode,
    )
    sw_extra_water_nodiff = EDMFExtraScalarTwoStreamSweep{false, FT}(
        kh_fac, Δt_FT, false, false, sgsflux_upw_mode, updraft_upw_mode,
    )
    sw_extra_tr_diff = EDMFExtraScalarTwoStreamSweep{true, FT}(
        kh_fac,
        Δt_FT,
        has_env_diff,
        has_updraft_diff,
        sgsflux_upw_mode,
        updraft_tr_upw_mode,
    )
    sw_extra_tr_nodiff = EDMFExtraScalarTwoStreamSweep{false, FT}(
        kh_fac, Δt_FT, false, false, sgsflux_upw_mode, updraft_tr_upw_mode,
    )
    MatrixFields.unrolled_foreach(sgs_tracer_names(Y)) do χ_name
        (
            χ_name == @name(q_lcl) ||
            χ_name == @name(q_icl) ||
            χ_name == @name(q_rai) ||
            χ_name == @name(q_sno)
        ) && return nothing
        ρχ_name = get_ρχ_name(χ_name)
        MatrixFields.has_field(Y.c, ρχ_name) || return nothing
        ᶜχ1 = MatrixFields.get_field(Y.c.sgsʲs.:(1), χ_name)
        ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
        is_water_num = χ_name == @name(n_lcl) || χ_name == @name(n_rai)
        is_precip = (χ_name, ρχ_name) in precipitation_species
        sw_ex = if is_water_num
            is_precip ? sw_extra_water_nodiff : sw_extra_water_diff
        else
            is_precip ? sw_extra_tr_nodiff : sw_extra_tr_diff
        end
        DataLayouts.foreach_column(
            sw_ex,
            ᶜχ1,
            ᶜρχ,
            Y.c,
            extra_geom,
            extra_pre,
            scratch_s,
            scratch_v;
            shmem = Val(:none),
        )
        return nothing
    end

    # Momentum diffusion (`uₕ`) and Robin/Neumann surface drag. Uses
    # `ᶠtemp_scalar` so `ᶠtemp_scalar_2` (`JF1_rel`) stays intact.
    if (has_env_diff || has_sfc) &&
       !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
        ᶠw_u = p.scratch.ᶠtemp_scalar
        if has_env_diff
            @. ᶠw_u = ᶠinterp(Y.c.ρ) * p.precomputed.ᶠK_u
        else
            fill!(ᶠw_u, zero(FT))
        end
        is_standalone = p.atmos.surface.flux_scheme !== nothing
        mom_geom = (; ᶜlg = Fields.local_geometry_field(Y.c), ᶠlg)
        mom_weights = (; ᶠw_u)
        mom_scratch = (;
            diag = p.scratch.ᶜtemp_scalar_3,
            upper = p.scratch.ᶜtemp_scalar_4,
            lower = p.scratch.ᶜtemp_scalar_5,
            rhs1 = p.scratch.ᶜtemp_scalar_6,
            rhs2 = p.scratch.ᶜtemp_scalar_7,
            sol1 = p.scratch.ᶜtemp_UVWxUVW.components.data.:1,
            sol2 = p.scratch.ᶜtemp_UVWxUVW.components.data.:2,
        )
        sw_mom = MomentumVDiffColumnSweep{false, FT}(
            Δt_FT, has_sfc, is_standalone,
        )
        DataLayouts.foreach_column(
            sw_mom,
            Y.c,
            Y.f,
            mom_geom,
            mom_weights,
            mom_scratch,
            p.precomputed.sfc_conditions;
            shmem = Val(:none),
        )
        p.precomputed.ᶜK .= compute_kinetic(Y.c.uₕ, Y.f.u₃)
    end

    # Diffuse scalars absent from `sgsʲs` (such as grid-scale P3 rime/number
    # or passive tracers) down their grid-mean gradients with `ᶠinterp(ρ) ᶠK_h`.
    passive_geom = (; ᶜJ, ᶠlg)
    passive_scratch = (;
        diag = p.scratch.ᶜtemp_scalar_3,
        upper = p.scratch.ᶜtemp_scalar_4,
        lower = p.scratch.ᶜtemp_scalar_5,
        rhs = p.scratch.ᶜtemp_scalar_6,
        sol = p.scratch.ᶜtemp_scalar_7,
    )
    if has_env_diff
        ᶠw_env = p.scratch.ᶠtemp_scalar
        @. ᶠw_env = ᶠinterp(Y.c.ρ) * p.precomputed.ᶠK_h
        sw_passive = PassiveScalarVDiffColumnSweep{false, FT}(Δt_FT)
        env_weights = (; ᶠw_h = ᶠw_env)
        jf_dummy = p.scratch.ᶠtemp_UVWxUVW.components.data.:9
        MatrixFields.unrolled_foreach(edmfx_diffused_scalars(Y)) do (χ_name, ρχ_name)
            MatrixFields.has_field(Y.c.sgsʲs.:(1), χ_name) && return nothing
            MatrixFields.has_field(Y.c, ρχ_name) || return nothing
            ᶜρχ = MatrixFields.get_field(Y.c, ρχ_name)
            DataLayouts.foreach_column(
                sw_passive,
                ᶜρχ,
                Y.c,
                passive_geom,
                env_weights,
                passive_scratch,
                jf_dummy;
                shmem = Val(:none),
            )
            return nothing
        end
    end

    # Refresh grid-mean, draft, and environment velocities and thermodynamics.
    if has_q_tot
        set_implicit_precomputed_quantities_core!(Y, p, t)
    else
        ᶠuₕ³ = _set_implicit_grid_mean_precomputed_quantities!(Y, p, t)
        set_velocity_quantities!(
            p.precomputed.ᶜuʲs.:(1),
            p.precomputed.ᶠu³ʲs.:(1),
            p.precomputed.ᶜKʲs.:(1),
            Y.f.sgsʲs.:(1).u₃,
            Y.c.uₕ,
            ᶠuₕ³,
        )
        @. p.precomputed.ᶜTʲs.:(1) = TD.air_temperature(
            thermo_params,
            TD.ph(),
            Y.c.sgsʲs.:(1).mse - p.core.ᶜΦ,
        )
        @. p.precomputed.ᶜρʲs.:(1) = TD.air_density(
            thermo_params,
            p.precomputed.ᶜTʲs.:(1),
            p.precomputed.ᶜp,
        )
        set_sgs_ᶠu₃!(u₃⁰, p.precomputed.ᶠu₃⁰, Y.c, Y.f, turbconv_model)
        set_velocity_quantities!(
            p.precomputed.ᶜu⁰,
            p.precomputed.ᶠu³⁰,
            p.precomputed.ᶜK⁰,
            p.precomputed.ᶠu₃⁰,
            Y.c.uₕ,
            ᶠuₕ³,
        )
        ᶜρa0_ref = p.scratch.ᶜtemp_scalar
        eps_FT = eps(FT)
        @. ᶜρa0_ref = max(eps_FT, Y.c.ρ - Y.c.sgsʲs.:(1).ρa)
        @. p.precomputed.ᶜT⁰ = TD.air_temperature(
            thermo_params,
            TD.ph(),
            (
                Y.c.ρ * (p.precomputed.ᶜh_tot - p.precomputed.ᶜK) -
                Y.c.sgsʲs.:(1).ρa * Y.c.sgsʲs.:(1).mse
            ) / ᶜρa0_ref - p.core.ᶜΦ,
        )
    end

    return nothing
end

# ============================================================================
# 5. step_prognostic_tke!
# ============================================================================

# Divergence `(J_{k+1/2} u³¹ ρa¹_don - J_{k-1/2} u³¹ ρa¹_don) / J_k` of the
# draft mass flux with the full contravariant `u³¹`, zero through the surface
# and the top. Each face takes the area of its donor cell, the cell below for
# `u³¹ ≥ 0` and the cell above otherwise, as in the `ρa¹` solve.
@inline function _draft_mass_flux_divergence(
    ᶠu31_fv,
    ᶠuh3_fv,
    ᶠlg_fv,
    ᶜρa1_fv,
    J_k,
    k::Int,
    nz::Int,
)
    FT = typeof(J_k)
    @inbounds begin
        mf_top = if k < nz
            u3_cont = _draft_face_u3_contravariant(
                ᶠuh3_fv[k + 1], ᶠlg_fv[k + 1], _c3_scalar(ᶠu31_fv[k + 1]),
            )
            ρa1_don = u3_cont >= zero(FT) ? ᶜρa1_fv[k] : ᶜρa1_fv[k + 1]
            ᶠlg_fv[k + 1].J * u3_cont * ρa1_don
        else
            zero(FT)
        end
        mf_bot = if k > 1
            u3_cont = _draft_face_u3_contravariant(
                ᶠuh3_fv[k], ᶠlg_fv[k], _c3_scalar(ᶠu31_fv[k]),
            )
            ρa1_don = u3_cont >= zero(FT) ? ᶜρa1_fv[k - 1] : ᶜρa1_fv[k]
            ᶠlg_fv[k].J * u3_cont * ρa1_don
        else
            zero(FT)
        end
    end
    return (mf_top - mf_bot) / J_k
end

struct TKEColumnSweep{HasEDMF, TurbParams, DetrModel, FT} <: ColumnSweep
    turbconv_params::TurbParams
    detr_model::DetrModel
    c_d::FT
    α_d::FT
    c_R::FT
    R_max::FT
    scale_height::FT
    a_min::FT
    a_max::FT
    entr_detr_buoy_inv_tau_max::FT
    Δt::FT
    has_nh_drag::Bool
    has_diff::Bool
end

@inline function (sw::TKEColumnSweep{HasEDMF})(
    Yc_col,
    Yf_col,
    geom_col,
    tke_pre_col,
    edmf_col,
    sfc_col,
    scratch_col,
) where {HasEDMF}
    FT = typeof(sw.Δt)
    Δt = sw.Δt
    ε = eps(FT)
    ε_tke = FT(1e-6)

    ᶜρ_fv = Fields.field_values(Yc_col.ρ)
    ᶜρtke_fv = Fields.field_values(Yc_col.ρtke)
    ᶜJ_fv = Fields.field_values(geom_col.ᶜJ)
    ᶜz_fv = Fields.field_values(geom_col.ᶜz)
    z_sfc_val = Fields.field_values(geom_col.z_sfc)[]
    ᶠlg_fv = Fields.field_values(geom_col.ᶠlg)
    ᶜlg_fv = Fields.field_values(geom_col.ᶜlg)
    ᶠK_u_fv = Fields.field_values(tke_pre_col.ᶠK_u)
    ᶠK_h_fv = Fields.field_values(tke_pre_col.ᶠK_h)
    ᶠbg_fv = Fields.field_values(tke_pre_col.ᶠbuoygrad)
    ᶜl_mix_fv = Fields.field_values(tke_pre_col.ᶜl_mix)
    ᶜshear_fv = Fields.field_values(tke_pre_col.ᶜshear_prod)
    ρtke_flux_c3 = Fields.field_values(sfc_col.ρtke_flux)[]

    diag_fv = Fields.field_values(scratch_col.diag)
    upper_fv = Fields.field_values(scratch_col.upper)
    lower_fv = Fields.field_values(scratch_col.lower)
    rhs_fv = Fields.field_values(scratch_col.rhs)
    sol_fv = Fields.field_values(scratch_col.sol)

    nz = Spaces.nlevels(axes(Yc_col.ρ))

    @inbounds for k in 1:nz
        J_k = ᶜJ_fv[k]
        ρ_k = ᶜρ_fv[k]
        ρtke_k = max(zero(FT), ᶜρtke_fv[k])
        tke_k = ρtke_k / max(ρ_k, ε)

        ρ_top = k < nz ? (ρ_k + ᶜρ_fv[k + 1]) / 2 : zero(FT)
        ρ_bot = k > 1 ? (ᶜρ_fv[k - 1] + ρ_k) / 2 : zero(FT)
        G_top =
            (sw.has_diff && k < nz) ?
            face_conductance(ᶠlg_fv[k + 1], ρ_top * ᶠK_u_fv[k + 1]) : zero(FT)
        G_bot =
            (sw.has_diff && k > 1) ?
            face_conductance(ᶠlg_fv[k], ρ_bot * ᶠK_u_fv[k]) : zero(FT)

        # Split buoyancy gradient into positive production (RHS) and negative
        # buoyancy destruction (Patankar sink on diagonal) across every cell.
        bg_top = ᶠbg_fv[k + 1]
        bg_bot = ᶠbg_fv[k]
        K_h_top = ᶠK_h_fv[k + 1]
        K_h_bot = ᶠK_h_fv[k]
        P_buoy =
            FT(0.5) *
            ρ_k *
            (K_h_top * max(zero(FT), -bg_top) + K_h_bot * max(zero(FT), -bg_bot))
        D_buoy =
            FT(0.5) *
            ρ_k *
            (K_h_top * max(zero(FT), bg_top) + K_h_bot * max(zero(FT), bg_bot))

        l_safe = max(ᶜl_mix_fv[k], one(FT))
        # D_diss / tke = c_d * ρ * √tke / max(l_mix, 1)
        sink_diss = sw.has_diff ? sw.c_d * ρ_k * sqrt(tke_k) / l_safe : zero(FT)
        sink_buoy = D_buoy / max(tke_k, ε_tke)
        sink_coeff = sink_diss + sink_buoy

        diag_fv[k] = J_k * ρ_k + Δt * (G_top + G_bot) + Δt * J_k * sink_coeff
        if k < nz
            upper_fv[k] = -Δt * G_top
        end
        if k > 1
            lower_fv[k - 1] = -Δt * G_bot
        end

        prod_k = ρ_k * max(zero(FT), ᶜshear_fv[k]) + P_buoy

        if HasEDMF
            ρa1_k = Fields.field_values(Yc_col.sgsʲs.:(1).ρa)[k]
            ρ1_k = max(Fields.field_values(edmf_col.ᶜρ1)[k], ε)
            a1_k = draft_area(ρa1_k, ρ1_k)
            a0_k = max(zero(FT), one(FT) - a1_k)
            u1_k = Fields.field_values(edmf_col.ᶜu1)[k]
            u0_k = Fields.field_values(edmf_col.ᶜu0)[k]
            Δw_k = get_physical_w(u1_k - u0_k, ᶜlg_fv[k])

            if sw.has_nh_drag
                drag_coeff = pressure_drag_coefficient(
                    sw.α_d,
                    sw.scale_height,
                    a1_k,
                    a0_k,
                    sw.a_min,
                    sw.a_max,
                    ᶜz_fv[k],
                    z_sfc_val,
                    sw.c_R,
                    sw.R_max,
                )
                prod_k += ρa1_k * a0_k * drag_coeff * abs(Δw_k)^3
            end

            # Detrainment shear-mixing production
            ᶠu31_fv = Fields.field_values(Yf_col.sgsʲs.:(1).u₃)
            ᶜρ_diff1_fv = Fields.field_values(edmf_col.ᶜρ_diff1)
            ᶠgradΦ_fv = Fields.field_values(geom_col.ᶠgradᵥ_ᶜΦ)
            w_f_top = k < nz ? get_physical_w(ᶠu31_fv[k + 1], ᶠlg_fv[k + 1]) : zero(FT)
            w_f_bot = k > 1 ? get_physical_w(ᶠu31_fv[k], ᶠlg_fv[k]) : zero(FT)
            vb_top =
                k < nz ?
                vertical_buoyancy_acceleration(
                    (ᶜρ_diff1_fv[k] + ᶜρ_diff1_fv[k + 1]) / 2,
                    _c3_scalar(ᶠgradΦ_fv[k + 1]),
                    ᶠlg_fv[k + 1],
                ) : zero(FT)
            vb_bot =
                k > 1 ?
                vertical_buoyancy_acceleration(
                    (ᶜρ_diff1_fv[k - 1] + ᶜρ_diff1_fv[k]) / 2,
                    _c3_scalar(ᶠgradΦ_fv[k]),
                    ᶠlg_fv[k],
                ) : zero(FT)
            bit_k =
                FT(0.5) * (
                    (
                        k < nz ?
                        detr_buoy_inv_time_scale(
                            w_f_top,
                            vb_top,
                            sw.entr_detr_buoy_inv_tau_max,
                        ) : zero(FT)
                    ) +
                    (
                        k > 1 ?
                        detr_buoy_inv_time_scale(
                            w_f_bot,
                            vb_bot,
                            sw.entr_detr_buoy_inv_tau_max,
                        ) : zero(FT)
                    )
                )
            div_mf = _draft_mass_flux_divergence(
                ᶠu31_fv,
                Fields.field_values(geom_col.ᶠuₕ³),
                ᶠlg_fv,
                Fields.field_values(Yc_col.sgsʲs.:(1).ρa),
                J_k,
                k,
                nz,
            )
            detr_k = compute_detrainment(
                sw.turbconv_params,
                a1_k,
                ρa1_k,
                bit_k,
                div_mf,
                area_bounding_entr_detr(sw.turbconv_params, a1_k),
                sw.detr_model,
            )
            prod_k += FT(0.5) * max(zero(FT), detr_k) * ρa1_k * (Δw_k * Δw_k)
        end

        rhs_fv[k] = J_k * ρtke_k + Δt * J_k * prod_k
    end

    # Surface TKE flux at bottom face
    if sw.has_diff
        @inbounds begin
            sfc_factor = surface_c3_flux_factor(ᶠlg_fv[1])
            rhs_fv[1] += Δt * sfc_factor * max(zero(FT), _c3_scalar(ρtke_flux_c3))
        end
    end

    column_thomas_solve!(sol_fv, rhs_fv, diag_fv, upper_fv, lower_fv, nz)

    @inbounds for k in 1:nz
        ᶜρtke_fv[k] = ᶜρ_fv[k] * max(zero(FT), sol_fv[k])
    end

    return nothing
end

"""
    step_prognostic_tke!(Y, p, Δt; refresh_diffusivities = true)

Implicit tridiagonal diffusion of `ρtke` (`K_u`) with Patankar-linearized sinks
(dissipation and negative buoyancy destruction), shear and positive-buoyancy
production, realized `Δt`, and refreshed `ᶠK_h, ᶠK_u` from `tke^{n+1}`.
"""
function step_prognostic_tke!(Y, p, Δt; refresh_diffusivities::Bool = true)
    MatrixFields.has_field(Y, @name(c.ρtke)) || return nothing

    FT = Spaces.undertype(axes(Y.c))
    Δt_FT = FT(Δt)
    turbconv_model = p.atmos.turbconv_model
    turbconv_params = CAP.turbconv_params(p.params)

    # Precompute shear production at faces from updated horizontal velocity
    # so implicit shear production stays bounded by dissipated mean kinetic energy.
    ᶜshear_prod = p.scratch.ᶜtemp_scalar
    ᶜu_cur = p.scratch.ᶜtemp_UVW
    @. ᶜu_cur = Geometry.UVWVector(Y.c.uₕ) + Geometry.UVWVector(ᶜinterp(Y.f.u₃))
    ᶠstrain_rate = compute_strain_rate_face_vertical(ᶜu_cur)
    @. ᶜshear_prod = ᶜinterp(
        2 * p.precomputed.ᶠK_u * vertical_shear_norm_sqr(ᶠstrain_rate),
    )

    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠlg = Fields.local_geometry_field(Y.f)
    z_sfc = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶠuₕ³ = p.scratch.ᶠtemp_CT3
    @. ᶠuₕ³ = $compute_ᶠuₕ³(Y.c.uₕ, Y.c.ρ)
    geom = (; ᶜJ, ᶜz, z_sfc, ᶜlg, ᶠlg, ᶠuₕ³, p.core.ᶠgradᵥ_ᶜΦ)

    tke_pre = (;
        ᶠK_u = p.precomputed.ᶠK_u,
        ᶠK_h = p.precomputed.ᶠK_h,
        ᶠbuoygrad = p.precomputed.ᶠbuoygrad,
        ᶜl_mix = p.precomputed.ᶜl_mix,
        ᶜshear_prod,
    )
    sfc = (; ρtke_flux = p.precomputed.ρtke_flux)

    has_edmf =
        turbconv_model isa PrognosticEDMFX &&
        n_mass_flux_subdomains(turbconv_model) > 0
    has_nh_drag = has_edmf && p.atmos.edmfx_model.nh_pressure
    has_diff =
        !(turbconv_model isa AbstractEDMF) || p.atmos.edmfx_model.sgs_diffusive_flux
    detr_model = has_edmf ? p.atmos.edmfx_model.detr_model : nothing

    scratch = (;
        diag = p.scratch.ᶜtemp_scalar_3,
        upper = p.scratch.ᶜtemp_scalar_4,
        lower = p.scratch.ᶜtemp_scalar_5,
        rhs = p.scratch.ᶜtemp_scalar_6,
        sol = p.scratch.ᶜtemp_scalar_7,
    )

    sw = TKEColumnSweep{
        has_edmf,
        typeof(turbconv_params),
        typeof(detr_model),
        FT,
    }(
        turbconv_params,
        detr_model,
        FT(tke_dissipation_coefficient(turbconv_params)),
        FT(CAP.pressure_normalmode_drag_coeff(turbconv_params)),
        FT(CAP.pressure_drag_plume_radius_slope(turbconv_params)),
        FT(CAP.pressure_drag_plume_radius_max(turbconv_params)),
        FT(CAP.R_d(p.params) * CAP.T_surf_ref(p.params) / CAP.grav(p.params)),
        FT(CAP.min_area(turbconv_params)),
        FT(CAP.max_area(turbconv_params)),
        FT(CAP.entr_detr_buoy_inv_tau_max(turbconv_params)),
        Δt_FT,
        has_nh_drag,
        has_diff,
    )

    if has_edmf
        edmf_col = (;
            ᶜρ1 = p.precomputed.ᶜρʲs.:(1),
            ᶜρ_diff1 = p.precomputed.ᶜρ_diffʲs.:(1),
            ᶜu1 = p.precomputed.ᶜuʲs.:(1),
            ᶜu0 = p.precomputed.ᶜu⁰,
        )
        DataLayouts.foreach_column(
            sw,
            Y.c,
            Y.f,
            geom,
            tke_pre,
            edmf_col,
            sfc,
            scratch;
            shmem = Val(:none),
        )
    else
        DataLayouts.foreach_column(
            sw,
            Y.c,
            Y.f,
            geom,
            tke_pre,
            (),
            sfc,
            scratch;
            shmem = Val(:none),
        )
    end

    # Refresh face diffusivities ᶠK_h, ᶠK_u from updated tke^{n+1}.
    refresh_diffusivities && set_face_diffusivities!(Y, p)

    return nothing
end

# ============================================================================
# 6. repair_column_water_conservatively!
# ============================================================================

struct WaterRepairSweep{HasNonEquil, TP, FT} <: ColumnSweep
    thermo_params::TP
end

@inline function (sw::WaterRepairSweep{HasNonEquil, TP, FT})(
    Yc_col,
    geom_col,
    pre_col,
    evap_scratch_col,
) where {HasNonEquil, TP, FT}
    ᶜρ_fv = Fields.field_values(Yc_col.ρ)
    ᶜρqt_fv = Fields.field_values(Yc_col.ρq_tot)
    ᶜρe_fv = Fields.field_values(Yc_col.ρe_tot)
    ᶜJ_fv = Fields.field_values(geom_col.ᶜJ)
    ᶜΦ_fv = Fields.field_values(geom_col.ᶜΦ)
    ᶜK_fv = Fields.field_values(pre_col.ᶜK)
    e_vap_fv = Fields.field_values(evap_scratch_col)
    nz = Spaces.nlevels(axes(Yc_col.ρ))
    ε = eps(FT)

    # Pass 1 clips negative condensates at fixed ρq_tot and totals any column
    # vapor deficit before diagnosing donor temperatures.
    total_deficit = zero(FT)
    if HasNonEquil
        ᶜρql_fv = Fields.field_values(Yc_col.ρq_lcl)
        ᶜρqi_fv = Fields.field_values(Yc_col.ρq_icl)
        ᶜρqr_fv = Fields.field_values(Yc_col.ρq_rai)
        ᶜρqs_fv = Fields.field_values(Yc_col.ρq_sno)
        @inbounds for k in 1:nz
            ql_k = ᶜρql_fv[k]
            qi_k = ᶜρqi_fv[k]
            qr_k = ᶜρqr_fv[k]
            qs_k = ᶜρqs_fv[k]
            ql_k < zero(FT) && (ql_k = zero(FT); ᶜρql_fv[k] = ql_k)
            qi_k < zero(FT) && (qi_k = zero(FT); ᶜρqi_fv[k] = qi_k)
            qr_k < zero(FT) && (qr_k = zero(FT); ᶜρqr_fv[k] = qr_k)
            qs_k < zero(FT) && (qs_k = zero(FT); ᶜρqs_fv[k] = qs_k)
            ρqv_k = ᶜρqt_fv[k] - (ql_k + qi_k + qr_k + qs_k)
            if ρqv_k < zero(FT)
                total_deficit += ᶜJ_fv[k] * (-ρqv_k)
            end
        end
    else
        @inbounds for k in 1:nz
            ρqv_k = ᶜρqt_fv[k]
            if ρqv_k < zero(FT)
                total_deficit += ᶜJ_fv[k] * (-ρqv_k)
            end
        end
    end
    total_deficit == zero(FT) && return nothing

    # Step 2: Redistribute any remaining negative q_vap = (ρq_tot - Σ ρq_cond)/ρ
    # conservatively within the column, updating ρ (to conserve ρ_d) and ρe_tot
    # (at mass-weighted donor vapor energy so ∫ ρe_tot dz is conserved).
    total_excess = zero(FT)
    excess_energy = zero(FT)
    col_mass = zero(FT)

    @inbounds for k in 1:nz
        ρ_k = ᶜρ_fv[k]
        inv_ρ = inv(max(ρ_k, ε))
        ρqt_k = ᶜρqt_fv[k]
        ρq_cond = if HasNonEquil
            Fields.field_values(Yc_col.ρq_lcl)[k] +
            Fields.field_values(Yc_col.ρq_icl)[k] +
            Fields.field_values(Yc_col.ρq_rai)[k] +
            Fields.field_values(Yc_col.ρq_sno)[k]
        else
            zero(FT)
        end

        ρqv_k = ρqt_k - ρq_cond
        J_k = ᶜJ_fv[k]
        col_mass += J_k * max(ρ_k, ε)

        if ρqv_k > zero(FT)
            ql_k =
                HasNonEquil ?
                (
                    Fields.field_values(Yc_col.ρq_lcl)[k] +
                    Fields.field_values(Yc_col.ρq_rai)[k]
                ) * inv_ρ : zero(FT)
            qi_k =
                HasNonEquil ?
                (
                    Fields.field_values(Yc_col.ρq_icl)[k] +
                    Fields.field_values(Yc_col.ρq_sno)[k]
                ) * inv_ρ : zero(FT)
            T_k = _cell_temperature(
                sw.thermo_params,
                Val(HasNonEquil),
                Val(true),
                ρ_k,
                ᶜρe_fv[k],
                ᶜΦ_fv[k],
                ᶜK_fv[k],
                ρqt_k * inv_ρ,
                ql_k,
                qi_k,
            )
            e_vap_k =
                TD.internal_energy_vapor(sw.thermo_params, T_k) +
                ᶜΦ_fv[k] +
                ᶜK_fv[k]
            e_vap_fv[k] = e_vap_k
            m_exc = J_k * ρqv_k
            total_excess += m_exc
            excess_energy += m_exc * e_vap_k
        end
    end

    if total_deficit > zero(FT) && total_excess > ε * col_mass
        scale_exc = min(total_deficit / total_excess, one(FT))
        scale_def = min(total_excess / total_deficit, one(FT))
        e_donor = excess_energy / total_excess

        @inbounds for k in 1:nz
            ρ_k = ᶜρ_fv[k]
            ρqt_k = ᶜρqt_fv[k]
            ρq_cond = if HasNonEquil
                Fields.field_values(Yc_col.ρq_lcl)[k] +
                Fields.field_values(Yc_col.ρq_icl)[k] +
                Fields.field_values(Yc_col.ρq_rai)[k] +
                Fields.field_values(Yc_col.ρq_sno)[k]
            else
                zero(FT)
            end

            ρqv_k = ρqt_k - ρq_cond
            if ρqv_k < zero(FT)
                Δρq = -scale_def * ρqv_k
                ᶜρqt_fv[k] = ρqt_k + Δρq
                ᶜρ_fv[k] = ρ_k + Δρq
                ᶜρe_fv[k] += Δρq * e_donor
            elseif ρqv_k > zero(FT)
                Δρq = -scale_exc * ρqv_k
                ᶜρqt_fv[k] = ρqt_k + Δρq
                ᶜρ_fv[k] = ρ_k + Δρq
                ᶜρe_fv[k] += Δρq * e_vap_fv[k]
            end
        end
    end

    return nothing
end

"""
    repair_column_water_conservatively!(Y, p)

Borrow negative condensates from vapor and redistribute any column-local
negative `ρq_tot` / `q_vap` conservatively within the column, updating `ρ` and
`ρe_tot` consistently so `∫ ρ_d dz`, `∫ ρq_tot dz`, and `∫ ρe_tot dz` are
conserved to round-off before coupled microphysics and sedimentation.
"""
function repair_column_water_conservatively!(Y, p)
    MatrixFields.has_field(Y.c, @name(ρq_tot)) || return nothing

    FT = Spaces.undertype(axes(Y.c))
    thermo_params = CAP.thermodynamics_params(p.params)
    has_nonequil =
        p.atmos.microphysics_model isa NonEquilibriumMicrophysics &&
        MatrixFields.has_field(Y.c, @name(ρq_lcl))

    p.precomputed.ᶜK .= compute_kinetic(Y.c.uₕ, Y.f.u₃)

    ᶜJ = Fields.local_geometry_field(Y.c).J
    geom = (; ᶜJ, p.core.ᶜΦ)
    pre = (; p.precomputed.ᶜK)

    sw = WaterRepairSweep{has_nonequil, typeof(thermo_params), FT}(thermo_params)
    DataLayouts.foreach_column(
        sw, Y.c, geom, pre, p.scratch.ᶜtemp_scalar; shmem = Val(:none),
    )
    return nothing
end

# ============================================================================
# 7. step_slab_ocean_turbulent_and_precip!
# ============================================================================

# Convert a `C3` surface flux vector to a horizontal-area flux matching `ᶜdivᵥ`
# into cell 1 on both flat and sloped-terrain grids.
@inline _sfc_c3_to_horiz_flux(v, lg) =
    (parent(lg.gⁱʲ)[end] / parent(lg.∂ξ∂x)[end]) * v.u₃

function _step_slab_ocean_turbulent!(Y, p, Δt)
    surface = p.atmos.surface
    !(surface.temperature isa SlabOceanTemperature) && return nothing
    p.atmos.disable_surface_flux_tendency && return nothing

    slab = surface.temperature
    FT = eltype(Y)
    Δt_FT = FT(Δt)
    hcap = slab.ρ_ocean * slab.cp_ocean * slab.depth_ocean
    sfc_cond = p.precomputed.sfc_conditions
    sfc_lg = Fields.local_geometry_field(sfc_cond)

    @. Y.sfc.T -= Δt_FT * _sfc_c3_to_horiz_flux(sfc_cond.ρ_flux_h_tot, sfc_lg) / hcap
    if !(p.atmos.microphysics_model isa DryModel)
        @. Y.sfc.water -=
            Δt_FT * _sfc_c3_to_horiz_flux(sfc_cond.ρ_flux_q_tot, sfc_lg)
    end
    return nothing
end

function _step_slab_ocean_precip!(Y, p, Δt)
    surface = p.atmos.surface
    !(surface.temperature isa SlabOceanTemperature) && return nothing
    p.atmos.microphysics_model isa DryModel && return nothing

    slab = surface.temperature
    FT = eltype(Y)
    Δt_FT = FT(Δt)
    hcap = slab.ρ_ocean * slab.cp_ocean * slab.depth_ocean
    P_liq = p.precomputed.surface_rain_flux
    P_snow = p.precomputed.surface_snow_flux
    pet = p.precomputed.col_integrated_precip_energy_tendency

    @. Y.sfc.water -= Δt_FT * (P_liq + P_snow)
    @. Y.sfc.T -= Δt_FT * pet / hcap
    return nothing
end

"""
    step_slab_ocean_turbulent_and_precip!(Y, p, Δt)

Advance `SlabOceanTemperature` (`Y.sfc.T`, `Y.sfc.water`) over realized `Δt`
from realized turbulent (`p.precomputed.sfc_conditions`) and precipitation
(`p.precomputed.surface_rain_flux`, `surface_snow_flux`,
`col_integrated_precip_energy_tendency`) fluxes so atmosphere + slab conserve
water and energy to round-off. Radiative and Q-flux terms remain in explicit
`surface_temp_tendency!`.
"""
function step_slab_ocean_turbulent_and_precip!(Y, p, Δt)
    _step_slab_ocean_turbulent!(Y, p, Δt)
    _step_slab_ocean_precip!(Y, p, Δt)
    return nothing
end

# ============================================================================
# 8. step_column_physics! (composite sequencer)
# ============================================================================

"""
    refresh_pre_microphysics_thermo!(Y, p, t)

Refresh core velocities and thermodynamics (plus `PrognosticEDMFX` draft and
environment thermodynamics) after column turbulence and
`repair_column_water_conservatively!`, prior to coupled microphysics and
sedimentation, along with SGS quadrature variances and cloud fraction when
`!not_quadrature(p.atmos.sgs_quadrature)`, or `0`/`1` `ᶜcloud_fraction` when
`p.atmos.cloud_model isa GridScaleCloud`.

Skips surface condition updates and mixing-length updates so realized turbulent
surface fluxes stay intact.
"""
function refresh_pre_microphysics_thermo!(Y, p, t)
    set_implicit_precomputed_quantities_core!(Y, p, t)
    if !not_quadrature(p.atmos.sgs_quadrature)
        set_covariance_cache_and_cloud_fraction!(Y, p)
    elseif p.atmos.cloud_model isa GridScaleCloud
        set_cloud_fraction!(
            Y,
            p,
            p.atmos.microphysics_model,
            p.atmos.cloud_model,
        )
    end
    return nothing
end

# Maximum substep duration [s] for coupled EDMFX turbulence + microphysics
# subcycling, bounding operator-splitting error between updraft ascent/detrainment
# and microphysical condensation/sedimentation at coarse GCM steps (`Δt > 60 s`).
const _EDMFX_MAX_SUBSTEP_DT = 60

"""
    step_column_physics!(Y, p, t, Δt; ᶠF̃ = nothing)

Advance column turbulence, convection, microphysics, sedimentation, and slab
ocean exchange over `Δt` [s]:

 1. Column turbulence/convection (`step_edmf_and_env_turbulence!` or
    `step_vertical_diffusion_and_surface!`) and `step_prognostic_tke!`.
 2. Slab ocean turbulent exchange from realized `sfc_conditions`.
 3. Column-conservative water repair (`repair_column_water_conservatively!`).
 4. Thermodynamic and cloud-fraction refresh (`refresh_pre_microphysics_thermo!`) followed
    by `step_coupled_microphysics_sedimentation!` (skipped under `DryModel`,
    whose surface precipitation fluxes stay zero; `PrognosticEDMFX` requires a
    moist model).
 5. Slab ocean precipitation exchange from realized bottom fluxes.

When `turbconv_model isa PrognosticEDMFX` and `Δt > _EDMFX_MAX_SUBSTEP_DT`
(`60 s`), subcycles the coupled sequence `1..5` with `n_sub = ceil(Int, Δt / 60)`
equal substeps `Δt_sub = Δt / n_sub`, refreshing draft constraints, entrainment/
detrainment closures, and eddy diffusivities between substeps and storing
step-mean realized surface stress and precipitation fluxes on `p.precomputed`.
"""
function step_column_physics!(Y, p, t, Δt; ᶠF̃ = nothing)
    (; turbconv_model, microphysics_model) = p.atmos
    FT = Spaces.undertype(axes(Y.c))

    if turbconv_model isa PrognosticEDMFX
        n_sub = max(1, ceil(Int, FT(Δt) / FT(_EDMFX_MAX_SUBSTEP_DT)))
        Δt_sub = FT(Δt) / FT(n_sub)
        inv_n = inv(FT(n_sub))
        accum_sfc = n_sub > 1
        has_tau_acc =
            accum_sfc &&
            !p.atmos.disable_surface_flux_tendency &&
            p.atmos.surface.flux_scheme !== nothing &&
            hasproperty(p.scratch, :temp_sfc_uₕ_level) &&
            !(p.scratch.temp_sfc_uₕ_level isa Tuple{})
        if accum_sfc
            fill!(p.scratch.temp_field_level, zero(FT))
            fill!(p.scratch.temp_field_level_2, zero(FT))
            fill!(p.scratch.temp_field_level_3, zero(FT))
            if has_tau_acc
                fill!(parent(p.scratch.temp_sfc_uₕ_level), zero(FT))
            end
        end
        couple_1m_updraft = true
        for m in 1:n_sub
            t_sub = t isa Real ? t + FT(m - 1) * Δt_sub : t
            step_edmf_and_env_turbulence!(
                Y,
                p,
                t_sub,
                Δt_sub;
                ᶠF̃,
                couple_updraft_condensation = couple_1m_updraft,
            )
            step_prognostic_tke!(Y, p, Δt_sub; refresh_diffusivities = m == n_sub)
            _step_slab_ocean_turbulent!(Y, p, Δt_sub)
            record_realized_increment!(Y, p, :turb)
            if has_tau_acc
                @. p.scratch.temp_sfc_uₕ_level +=
                    inv_n * p.precomputed.sfc_conditions.ρ_flux_uₕ
            end
            repair_column_water_conservatively!(Y, p)
            record_realized_increment!(Y, p, :wrepair)
            p.atmos.edmfx_model.filter &&
                enforce_edmf_updraft_constraints!(Y, p, t_sub, turbconv_model)
            record_realized_increment!(Y, p, :fixer)
            refresh_pre_microphysics_thermo!(Y, p, t_sub)
            step_coupled_microphysics_sedimentation!(
                Y,
                p,
                Δt_sub;
                skip_updraft_cloud_formation = couple_1m_updraft,
                diag_weight = inv_n,
                diag_reset = m == 1,
            )
            _step_slab_ocean_precip!(Y, p, Δt_sub)
            record_realized_increment!(Y, p, :micro)
            if accum_sfc
                @. p.scratch.temp_field_level +=
                    inv_n * p.precomputed.surface_rain_flux
                @. p.scratch.temp_field_level_2 +=
                    inv_n * p.precomputed.surface_snow_flux
                @. p.scratch.temp_field_level_3 +=
                    inv_n * p.precomputed.col_integrated_precip_energy_tendency
            end
            if m < n_sub
                t_next = t isa Real ? t + FT(m) * Δt_sub : t
                enforce_physical_constraints!(Y, p, t_next, p.atmos)
                record_realized_increment!(Y, p, :fixer)
                set_implicit_precomputed_quantities_core!(Y, p, t_next)
                if p.atmos.surface.temperature isa
                   SurfaceConditions.SlabOceanTemperature &&
                   !isnothing(p.atmos.surface.flux_scheme)
                    Fields.field_values(p.precomputed.sfc_conditions.T_sfc) .=
                        Fields.field_values(Y.sfc.T)
                end
                set_prognostic_edmf_precomputed_quantities_explicit_closures!(
                    Y,
                    p,
                    t_next,
                )
                set_covariance_cache_and_cloud_fraction!(Y, p)
                set_prognostic_edmf_draft_closures!(Y, p)
                set_face_diffusivities!(Y, p)
                if !uses_covariances(p.atmos)
                    p.precomputed.ᶜl_mix .= ᶜmixing_length(Y, p)
                end
            end
        end
        if accum_sfc
            p.precomputed.surface_rain_flux .= p.scratch.temp_field_level
            p.precomputed.surface_snow_flux .= p.scratch.temp_field_level_2
            p.precomputed.col_integrated_precip_energy_tendency .=
                p.scratch.temp_field_level_3
            if has_tau_acc
                p.precomputed.sfc_conditions.ρ_flux_uₕ .=
                    p.scratch.temp_sfc_uₕ_level
            end
        end
        t_end = t isa Real ? t + FT(Δt) : t
        set_implicit_precomputed_quantities_core!(Y, p, t_end)
    else
        step_vertical_diffusion_and_surface!(Y, p, t, Δt)
        step_prognostic_tke!(Y, p, Δt)
        _step_slab_ocean_turbulent!(Y, p, Δt)
        record_realized_increment!(Y, p, :turb)
        repair_column_water_conservatively!(Y, p)
        record_realized_increment!(Y, p, :wrepair)
        refresh_pre_microphysics_thermo!(Y, p, t)
        if !(microphysics_model isa DryModel)
            step_coupled_microphysics_sedimentation!(Y, p, Δt)
            _step_slab_ocean_precip!(Y, p, Δt)
            record_realized_increment!(Y, p, :micro)
            t_end = t isa Real ? t + FT(Δt) : t
            set_implicit_precomputed_quantities_core!(Y, p, t_end)
        end
    end

    return nothing
end
