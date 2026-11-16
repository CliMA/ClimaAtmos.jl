# ============================================================================
# Post-ARK DSS-Consistent Mass-Flux Reconstruction and Vertical Scalar Transport
# ============================================================================
#
# Reconstructs the continuous step-integrated vertical mass-flux divergence `D`,
# pre-transport density `ρ⁰`, face mass flux `ᶠF̃`, and pre-transport extensive
# scalars `(ρχ)⁰` and `(ρχ_e)⁰` from the ARK implicit stage tendencies `T_imp`,
# then advances grid-mean scalars with a single per-column tridiagonal M-matrix
# factorization and metric-aware third-order Zalesak flux-corrected transport.
# For `PrognosticEDMFX`, updraft states (`sgsʲs`) stay unchanged here and
# evolve under their full updraft velocity in `column_physics_step.jl`.

import ClimaCore: DataLayouts, Fields, Geometry, Operators, Spaces
import ClimaCore.MatrixFields
import ClimaCore.MatrixFields: @name
import ClimaTimeSteppers as CTS
import UnrolledUtilities: unrolled_foreach, unrolled_filter

@inline function _write_stage_entry_χ_e!(ᶠê, Y, ᶜΦ)
    @. ᶠê = ᶠinterp(specific(Y.c.ρe_tot, Y.c.ρ) - ᶜΦ)
    return nothing
end

"""
    record_implicit_stage_entry_energy!(Y, p)

Record the stage-entry face specific internal-plus-kinetic energy
`ê_i = ᶠinterp(e_{tot,i} - Φ)` on implicit stages with `a_imp[i, i] != 0`.
"""
@inline function record_implicit_stage_entry_energy!(Y, p)
    hasproperty(p, :scratch) && hasproperty(p.scratch, :post_step) ||
        return nothing
    isnothing(post_step_integrator_cache(p)) && return nothing
    stage_bufs = p.scratch.ᶠstage_entry_χ_e
    isempty(stage_bufs) && return nothing
    k_imp = advance_post_step_stage_index!(p)
    imp_map = p.scratch.post_step.imp_stage_map
    stage_i = (1 <= k_imp <= length(imp_map)) ? imp_map[k_imp] : 0
    if 1 <= stage_i <= length(stage_bufs)
        _write_stage_entry_χ_e!(stage_bufs[stage_i], Y, p.core.ᶜΦ)
        p.scratch.stage_entry_χ_e_mask[] |= UInt16(1) << (stage_i - 1)
    end
    return nothing
end

"""
    maybe_record_explicit_imp_stage_energy!(Yₜ, Y, p)

Record `ê_i = ᶠinterp(e_{tot,i} - Φ)` when `implicit_tendency!` evaluates an
explicit-diagonal IMEX stage (`a_imp[i, i] == 0`, identified by
`Yₜ === integrator_cache.T_imp[i]`).
"""
@inline function maybe_record_explicit_imp_stage_energy!(Yₜ, Y, p)
    hasproperty(p, :scratch) && hasproperty(p.scratch, :post_step) ||
        return nothing
    explicit_stages = p.scratch.post_step.explicit_imp_stages.rc
    isempty(explicit_stages) && return nothing
    stage_bufs = p.scratch.ᶠstage_entry_χ_e
    isempty(stage_bufs) && return nothing
    for (i, T_imp_i) in explicit_stages
        if Yₜ === T_imp_i && 1 <= i <= length(stage_bufs)
            _write_stage_entry_χ_e!(stage_bufs[i], Y, p.core.ᶜΦ)
            p.scratch.stage_entry_χ_e_mask[] |= UInt16(1) << (i - 1)
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------
# Stage tendency accumulation and mass-projected flux helpers
# ----------------------------------------------------------------------------

@inline function _weighted_dss_scalar!(ᶜfield, p)
    if do_dss(axes(ᶜfield))
        Spaces.weighted_dss!(ᶜfield, p.scratch.ᶜscalar_dss_buffer)
    end
    return ᶜfield
end

# Return `true` when any stage with `b_imp[i] != 0` has `a_imp[i, i] == 0`.
# Implicit-diagonal stages compute `T_imp[i] = (U - temp) / dtγ` from DSSed
# states and are continuous across spectral-element boundaries; explicit-
# diagonal stages evaluate `T_imp!` directly without DSS and require
# `weighted_dss!` on C⁰ spectral-element grids.
@inline function _has_explicit_b_imp_stage(integrator_cache)
    has_implicit_stage_tendencies(integrator_cache) || return false
    a_imp = integrator_cache.tableau.a_imp
    b_imp = integrator_cache.tableau.b_imp
    s = CTS.val_to_int(CTS.get_val_S(b_imp))
    @inbounds for i in 1:s
        if iszero(a_imp[i, i]) && !iszero(b_imp[i])
            return true
        end
    end
    return false
end

struct _StageScalarAccumulator{F, N, FT}
    ᶜdest::F
    field_name::N
    z_ft::FT
end

@inline function (acc::_StageScalarAccumulator)(_i, w_i, T_imp_i)
    w_ft = oftype(acc.z_ft, w_i)
    ᶜsrc = MatrixFields.get_field(T_imp_i.c, acc.field_name)
    @. acc.ᶜdest += w_ft * ᶜsrc
    return nothing
end

function _accumulate_stage_tendency!(ᶜdest, p, integrator_cache, field_name)
    FT = Spaces.undertype(axes(ᶜdest))
    z = zero(FT)
    fill!(ᶜdest, z)
    acc = _StageScalarAccumulator(ᶜdest, field_name, z)
    foreach_b_imp_stage(acc, integrator_cache)
    if _has_explicit_b_imp_stage(integrator_cache)
        _weighted_dss_scalar!(ᶜdest, p)
    end
    return ᶜdest
end

struct _StageScalarSubtractor{F, N, FT}
    ᶜdest::F
    field_name::N
    Δt::FT
end

@inline function (sub::_StageScalarSubtractor)(_i, w_i, T_imp_i)
    c = sub.Δt * oftype(sub.Δt, w_i)
    ᶜsrc = MatrixFields.get_field(T_imp_i.c, sub.field_name)
    @. sub.ᶜdest -= c * ᶜsrc
    return nothing
end

function _subtract_stage_tendency!(
    ᶜdest,
    p,
    Δt,
    integrator_cache,
    field_name,
    ᶜaccum,
)
    if _has_explicit_b_imp_stage(integrator_cache) && do_dss(axes(ᶜdest))
        _accumulate_stage_tendency!(ᶜaccum, p, integrator_cache, field_name)
        @. ᶜdest -= Δt * ᶜaccum
    else
        sub = _StageScalarSubtractor(ᶜdest, field_name, Δt)
        foreach_b_imp_stage(sub, integrator_cache)
    end
    return ᶜdest
end

"""
    _project_and_integrate_mass_flux_column!(ᶠF_col, ᶜD_raw_col, ᶜρ_new_col, ᶜlg_col, ᶠlg_col)

Apply the mass-weighted column-sum projection to `ᶜD_raw_col` against
`ᶜρ_new_col` and integrate `∂_z F̃ = -D` upward from `F̃_{1/2} = 0` with
`F̃_{nz+1/2} = 0`. Accumulates column sums and partial integrals in `Float64`
registers so `Float32` columns retain full relative precision into low-density
upper levels.
"""
@inline function _project_and_integrate_mass_flux_column!(
    ᶠF_col,
    ᶜD_raw_col,
    ᶜρ_new_col,
    ᶜlg_col,
    ᶠlg_col,
)
    FT = Spaces.undertype(axes(ᶜρ_new_col))
    F_fv = Fields.field_values(ᶠF_col)
    D_raw_fv = Fields.field_values(ᶜD_raw_col)
    ρ_new_fv = Fields.field_values(ᶜρ_new_col)
    ᶜlg_fv = Fields.field_values(ᶜlg_col)
    ᶠlg_fv = Fields.field_values(ᶠlg_col)
    nz = Spaces.nlevels(axes(ᶜρ_new_col))

    sum_JD = 0.0
    sum_Jρ = 0.0
    @inbounds for k in 1:nz
        J_k = Float64(ᶜlg_fv[k].J)
        sum_JD = muladd(J_k, Float64(D_raw_fv[k]), sum_JD)
        sum_Jρ = muladd(J_k, Float64(ρ_new_fv[k]), sum_Jρ)
    end
    α_proj = sum_JD / sum_Jρ

    @inbounds F_fv[1] = CT3(zero(FT))
    JF_curr = 0.0
    @inbounds for k in 1:(nz - 1)
        J_k = Float64(ᶜlg_fv[k].J)
        D_k = muladd(-α_proj, Float64(ρ_new_fv[k]), Float64(D_raw_fv[k]))
        JF_curr -= J_k * D_k
        J_f = Float64(ᶠlg_fv[k + 1].J)
        F_fv[k + 1] = CT3(FT(JF_curr / J_f))
    end
    @inbounds F_fv[nz + 1] = CT3(zero(FT))
    return α_proj
end

struct _StageEnergyFluxColumnAccum{FT} <: ColumnSweep
    w_i::FT
end

@inline function (op::_StageEnergyFluxColumnAccum{FT})(
    ᶠF_χe_col,
    ᶜD_stage_col,
    ᶜρ_new_col,
    ᶠê_col,
    ᶜlg_col,
    ᶠlg_col,
) where {FT}
    F_χe_fv = Fields.field_values(ᶠF_χe_col)
    D_fv = Fields.field_values(ᶜD_stage_col)
    ρ_new_fv = Fields.field_values(ᶜρ_new_col)
    ê_fv = Fields.field_values(ᶠê_col)
    ᶜlg_fv = Fields.field_values(ᶜlg_col)
    ᶠlg_fv = Fields.field_values(ᶠlg_col)
    nz = Spaces.nlevels(axes(ᶜρ_new_col))

    sum_JD = 0.0
    sum_Jρ = 0.0
    @inbounds for k in 1:nz
        J_k = Float64(ᶜlg_fv[k].J)
        sum_JD = muladd(J_k, Float64(D_fv[k]), sum_JD)
        sum_Jρ = muladd(J_k, Float64(ρ_new_fv[k]), sum_Jρ)
    end
    α_i = sum_JD / sum_Jρ

    JF_curr = 0.0
    w_i = op.w_i
    @inbounds for k in 1:(nz - 1)
        J_k = Float64(ᶜlg_fv[k].J)
        D_k = muladd(-α_i, Float64(ρ_new_fv[k]), Float64(D_fv[k]))
        JF_curr -= J_k * D_k
        f = k + 1
        F_i_u3 = FT(JF_curr / Float64(ᶠlg_fv[f].J))
        F_χe_fv[f] = CT3(muladd(w_i * F_i_u3, ê_fv[f], F_χe_fv[f].u³))
    end
    return nothing
end

struct _StageEnergyFluxAccumulator{FX, YV, PV, SB, FB, AC, AI, LG, FLG, FT}
    ᶠF_χe_accum::FX
    Y::YV
    p::PV
    stage_bufs::SB
    ᶠê_fallback::FB
    ᶜD_dss::AC
    a_imp::AI
    ᶜlg::LG
    ᶠlg::FLG
    mask::UInt16
    z_ft::FT
end

@inline function (acc::_StageEnergyFluxAccumulator)(i, w_i, T_imp_i)
    w_ft = oftype(acc.z_ft, w_i)
    has_recorded =
        (1 <= i <= length(acc.stage_bufs)) &&
        !iszero(acc.mask & (UInt16(1) << (i - 1)))
    ᶠê_i = has_recorded ? acc.stage_bufs[i] : acc.ᶠê_fallback
    col_op = _StageEnergyFluxColumnAccum(w_ft)
    if iszero(acc.a_imp[i, i]) && do_dss(axes(acc.ᶜD_dss))
        @. acc.ᶜD_dss = T_imp_i.c.ρ
        _weighted_dss_scalar!(acc.ᶜD_dss, acc.p)
        DataLayouts.foreach_column(
            col_op,
            acc.ᶠF_χe_accum,
            acc.ᶜD_dss,
            acc.Y.c.ρ,
            ᶠê_i,
            acc.ᶜlg,
            acc.ᶠlg;
            shmem = Val(:none),
        )
    else
        DataLayouts.foreach_column(
            col_op,
            acc.ᶠF_χe_accum,
            T_imp_i.c.ρ,
            acc.Y.c.ρ,
            ᶠê_i,
            acc.ᶜlg,
            acc.ᶠlg;
            shmem = Val(:none),
        )
    end
    return nothing
end

# Return `true` when a stage with `b_imp[i] != 0` has no entry energy recorded
# in the bit `i` of `mask` (or no buffer `i` among the `n_bufs` stage buffers).
@inline function _needs_stage_entry_fallback(mask::UInt16, n_bufs, integrator_cache)
    b_imp = integrator_cache.tableau.b_imp
    s = CTS.val_to_int(CTS.get_val_S(b_imp))
    @inbounds for i in 1:s
        recorded = (i <= n_bufs) && !iszero(mask & (UInt16(1) << (i - 1)))
        if !iszero(b_imp[i]) && !recorded
            return true
        end
    end
    return false
end

function _reconstruct_pre_transport_energy!(
    Y,
    p,
    Δt,
    integrator_cache,
    ᶜD_dss,
    ᶠF_χe_accum,
)
    FT = typeof(Δt)
    z = zero(FT)
    fill!(ᶠF_χe_accum, CT3(z))
    mask = p.scratch.stage_entry_χ_e_mask[]
    stage_bufs = p.scratch.ᶠstage_entry_χ_e
    ᶠê_fallback = p.scratch.ᶠtemp_scalar
    # Fallback face energy field for the stages without a recorded entry
    # energy, as when `step_vertical_scalar_transport!` is invoked directly on
    # synthetic `T_imp` buffers outside `CTS.step!`.
    if _needs_stage_entry_fallback(mask, length(stage_bufs), integrator_cache)
        _write_stage_entry_χ_e!(ᶠê_fallback, Y, p.core.ᶜΦ)
    end
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶠlg = Fields.local_geometry_field(Y.f)
    acc = _StageEnergyFluxAccumulator(
        ᶠF_χe_accum,
        Y,
        p,
        stage_bufs,
        ᶠê_fallback,
        ᶜD_dss,
        integrator_cache.tableau.a_imp,
        ᶜlg,
        ᶠlg,
        mask,
        z,
    )
    foreach_b_imp_stage(acc, integrator_cache)
    @. Y.c.ρe_tot =
        Y.c.ρe_tot - Y.c.ρ * p.core.ᶜΦ + Δt * ᶜadvdivᵥ(ᶠF_χe_accum)
    return nothing
end

# ----------------------------------------------------------------------------
# Column M-matrix factorization and metric-aware Zalesak FCT kernels
# ----------------------------------------------------------------------------

"""
    _vt_fct_ratio(P, Q)

Return the Zalesak flux-correction ratio `min(1, Q / P)` for non-negative
antidiffusive demand `P` and capacity `Q >= 0`, returning `1` when `P <= 0` so
zero-flux species (such as clear-air condensates) impose no limiter clipping on
co-transported thermodynamic scalars.
"""
@inline function _vt_fct_ratio(P, Q)
    FT = typeof(P)
    return ifelse(P <= zero(FT), one(FT), ifelse(Q >= P, one(FT), Q / P))
end

"""
    _upwind3_antidiff_delta(JF_f, f, nz, χ_fv, ᶜlg_fv)

Metric-aware third-order finite-volume upwind antidiffusive face increment
`χ_HO - χ_up` at interior face `f ∈ 2:nz` (between cells `f - 1` and `f`).
Reconstructs the quadratic primitive over cell metric volumes `V_m = ᶜlg[m].J`,
reducing to `(-2 χ_u + 10 χ_c + 4 χ_d) / 12 - χ_c` on uniform grids and to
two-cell central interpolation at the two faces adjacent to the boundaries.
"""
@inline function _upwind3_antidiff_delta(
    JF_f,
    f::Int,
    nz::Int,
    χ_fv,
    ᶜlg_fv,
)
    FT = typeof(JF_f)
    if JF_f >= zero(FT)
        c = f - 1
        d = f
        χ_c = @inbounds χ_fv[c]
        χ_d = @inbounds χ_fv[d]
        V_c = @inbounds ᶜlg_fv[c].J
        V_d = @inbounds ᶜlg_fv[d].J
        if f == 2
            return V_c * (χ_d - χ_c) / (V_c + V_d)
        else
            u = f - 2
            χ_u = @inbounds χ_fv[u]
            V_u = @inbounds ᶜlg_fv[u].J
            V_uc = V_u + V_c
            return (V_c / (V_uc + V_d)) * (
                (V_uc / (V_c + V_d)) * (χ_d - χ_c) +
                (V_d / V_uc) * (χ_c - χ_u)
            )
        end
    else
        c = f
        d = f - 1
        χ_c = @inbounds χ_fv[c]
        χ_d = @inbounds χ_fv[d]
        V_c = @inbounds ᶜlg_fv[c].J
        V_d = @inbounds ᶜlg_fv[d].J
        if f == nz
            return V_c * (χ_d - χ_c) / (V_c + V_d)
        else
            u = f + 1
            χ_u = @inbounds χ_fv[u]
            V_u = @inbounds ᶜlg_fv[u].J
            V_uc = V_u + V_c
            return (V_c / (V_uc + V_d)) * (
                (V_uc / (V_c + V_d)) * (χ_d - χ_c) +
                (V_d / V_uc) * (χ_c - χ_u)
            )
        end
    end
end

# Integer upwinding mode tags passed into GPU column kernels (`isbits`).
const _UPW_NONE = Int8(0)
const _UPW_FIRST_ORDER = Int8(1)
const _UPW_VANLEER = Int8(2)
const _UPW_THIRD_ORDER = Int8(3)

@inline _vt_upw_code(::Val{:none}) = _UPW_NONE
@inline _vt_upw_code(::Val{:first_order}) = _UPW_FIRST_ORDER
@inline _vt_upw_code(::Val{:vanleer_limiter}) = _UPW_VANLEER
@inline _vt_upw_code(::Val) = _UPW_THIRD_ORDER

@inline function _vt_rho0_k(
    ρ_new_fv,
    F_fv,
    ᶜlg_fv,
    ᶠlg_fv,
    Δt,
    k::Int,
)
    @inbounds begin
        JF_bot = ᶠlg_fv[k].J * F_fv[k].u³
        JF_top = ᶠlg_fv[k + 1].J * F_fv[k + 1].u³
        return ρ_new_fv[k] + (Δt * ᶜlg_fv[k].invJ) * (JF_top - JF_bot)
    end
end

@inline function _vt_face_courant(
    ρ_new_fv,
    F_fv,
    ᶜlg_fv,
    ᶠlg_fv,
    Δt,
    f::Int,
    nz::Int,
)
    FT = typeof(Δt)
    (f <= 1 || f > nz) && return zero(FT)
    @inbounds begin
        JF_abs = abs(ᶠlg_fv[f].J * F_fv[f].u³)
        M_lo = ᶜlg_fv[f - 1].J * ρ_new_fv[f - 1]
        M_hi = ᶜlg_fv[f].J * ρ_new_fv[f]
        M_min = max(min(M_lo, M_hi), floatmin(FT))
        return (Δt * JF_abs) / M_min
    end
end

# Factor the implicit donor-cell M-matrix `M = diag(ρ^{n+1}) + Δt ∂_z^{up}(F̃)`
# per column: row `k` has `dl_k = -s_k max(JF_k, 0)`, `du_k = -s_k max(-JF_{k+1}, 0)`,
# and `d_k = ρ^{n+1}_k + s_k (max(JF_{k+1}, 0) + max(-JF_k, 0))` (`s_k = Δt / J_k`).
# Each `J_k`-weighted column sum equals `J_k ρ^{n+1}_k > 0`, making `M` a strict
# nonsingular M-matrix whose Thomas upper pivots `u_curr` are strictly positive.
@inline function _factor_inv_u_column!(
    inv_u_fv,
    ᶠF̃_col,
    ᶜD_raw_col,
    ᶜρ_new_col,
    ᶜlg_col,
    ᶠlg_col,
    Δt,
)
    FT = typeof(Δt)
    _project_and_integrate_mass_flux_column!(
        ᶠF̃_col,
        ᶜD_raw_col,
        ᶜρ_new_col,
        ᶜlg_col,
        ᶠlg_col,
    )
    F_fv = Fields.field_values(ᶠF̃_col)
    ρ_new_fv = Fields.field_values(ᶜρ_new_col)
    ᶜlg_fv = Fields.field_values(ᶜlg_col)
    ᶠlg_fv = Fields.field_values(ᶠlg_col)
    nz = Spaces.nlevels(axes(ᶜρ_new_col))
    z = zero(FT)
    u_prev = one(FT)
    du_prev = z
    @inbounds for k in 1:nz
        ρ_new_k = ρ_new_fv[k]
        s_k = Δt * ᶜlg_fv[k].invJ
        JF_bot = ᶠlg_fv[k].J * F_fv[k].u³
        JF_top = ᶠlg_fv[k + 1].J * F_fv[k + 1].u³

        dl_k = -s_k * max(JF_bot, z)
        du_k = -s_k * max(-JF_top, z)
        d_k = ρ_new_k + s_k * (max(JF_top, z) + max(-JF_bot, z))
        u_curr = k == 1 ? d_k : muladd(-(dl_k / u_prev), du_prev, d_k)
        inv_u_fv[k] = inv(max(u_curr, floatmin(FT)))
        u_prev = max(u_curr, floatmin(FT))
        du_prev = du_k
    end
    return nothing
end

@inline function _thomas_solve_m!(
    χL_fv,
    ρχ0_fv,
    inv_u_fv,
    F_fv,
    ᶜlg_fv,
    ᶠlg_fv,
    Δt,
    nz::Int,
)
    FT = typeof(Δt)
    z = zero(FT)
    @inbounds begin
        χL_fv[1] = ρχ0_fv[1]
        for k in 2:nz
            s_k = Δt * ᶜlg_fv[k].invJ
            dl_k = -s_k * max(ᶠlg_fv[k].J * F_fv[k].u³, z)
            l_k = dl_k * inv_u_fv[k - 1]
            χL_fv[k] = muladd(-l_k, χL_fv[k - 1], ρχ0_fv[k])
        end
        χL_fv[nz] = χL_fv[nz] * inv_u_fv[nz]
        for k in (nz - 1):-1:1
            s_k = Δt * ᶜlg_fv[k].invJ
            du_k = -s_k * max(-(ᶠlg_fv[k + 1].J * F_fv[k + 1].u³), z)
            χL_fv[k] = muladd(-du_k, χL_fv[k + 1], χL_fv[k]) * inv_u_fv[k]
        end
    end
    return nothing
end

@inline function _compute_raw_antidiff_transfers!(
    Φ_fv,
    χL_fv,
    F_fv,
    ρ_new_fv,
    ᶜlg_fv,
    ᶠlg_fv,
    Δt,
    nz::Int,
)
    FT = typeof(Δt)
    z = zero(FT)
    o = one(FT)
    @inbounds Φ_fv[1] = z
    @inbounds for f in 2:nz
        JF_f = ᶠlg_fv[f].J * F_fv[f].u³
        c_f = _vt_face_courant(ρ_new_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, f, nz)
        w_C_f = ifelse(c_f > o, inv(c_f), o)
        Δχ_f = _upwind3_antidiff_delta(JF_f, f, nz, χL_fv, ᶜlg_fv)
        Φ_fv[f] = Δt * JF_f * w_C_f * Δχ_f
    end
    @inbounds Φ_fv[nz + 1] = z
    return nothing
end

"""
    _narrow_face_fct_limiter!(α_fv, Φ_fv, χL_fv, ρχ0_fv, F_fv, ρ_new_fv, ᶜlg_fv, ᶠlg_fv, Δt, nz, is_specific_χ0)

Intersect `α_fv[2:nz]` in place with the Zalesak FCT face limiters of a scalar
whose low-order solution is `χL_fv` and whose pre-transport profile is either
extensive (`ρχ0_fv / ρ0_k` when `is_specific_χ0 == false`) or already specific
(`ρχ0_fv` when `is_specific_χ0 == true`, as for `q_vap⁰`).
"""
@inline function _narrow_face_fct_limiter!(
    α_fv,
    Φ_fv,
    χL_fv,
    ρχ0_fv,
    F_fv,
    ρ_new_fv,
    ᶜlg_fv,
    ᶠlg_fv,
    Δt::FT,
    nz::Int,
    ::Val{is_specific_χ0},
) where {FT <: AbstractFloat, is_specific_χ0}
    z = zero(FT)
    o = one(FT)
    @inbounds for k in 1:nz
        km = max(1, k - 1)
        kp = min(nz, k + 1)

        χL_km = χL_fv[km]
        χL_k = χL_fv[k]
        χL_kp = χL_fv[kp]
        χ_max = max(χL_km, χL_k, χL_kp)
        χ_min = min(χL_km, χL_k, χL_kp)

        c_bot = _vt_face_courant(ρ_new_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, k, nz)
        c_top =
            _vt_face_courant(ρ_new_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, k + 1, nz)
        if max(c_bot, c_top) <= o
            if is_specific_χ0
                χ0_km = ρχ0_fv[km]
                χ0_k = ρχ0_fv[k]
                χ0_kp = ρχ0_fv[kp]
            else
                χ0_km =
                    ρχ0_fv[km] /
                    _vt_rho0_k(ρ_new_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, km)
                χ0_k =
                    ρχ0_fv[k] /
                    _vt_rho0_k(ρ_new_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, k)
                χ0_kp =
                    ρχ0_fv[kp] /
                    _vt_rho0_k(ρ_new_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, kp)
            end
            χ_max = max(χ_max, χ0_km, χ0_k, χ0_kp)
            χ_min = min(χ_min, χ0_km, χ0_k, χ0_kp)
        end

        M_k = ᶜlg_fv[k].J * ρ_new_fv[k]
        Q_plus = M_k * max(z, χ_max - χL_k)
        Q_minus = M_k * max(z, χL_k - χ_min)

        Φ_bot = Φ_fv[k]
        Φ_top = Φ_fv[k + 1]
        P_plus = max(z, Φ_bot) - min(z, Φ_top)
        P_minus = max(z, Φ_top) - min(z, Φ_bot)

        R_plus = _vt_fct_ratio(P_plus, Q_plus)
        R_minus = _vt_fct_ratio(P_minus, Q_minus)

        if k >= 2
            R_for_bot = ifelse(Φ_bot >= z, R_plus, R_minus)
            α_fv[k] = min(α_fv[k], R_for_bot)
        end
        if k <= nz - 1
            R_for_top = ifelse(Φ_top >= z, R_minus, R_plus)
            α_fv[k + 1] = min(α_fv[k + 1], R_for_top)
        end
    end
    return nothing
end

@inline function _apply_limited_antidiff_update!(
    ρχ_fv,
    χL_fv,
    Φ_fv,
    α_fv,
    ρ_new_fv,
    ᶜlg_fv,
    nz::Int,
)
    @inbounds for k in 1:nz
        invJ_k = ᶜlg_fv[k].invJ
        flux_diff = muladd(α_fv[k], Φ_fv[k], -α_fv[k + 1] * Φ_fv[k + 1])
        ρχ_fv[k] = muladd(ρ_new_fv[k], χL_fv[k], flux_diff * invJ_k)
    end
    return nothing
end

@inline function _transport_single_scalar_column!(
    ρχ_fv,
    χL_fv,
    Φ_fv,
    α_fv,
    F_fv,
    ρ_new_fv,
    inv_u_fv,
    ᶜlg_fv,
    ᶠlg_fv,
    Δt,
    nz::Int,
    upw_code::Int8,
)
    FT = typeof(Δt)
    if upw_code == _UPW_NONE
        # Central reconstruction `-Δt ᶜadvdivᵥ(ᶠF̃ * ᶠinterp(χ⁰))` on `(ρχ)⁰`,
        # preserving `χ ≡ 1` to round-off when `ρ⁰ = ρ^{n+1} + Δt ᶜadvdivᵥ(ᶠF̃)`.
        z = zero(FT)
        @inbounds Φ_fv[1] = z
        @inbounds for f in 2:nz
            JF_f = ᶠlg_fv[f].J * F_fv[f].u³
            χ0_lo =
                ρχ_fv[f - 1] /
                _vt_rho0_k(ρ_new_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, f - 1)
            χ0_hi =
                ρχ_fv[f] /
                _vt_rho0_k(ρ_new_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, f)
            Φ_fv[f] = Δt * JF_f * FT(0.5) * (χ0_lo + χ0_hi)
        end
        @inbounds Φ_fv[nz + 1] = z
        @inbounds for k in 1:nz
            invJ_k = ᶜlg_fv[k].invJ
            ρχ_fv[k] = ρχ_fv[k] - (Φ_fv[k + 1] - Φ_fv[k]) * invJ_k
        end
        return nothing
    end

    _thomas_solve_m!(χL_fv, ρχ_fv, inv_u_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, nz)

    if upw_code == _UPW_FIRST_ORDER || nz < 2
        @inbounds for k in 1:nz
            ρχ_fv[k] = ρ_new_fv[k] * χL_fv[k]
        end
        return nothing
    end

    _compute_raw_antidiff_transfers!(
        Φ_fv,
        χL_fv,
        F_fv,
        ρ_new_fv,
        ᶜlg_fv,
        ᶠlg_fv,
        Δt,
        nz,
    )
    o = one(FT)
    @inbounds for f in 1:(nz + 1)
        α_fv[f] = o
    end
    if upw_code == _UPW_VANLEER
        _narrow_face_fct_limiter!(
            α_fv,
            Φ_fv,
            χL_fv,
            ρχ_fv,
            F_fv,
            ρ_new_fv,
            ᶜlg_fv,
            ᶠlg_fv,
            Δt,
            nz,
            Val(false),
        )
    end
    _apply_limited_antidiff_update!(
        ρχ_fv,
        χL_fv,
        Φ_fv,
        α_fv,
        ρ_new_fv,
        ᶜlg_fv,
        nz,
    )
    return nothing
end

@generated function _vt_cond_names(::Type{YcEl}) where {YcEl}
    fnames = fieldnames(YcEl)
    cond_syms = (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
    cond_names =
        Tuple(MatrixFields.FieldName(s) for s in cond_syms if s in fnames)
    return :($cond_names)
end

"""
    _unpack_vt_yc(Yc)

Unpack composite center state `Y.c` into leaf scalar `Field`s and homogeneous
tuples of leaf `Field`s (`()` when a moisture or condensate group is absent, as
under `DryModel`) so `DataLayouts.foreach_column` kernel arguments stay flat and
within GPU tuple-specialization limits on warped grids.
"""
@generated function _unpack_vt_yc(Yc::Fields.Field)
    YcEl = eltype(Yc)
    fnames = fieldnames(YcEl)
    cond_syms = (:ρq_lcl, :ρq_icl, :ρq_rai, :ρq_sno)
    indiv_syms = (:ρn_lcl, :ρn_rai, :ρn_ice, :ρq_rim, :ρb_rim)
    non_passive =
        (:ρ, :uₕ, :ρe_tot, :ρq_tot, :ρtke, :sgsʲs, cond_syms..., indiv_syms...)

    has_qt = :ρq_tot in fnames
    has_tke = :ρtke in fnames
    cond_exprs = [:(Yc.$s) for s in cond_syms if s in fnames]
    indiv_exprs = [:(Yc.$s) for s in indiv_syms if s in fnames]
    pass_exprs = [
        :(Yc.$s) for
        s in fnames if startswith(String(s), "ρ") && !(s in non_passive)
    ]

    return quote
        Base.@_inline_meta
        (;
            ρ = Yc.ρ,
            ρe_tot = Yc.ρe_tot,
            ρq_tot = $(has_qt ? :(Yc.ρq_tot) : :(())),
            cond = ($(cond_exprs...),),
            indiv = ($(indiv_exprs...),),
            passive = ($(pass_exprs...),),
            ρtke = $(has_tke ? :(Yc.ρtke) : :nothing),
        )
    end
end

"""
    _vt_gm_thermo_column!(...)

Project `ᶜD_raw` per column, integrate `ᶠF̃`, factor the grid-mean implicit
donor-cell M-matrix into `c_work_cols[1]` (`inv_u`), and advance grid-mean
`(ρχ_e, ρq_tot, cond)` in a two-pass Zalesak FCT sweep (narrowing a single
shared face limiter `α_fv` across `{χ_e, q_tot, q_c, q_vap}` in pass 1 before
applying the limited antidiffusive fluxes identically in pass 2).
"""
@inline function _vt_gm_thermo_column!(
    ρ_col,
    ρe_col,
    ρqt_col,
    cond_cols,
    ᶜlg_col,
    ᶠlg_col,
    ᶜΦ_col,
    ᶜD_raw_col,
    ᶠF̃_col,
    f_work_cols,
    c_work_cols,
    Δt,
    eh_code::Int8,
)
    FT = typeof(Δt)
    inv_u_fv = Fields.field_values(c_work_cols[1])
    χL_fv = Fields.field_values(c_work_cols[2])
    q_vap_L_fv = Fields.field_values(c_work_cols[3])
    q_vap_0_fv = Fields.field_values(c_work_cols[4])
    α_fv = Fields.field_values(f_work_cols[1])
    Φ_fv = Fields.field_values(f_work_cols[2])

    _factor_inv_u_column!(
        inv_u_fv,
        ᶠF̃_col,
        ᶜD_raw_col,
        ρ_col,
        ᶜlg_col,
        ᶠlg_col,
        Δt,
    )
    eh_code == _UPW_NONE && return nothing

    ρ_new_fv = Fields.field_values(ρ_col)
    ρe_fv = Fields.field_values(ρe_col)
    Φ_grav_fv = Fields.field_values(ᶜΦ_col)
    ᶜlg_fv = Fields.field_values(ᶜlg_col)
    ᶠlg_fv = Fields.field_values(ᶠlg_col)
    F_fv = Fields.field_values(ᶠF̃_col)
    nz = Spaces.nlevels(axes(ρ_col))
    has_q_tot = !(ρqt_col isa Tuple{})
    has_cond = !isempty(cond_cols)

    if eh_code == _UPW_FIRST_ORDER || nz < 2
        _thomas_solve_m!(χL_fv, ρe_fv, inv_u_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, nz)
        @inbounds for k in 1:nz
            ρe_fv[k] = ρ_new_fv[k] * (χL_fv[k] + Φ_grav_fv[k])
        end
        if has_q_tot
            ρqt_fv = Fields.field_values(ρqt_col)
            _thomas_solve_m!(
                χL_fv,
                ρqt_fv,
                inv_u_fv,
                F_fv,
                ᶜlg_fv,
                ᶠlg_fv,
                Δt,
                nz,
            )
            @inbounds for k in 1:nz
                ρqt_fv[k] = ρ_new_fv[k] * χL_fv[k]
            end
        end
        unrolled_foreach(cond_cols) do ρqc_col
            ρqc_fv = Fields.field_values(ρqc_col)
            _thomas_solve_m!(
                χL_fv,
                ρqc_fv,
                inv_u_fv,
                F_fv,
                ᶜlg_fv,
                ᶠlg_fv,
                Δt,
                nz,
            )
            @inbounds for k in 1:nz
                ρqc_fv[k] = ρ_new_fv[k] * χL_fv[k]
            end
        end
        return nothing
    end

    # Pass 1: narrow the shared face limiter `α_fv` across `{χ_e, q_tot, q_c, q_vap}`.
    o = one(FT)
    @inbounds for f in 1:(nz + 1)
        α_fv[f] = o
    end

    if eh_code == _UPW_VANLEER
        _thomas_solve_m!(χL_fv, ρe_fv, inv_u_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, nz)
        _compute_raw_antidiff_transfers!(
            Φ_fv,
            χL_fv,
            F_fv,
            ρ_new_fv,
            ᶜlg_fv,
            ᶠlg_fv,
            Δt,
            nz,
        )
        _narrow_face_fct_limiter!(
            α_fv,
            Φ_fv,
            χL_fv,
            ρe_fv,
            F_fv,
            ρ_new_fv,
            ᶜlg_fv,
            ᶠlg_fv,
            Δt,
            nz,
            Val(false),
        )

        if has_q_tot
            ρqt_fv = Fields.field_values(ρqt_col)
            _thomas_solve_m!(
                χL_fv,
                ρqt_fv,
                inv_u_fv,
                F_fv,
                ᶜlg_fv,
                ᶠlg_fv,
                Δt,
                nz,
            )
            _compute_raw_antidiff_transfers!(
                Φ_fv,
                χL_fv,
                F_fv,
                ρ_new_fv,
                ᶜlg_fv,
                ᶠlg_fv,
                Δt,
                nz,
            )
            _narrow_face_fct_limiter!(
                α_fv,
                Φ_fv,
                χL_fv,
                ρqt_fv,
                F_fv,
                ρ_new_fv,
                ᶜlg_fv,
                ᶠlg_fv,
                Δt,
                nz,
                Val(false),
            )
            if has_cond
                @inbounds for k in 1:nz
                    q_vap_L_fv[k] = χL_fv[k]
                    q_vap_0_fv[k] = ρqt_fv[k]
                end
            end
        end

        if has_cond
            unrolled_foreach(cond_cols) do ρqc_col
                ρqc_fv = Fields.field_values(ρqc_col)
                _thomas_solve_m!(
                    χL_fv,
                    ρqc_fv,
                    inv_u_fv,
                    F_fv,
                    ᶜlg_fv,
                    ᶠlg_fv,
                    Δt,
                    nz,
                )
                _compute_raw_antidiff_transfers!(
                    Φ_fv,
                    χL_fv,
                    F_fv,
                    ρ_new_fv,
                    ᶜlg_fv,
                    ᶠlg_fv,
                    Δt,
                    nz,
                )
                _narrow_face_fct_limiter!(
                    α_fv,
                    Φ_fv,
                    χL_fv,
                    ρqc_fv,
                    F_fv,
                    ρ_new_fv,
                    ᶜlg_fv,
                    ᶠlg_fv,
                    Δt,
                    nz,
                    Val(false),
                )
                if has_q_tot
                    @inbounds for k in 1:nz
                        q_vap_L_fv[k] -= χL_fv[k]
                        q_vap_0_fv[k] -= ρqc_fv[k]
                    end
                end
            end

            if has_q_tot
                @inbounds for k in 1:nz
                    q_vap_0_fv[k] /= _vt_rho0_k(
                        ρ_new_fv,
                        F_fv,
                        ᶜlg_fv,
                        ᶠlg_fv,
                        Δt,
                        k,
                    )
                end
                _compute_raw_antidiff_transfers!(
                    Φ_fv,
                    q_vap_L_fv,
                    F_fv,
                    ρ_new_fv,
                    ᶜlg_fv,
                    ᶠlg_fv,
                    Δt,
                    nz,
                )
                _narrow_face_fct_limiter!(
                    α_fv,
                    Φ_fv,
                    q_vap_L_fv,
                    q_vap_0_fv,
                    F_fv,
                    ρ_new_fv,
                    ᶜlg_fv,
                    ᶠlg_fv,
                    Δt,
                    nz,
                    Val(true),
                )
            end
        end
    end

    # Pass 2: apply the shared face limiter `α_fv` identically to `{χ_e, q_tot, q_c}`.
    _thomas_solve_m!(χL_fv, ρe_fv, inv_u_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, nz)
    _compute_raw_antidiff_transfers!(
        Φ_fv,
        χL_fv,
        F_fv,
        ρ_new_fv,
        ᶜlg_fv,
        ᶠlg_fv,
        Δt,
        nz,
    )
    _apply_limited_antidiff_update!(
        ρe_fv,
        χL_fv,
        Φ_fv,
        α_fv,
        ρ_new_fv,
        ᶜlg_fv,
        nz,
    )
    @inbounds for k in 1:nz
        ρe_fv[k] = muladd(ρ_new_fv[k], Φ_grav_fv[k], ρe_fv[k])
    end

    if has_q_tot
        ρqt_fv = Fields.field_values(ρqt_col)
        _thomas_solve_m!(χL_fv, ρqt_fv, inv_u_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, nz)
        _compute_raw_antidiff_transfers!(
            Φ_fv,
            χL_fv,
            F_fv,
            ρ_new_fv,
            ᶜlg_fv,
            ᶠlg_fv,
            Δt,
            nz,
        )
        _apply_limited_antidiff_update!(
            ρqt_fv,
            χL_fv,
            Φ_fv,
            α_fv,
            ρ_new_fv,
            ᶜlg_fv,
            nz,
        )
    end

    unrolled_foreach(cond_cols) do ρqc_col
        ρqc_fv = Fields.field_values(ρqc_col)
        _thomas_solve_m!(χL_fv, ρqc_fv, inv_u_fv, F_fv, ᶜlg_fv, ᶠlg_fv, Δt, nz)
        _compute_raw_antidiff_transfers!(
            Φ_fv,
            χL_fv,
            F_fv,
            ρ_new_fv,
            ᶜlg_fv,
            ᶠlg_fv,
            Δt,
            nz,
        )
        _apply_limited_antidiff_update!(
            ρqc_fv,
            χL_fv,
            Φ_fv,
            α_fv,
            ρ_new_fv,
            ᶜlg_fv,
            nz,
        )
    end
    return nothing
end

@inline function _vt_dispatch_extensive!(
    ρs_field,
    ρ_field,
    ᶜlg,
    ᶠlg,
    ᶠF̃,
    f_work,
    c_work_2,
    Δt_ft,
    upw_code::Int8,
)
    DataLayouts.foreach_column(
        ρs_field,
        ρ_field,
        ᶜlg,
        ᶠlg,
        ᶠF̃,
        f_work,
        c_work_2;
        shmem = Val(:none),
    ) do ρs_c, ρ_c, lg_c, flg_c, F_c, fw_c, cw_c
        nz = Spaces.nlevels(axes(ρ_c))
        _transport_single_scalar_column!(
            Fields.field_values(ρs_c),
            Fields.field_values(cw_c[2]),
            Fields.field_values(fw_c[2]),
            Fields.field_values(fw_c[1]),
            Fields.field_values(F_c),
            Fields.field_values(ρ_c),
            Fields.field_values(cw_c[1]),
            Fields.field_values(lg_c),
            Fields.field_values(flg_c),
            Δt_ft,
            nz,
            upw_code,
        )
    end
    return nothing
end

"""
    step_vertical_scalar_transport!(Y, p, Δt[, integrator_cache])
    step_vertical_scalar_transport!(Y, p, t, Δt, integrator_cache)

Execute post-ARK DSS-consistent mass-flux reconstruction and monotone vertical
scalar transport over the realized step `Δt` [s], mutating `Y.c` in place and
writing the resolved vertical mass flux `ᶠF̃` into `p.scratch.ᶠF̃`.

Steps:

 1. Accumulate `D_raw = Σ_i b_imp[i] T_imp[i].c.ρ` across all non-zero `b_imp`
    stages (including explicit-diagonal `dtγ == 0` stages), applying
    `weighted_dss!` on spectral-element grids only when the tableau has an
    explicit-diagonal stage with `b_imp[i] != 0` (implicit-diagonal `T_imp[i]`
    buffers are already continuous across element boundaries).
 2. Recover pre-transport extensive scalars `(ρχ)⁰` by subtracting
    `Δt * Σ_i b_imp[i] T_imp[i].c.ρχ` for scalars advected centrally in `T_imp`
    (`ρq_tot` and condensate mass tracers), and recover
    `(ρχ_e)⁰ = (ρe_tot)^{n+1} - ρ^{n+1} Φ + Δt ᶜadvdivᵥ(Σ_i b_imp[i] ᶠF̃_i ê_i)`
    from the stage-paired face energies `ê_i` and mass-projected stage fluxes
    `ᶠF̃_i`.
 3. Mass-project `D` against `ρ^{n+1}`, integrate `ᶠF̃` (`ᶜadvdivᵥ(ᶠF̃) = -D` with
    zero boundary fluxes), factor the tridiagonal M-matrix
    `diag(ρ^{n+1}) + Δt ∂_z^{up}(F̃)` once per column into `ᶜtemp_scalar`,
    advance `{χ_e, q_tot, q_lcl, q_icl, q_rai, q_sno}` with a shared face
    Zalesak FCT factor (including `q_vap = q_tot - Σ q_c` in the bounds),
    reconstruct `(ρe_tot)^{n+1} = ρ^{n+1} χ_e^{n+1} + ρ^{n+1} Φ`, and advance
    all active number concentrations, rime species, `ρtke`, and passive tracers
    reusing `inv_u`.
 4. For `PrognosticEDMFX`, `sgsʲs` stays unchanged here and advances under
    `u₃ʲ` in `column_physics_step.jl`.
"""
function step_vertical_scalar_transport!(
    Y,
    p,
    Δt,
    integrator_cache = post_step_integrator_cache(p),
)
    !isnothing(integrator_cache) || error(
        "step_vertical_scalar_transport! requires a bound integrator_cache",
    )
    has_implicit_stage_tendencies(integrator_cache) || return p.scratch.ᶠF̃

    FT = Spaces.undertype(axes(Y.c))
    Δt_ft = FT(Δt)
    eh_upw = p.atmos.numerics.energy_humidity_upwinding
    tr_upw = p.atmos.numerics.tracer_upwinding
    eh_code = _vt_upw_code(eh_upw)
    tr_code = _vt_upw_code(tr_upw)

    cond_names = _vt_cond_names(eltype(Y.c))
    ᶜaccum = p.scratch.ᶜtemp_scalar_2

    # Recover `(ρχ)⁰` and `(ρχ_e)⁰` in `Y.c`, then accumulate `ᶜD_raw` into
    # `p.scratch.ᶜtemp_scalar_2` for the column sweep.
    if eh_code != _UPW_NONE
        if MatrixFields.has_field(Y.c, @name(ρq_tot))
            _subtract_stage_tendency!(
                Y.c.ρq_tot,
                p,
                Δt_ft,
                integrator_cache,
                @name(ρq_tot),
                ᶜaccum,
            )
        end
        unrolled_foreach(cond_names) do cname
            ᶜρqc = MatrixFields.get_field(Y.c, cname)
            _subtract_stage_tendency!(
                ᶜρqc,
                p,
                Δt_ft,
                integrator_cache,
                cname,
                ᶜaccum,
            )
        end
        _reconstruct_pre_transport_energy!(
            Y,
            p,
            Δt_ft,
            integrator_cache,
            ᶜaccum,
            p.scratch.ᶠtemp_CT3,
        )
    end
    p.scratch.stage_entry_χ_e_mask[] = UInt16(0)

    ᶜD_raw = ᶜaccum
    _accumulate_stage_tendency!(ᶜD_raw, p, integrator_cache, @name(ρ))

    unp = _unpack_vt_yc(Y.c)
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶠF̃ = p.scratch.ᶠF̃
    f_work = (p.scratch.ᶠtemp_scalar, p.scratch.ᶠtemp_scalar_2)
    c_work = (
        p.scratch.ᶜtemp_scalar,
        p.scratch.ᶜtemp_scalar_3,
        p.scratch.ᶜtemp_scalar_4,
        p.scratch.ᶜtemp_scalar_5,
    )

    DataLayouts.foreach_column(
        unp.ρ,
        unp.ρe_tot,
        unp.ρq_tot,
        unp.cond,
        ᶜlg,
        ᶠlg,
        p.core.ᶜΦ,
        ᶜD_raw,
        ᶠF̃,
        f_work,
        c_work;
        shmem = Val(:none),
    ) do ρ_c, ρe_c, ρqt_c, cond_c, lg_c, flg_c, Φ_c, D_c, F_c, fw_c, cw_c
        _vt_gm_thermo_column!(
            ρ_c,
            ρe_c,
            ρqt_c,
            cond_c,
            lg_c,
            flg_c,
            Φ_c,
            D_c,
            F_c,
            fw_c,
            cw_c,
            Δt_ft,
            eh_code,
        )
    end

    # The wave core does not transport the number and rime species or `ρtke`,
    # so `:none` has no central implicit transport to keep for them; they take
    # the limited scheme then.
    indiv_code = eh_code == _UPW_NONE ? _UPW_VANLEER : eh_code
    c_work_2 = (p.scratch.ᶜtemp_scalar, p.scratch.ᶜtemp_scalar_3)
    unrolled_foreach(unp.indiv) do ρm_field
        _vt_dispatch_extensive!(
            ρm_field,
            unp.ρ,
            ᶜlg,
            ᶠlg,
            ᶠF̃,
            f_work,
            c_work_2,
            Δt_ft,
            indiv_code,
        )
    end

    if !isnothing(unp.ρtke)
        tke_code =
            eh_code == _UPW_FIRST_ORDER ? _UPW_FIRST_ORDER : _UPW_VANLEER
        _vt_dispatch_extensive!(
            unp.ρtke,
            unp.ρ,
            ᶜlg,
            ᶠlg,
            ᶠF̃,
            f_work,
            c_work_2,
            Δt_ft,
            tke_code,
        )
    end

    unrolled_foreach(unp.passive) do ρp_field
        _vt_dispatch_extensive!(
            ρp_field,
            unp.ρ,
            ᶜlg,
            ᶠlg,
            ᶠF̃,
            f_work,
            c_work_2,
            Δt_ft,
            tr_code,
        )
    end

    return ᶠF̃
end

@inline function step_vertical_scalar_transport!(
    Y,
    p,
    _t,
    Δt,
    integrator_cache,
)
    return step_vertical_scalar_transport!(Y, p, Δt, integrator_cache)
end
