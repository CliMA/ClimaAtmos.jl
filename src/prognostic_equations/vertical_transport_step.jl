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
    _project_and_integrate_mass_flux_column_j!(ᶠF_col, ᶜD_raw_col, ᶜρ_new_col, ᶜJ_col, ᶠJ_col)

Apply the mass-weighted column-sum projection to `ᶜD_raw_col` against
`ᶜρ_new_col` and integrate `∂_z F̃ = -D` upward from `F̃_{1/2} = 0` with
`F̃_{nz+1/2} = 0`. Accumulates column sums and partial integrals in `Float64`
registers so `Float32` columns retain full relative precision into low-density
upper levels.
"""
@inline function _project_and_integrate_mass_flux_column_j!(
    ᶠF_col,
    ᶜD_raw_col,
    ᶜρ_new_col,
    ᶜJ_col,
    ᶠJ_col,
)
    FT = Spaces.undertype(axes(ᶜρ_new_col))
    F_fv = Fields.field_values(ᶠF_col)
    D_raw_fv = Fields.field_values(ᶜD_raw_col)
    ρ_new_fv = Fields.field_values(ᶜρ_new_col)
    ᶜJ_fv = Fields.field_values(ᶜJ_col)
    ᶠJ_fv = Fields.field_values(ᶠJ_col)
    nz = Spaces.nlevels(axes(ᶜρ_new_col))

    sum_JD = 0.0
    sum_Jρ = 0.0
    @inbounds for k in 1:nz
        J_k = Float64(ᶜJ_fv[k])
        sum_JD = muladd(J_k, Float64(D_raw_fv[k]), sum_JD)
        sum_Jρ = muladd(J_k, Float64(ρ_new_fv[k]), sum_Jρ)
    end
    α_proj = sum_JD / sum_Jρ

    @inbounds F_fv[1] = CT3(zero(FT))
    JF_curr = 0.0
    @inbounds for k in 1:(nz - 1)
        J_k = Float64(ᶜJ_fv[k])
        D_k = muladd(-α_proj, Float64(ρ_new_fv[k]), Float64(D_raw_fv[k]))
        JF_curr -= J_k * D_k
        J_f = Float64(ᶠJ_fv[k + 1])
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
    ᶜJ_col,
    ᶠJ_col,
) where {FT}
    F_χe_fv = Fields.field_values(ᶠF_χe_col)
    D_fv = Fields.field_values(ᶜD_stage_col)
    ρ_new_fv = Fields.field_values(ᶜρ_new_col)
    ê_fv = Fields.field_values(ᶠê_col)
    ᶜJ_fv = Fields.field_values(ᶜJ_col)
    ᶠJ_fv = Fields.field_values(ᶠJ_col)
    nz = Spaces.nlevels(axes(ᶜρ_new_col))

    sum_JD = 0.0
    sum_Jρ = 0.0
    @inbounds for k in 1:nz
        J_k = Float64(ᶜJ_fv[k])
        sum_JD = muladd(J_k, Float64(D_fv[k]), sum_JD)
        sum_Jρ = muladd(J_k, Float64(ρ_new_fv[k]), sum_Jρ)
    end
    α_i = sum_JD / sum_Jρ

    JF_curr = 0.0
    w_i = op.w_i
    @inbounds for k in 1:(nz - 1)
        J_k = Float64(ᶜJ_fv[k])
        D_k = muladd(-α_i, Float64(ρ_new_fv[k]), Float64(D_fv[k]))
        JF_curr -= J_k * D_k
        f = k + 1
        F_i_u3 = FT(JF_curr / Float64(ᶠJ_fv[f]))
        F_χe_fv[f] = CT3(muladd(w_i * F_i_u3, ê_fv[f], F_χe_fv[f].u³))
    end
end

struct _StageEnergyFluxAccumulator{FX, YV, PV, SB, FB, AC, AI, CJ, FJ, FT}
    ᶠF_χe_accum::FX
    Y::YV
    p::PV
    stage_bufs::SB
    ᶠê_fallback::FB
    ᶜD_dss::AC
    a_imp::AI
    ᶜJ::CJ
    ᶠJ::FJ
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
            acc.ᶜJ,
            acc.ᶠJ;
            shmem = Val(:none),
        )
    else
        DataLayouts.foreach_column(
            col_op,
            acc.ᶠF_χe_accum,
            T_imp_i.c.ρ,
            acc.Y.c.ρ,
            ᶠê_i,
            acc.ᶜJ,
            acc.ᶠJ;
            shmem = Val(:none),
        )
    end
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
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    acc = _StageEnergyFluxAccumulator(
        ᶠF_χe_accum,
        Y,
        p,
        stage_bufs,
        ᶠê_fallback,
        ᶜD_dss,
        integrator_cache.tableau.a_imp,
        ᶜJ,
        ᶠJ,
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
    _upwind3_antidiff_delta_j(JF_f, f, nz, χ_fv, ᶜJ_fv)

Metric-aware third-order finite-volume upwind antidiffusive face increment
`χ_HO - χ_up` at interior face `f ∈ 2:nz` (between cells `f - 1` and `f`).
Reconstructs the quadratic primitive over cell metric volumes `V_m = ᶜJ_fv[m]`,
reducing to `(-2 χ_u + 10 χ_c + 4 χ_d) / 12 - χ_c` on uniform grids and to
two-cell central interpolation at the two faces adjacent to the boundaries.
"""
@inline function _upwind3_antidiff_delta_j(
    JF_f,
    f::Int,
    nz::Int,
    χ_fv,
    ᶜJ_fv,
)
    FT = typeof(JF_f)
    if JF_f >= zero(FT)
        c = f - 1
        d = f
        χ_c = @inbounds χ_fv[c]
        χ_d = @inbounds χ_fv[d]
        V_c = @inbounds ᶜJ_fv[c]
        V_d = @inbounds ᶜJ_fv[d]
        if f == 2
            return V_c * (χ_d - χ_c) / (V_c + V_d)
        else
            u = f - 2
            χ_u = @inbounds χ_fv[u]
            V_u = @inbounds ᶜJ_fv[u]
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
        V_c = @inbounds ᶜJ_fv[c]
        V_d = @inbounds ᶜJ_fv[d]
        if f == nz
            return V_c * (χ_d - χ_c) / (V_c + V_d)
        else
            u = f + 1
            χ_u = @inbounds χ_fv[u]
            V_u = @inbounds ᶜJ_fv[u]
            V_uc = V_u + V_c
            return (V_c / (V_uc + V_d)) * (
                (V_uc / (V_c + V_d)) * (χ_d - χ_c) +
                (V_d / V_uc) * (χ_c - χ_u)
            )
        end
    end
end

# Integer upwinding mode tags (`isbits`).
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

# Project `ᶜD_raw_col` onto `ᶠF̃_col`, cache face invariants
# `(JF, c_f, Δt * JF * w_C)` and pre-transport density `ρ⁰`, and factor the
# implicit donor-cell M-matrix `M = diag(ρ^{n+1}) + Δt ∂_z^{up}(F̃)` into
# `inv_u_fv`. Each `J_k`-weighted column sum equals `J_k ρ^{n+1}_k > 0`, making
# `M` a strict nonsingular M-matrix with positive Thomas upper pivots `u_curr`.
@inline function _factor_column_invariants!(
    ᶠF̃_col,
    inv_u_fv,
    ρ0_fv,
    ᶜinvJ_fv,
    JF_fv,
    c_fv,
    dt_JF_wC_fv,
    ᶜD_raw_col,
    ᶜρ_new_col,
    ᶜJ_col,
    ᶠJ_col,
    Δt,
)
    FT = typeof(Δt)
    _project_and_integrate_mass_flux_column_j!(
        ᶠF̃_col,
        ᶜD_raw_col,
        ᶜρ_new_col,
        ᶜJ_col,
        ᶠJ_col,
    )
    F_fv = Fields.field_values(ᶠF̃_col)
    ρ_new_fv = Fields.field_values(ᶜρ_new_col)
    ᶜJ_fv = Fields.field_values(ᶜJ_col)
    ᶠJ_fv = Fields.field_values(ᶠJ_col)
    nz = Spaces.nlevels(axes(ᶜρ_new_col))
    z = zero(FT)
    o = one(FT)

    @inbounds begin
        JF_fv[1] = ᶠJ_fv[1] * F_fv[1].u³
        c_fv[1] = z
        dt_JF_wC_fv[1] = z
        M_lo = ᶜJ_fv[1] * ρ_new_fv[1]
        for f in 2:nz
            JF_f = ᶠJ_fv[f] * F_fv[f].u³
            JF_fv[f] = JF_f
            M_hi = ᶜJ_fv[f] * ρ_new_fv[f]
            M_min = max(min(M_lo, M_hi), floatmin(FT))
            c_f = (Δt * abs(JF_f)) / M_min
            c_fv[f] = c_f
            w_C_f = ifelse(c_f > o, inv(c_f), o)
            dt_JF_wC_fv[f] = Δt * JF_f * w_C_f
            M_lo = M_hi
        end
        JF_fv[nz + 1] = ᶠJ_fv[nz + 1] * F_fv[nz + 1].u³
        c_fv[nz + 1] = z
        dt_JF_wC_fv[nz + 1] = z

        u_prev = o
        du_prev = z
        JF_bot = JF_fv[1]
        for k in 1:nz
            ρ_new_k = ρ_new_fv[k]
            invJ_k = inv(ᶜJ_fv[k])
            ᶜinvJ_fv[k] = invJ_k
            s_k = Δt * invJ_k
            JF_top = JF_fv[k + 1]
            ρ0_fv[k] = ρ_new_k + s_k * (JF_top - JF_bot)

            dl_k = -s_k * max(JF_bot, z)
            du_k = -s_k * max(-JF_top, z)
            d_k = ρ_new_k + s_k * (max(JF_top, z) + max(-JF_bot, z))
            u_curr = k == 1 ? d_k : muladd(-(dl_k / u_prev), du_prev, d_k)
            u_safe = max(u_curr, floatmin(FT))
            inv_u_fv[k] = inv(u_safe)
            u_prev = u_safe
            du_prev = du_k
            JF_bot = JF_top
        end
    end
    return nothing
end

@inline function _thomas_solve_m_cached!(
    χL_fv,
    ρχ0_fv,
    inv_u_fv,
    JF_fv,
    ᶜinvJ_fv,
    Δt,
    nz::Int,
)
    FT = typeof(Δt)
    z = zero(FT)
    @inbounds begin
        χL_prev = ρχ0_fv[1]
        χL_fv[1] = χL_prev
        for k in 2:nz
            s_k = Δt * ᶜinvJ_fv[k]
            dl_k = -s_k * max(JF_fv[k], z)
            l_k = dl_k * inv_u_fv[k - 1]
            χL_prev = muladd(-l_k, χL_prev, ρχ0_fv[k])
            χL_fv[k] = χL_prev
        end
        χL_next = χL_fv[nz] * inv_u_fv[nz]
        χL_fv[nz] = χL_next
        for k in (nz - 1):-1:1
            s_k = Δt * ᶜinvJ_fv[k]
            du_k = -s_k * max(-JF_fv[k + 1], z)
            χL_next = muladd(-du_k, χL_next, χL_fv[k]) * inv_u_fv[k]
            χL_fv[k] = χL_next
        end
    end
    return nothing
end

@inline function _compute_raw_antidiff_cached!(
    Φ_fv,
    χL_fv,
    JF_fv,
    dt_JF_wC_fv,
    ᶜJ_fv,
    nz::Int,
)
    FT = eltype(Φ_fv)
    z = zero(FT)
    @inbounds Φ_fv[1] = z
    @inbounds for f in 2:nz
        JF_f = JF_fv[f]
        Δχ_f = _upwind3_antidiff_delta_j(JF_f, f, nz, χL_fv, ᶜJ_fv)
        Φ_fv[f] = dt_JF_wC_fv[f] * Δχ_f
    end
    @inbounds Φ_fv[nz + 1] = z
    return nothing
end

"""
    _narrow_face_fct_limiter_cached!(α_fv, Φ_fv, χL_fv, ρχ0_fv, ρ0_fv, c_fv, ρ_new_fv, ᶜJ_fv, nz, is_specific_χ0)

Intersect `α_fv[2:nz]` in place with the Zalesak FCT face limiters of a scalar
whose low-order solution is `χL_fv` and whose pre-transport profile is either
extensive (`ρχ0_fv / ρ0_fv` when `is_specific_χ0 == false`) or already specific
(`ρχ0_fv` when `is_specific_χ0 == true`, as for `q_vap⁰`).
"""
@inline function _narrow_face_fct_limiter_cached!(
    α_fv,
    Φ_fv,
    χL_fv,
    ρχ0_fv,
    ρ0_fv,
    c_fv,
    ρ_new_fv,
    ᶜJ_fv,
    nz::Int,
    ::Val{is_specific_χ0},
) where {is_specific_χ0}
    FT = eltype(α_fv)
    z = zero(FT)
    o = one(FT)
    @inbounds begin
        c_bot = c_fv[1]
        Φ_bot = Φ_fv[1]
        χL_km = χL_fv[1]
        χL_k = χL_km
        prev_sub1 = false
        χ0_k = z
        χ0_kp = z
        for k in 1:nz
            χL_kp = k < nz ? χL_fv[k + 1] : χL_k
            χ_max = max(χL_km, χL_k, χL_kp)
            χ_min = min(χL_km, χL_k, χL_kp)

            c_top = c_fv[k + 1]
            if max(c_bot, c_top) <= o
                if is_specific_χ0
                    if prev_sub1
                        χ0_km = χ0_k
                        χ0_k = χ0_kp
                        χ0_kp = k < nz ? ρχ0_fv[k + 1] : χ0_k
                    else
                        km = max(1, k - 1)
                        χ0_km = ρχ0_fv[km]
                        χ0_k = k == km ? χ0_km : ρχ0_fv[k]
                        χ0_kp = k < nz ? ρχ0_fv[k + 1] : χ0_k
                        prev_sub1 = true
                    end
                else
                    if prev_sub1
                        χ0_km = χ0_k
                        χ0_k = χ0_kp
                        χ0_kp = k < nz ? (ρχ0_fv[k + 1] / ρ0_fv[k + 1]) : χ0_k
                    else
                        km = max(1, k - 1)
                        χ0_km = ρχ0_fv[km] / ρ0_fv[km]
                        χ0_k = k == km ? χ0_km : (ρχ0_fv[k] / ρ0_fv[k])
                        χ0_kp = k < nz ? (ρχ0_fv[k + 1] / ρ0_fv[k + 1]) : χ0_k
                        prev_sub1 = true
                    end
                end
                χ_max = max(χ_max, χ0_km, χ0_k, χ0_kp)
                χ_min = min(χ_min, χ0_km, χ0_k, χ0_kp)
            else
                prev_sub1 = false
            end

            M_k = ᶜJ_fv[k] * ρ_new_fv[k]
            Q_plus = M_k * max(z, χ_max - χL_k)
            Q_minus = M_k * max(z, χL_k - χ_min)

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

            χL_km = χL_k
            χL_k = χL_kp
            c_bot = c_top
            Φ_bot = Φ_top
        end
    end
    return nothing
end

@inline function _apply_limited_antidiff_update_cached!(
    ρχ_fv,
    χL_fv,
    Φ_fv,
    α_fv,
    ρ_new_fv,
    ᶜinvJ_fv,
    nz::Int,
)
    @inbounds begin
        α_bot = α_fv[1]
        Φ_bot = Φ_fv[1]
        for k in 1:nz
            α_top = α_fv[k + 1]
            Φ_top = Φ_fv[k + 1]
            flux_diff = muladd(α_bot, Φ_bot, -α_top * Φ_top)
            ρχ_fv[k] = muladd(ρ_new_fv[k], χL_fv[k], flux_diff * ᶜinvJ_fv[k])
            α_bot = α_top
            Φ_bot = Φ_top
        end
    end
    return nothing
end

@inline function _apply_unlimited_antidiff_update_cached!(
    ρχ_fv,
    χL_fv,
    Φ_fv,
    ρ_new_fv,
    ᶜinvJ_fv,
    nz::Int,
)
    @inbounds begin
        Φ_bot = Φ_fv[1]
        for k in 1:nz
            Φ_top = Φ_fv[k + 1]
            flux_diff = Φ_bot - Φ_top
            ρχ_fv[k] = muladd(ρ_new_fv[k], χL_fv[k], flux_diff * ᶜinvJ_fv[k])
            Φ_bot = Φ_top
        end
    end
    return nothing
end

@inline function _transport_single_scalar_cached!(
    ρχ_fv,
    χL_fv,
    Φ_fv,
    α_fv,
    inv_u_fv,
    JF_fv,
    dt_JF_wC_fv,
    ρ0_fv,
    c_fv,
    ρ_new_fv,
    ᶜJ_fv,
    ᶜinvJ_fv,
    Δt,
    nz::Int,
    ::Val{C},
) where {C}
    FT = typeof(Δt)
    if C == _UPW_NONE
        # Central reconstruction `-Δt ᶜadvdivᵥ(ᶠF̃ * ᶠinterp(χ⁰))` on `(ρχ)⁰`,
        # preserving `χ ≡ 1` to round-off when `ρ⁰ = ρ^{n+1} + Δt ᶜadvdivᵥ(ᶠF̃)`.
        z = zero(FT)
        @inbounds if nz >= 1
            Φ_bot = z
            χ0_lo = ρχ_fv[1] / ρ0_fv[1]
            for f in 2:nz
                χ0_hi = ρχ_fv[f] / ρ0_fv[f]
                Φ_top = Δt * JF_fv[f] * FT(0.5) * (χ0_lo + χ0_hi)
                ρχ_fv[f - 1] = ρχ_fv[f - 1] - (Φ_top - Φ_bot) * ᶜinvJ_fv[f - 1]
                Φ_bot = Φ_top
                χ0_lo = χ0_hi
            end
            ρχ_fv[nz] = ρχ_fv[nz] - (z - Φ_bot) * ᶜinvJ_fv[nz]
        end
        return nothing
    end

    _thomas_solve_m_cached!(χL_fv, ρχ_fv, inv_u_fv, JF_fv, ᶜinvJ_fv, Δt, nz)

    if C == _UPW_FIRST_ORDER || nz < 2
        @inbounds for k in 1:nz
            ρχ_fv[k] = ρ_new_fv[k] * χL_fv[k]
        end
        return nothing
    end

    _compute_raw_antidiff_cached!(Φ_fv, χL_fv, JF_fv, dt_JF_wC_fv, ᶜJ_fv, nz)

    if C == _UPW_THIRD_ORDER
        _apply_unlimited_antidiff_update_cached!(
            ρχ_fv,
            χL_fv,
            Φ_fv,
            ρ_new_fv,
            ᶜinvJ_fv,
            nz,
        )
        return nothing
    end

    o = one(FT)
    @inbounds for f in 1:(nz + 1)
        α_fv[f] = o
    end
    _narrow_face_fct_limiter_cached!(
        α_fv,
        Φ_fv,
        χL_fv,
        ρχ_fv,
        ρ0_fv,
        c_fv,
        ρ_new_fv,
        ᶜJ_fv,
        nz,
        Val(false),
    )
    _apply_limited_antidiff_update_cached!(
        ρχ_fv,
        χL_fv,
        Φ_fv,
        α_fv,
        ρ_new_fv,
        ᶜinvJ_fv,
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
tuples of leaf `Field`s (`()` when a moisture, condensate, or TKE group is
absent, as under `DryModel`) so `DataLayouts.foreach_column` kernel arguments
stay flat and within GPU tuple-specialization limits on warped grids.
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
            ρtke = $(has_tke ? :(Yc.ρtke) : :(())),
        )
    end
end

@inline _uvwxuvw_cond_slots(_m, ::Val{0}) = ()
@inline _uvwxuvw_cond_slots(m, ::Val{1}) = (m.components.data.:3,)
@inline _uvwxuvw_cond_slots(m, ::Val{2}) =
    (m.components.data.:3, m.components.data.:4)
@inline _uvwxuvw_cond_slots(m, ::Val{3}) =
    (m.components.data.:3, m.components.data.:4, m.components.data.:5)
@inline _uvwxuvw_cond_slots(m, ::Val{4}) = (
    m.components.data.:3,
    m.components.data.:4,
    m.components.data.:5,
    m.components.data.:6,
)

@inline _vt_first_order_cond!(
    ::Tuple{},
    _χL_fv,
    _inv_u_fv,
    _JF_fv,
    _ᶜinvJ_fv,
    _ρ_new_fv,
    _Δt,
    _nz::Int,
) = nothing

@inline function _vt_first_order_cond!(
    cond_cols::Tuple,
    χL_fv,
    inv_u_fv,
    JF_fv,
    ᶜinvJ_fv,
    ρ_new_fv,
    Δt,
    nz::Int,
)
    ρqc_fv = Fields.field_values(first(cond_cols))
    _thomas_solve_m_cached!(χL_fv, ρqc_fv, inv_u_fv, JF_fv, ᶜinvJ_fv, Δt, nz)
    @inbounds for k in 1:nz
        ρqc_fv[k] = ρ_new_fv[k] * χL_fv[k]
    end
    return _vt_first_order_cond!(
        Base.tail(cond_cols),
        χL_fv,
        inv_u_fv,
        JF_fv,
        ᶜinvJ_fv,
        ρ_new_fv,
        Δt,
        nz,
    )
end

@inline _vt_third_order_cond!(
    ::Tuple{},
    _χL_fv,
    _Φ_fv,
    _inv_u_fv,
    _JF_fv,
    _dt_JF_wC_fv,
    _ρ_new_fv,
    _ᶜJ_fv,
    _ᶜinvJ_fv,
    _Δt,
    _nz::Int,
) = nothing

@inline function _vt_third_order_cond!(
    cond_cols::Tuple,
    χL_fv,
    Φ_fv,
    inv_u_fv,
    JF_fv,
    dt_JF_wC_fv,
    ρ_new_fv,
    ᶜJ_fv,
    ᶜinvJ_fv,
    Δt,
    nz::Int,
)
    ρqc_fv = Fields.field_values(first(cond_cols))
    _thomas_solve_m_cached!(χL_fv, ρqc_fv, inv_u_fv, JF_fv, ᶜinvJ_fv, Δt, nz)
    _compute_raw_antidiff_cached!(Φ_fv, χL_fv, JF_fv, dt_JF_wC_fv, ᶜJ_fv, nz)
    _apply_unlimited_antidiff_update_cached!(
        ρqc_fv,
        χL_fv,
        Φ_fv,
        ρ_new_fv,
        ᶜinvJ_fv,
        nz,
    )
    return _vt_third_order_cond!(
        Base.tail(cond_cols),
        χL_fv,
        Φ_fv,
        inv_u_fv,
        JF_fv,
        dt_JF_wC_fv,
        ρ_new_fv,
        ᶜJ_fv,
        ᶜinvJ_fv,
        Δt,
        nz,
    )
end

@inline _vt_pass1_cond!(
    ::Tuple{},
    ::Tuple{},
    ::Tuple{},
    _α_fv,
    _inv_u_fv,
    _JF_fv,
    _dt_JF_wC_fv,
    _ρ0_fv,
    _c_fv,
    _ρ_new_fv,
    _ᶜJ_fv,
    _ᶜinvJ_fv,
    _q_vap_L_fv,
    _q_vap_0_fv,
    _Δt,
    _nz::Int,
    _has_q_tot::Bool,
) = nothing

@inline function _vt_pass1_cond!(
    cond_cols::Tuple,
    χL_cond_cols::Tuple,
    Φ_cond_cols::Tuple,
    α_fv,
    inv_u_fv,
    JF_fv,
    dt_JF_wC_fv,
    ρ0_fv,
    c_fv,
    ρ_new_fv,
    ᶜJ_fv,
    ᶜinvJ_fv,
    q_vap_L_fv,
    q_vap_0_fv,
    Δt,
    nz::Int,
    has_q_tot::Bool,
)
    ρqc_fv = Fields.field_values(first(cond_cols))
    χL_c_fv = Fields.field_values(first(χL_cond_cols))
    Φ_c_fv = Fields.field_values(first(Φ_cond_cols))

    _thomas_solve_m_cached!(χL_c_fv, ρqc_fv, inv_u_fv, JF_fv, ᶜinvJ_fv, Δt, nz)
    _compute_raw_antidiff_cached!(
        Φ_c_fv,
        χL_c_fv,
        JF_fv,
        dt_JF_wC_fv,
        ᶜJ_fv,
        nz,
    )
    _narrow_face_fct_limiter_cached!(
        α_fv,
        Φ_c_fv,
        χL_c_fv,
        ρqc_fv,
        ρ0_fv,
        c_fv,
        ρ_new_fv,
        ᶜJ_fv,
        nz,
        Val(false),
    )
    if has_q_tot
        @inbounds for k in 1:nz
            q_vap_L_fv[k] -= χL_c_fv[k]
            q_vap_0_fv[k] -= ρqc_fv[k]
        end
    end
    return _vt_pass1_cond!(
        Base.tail(cond_cols),
        Base.tail(χL_cond_cols),
        Base.tail(Φ_cond_cols),
        α_fv,
        inv_u_fv,
        JF_fv,
        dt_JF_wC_fv,
        ρ0_fv,
        c_fv,
        ρ_new_fv,
        ᶜJ_fv,
        ᶜinvJ_fv,
        q_vap_L_fv,
        q_vap_0_fv,
        Δt,
        nz,
        has_q_tot,
    )
end

@inline _vt_pass2_cond!(
    ::Tuple{},
    ::Tuple{},
    ::Tuple{},
    _α_fv,
    _ρ_new_fv,
    _ᶜinvJ_fv,
    _nz::Int,
) = nothing

@inline function _vt_pass2_cond!(
    cond_cols::Tuple,
    χL_cond_cols::Tuple,
    Φ_cond_cols::Tuple,
    α_fv,
    ρ_new_fv,
    ᶜinvJ_fv,
    nz::Int,
)
    ρqc_fv = Fields.field_values(first(cond_cols))
    χL_c_fv = Fields.field_values(first(χL_cond_cols))
    Φ_c_fv = Fields.field_values(first(Φ_cond_cols))
    _apply_limited_antidiff_update_cached!(
        ρqc_fv,
        χL_c_fv,
        Φ_c_fv,
        α_fv,
        ρ_new_fv,
        ᶜinvJ_fv,
        nz,
    )
    return _vt_pass2_cond!(
        Base.tail(cond_cols),
        Base.tail(χL_cond_cols),
        Base.tail(Φ_cond_cols),
        α_fv,
        ρ_new_fv,
        ᶜinvJ_fv,
        nz,
    )
end

struct _VTGmThermoSweep{FT, C} <: ColumnSweep
    Δt::FT
end

"""
    (_VTGmThermoSweep{FT, C})(...)

Project `ᶜD_raw` per column, integrate `ᶠF̃`, cache face invariants and `inv_u`,
advance grid-mean `(ρχ_e, ρq_tot, cond)` with static upwind mode `C` (caching
pass-1 `(χL, Φ)` in `UVWxUVW` scratch slots under `_UPW_VANLEER` so pass 2
applies the limited fluxes without re-solving), and advance `ρtke` in the same
column pass.
"""
@inline function (op::_VTGmThermoSweep{FT, C})(
    ρ_col,
    ρe_col,
    ρqt_col,
    cond_cols,
    ρtke_col,
    geom_cols,
    ᶜΦ_col,
    ᶜD_raw_col,
    inv_cols,
    work_cols,
) where {FT, C}
    Δt = op.Δt
    ᶜJ_col, ᶠJ_col = geom_cols
    ᶠF̃_col, ᶜinv_u_col, ᶜρ0_col, ᶜinvJ_col, ᶠJF_col, ᶠc_col, ᶠdt_JF_wC_col =
        inv_cols

    inv_u_fv = Fields.field_values(ᶜinv_u_col)
    ρ0_fv = Fields.field_values(ᶜρ0_col)
    ᶜinvJ_fv = Fields.field_values(ᶜinvJ_col)
    JF_fv = Fields.field_values(ᶠJF_col)
    c_fv = Fields.field_values(ᶠc_col)
    dt_JF_wC_fv = Fields.field_values(ᶠdt_JF_wC_col)

    _factor_column_invariants!(
        ᶠF̃_col,
        inv_u_fv,
        ρ0_fv,
        ᶜinvJ_fv,
        JF_fv,
        c_fv,
        dt_JF_wC_fv,
        ᶜD_raw_col,
        ρ_col,
        ᶜJ_col,
        ᶠJ_col,
        Δt,
    )

    ρ_new_fv = Fields.field_values(ρ_col)
    ᶜJ_fv = Fields.field_values(ᶜJ_col)
    nz = Spaces.nlevels(axes(ρ_col))
    has_q_tot = !(ρqt_col isa Tuple{})
    has_cond = !isempty(cond_cols)
    has_tke = !(ρtke_col isa Tuple{})

    ᶠα_col,
    ᶠΦ_tmp_col,
    ᶜq_vap_L_col,
    ᶜq_vap_0_col,
    ᶜχL_e_col,
    ᶜχL_qt_col,
    ᶜχL_cond_cols,
    ᶠΦ_e_col,
    ᶠΦ_qt_col,
    ᶠΦ_cond_cols =
        work_cols
    α_fv = Fields.field_values(ᶠα_col)
    Φ_tmp_fv = Fields.field_values(ᶠΦ_tmp_col)
    χL_e_fv = Fields.field_values(ᶜχL_e_col)

    if C != _UPW_NONE
        ρe_fv = Fields.field_values(ρe_col)
        Φ_grav_fv = Fields.field_values(ᶜΦ_col)

        if C == _UPW_FIRST_ORDER || nz < 2
            _thomas_solve_m_cached!(
                χL_e_fv,
                ρe_fv,
                inv_u_fv,
                JF_fv,
                ᶜinvJ_fv,
                Δt,
                nz,
            )
            @inbounds for k in 1:nz
                ρe_fv[k] = ρ_new_fv[k] * (χL_e_fv[k] + Φ_grav_fv[k])
            end
            if has_q_tot
                ρqt_fv = Fields.field_values(ρqt_col)
                _thomas_solve_m_cached!(
                    χL_e_fv,
                    ρqt_fv,
                    inv_u_fv,
                    JF_fv,
                    ᶜinvJ_fv,
                    Δt,
                    nz,
                )
                @inbounds for k in 1:nz
                    ρqt_fv[k] = ρ_new_fv[k] * χL_e_fv[k]
                end
            end
            _vt_first_order_cond!(
                cond_cols,
                χL_e_fv,
                inv_u_fv,
                JF_fv,
                ᶜinvJ_fv,
                ρ_new_fv,
                Δt,
                nz,
            )
        elseif C == _UPW_THIRD_ORDER
            _thomas_solve_m_cached!(
                χL_e_fv,
                ρe_fv,
                inv_u_fv,
                JF_fv,
                ᶜinvJ_fv,
                Δt,
                nz,
            )
            _compute_raw_antidiff_cached!(
                Φ_tmp_fv,
                χL_e_fv,
                JF_fv,
                dt_JF_wC_fv,
                ᶜJ_fv,
                nz,
            )
            _apply_unlimited_antidiff_update_cached!(
                ρe_fv,
                χL_e_fv,
                Φ_tmp_fv,
                ρ_new_fv,
                ᶜinvJ_fv,
                nz,
            )
            @inbounds for k in 1:nz
                ρe_fv[k] = muladd(ρ_new_fv[k], Φ_grav_fv[k], ρe_fv[k])
            end
            if has_q_tot
                ρqt_fv = Fields.field_values(ρqt_col)
                _thomas_solve_m_cached!(
                    χL_e_fv,
                    ρqt_fv,
                    inv_u_fv,
                    JF_fv,
                    ᶜinvJ_fv,
                    Δt,
                    nz,
                )
                _compute_raw_antidiff_cached!(
                    Φ_tmp_fv,
                    χL_e_fv,
                    JF_fv,
                    dt_JF_wC_fv,
                    ᶜJ_fv,
                    nz,
                )
                _apply_unlimited_antidiff_update_cached!(
                    ρqt_fv,
                    χL_e_fv,
                    Φ_tmp_fv,
                    ρ_new_fv,
                    ᶜinvJ_fv,
                    nz,
                )
            end
            _vt_third_order_cond!(
                cond_cols,
                χL_e_fv,
                Φ_tmp_fv,
                inv_u_fv,
                JF_fv,
                dt_JF_wC_fv,
                ρ_new_fv,
                ᶜJ_fv,
                ᶜinvJ_fv,
                Δt,
                nz,
            )
        else
            # `_UPW_VANLEER`: cache `(χL, Φ)` per species in pass 1 while
            # narrowing `α_fv` across `{χ_e, q_tot, q_c, q_vap}`, then apply in
            # pass 2.
            Φ_e_fv = Fields.field_values(ᶠΦ_e_col)
            χL_qt_fv = Fields.field_values(ᶜχL_qt_col)
            Φ_qt_fv = Fields.field_values(ᶠΦ_qt_col)
            q_vap_L_fv = Fields.field_values(ᶜq_vap_L_col)
            q_vap_0_fv = Fields.field_values(ᶜq_vap_0_col)

            o = one(FT)
            @inbounds for f in 1:(nz + 1)
                α_fv[f] = o
            end

            _thomas_solve_m_cached!(
                χL_e_fv,
                ρe_fv,
                inv_u_fv,
                JF_fv,
                ᶜinvJ_fv,
                Δt,
                nz,
            )
            _compute_raw_antidiff_cached!(
                Φ_e_fv,
                χL_e_fv,
                JF_fv,
                dt_JF_wC_fv,
                ᶜJ_fv,
                nz,
            )
            _narrow_face_fct_limiter_cached!(
                α_fv,
                Φ_e_fv,
                χL_e_fv,
                ρe_fv,
                ρ0_fv,
                c_fv,
                ρ_new_fv,
                ᶜJ_fv,
                nz,
                Val(false),
            )

            if has_q_tot
                ρqt_fv = Fields.field_values(ρqt_col)
                _thomas_solve_m_cached!(
                    χL_qt_fv,
                    ρqt_fv,
                    inv_u_fv,
                    JF_fv,
                    ᶜinvJ_fv,
                    Δt,
                    nz,
                )
                _compute_raw_antidiff_cached!(
                    Φ_qt_fv,
                    χL_qt_fv,
                    JF_fv,
                    dt_JF_wC_fv,
                    ᶜJ_fv,
                    nz,
                )
                _narrow_face_fct_limiter_cached!(
                    α_fv,
                    Φ_qt_fv,
                    χL_qt_fv,
                    ρqt_fv,
                    ρ0_fv,
                    c_fv,
                    ρ_new_fv,
                    ᶜJ_fv,
                    nz,
                    Val(false),
                )
                if has_cond
                    @inbounds for k in 1:nz
                        q_vap_L_fv[k] = χL_qt_fv[k]
                        q_vap_0_fv[k] = ρqt_fv[k]
                    end
                end
            end

            if has_cond
                _vt_pass1_cond!(
                    cond_cols,
                    ᶜχL_cond_cols,
                    ᶠΦ_cond_cols,
                    α_fv,
                    inv_u_fv,
                    JF_fv,
                    dt_JF_wC_fv,
                    ρ0_fv,
                    c_fv,
                    ρ_new_fv,
                    ᶜJ_fv,
                    ᶜinvJ_fv,
                    q_vap_L_fv,
                    q_vap_0_fv,
                    Δt,
                    nz,
                    has_q_tot,
                )
                if has_q_tot
                    @inbounds for k in 1:nz
                        q_vap_0_fv[k] /= ρ0_fv[k]
                    end
                    _compute_raw_antidiff_cached!(
                        Φ_tmp_fv,
                        q_vap_L_fv,
                        JF_fv,
                        dt_JF_wC_fv,
                        ᶜJ_fv,
                        nz,
                    )
                    _narrow_face_fct_limiter_cached!(
                        α_fv,
                        Φ_tmp_fv,
                        q_vap_L_fv,
                        q_vap_0_fv,
                        ρ0_fv,
                        c_fv,
                        ρ_new_fv,
                        ᶜJ_fv,
                        nz,
                        Val(true),
                    )
                end
            end

            # Pass 2: apply the shared face limiter `α_fv` to cached `(χL, Φ)`.
            _apply_limited_antidiff_update_cached!(
                ρe_fv,
                χL_e_fv,
                Φ_e_fv,
                α_fv,
                ρ_new_fv,
                ᶜinvJ_fv,
                nz,
            )
            @inbounds for k in 1:nz
                ρe_fv[k] = muladd(ρ_new_fv[k], Φ_grav_fv[k], ρe_fv[k])
            end

            if has_q_tot
                _apply_limited_antidiff_update_cached!(
                    Fields.field_values(ρqt_col),
                    χL_qt_fv,
                    Φ_qt_fv,
                    α_fv,
                    ρ_new_fv,
                    ᶜinvJ_fv,
                    nz,
                )
            end

            if has_cond
                _vt_pass2_cond!(
                    cond_cols,
                    ᶜχL_cond_cols,
                    ᶠΦ_cond_cols,
                    α_fv,
                    ρ_new_fv,
                    ᶜinvJ_fv,
                    nz,
                )
            end
        end
    end

    if has_tke
        tke_mode =
            C == _UPW_FIRST_ORDER ? Val(_UPW_FIRST_ORDER) : Val(_UPW_VANLEER)
        _transport_single_scalar_cached!(
            Fields.field_values(ρtke_col),
            χL_e_fv,
            Φ_tmp_fv,
            α_fv,
            inv_u_fv,
            JF_fv,
            dt_JF_wC_fv,
            ρ0_fv,
            c_fv,
            ρ_new_fv,
            ᶜJ_fv,
            ᶜinvJ_fv,
            Δt,
            nz,
            tke_mode,
        )
    end
end

@noinline function _launch_vt_gm_thermo!(
    ::Val{C},
    unp,
    geom,
    ᶜΦ,
    ᶜD_raw,
    invariants,
    thermo_work,
    Δt_ft::FT,
) where {FT, C}
    DataLayouts.foreach_column(
        _VTGmThermoSweep{FT, C}(Δt_ft),
        unp.ρ,
        unp.ρe_tot,
        unp.ρq_tot,
        unp.cond,
        unp.ρtke,
        geom,
        ᶜΦ,
        ᶜD_raw,
        invariants,
        thermo_work;
        shmem = Val(:none),
    )
    return nothing
end

struct _VTExtensiveSweep{FT, C} <: ColumnSweep
    Δt::FT
end

@inline function (op::_VTExtensiveSweep{FT, C})(
    ρs_col,
    ρ_col,
    geom_cols,
    inv_cols,
    ext_work_cols,
) where {FT, C}
    ᶜJ_col, _ᶠJ_col = geom_cols
    _ᶠF̃_col, ᶜinv_u_col, ᶜρ0_col, ᶜinvJ_col, ᶠJF_col, ᶠc_col, ᶠdt_JF_wC_col =
        inv_cols
    ᶠα_col, ᶠΦ_col, ᶜχL_col = ext_work_cols
    nz = Spaces.nlevels(axes(ρ_col))
    _transport_single_scalar_cached!(
        Fields.field_values(ρs_col),
        Fields.field_values(ᶜχL_col),
        Fields.field_values(ᶠΦ_col),
        Fields.field_values(ᶠα_col),
        Fields.field_values(ᶜinv_u_col),
        Fields.field_values(ᶠJF_col),
        Fields.field_values(ᶠdt_JF_wC_col),
        Fields.field_values(ᶜρ0_col),
        Fields.field_values(ᶠc_col),
        Fields.field_values(ρ_col),
        Fields.field_values(ᶜJ_col),
        Fields.field_values(ᶜinvJ_col),
        op.Δt,
        nz,
        Val(C),
    )
end

@noinline function _launch_vt_extensive!(
    ::Val{C},
    ρs_field,
    ρ_field,
    geom,
    invariants,
    ext_work,
    Δt_ft::FT,
) where {FT, C}
    DataLayouts.foreach_column(
        _VTExtensiveSweep{FT, C}(Δt_ft),
        ρs_field,
        ρ_field,
        geom,
        invariants,
        ext_work;
        shmem = Val(:none),
    )
    return nothing
end

@inline function _vt_dispatch_extensive!(
    ρs_field,
    ρ_field,
    geom,
    invariants,
    ext_work,
    Δt_ft,
    upw_code::Int8,
)
    if upw_code == _UPW_VANLEER
        _launch_vt_extensive!(
            Val(_UPW_VANLEER),
            ρs_field,
            ρ_field,
            geom,
            invariants,
            ext_work,
            Δt_ft,
        )
    elseif upw_code == _UPW_FIRST_ORDER
        _launch_vt_extensive!(
            Val(_UPW_FIRST_ORDER),
            ρs_field,
            ρ_field,
            geom,
            invariants,
            ext_work,
            Δt_ft,
        )
    elseif upw_code == _UPW_NONE
        _launch_vt_extensive!(
            Val(_UPW_NONE),
            ρs_field,
            ρ_field,
            geom,
            invariants,
            ext_work,
            Δt_ft,
        )
    else
        _launch_vt_extensive!(
            Val(_UPW_THIRD_ORDER),
            ρs_field,
            ρ_field,
            geom,
            invariants,
            ext_work,
            Δt_ft,
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
    `diag(ρ^{n+1}) + Δt ∂_z^{up}(F̃)` and face Courant invariants once per column,
    advance `{χ_e, q_tot, q_lcl, q_icl, q_rai, q_sno}` with a shared face
    Zalesak FCT factor (including `q_vap = q_tot - Σ q_c` in the bounds),
    reconstruct `(ρe_tot)^{n+1} = ρ^{n+1} χ_e^{n+1} + ρ^{n+1} Φ`, advance `ρtke`
    in the same column sweep, and advance all active number concentrations, rime
    species, and passive tracers reusing the cached column invariants.
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
    geom = (ᶜlg.J, ᶠlg.J)

    ᶠF̃ = p.scratch.ᶠF̃
    ᶠu_inv = p.scratch.ᶠtemp_UVW
    ᶜm_cache = p.scratch.ᶜtemp_UVWxUVW
    ᶠm_cache = p.scratch.ᶠtemp_UVWxUVW
    invariants = (
        ᶠF̃,
        p.scratch.ᶜtemp_scalar,
        p.scratch.ᶜtemp_scalar_6,
        ᶜm_cache.components.data.:7,
        ᶠu_inv.components.data.:1,
        ᶠu_inv.components.data.:2,
        ᶠu_inv.components.data.:3,
    )

    n_cond = Val(length(unp.cond))
    thermo_work = (
        p.scratch.ᶠtemp_scalar,
        p.scratch.ᶠtemp_scalar_2,
        p.scratch.ᶜtemp_scalar_4,
        p.scratch.ᶜtemp_scalar_5,
        ᶜm_cache.components.data.:1,
        ᶜm_cache.components.data.:2,
        _uvwxuvw_cond_slots(ᶜm_cache, n_cond),
        ᶠm_cache.components.data.:1,
        ᶠm_cache.components.data.:2,
        _uvwxuvw_cond_slots(ᶠm_cache, n_cond),
    )

    if eh_code == _UPW_VANLEER
        _launch_vt_gm_thermo!(
            Val(_UPW_VANLEER),
            unp,
            geom,
            p.core.ᶜΦ,
            ᶜD_raw,
            invariants,
            thermo_work,
            Δt_ft,
        )
    elseif eh_code == _UPW_FIRST_ORDER
        _launch_vt_gm_thermo!(
            Val(_UPW_FIRST_ORDER),
            unp,
            geom,
            p.core.ᶜΦ,
            ᶜD_raw,
            invariants,
            thermo_work,
            Δt_ft,
        )
    elseif eh_code == _UPW_NONE
        _launch_vt_gm_thermo!(
            Val(_UPW_NONE),
            unp,
            geom,
            p.core.ᶜΦ,
            ᶜD_raw,
            invariants,
            thermo_work,
            Δt_ft,
        )
    else
        _launch_vt_gm_thermo!(
            Val(_UPW_THIRD_ORDER),
            unp,
            geom,
            p.core.ᶜΦ,
            ᶜD_raw,
            invariants,
            thermo_work,
            Δt_ft,
        )
    end

    # The wave core does not transport the number and rime species, so `:none`
    # has no central implicit transport to keep for them; they take the limited
    # scheme then.
    indiv_code = eh_code == _UPW_NONE ? _UPW_VANLEER : eh_code
    ext_work = (
        p.scratch.ᶠtemp_scalar,
        p.scratch.ᶠtemp_scalar_2,
        p.scratch.ᶜtemp_scalar_3,
    )
    unrolled_foreach(unp.indiv) do ρm_field
        _vt_dispatch_extensive!(
            ρm_field,
            unp.ρ,
            geom,
            invariants,
            ext_work,
            Δt_ft,
            indiv_code,
        )
    end

    unrolled_foreach(unp.passive) do ρp_field
        _vt_dispatch_extensive!(
            ρp_field,
            unp.ρ,
            geom,
            invariants,
            ext_work,
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
