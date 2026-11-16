import LinearAlgebra: I, Adjoint

using ClimaCore.MatrixFields
import ClimaCore.MatrixFields: @name
import UnrolledUtilities:
    unrolled_any, unrolled_filter, unrolled_flatmap, unrolled_map, unrolled_reduce
import Thermodynamics as TD

"""
    DerivativeFlag

Whether a group of tendency derivatives is included in the manually computed
Jacobian.

Subtypes:

  - `UseDerivative`: include the derivatives.
  - `IgnoreDerivative`: omit the derivatives.

The flags are derived from the `AtmosModel` by `_derivative_flags`, and are
queried with `use_derivative`. Using a singleton type rather than a `Bool`
keeps the branches statically inferrable.
"""
abstract type DerivativeFlag end
struct UseDerivative <: DerivativeFlag end
struct IgnoreDerivative <: DerivativeFlag end

"""
    DerivativeFlag(value::Bool)

Construct a `UseDerivative()` or `IgnoreDerivative()` flag from a `Bool`.
"""
DerivativeFlag(value) = value ? UseDerivative() : IgnoreDerivative()

"""
    use_derivative(flag)

Return `true` for `UseDerivative()` and `false` for `IgnoreDerivative()`.
"""
use_derivative(::UseDerivative) = true
use_derivative(::IgnoreDerivative) = false

include("moist_pressure_derivatives.jl")

"""
    ManualSparseJacobian()

A [`JacobianAlgorithm`](@ref) that computes the analytical wave-core Jacobian
and inverts it with an exact `BlockArrowheadSolve` Schur complement onto `f.u₃`.
See the implicit Jacobian section of [Implicit Solver](@ref).
"""
struct ManualSparseJacobian <: SparseJacobian end

"""
    _derivative_flags(atmos, Y)

Return the `DerivativeFlag`s that specialize the manual Jacobian cache
at build time, as a `NamedTuple` with field `topography_flag` (set from
`has_topography(axes(Y.c))`). Called from `jacobian_cache`.
"""
function _derivative_flags(atmos, Y)
    return (;
        topography_flag = DerivativeFlag(has_topography(axes(Y.c))),
    )
end

# ============================================================================
# Jacobian matrix structure
# ============================================================================
#
# The wave-core Jacobian pairs `advection_jacobian_blocks` (continuity, energy
# + pressure work, `q_tot`, condensate-mass transport, and vertical momentum)
# with identity `-I` diagonal blocks for every remaining state variable.

"""
    jacobian_row_types(FT)

Return a `NamedTuple` of the `MatrixFields` band-matrix row types used by the
manual Jacobian blocks, for a state with floating-point type `FT`.

The name of each row type encodes its bandwidth and the type of its entries:
`TridiagonalRow` has scalar entries, `BidiagonalRow_C3` has `C3` vector
entries, `TridiagonalRow_ACT12` has adjoint `CT12` entries, and so on. The
entry type of a block is the outer product of the row variable's type with the
adjoint of the column variable's type.
"""
function jacobian_row_types(FT)
    return (;
        TridiagonalRow = TridiagonalMatrixRow{FT},
        BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}},
        TridiagonalRow_ACT12 = TridiagonalMatrixRow{typeof(CT12(FT(0), FT(0))')},
        BidiagonalRow_ACT3 = BidiagonalMatrixRow{typeof(CT3(FT(0))')},
        BidiagonalRow_C3xACT12 = BidiagonalMatrixRow{
            typeof(zero(C3{FT}) * zero(CT12{FT})'),
        },
        TridiagonalRow_C3xACT3 = TridiagonalMatrixRow{
            typeof(zero(C3{FT}) * zero(CT3{FT})'),
        },
    )
end

"""
    advection_jacobian_blocks(Y, atmos, topography_flag)

Allocate the wave-core Jacobian blocks for vertical continuity, `ρe_tot`, `ρq_tot`,
condensate masses, and `f.u₃`.

The `(f.u₃, condensate mass)` blocks hold the pressure-gradient derivatives with
respect to sedimenting condensate masses. The `(scalar, c.uₕ)` blocks are
allocated only when `topography_flag` is `UseDerivative()`, since horizontal
velocity enters vertical mass flux only through terrain-following metric terms.

# Returns

`Tuple` of `(row_name, col_name) => block` pairs, where each block is a
`Field` of the band-matrix row type from `jacobian_row_types`.
"""
function advection_jacobian_blocks(Y, atmos, topography_flag)
    FT = Spaces.undertype(axes(Y.c))
    (;
        TridiagonalRow_ACT12,
        BidiagonalRow_ACT3,
        BidiagonalRow_C3,
        BidiagonalRow_C3xACT12,
        TridiagonalRow_C3xACT3,
    ) = jacobian_row_types(FT)
    active_scalar_names =
        unrolled_map(center_state_name, advected_gs_scalar_names(Y))
    mass_names = unrolled_map(center_state_name, sedimenting_mass_names(Y))
    transported_names = (active_scalar_names..., mass_names...)
    return (
        (
            use_derivative(topography_flag) ?
            map(
                name ->
                    (name, @name(c.uₕ)) =>
                        similar(Y.c, TridiagonalRow_ACT12),
                transported_names,
            ) : ()
        )...,
        map(
            name -> (name, @name(f.u₃)) => similar(Y.c, BidiagonalRow_ACT3),
            transported_names,
        )...,
        map(
            name -> (@name(f.u₃), name) => similar(Y.f, BidiagonalRow_C3),
            active_scalar_names,
        )...,
        map(
            name -> (@name(f.u₃), name) => similar(Y.f, BidiagonalRow_C3),
            mass_names,
        )...,
        (@name(f.u₃), @name(c.uₕ)) => similar(Y.f, BidiagonalRow_C3xACT12),
        (@name(f.u₃), @name(f.u₃)) => similar(Y.f, TridiagonalRow_C3xACT3),
    )
end

"""
    merge_jacobian_blocks(block_pairs)

De-duplicate Jacobian block pairs requested by multiple process builders,
keeping the first occurrence of each `(row_name, col_name)` key.

Requesting the same block with two different band-matrix types throws an error.
In principle two different types are possible (for example a `DiagonalRow` from
thermodynamics and a `TridiagonalRow` from diffusion); there is no such case at
present, but if one arises the wider type should be used.
"""
merge_jacobian_blocks(block_pairs) =
    unrolled_reduce(block_pairs, ()) do merged, block_pair
        matching = unrolled_filter(pair -> pair.first == block_pair.first, merged)
        if isempty(matching)
            (merged..., block_pair)
        else
            typeof(matching[1].second) == typeof(block_pair.second) ||
                error("Jacobian block requested with two different types")
            merged
        end
    end

"""
    jacobian_diagonal_names(Y)

Return a `Tuple` of the state variable names that require a diagonal Jacobian
block: every top-level field of `Y.c` and `Y.f`, every field of the first
updraft `Y.c.sgsʲs.:(1)` together with `f.sgsʲs.:(1).u₃`, and `sfc` when the
state has a surface field. Called from `fallback_identity_blocks`.
"""
function jacobian_diagonal_names(Y)
    center_names = unrolled_map(
        center_state_name,
        unrolled_filter(
            name -> name != @name(sgsʲs),
            MatrixFields.top_level_names(Y.c),
        ),
    )
    face_names = unrolled_map(
        name -> MatrixFields.append_internal_name(@name(f), name),
        unrolled_filter(
            name -> name != @name(sgsʲs),
            MatrixFields.top_level_names(Y.f),
        ),
    )
    sgs_names =
        hasproperty(Y.c, :sgsʲs) ?
        (
            unrolled_map(
                sgs_state_name,
                MatrixFields.top_level_names(Y.c.sgsʲs.:(1)),
            )...,
            @name(f.sgsʲs.:(1).u₃),
        ) : ()
    sfc_names = MatrixFields.has_field(Y, @name(sfc)) ? (@name(sfc),) : ()
    return (center_names..., face_names..., sgs_names..., sfc_names...)
end

"""
    fallback_identity_blocks(block_pairs, Y, FT)

Return `(name, name) => FT(-1) * I` pairs for all state variables that did not
receive a diagonal block from any process builder in `block_pairs`. For these
variables the implicit residual is `-ΔY`, so only the explicit tendency
contributes to the implicit error.

`FT(-1) * I` is used instead of `-I` because `inv(-1) == -1.0`, so multiplying
`inv(-1)` by a `Float32` would yield a `Float64`.
"""
function fallback_identity_blocks(block_pairs, Y, FT)
    missing_names = unrolled_filter(jacobian_diagonal_names(Y)) do name
        !unrolled_any(pair -> pair.first == (name, name), block_pairs)
    end
    return unrolled_map(name -> (name, name) => FT(-1) * I, missing_names)
end

"""
    jacobian_solver_algorithm(Y)

Return the exact nested `MatrixFields.BlockArrowheadSolve` algorithm used to
invert the wave-core sparse Jacobian onto `f.u₃` after `c.uₕ`.
"""
function jacobian_solver_algorithm(Y)
    # Every scalar, SGS, and surface diagonal block is `-I`, and only
    # `(scalar, u₃)`, `(scalar, uₕ)`, `(u₃, scalar)`, and `(u₃, uₕ)` are
    # off-diagonal, so eliminating all non-`(uₕ, u₃)` fields produces an
    # exact tridiagonal Schur complement onto `f.u₃` after `c.uₕ`.
    diagonal_names = unrolled_filter(
        name -> name != @name(c.uₕ) && name != @name(f.u₃),
        jacobian_diagonal_names(Y),
    )
    return MatrixFields.BlockArrowheadSolve(
        diagonal_names...;
        alg₂ = MatrixFields.BlockLowerTriangularSolve(@name(c.uₕ)),
    )
end

"""
    jacobian_cache(alg::ManualSparseJacobian, Y, atmos)

Allocate the sparse `∂R/∂Y` matrix and its solver for a
[`ManualSparseJacobian`](@ref).

The nonzero blocks are collected from `advection_jacobian_blocks`,
de-duplicated with `merge_jacobian_blocks`, and completed with
`fallback_identity_blocks`; the solver comes from `jacobian_solver_algorithm`.

# Returns

`NamedTuple` with fields `matrix` (a `MatrixFields.FieldMatrixWithSolver`) and
`derivative_flags` (the flags from `_derivative_flags`, which
`update_jacobian!` passes back to `update_advection_jacobian!`).
"""
function jacobian_cache(alg::ManualSparseJacobian, Y, atmos)
    derivative_flags = _derivative_flags(atmos, Y)
    (; topography_flag) = derivative_flags
    FT = Spaces.undertype(axes(Y.c))

    process_block_pairs = merge_jacobian_blocks(
        advection_jacobian_blocks(Y, atmos, topography_flag),
    )
    block_pairs = (
        process_block_pairs...,
        fallback_identity_blocks(process_block_pairs, Y, FT)...,
    )
    matrix = MatrixFields.FieldMatrix(block_pairs...)
    full_alg = jacobian_solver_algorithm(Y)

    return (;
        matrix = MatrixFields.FieldMatrixWithSolver(matrix, Y, full_alg),
        derivative_flags,
    )
end

# ============================================================================
# Jacobian matrix entries
# ============================================================================
#
# `update_jacobian!` delegates to one update function per process, mirroring
# the structure of `implicit_tendency!`. The update functions communicate
# through the matrix blocks and through `p.scratch`, so their relative order
# matters; see the ordering contract in `update_jacobian!`.
#
# TODO: There are a few for loops in these functions. This is because
# using unrolled_foreach allocates (breaks the flame tests)

"""
    update_advection_jacobian!(matrix, Y, p, dtγ, topography_flag)

Update the Jacobian blocks for implicit vertical advection of the active
scalars and sedimenting tracers with respect to velocity, and for the vertical
momentum equation (pressure gradient, buoyancy, and Rayleigh sponge).

All of these blocks are assigned, not accumulated, so this update must run
first. The velocity derivatives of `ρe_tot`, `ρq_tot`, and the sedimenting
tracers use central face values (`ᶠinterp`, with face energy flux per unit
contravariant velocity `ᶠρ ᶠinterp(ᶜe_tot) + ᶠinterp(ᶜp)` and `ᶜp` frozen,
omitting the `O(w²)` kinetic-energy derivative `∂p/∂u₃`), keeping their
velocity blocks independent of `sign(u³)` while preserving affine relations
among material scalars at fixed pressure and geopotential (`ρtke` omits velocity
blocks, as freezing the `ᶠupwind1` donor at the guess velocity turns downwind
when `u³` reverses sign). The `(ρ, uₕ)` and `(scalar, uₕ)` blocks are written
in a separate column loop when `topography_flag` is `UseDerivative()`, keeping
each GPU kernel within the 4 KiB parameter limit on multi-moment microphysics.
The `(f.u₃, f.u₃)` block includes the `-I` term of the residual, and the
Rayleigh sponge damping when `p.atmos.rayleigh_sponge` is a `RayleighSponge`.

The linearization of the Exner-form pressure gradient is exact: the reference
profile terms cancel identically, and the thermal and pressure buoyancy
contributions combine into a single term proportional to the thermodynamic
density derivative, which is nonzero only in the `ρ` column. When
`use_moist_wave_jacobian(p)` holds (`EquilibriumMicrophysics0M`), the pressure
derivatives `(∂p/∂ρe_tot, ∂p/∂ρq_tot, ∂p/∂ρ)` and kinetic-energy
`(f.u₃, c.uₕ)` and `(f.u₃, f.u₃)` blocks blend frozen-condensate and
saturation-equilibrium derivatives by the saturated fraction `f`. Non-equilibrium
microphysics schemes (`1M`, `2M`, `2M+P3`) use frozen-condensate pressure
derivatives matching the wave-core residual. Sound and gravity waves are both
treated implicitly. See the comment in `_update_u₃_advection_jacobian!` for the
algebra.

Runs in two column loops without topography (three with topography), may write
to `p.scratch.ᶜtemp_scalar`..`ᶜtemp_scalar_3`, mutates `matrix`, and returns
`nothing`.
"""
function update_advection_jacobian!(
    matrix,
    Y,
    p,
    dtγ,
    topography_flag,
)
    (; ᶜp) = p.precomputed

    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜgⁱʲ = Fields.local_geometry_field(Y.c).gⁱʲ
    ᶠgⁱʲ = Fields.local_geometry_field(Y.f).gⁱʲ
    has_q_tot = MatrixFields.has_field(Y, @name(c.ρq_tot))

    # Scalar rows with respect to u₃ and, in a second loop when terrain is
    # active, to uₕ (through the terrain-following contravariant velocity).
    sedimenting_names = sedimenting_mass_names(Y)
    central_tracer_names = (
        (has_q_tot ? (@name(c.ρq_tot),) : ())...,
        unrolled_map(center_state_name, sedimenting_names)...,
    )
    ∂ᶜρχ_err_∂ᶠu₃s =
        unrolled_map(n -> matrix[n, @name(f.u₃)], central_tracer_names)
    DataLayouts.foreach_column(
        matrix[@name(c.ρ), @name(f.u₃)],
        matrix[@name(c.ρe_tot), @name(f.u₃)],
        Y.c,
        ᶜp,
        ᶜJ,
        ᶠJ,
        ᶠgⁱʲ,
        ∂ᶜρχ_err_∂ᶠu₃s,
    ) do ∂ᶜρ_err_∂ᶠu₃,
    ∂ᶜρe_tot_err_∂ᶠu₃,
    Yc,
    ᶜp,
    ᶜJ,
    ᶠJ,
    ᶠgⁱʲ,
    u₃_blocks
        ᶜχs = (
            (
                MatrixFields.has_field(Yc, @name(ρq_tot)) ?
                ((@. lazy(specific(Yc.ρq_tot, Yc.ρ))),) : ()
            )...,
            unrolled_map(sedimenting_names) do n
                @inline
                return @. lazy(specific(MatrixFields.get_field(Yc, n), Yc.ρ))
            end...,
        )
        ᶠρ = @. lazy(ᶠinterp(Yc.ρ * ᶜJ) / ᶠJ)
        ᶜe_tot = @. lazy(specific(Yc.ρe_tot, Yc.ρ))
        ᶜadv_mat = @. -(ᶜadvdivᵥ_matrix()) * DiagonalMatrixRow(ᶠρ)
        @. ∂ᶜρ_err_∂ᶠu₃ = dtγ * ᶜadv_mat * DiagonalMatrixRow(g³³(ᶠgⁱʲ))
        @. ∂ᶜρe_tot_err_∂ᶠu₃ =
            dtγ *
            (-(ᶜadvdivᵥ_matrix())) *
            DiagonalMatrixRow((ᶠρ * ᶠinterp(ᶜe_tot) + ᶠinterp(ᶜp)) * g³³(ᶠgⁱʲ))
        MatrixFields.unrolled_foreach(u₃_blocks, ᶜχs) do ∂ᶜρχ_err_∂ᶠu₃, ᶜχ
            @inline
            @. ∂ᶜρχ_err_∂ᶠu₃ =
                dtγ * ᶜadv_mat * DiagonalMatrixRow(ᶠinterp(ᶜχ) * g³³(ᶠgⁱʲ))
            return nothing
        end
        return nothing
    end

    if use_derivative(topography_flag)
        ∂ᶜρχ_err_∂ᶜuₕs =
            unrolled_map(n -> matrix[n, @name(c.uₕ)], central_tracer_names)
        DataLayouts.foreach_column(
            matrix[@name(c.ρ), @name(c.uₕ)],
            matrix[@name(c.ρe_tot), @name(c.uₕ)],
            Y.c,
            ᶜp,
            ᶜJ,
            ᶠJ,
            ᶜgⁱʲ,
            ∂ᶜρχ_err_∂ᶜuₕs,
        ) do ∂ᶜρ_err_∂ᶜuₕ,
        ∂ᶜρe_tot_err_∂ᶜuₕ,
        Yc,
        ᶜp,
        ᶜJ,
        ᶠJ,
        ᶜgⁱʲ,
        uₕ_blocks
            ᶜχs = (
                (
                    MatrixFields.has_field(Yc, @name(ρq_tot)) ?
                    ((@. lazy(specific(Yc.ρq_tot, Yc.ρ))),) : ()
                )...,
                unrolled_map(sedimenting_names) do n
                    @inline
                    return @. lazy(
                        specific(MatrixFields.get_field(Yc, n), Yc.ρ),
                    )
                end...,
            )
            ᶠρ = @. lazy(ᶠinterp(Yc.ρ * ᶜJ) / ᶠJ)
            ᶜe_tot = @. lazy(specific(Yc.ρe_tot, Yc.ρ))
            ᶜadv_mat = @. -(ᶜadvdivᵥ_matrix()) * DiagonalMatrixRow(ᶠρ)
            ᶠwinterp_g³ʰ =
                @. ᶠwinterp_matrix(ᶜJ * Yc.ρ) * DiagonalMatrixRow(g³ʰ(ᶜgⁱʲ))
            @. ∂ᶜρ_err_∂ᶜuₕ = dtγ * ᶜadv_mat * ᶠwinterp_g³ʰ
            @. ∂ᶜρe_tot_err_∂ᶜuₕ =
                dtγ *
                (-(ᶜadvdivᵥ_matrix())) *
                DiagonalMatrixRow(ᶠρ * ᶠinterp(ᶜe_tot) + ᶠinterp(ᶜp)) *
                ᶠwinterp_g³ʰ
            MatrixFields.unrolled_foreach(uₕ_blocks, ᶜχs) do ∂ᶜρχ_err_∂ᶜuₕ, ᶜχ
                @inline
                @. ∂ᶜρχ_err_∂ᶜuₕ =
                    dtγ *
                    ᶜadv_mat *
                    DiagonalMatrixRow(ᶠinterp(ᶜχ)) *
                    ᶠwinterp_g³ʰ
                return nothing
            end
            return nothing
        end
    end

    # Loop 2: the u₃ row. Exner-form PGF tendency: grad(Φ) - grad(Φ_r) + cp_d
    # int(θ_v - θ_vr) grad(Π). The reference terms satisfy grad(Φ_r(p)) + cp_d
    # θ_vr(p) grad(Π(p)) ≡ 0 pointwise for ANY p field (Φ_r′ = -cp_d θ_vr Π′ by
    # construction), so their state derivative vanishes identically and the
    # exact linearization carries the full θ_v:
    #   d(PGF)·δχ = cp_d (∂θ_v/∂χ δχ) grad(Π)            [thermal buoyancy]
    #             + cp_d θ_v (κ_d Π / p) grad(∂p/∂χ δχ)  [acoustic]
    #             + cp_d θ_v grad(κ_d Π / p) ∂p/∂χ δχ.   [pressure buoyancy]
    # With grad(κ_d Π/p) = -(1 - κ_d) (κ_d Π/p) grad(p)/p and the equation
    # of state ρ = p^(1-κ_d) p₀^κ_d / (R_d θ_v), i.e.
    #   δρ/ρ = (1 - κ_d) δp/p - δθ_v/θ_v,
    # the two buoyancy terms combine into a single term proportional to the
    # thermodynamic density derivative:
    #   thermal + pressure buoyancy = -cp_d θ_v grad(Π) (∂ρ/∂χ δχ) / ρ.
    # It therefore vanishes identically in the columns perturbed at fixed ρ
    # (ρe_tot, the ρq's, and K through the velocities), where thermal and
    # pressure buoyancy are equal and opposite — cancelling them
    # analytically here avoids computing two large opposing terms — and it
    # reduces to grad(p) δρ/ρ² in the ρ column. Since cp_d θ_v κ_d Π/p =
    # R_m T/p = 1/ρ pointwise, the acoustic factor is the familiar
    # -grad(·)/ρ operator. Sound and gravity waves are thus both treated fully
    # implicitly: acoustics couple ᶠu₃ to every thermodynamic column through
    # ∂p/∂χ, and buoyancy couples it to the ρ column.
    if use_moist_wave_jacobian(p)
        _update_u₃_advection_jacobian!(
            matrix,
            Y,
            p,
            dtγ,
            topography_flag,
            UseDerivative(),
        )
    else
        _update_u₃_advection_jacobian!(
            matrix,
            Y,
            p,
            dtγ,
            topography_flag,
            IgnoreDerivative(),
        )
    end
    return nothing
end

function _update_u₃_advection_jacobian!(
    matrix,
    Y,
    p,
    dtγ,
    topography_flag,
    moist_flag::DerivativeFlag,
)
    (; params) = p
    (; ᶜΦ) = p.core
    (; ᶜK, ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    rs = p.atmos.rayleigh_sponge

    FT = Spaces.undertype(axes(Y.c))
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'

    cv_d = FT(CAP.cv_d(params))
    Δcv_v = FT(CAP.cv_v(params)) - cv_d
    T_0 = FT(CAP.T_0(params))
    R_d = FT(CAP.R_d(params))
    R_v = FT(CAP.R_v(params))
    ΔR_v = R_v - R_d
    e_int_v0 = FT(CAP.e_int_v0(params))
    cp_d = FT(CAP.cp_d(params))
    thermo_params = CAP.thermodynamics_params(params)

    ᶜgⁱʲ = Fields.local_geometry_field(Y.c).gⁱʲ
    ᶠz = Fields.coordinate_field(Y.f).z
    zmax = Spaces.z_max(axes(Y.f))
    has_q_tot = MatrixFields.has_field(Y, @name(c.ρq_tot))

    sgs_cf_flag = _sgs_cf_flag_for_cache(moist_flag, p.atmos, p.precomputed)
    if use_derivative(moist_flag)
        _compute_moist_wave_jacobian_scratch!(
            p,
            Y,
            thermo_params,
            sgs_cf_flag,
        )
    end
    moist_scratch = _moist_scratch_fields(moist_flag, p.scratch)

    mass_names = sedimenting_mass_names(Y)
    ∂ᶠu₃_err_∂ᶜρq_masses(phase) = unrolled_map(
        n -> matrix[@name(f.u₃), center_state_name(n)],
        unrolled_filter(n -> condensate_phase(n) == phase, mass_names),
    )
    phase_consts = unrolled_map((TD.Liquid(), TD.Ice())) do phase
        (
            condensate_e_int_offset(phase, params),
            condensate_cv_difference(phase, params),
        )
    end
    I_u₃ = DiagonalMatrixRow(one_C3xACT3)
    ∂ᶠu₃_err_∂ᶜρq_tots =
        has_q_tot ? (matrix[@name(f.u₃), @name(c.ρq_tot)],) : ()

    DataLayouts.foreach_column(
        (
            matrix[@name(f.u₃), @name(c.ρ)],
            matrix[@name(f.u₃), @name(c.ρe_tot)],
            matrix[@name(f.u₃), @name(c.uₕ)],
            matrix[@name(f.u₃), @name(f.u₃)],
        ),
        (Y.c, Y.f.u₃),
        (ᶜK, ᶜp, ᶜT, ᶜΦ),
        (ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice),
        (ᶠz, ᶜgⁱʲ),
        moist_scratch,
        ∂ᶠu₃_err_∂ᶜρq_tots,
        (∂ᶠu₃_err_∂ᶜρq_masses(TD.Liquid()), ∂ᶠu₃_err_∂ᶜρq_masses(TD.Ice())),
    ) do (∂ᶠu₃_err_∂ᶜρ, ∂ᶠu₃_err_∂ᶜρe_tot, ∂ᶠu₃_err_∂ᶜuₕ, ∂ᶠu₃_err_∂ᶠu₃),
    (Yc, ᶠu₃),
    (ᶜK, ᶜp, ᶜT, ᶜΦ),
    (ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice),
    (ᶠz, ᶜgⁱʲ),
    moist_scratch_cols,
    q_tot_cols,
    mass_phase_cols
        @inline
        ᶜkappa_m = @. lazy(
            TD.gas_constant_air(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) /
            TD.cv_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice),
        )
        ᶠp_grad = @. DiagonalMatrixRow(-1 / ᶠinterp(Yc.ρ)) * ᶠgradᵥ_matrix()
        ᶜkappa_eff =
            !isempty(moist_scratch_cols) ? moist_scratch_cols[1] : ᶜkappa_m

        ᶜθ_v = @. lazy(
            theta_v(thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice),
        )
        ᶜΠ = @. lazy(TD.exner_given_pressure(thermo_params, ᶜp))
        # ∂p/∂ρ|_{T, ρq_tot} = R_d T and ∂(ρe_int)/∂ρ|_{T, ρq_tot} = e_d(T) hold
        # in both frozen and saturated regimes, so ∂p/∂ρ has the same affine
        # dependence on κ = ∂p/∂(ρe_tot) in both and blends via `ᶜkappa_eff`.
        ᶜ∂p∂ρ = @. lazy(
            ᶜkappa_eff * (T_0 * cp_d - ᶜK - ᶜΦ) + (R_d - ᶜkappa_eff * cv_d) * ᶜT,
        )
        @. ∂ᶠu₃_err_∂ᶜρ =
            dtγ * (
                ᶠp_grad * DiagonalMatrixRow(ᶜ∂p∂ρ) +
                DiagonalMatrixRow(
                    cp_d * ᶠinterp(ᶜθ_v) * ᶠgradᵥ(ᶜΠ) / ᶠinterp(Yc.ρ),
                ) * ᶠinterp_matrix()
            )
        @. ∂ᶠu₃_err_∂ᶜρe_tot = dtγ * ᶠp_grad * DiagonalMatrixRow(ᶜkappa_eff)

        if !isempty(q_tot_cols)
            ᶜ∂p∂ρq_tot = if !isempty(moist_scratch_cols)
                moist_scratch_cols[3]
            else
                @. lazy(
                    ᶜkappa_m * (-e_int_v0 - R_d * T_0 - Δcv_v * (ᶜT - T_0)) +
                    ΔR_v * ᶜT,
                )
            end
            ᶠblock_q_tot = @. dtγ * ᶠp_grad * DiagonalMatrixRow(ᶜ∂p∂ρq_tot)
            MatrixFields.unrolled_foreach(q_tot_cols) do ∂ᶠu₃_err_∂ᶜρq_tot
                @inline
                @. ∂ᶠu₃_err_∂ᶜρq_tot = ᶠblock_q_tot
            end
        end

        # The -R_v T term is ∂p/∂q_c at fixed q_tot: condensate replaces vapor,
        # ∂R_m/∂q_c = -R_v (R_m = (1 - q_tot) R_d + q_vap R_v). Non-equilibrium
        # condensates retain full frozen loading matching the wave-core residual.
        MatrixFields.unrolled_foreach(
            map(tuple, mass_phase_cols, phase_consts),
        ) do (mass_cols, (e_int_q, ∂cv∂q))
            @inline
            isempty(mass_cols) && return nothing
            ᶜ∂p∂ρq = @. lazy(
                ᶜkappa_m * (e_int_q - ∂cv∂q * (ᶜT - T_0)) - R_v * ᶜT,
            )
            MatrixFields.unrolled_foreach(mass_cols) do ∂ᶠu₃_err_∂ᶜρq
                @inline
                @. ∂ᶠu₃_err_∂ᶜρq = dtγ * ᶠp_grad * DiagonalMatrixRow(ᶜ∂p∂ρq)
            end
            return nothing
        end

        # ∂K/∂uₕ, with the terrain-following part of u³ with topography
        ᶜ∂K∂uₕ =
            use_derivative(topography_flag) ?
            (@. lazy(adjoint(CT12(Yc.uₕ)) + adjoint(ᶜinterp(ᶠu₃)) * g³ʰ(ᶜgⁱʲ))) :
            (@. lazy(adjoint(CT12(Yc.uₕ))))
        @. ∂ᶠu₃_err_∂ᶜuₕ =
            dtγ * ᶠp_grad * DiagonalMatrixRow(-(ᶜkappa_eff) * Yc.ρ) *
            DiagonalMatrixRow(ᶜ∂K∂uₕ)
        ∂ᶜK_∂ᶠu₃ = @. lazy(
            ᶜinterp_matrix() * DiagonalMatrixRow(adjoint(CT3(ᶠu₃))) +
            DiagonalMatrixRow(adjoint(CT3(Yc.uₕ))) * ᶜinterp_matrix(),
        )
        if rs isa RayleighSponge
            @. ∂ᶠu₃_err_∂ᶠu₃ =
                dtγ * (
                    ᶠp_grad *
                    DiagonalMatrixRow(-(ᶜkappa_eff) * Yc.ρ) *
                    ∂ᶜK_∂ᶠu₃ + DiagonalMatrixRow(
                        -β_rayleigh_u₃(rs, ᶠz, zmax) * (one_C3xACT3,),
                    )
                ) - (I_u₃,)
        else
            @. ∂ᶠu₃_err_∂ᶠu₃ =
                dtγ *
                ᶠp_grad *
                DiagonalMatrixRow(-(ᶜkappa_eff) * Yc.ρ) *
                ∂ᶜK_∂ᶠu₃ - (I_u₃,)
        end
    end
    return nothing
end

"""
    update_jacobian!(alg::ManualSparseJacobian, cache, Y, p, dtγ, t)

Update all blocks of the sparse `∂R/∂Y = dtγ ∂Yₜ/∂Y - I` in `cache.matrix`.

Reads the precomputed quantities set by `set_implicit_precomputed_quantities!`,
mutates `cache.matrix` and `p.scratch`, and returns `nothing`.
"""
function update_jacobian!(alg::ManualSparseJacobian, cache, Y, p, dtγ, t)
    (; topography_flag) = cache.derivative_flags
    (; matrix) = cache
    update_advection_jacobian!(matrix, Y, p, dtγ, topography_flag)
    zero_velocity_jacobian!(matrix, Y, p, t)
    return nothing
end

"""
    invert_jacobian!(alg::ManualSparseJacobian, cache, ΔY, R)

Solve `(∂R/∂Y) ΔY = R` for `ΔY` with the nested block solver from
`jacobian_solver_algorithm`. Mutates `ΔY`.
"""
invert_jacobian!(::ManualSparseJacobian, cache, ΔY, R) =
    LinearAlgebra.ldiv!(ΔY, cache.matrix, R)
