import LinearAlgebra: I, Adjoint

using ClimaCore.MatrixFields
import ClimaCore.MatrixFields: @name
import UnrolledUtilities: unrolled_any, unrolled_filter, unrolled_map, unrolled_reduce

abstract type DerivativeFlag end
struct UseDerivative <: DerivativeFlag end
struct IgnoreDerivative <: DerivativeFlag end

DerivativeFlag(value) = value ? UseDerivative() : IgnoreDerivative()
DerivativeFlag(mode::AbstractTimesteppingMode) =
    DerivativeFlag(mode == Implicit())

use_derivative(::UseDerivative) = true
use_derivative(::IgnoreDerivative) = false

"""
    ManualSparseJacobian(; approximate_solve_iters = 1)

A [`JacobianAlgorithm`](@ref) that approximates the Jacobian using analytically
derived tendency derivatives and inverts it using a specialized nested linear
solver.

Which derivative blocks are computed is determined automatically from the
`AtmosModel` (topography, diffusion mode, EDMF modes) when the cache is
built — users do not configure them directly.

# Arguments

  - `approximate_solve_iters::Int = 1`: number of iterations to take for the
    approximate linear solve required when grid-scale diffusion is treated
    implicitly.
"""
struct ManualSparseJacobian <: SparseJacobian
    approximate_solve_iters::Int
end
ManualSparseJacobian(; approximate_solve_iters::Int = 1) =
    ManualSparseJacobian(approximate_solve_iters)

# Topography and diffusion flags specialize the cache at build time.
# SGS modes (advection, entr/detr, mass flux, NH pressure, vertdiff) are
# always implicit — no flags needed for them.
function _derivative_flags(atmos, Y)
    return (;
        topography_flag = DerivativeFlag(has_topography(axes(Y.c))),
        diffusion_flag = DerivativeFlag(atmos.diff_mode),
    )
end

# ============================================================================
# Jacobian matrix structure
# ============================================================================
#
# The set of nonzero Jacobian blocks is assembled from per-process "builders"
# (advection, diffusion, sedimentation/moisture, SGS advection, SGS mass
# flux). Each builder returns `(row_name, col_name) => block` pairs for the
# tracers that participate in its process, based on the process-based tracer
# lists in `utils/tracer_processes.jl`. Blocks requested by multiple processes
# are de-duplicated, and every state variable that ends up without a diagonal
# block is given the identity block `-I` (i.e., only its explicit tendency
# contributes to its implicit error).

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

Jacobian blocks for implicit vertical advection of the active scalars and for
the vertical momentum equation (pressure gradient, buoyancy, and Rayleigh
sponge). The `(f.u₃, condensate mass)` blocks hold the derivatives of the
pressure gradient with respect to condensate masses.
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
    return (
        (
            use_derivative(topography_flag) ?
            map(
                name ->
                    (name, @name(c.uₕ)) =>
                        similar(Y.c, TridiagonalRow_ACT12),
                active_scalar_names,
            ) : ()
        )...,
        map(
            name -> (name, @name(f.u₃)) => similar(Y.c, BidiagonalRow_ACT3),
            active_scalar_names,
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
    diffusion_jacobian_blocks(Y, atmos, diffusion_flag)

Jacobian blocks for implicit vertical diffusion of scalars and momentum.
Empty when diffusion is treated explicitly (the affected variables then get
their diagonal blocks from sedimentation or from the `-I` fallback).
"""
function diffusion_jacobian_blocks(Y, atmos, diffusion_flag)
    use_derivative(diffusion_flag) || return ()
    FT = Spaces.undertype(axes(Y.c))
    (; TridiagonalRow) = jacobian_row_types(FT)
    is_in_Y(name) = MatrixFields.has_field(Y, name)
    diffused_scalar_names =
        unrolled_map(center_state_name, diffused_gs_scalar_names(Y))
    mass_names = unrolled_map(center_state_name, sedimenting_mass_names(Y))
    sedimenting_names =
        unrolled_map(center_state_name, sedimenting_tracer_names(Y))
    passive_names =
        unrolled_map(center_state_name, passive_gs_tracer_names(Y))
    ρtke_if_available =
        is_in_Y(@name(c.ρtke)) ? (@name(c.ρtke),) : ()
    return (
        # (·, ρ) blocks exist only where they receive values: (ρe_tot, ρ) and
        # (ρq_tot, ρ) accumulate the SGS mass-flux Jacobian, and (ρtke, ρ)
        # holds the dissipation derivative. The diffusive fluxes' own
        # ρ-dependence — through χ = ρχ/ρ and the ρ factor in ρ K ∇χ, and
        # through the Yₜ.c.ρ counterpart of the ρq_tot diffusion — is
        # neglected everywhere (like the other ∂K/∂state terms), so the
        # microphysics tracers and passive tracers carry no (·, ρ) blocks at
        # all: they were identically zero, and the condensate-mass rows must
        # in any case precede ρ in the scalar solve because the ρ row holds
        # their sedimentation derivatives (see `sedimentation_jacobian_blocks`
        # and `jacobian_solver_algorithm`).
        map(
            name -> (name, @name(c.ρ)) => similar(Y.c, TridiagonalRow),
            (
                unrolled_filter(
                    name -> !(name in sedimenting_names),
                    diffused_scalar_names,
                )...,
                ρtke_if_available...,
            ),
        )...,
        map(
            name -> (name, name) => similar(Y.c, TridiagonalRow),
            (
                diffused_scalar_names...,
                passive_names...,
                ρtke_if_available...,
            ),
        )...,
        (
            is_in_Y(@name(c.ρq_tot)) ?
            (
                (@name(c.ρe_tot), @name(c.ρq_tot)) =>
                    similar(Y.c, TridiagonalRow),
            ) : ()
        )...,
        map(
            name -> (@name(c.ρe_tot), name) => similar(Y.c, TridiagonalRow),
            mass_names,
        )...,
        # TODO should we check is_in_Y(@name(c.ρq_tot)) here
        map(
            name -> (@name(c.ρq_tot), name) => similar(Y.c, TridiagonalRow),
            mass_names,
        )...,
        (
            !isnothing(atmos.turbconv_model) ||
            !disable_momentum_vertical_diffusion(atmos.vertical_diffusion) ?
            ((@name(c.uₕ), @name(c.uₕ)) => similar(Y.c, TridiagonalRow),) : ()
        )...,
    )
end

"""
    sedimentation_jacobian_blocks(Y, atmos)

Jacobian blocks for implicit sedimentation of condensate tracers, including
the couplings of sedimenting condensate masses to `ρq_tot` and `ρe_tot`.
Also allocates the `ρe_tot` and `ρq_tot` diagonal blocks (and their mutual
coupling), which default to `-I` and are accumulated into by diffusion and
SGS mass flux in moist configurations.
"""
function sedimentation_jacobian_blocks(Y, atmos)
    atmos.microphysics_model isa DryModel && return ()
    FT = Spaces.undertype(axes(Y.c))
    (; TridiagonalRow) = jacobian_row_types(FT)
    sedimenting_names =
        unrolled_map(center_state_name, sedimenting_tracer_names(Y))
    mass_names = unrolled_map(center_state_name, sedimenting_mass_names(Y))
    return (
        (@name(c.ρe_tot), @name(c.ρe_tot)) => similar(Y.c, TridiagonalRow),
        (@name(c.ρq_tot), @name(c.ρq_tot)) => similar(Y.c, TridiagonalRow),
        (@name(c.ρe_tot), @name(c.ρq_tot)) => similar(Y.c, TridiagonalRow),
        map(
            name -> (name, name) => similar(Y.c, TridiagonalRow),
            sedimenting_names,
        )...,
        map(
            name -> (@name(c.ρq_tot), name) => similar(Y.c, TridiagonalRow),
            mass_names,
        )...,
        map(
            name -> (@name(c.ρe_tot), name) => similar(Y.c, TridiagonalRow),
            mass_names,
        )...,
        # Sedimentation moves mass: ∂(ρ tendency)/∂(ρq_x), matching the
        # identical vtt added to Yₜ.c.ρ and Yₜ.c.ρq_tot in
        # `vertical_advection_of_water_tendency!`.
        map(
            name -> (@name(c.ρ), name) => similar(Y.c, TridiagonalRow),
            mass_names,
        )...,
    )
end

"""
    sgs_advection_jacobian_blocks(Y, atmos)

Jacobian blocks for implicit vertical advection, sedimentation, diffusion,
and entrainment of the updraft scalars, including the couplings of
sedimenting SGS condensate masses to the updraft `q_tot`.
"""
function sgs_advection_jacobian_blocks(Y, atmos)
    atmos.turbconv_model isa PrognosticEDMFX || return ()
    FT = Spaces.undertype(axes(Y.c))
    (; TridiagonalRow) = jacobian_row_types(FT)
    sgs_scalar_names =
        unrolled_map(sgs_state_name, advected_sgs_scalar_names(Y))
    sgs_mass_names =
        unrolled_map(sgs_state_name, sedimenting_sgs_mass_names(Y))
    return (
        map(
            name -> (name, name) => similar(Y.c, TridiagonalRow),
            sgs_scalar_names,
        )...,
        map(
            name ->
                (@name(c.sgsʲs.:(1).q_tot), name) =>
                    similar(Y.c, TridiagonalRow),
            sgs_mass_names,
        )...,
    )
end

"""
    sgs_massflux_jacobian_blocks(Y, atmos)

Jacobian blocks for the contributions of the SGS mass flux to the grid-mean
scalars.
"""
function sgs_massflux_jacobian_blocks(Y, atmos)
    (
        atmos.turbconv_model isa PrognosticEDMFX &&
        atmos.edmfx_model.sgs_mass_flux isa Val{true}
    ) || return ()
    FT = Spaces.undertype(axes(Y.c))
    (; TridiagonalRow, BidiagonalRow_ACT3) = jacobian_row_types(FT)
    tracer_names =
        unrolled_map(center_state_name, microphysics_tracer_names(Y))
    sedimenting_names =
        unrolled_map(center_state_name, sedimenting_tracer_names(Y))
    return (
        map(
            name ->
                (name, get_χʲ_name_from_ρχ_name(name)) =>
                    similar(Y.c, TridiagonalRow),
            tracer_names,
        )...,
        map(
            name -> (name, @name(f.u₃)) => similar(Y.c, BidiagonalRow_ACT3),
            sedimenting_names,
        )...,
        (@name(c.ρe_tot), @name(c.sgsʲs.:(1).mse)) =>
            similar(Y.c, TridiagonalRow),
        (@name(c.ρe_tot), @name(c.ρ)) => similar(Y.c, TridiagonalRow),
        (@name(c.ρq_tot), @name(c.ρ)) => similar(Y.c, TridiagonalRow),
    )
end

"""
    merge_jacobian_blocks(block_pairs)

De-duplicate Jacobian block pairs requested by multiple process builders,
keeping the first occurrence of each `(row_name, col_name)` key. Requesting
the same block with two different types is an error.

Note: In principle two different types are possible
(for example DiagonalRow from thermodynamics and TridiagonalRow from diffusion).
We don't have an example like that right now, but in the future we might
want to use the larger type.
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

`Tuple` of the state variable names that require a diagonal Jacobian block.
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

`(name, name) => -I` pairs for all state variables that did not receive a
diagonal block from any process builder. For these variables, only the
explicit tendency contributes to the implicit error.

Note: We have to use FT(-1) * I instead of -I because inv(-1) == -1.0,
which means that multiplying inv(-1) by a Float32 will yield a Float64.
"""
function fallback_identity_blocks(block_pairs, Y, FT)
    missing_names = unrolled_filter(jacobian_diagonal_names(Y)) do name
        !unrolled_any(pair -> pair.first == (name, name), block_pairs)
    end
    return unrolled_map(name -> (name, name) => FT(-1) * I, missing_names)
end

"""
    jacobian_solver_algorithm(Y, atmos, diffusion_flag, approximate_solve_iters)

The nested `MatrixFields` solver algorithm used to invert the sparse Jacobian.
"""
function jacobian_solver_algorithm(
    Y,
    atmos,
    diffusion_flag,
    approximate_solve_iters,
)
    is_in_Y(name) = MatrixFields.has_field(Y, name)
    sfc_if_available = is_in_Y(@name(sfc)) ? (@name(sfc),) : ()
    ρtke_if_available = is_in_Y(@name(c.ρtke)) ? (@name(c.ρtke),) : ()
    sgs_ρa_if_available =
        is_in_Y(@name(c.sgsʲs.:(1).ρa)) ? (@name(c.sgsʲs.:(1).ρa),) : ()
    sgs_u³_if_available =
        is_in_Y(@name(f.sgsʲs.:(1).u₃)) ? (@name(f.sgsʲs.:(1).u₃),) : ()

    mass_names = unrolled_map(center_state_name, sedimenting_mass_names(Y))
    sgs_scalar_names =
        unrolled_map(sgs_state_name, advected_sgs_scalar_names(Y))
    sgs_sedimenting_names =
        unrolled_map(sgs_state_name, sedimenting_sgs_tracer_names(Y))

    mass_and_surface_names = (@name(c.ρ), sfc_if_available...)
    available_scalar_names = (
        mass_and_surface_names...,
        unrolled_map(center_state_name, microphysics_tracer_names(Y))...,
        @name(c.ρe_tot),
        ρtke_if_available...,
        sgs_scalar_names...,
        sgs_ρa_if_available...,
    )

    velocity_alg = MatrixFields.BlockLowerTriangularSolve(
        @name(c.uₕ),
        sgs_u³_if_available...,
    )
    if use_derivative(diffusion_flag) ||
       !(atmos.microphysics_model isa DryModel)
        # Scalar solve order: gs condensate masses precede ρ because the ρ row
        # carries their sedimentation derivatives ∂(ρ tendency)/∂ρq_x; ρ
        # precedes ρq_tot and ρe_tot, whose rows depend on the ρ column (SGS
        # mass flux). The condensate-mass rows carry no (·, ρ) blocks (see
        # `diffusion_jacobian_blocks`), so this order is block lower
        # triangular.
        gs_scalar_subalg = if !(atmos.microphysics_model isa DryModel)
            MatrixFields.BlockLowerTriangularSolve(
                mass_names...,
                alg₂ = MatrixFields.BlockLowerTriangularSolve(
                    mass_and_surface_names...;
                    alg₂ = MatrixFields.BlockLowerTriangularSolve(
                        @name(c.ρq_tot),
                    ),
                ),
            )
        else
            MatrixFields.BlockLowerTriangularSolve(mass_and_surface_names...)
        end
        scalar_alg =
            if atmos.turbconv_model isa PrognosticEDMFX
                MatrixFields.BlockLowerTriangularSolve(
                    sgs_sedimenting_names...;
                    alg₂ = MatrixFields.BlockLowerTriangularSolve(
                        @name(c.sgsʲs.:(1).q_tot),
                        @name(c.sgsʲs.:(1).mse);
                        alg₂ = gs_scalar_subalg,
                    ),
                )
            else
                gs_scalar_subalg
            end
        return MatrixFields.ApproximateBlockArrowheadIterativeSolve(
            available_scalar_names...;
            alg₁ = scalar_alg,
            alg₂ = velocity_alg,
            P_alg₁ = MatrixFields.MainDiagonalPreconditioner(),
            n_iters = approximate_solve_iters,
        )
    else
        return MatrixFields.BlockArrowheadSolve(
            available_scalar_names...;
            alg₂ = velocity_alg,
        )
    end
end

function jacobian_cache(alg::ManualSparseJacobian, Y, atmos)
    derivative_flags = _derivative_flags(atmos, Y)
    (; topography_flag, diffusion_flag) = derivative_flags
    FT = Spaces.undertype(axes(Y.c))

    process_block_pairs = merge_jacobian_blocks((
        sgs_advection_jacobian_blocks(Y, atmos)...,
        advection_jacobian_blocks(Y, atmos, topography_flag)...,
        diffusion_jacobian_blocks(Y, atmos, diffusion_flag)...,
        sedimentation_jacobian_blocks(Y, atmos)...,
        sgs_massflux_jacobian_blocks(Y, atmos)...,
    ))
    block_pairs = (
        process_block_pairs...,
        fallback_identity_blocks(process_block_pairs, Y, FT)...,
    )
    matrix = MatrixFields.FieldMatrix(block_pairs...)

    full_alg = jacobian_solver_algorithm(
        Y,
        atmos,
        diffusion_flag,
        alg.approximate_solve_iters,
    )

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

# ᶜkappa_m = R_m / cv_m. Recomputed by each process update that needs it,
# since the scratch field it lives in is reused by other processes.
function ᶜkappa_m_field!(Y, p)
    (; ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    thermo_params = CAP.thermodynamics_params(p.params)
    ᶜkappa_m = p.scratch.ᶜtemp_scalar
    @. ᶜkappa_m =
        TD.gas_constant_air(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) /
        TD.cv_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)
    return ᶜkappa_m
end

# Derivative of pressure with respect to ρq_tot at constant ρ, ρe_tot.
function ᶜ∂p∂ρq_tot_field!(Y, p, ᶜkappa_m)
    (; params) = p
    (; ᶜT) = p.precomputed
    FT = Spaces.undertype(axes(Y.c))
    cv_d = FT(CAP.cv_d(params))
    Δcv_v = FT(CAP.cv_v(params)) - cv_d
    T_0 = FT(CAP.T_0(params))
    R_d = FT(CAP.R_d(params))
    R_v = FT(CAP.R_v(params))
    ΔR_v = R_v - R_d
    e_int_v0 = FT(CAP.e_int_v0(params))
    ᶜ∂p∂ρq_tot = p.scratch.ᶜtemp_scalar_2
    @. ᶜ∂p∂ρq_tot =
        ᶜkappa_m * (-e_int_v0 - R_d * T_0 - Δcv_v * (ᶜT - T_0)) + ΔR_v * ᶜT
    return ᶜ∂p∂ρq_tot
end

"""
    update_advection_jacobian!(matrix, Y, p, dtγ, topography_flag)

Updates the Jacobian blocks for implicit vertical advection of the active
scalars and for the vertical momentum equation (pressure gradient, buoyancy,
and Rayleigh sponge).

Computes `∂ᶜK_∂ᶜuₕ`, `∂ᶜK_∂ᶠu₃`, `ᶠp_grad_matrix`, and `ᶜadvection_matrix` in
`p.scratch`; must run before `update_diffusion_jacobian!`, which reuses
`ᶠp_grad_matrix` as scratch space.
"""
function update_advection_jacobian!(matrix, Y, p, dtγ, topography_flag)
    (; params) = p
    (; ᶜΦ) = p.core
    (; ᶠu³, ᶜK, ᶜp, ᶜT, ᶜh_tot) = p.precomputed
    (; ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; ∂ᶜK_∂ᶜuₕ, ∂ᶜK_∂ᶠu₃, ᶠp_grad_matrix, ᶜadvection_matrix) = p.scratch
    rs = p.atmos.rayleigh_sponge

    FT = Spaces.undertype(axes(Y.c))
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'

    cv_d = FT(CAP.cv_d(params))
    T_0 = FT(CAP.T_0(params))
    R_d = FT(CAP.R_d(params))
    R_v = FT(CAP.R_v(params))
    cp_d = FT(CAP.cp_d(params))
    e_int_v0 = FT(CAP.e_int_v0(params))
    thermo_params = CAP.thermodynamics_params(params)

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜgⁱʲ = Fields.local_geometry_field(Y.c).gⁱʲ
    ᶠgⁱʲ = Fields.local_geometry_field(Y.f).gⁱʲ
    ᶠz = Fields.coordinate_field(Y.f).z
    zmax = Spaces.z_max(axes(Y.f))

    ᶜkappa_m = ᶜkappa_m_field!(Y, p)
    ᶜ∂p∂ρq_tot = ᶜ∂p∂ρq_tot_field!(Y, p, ᶜkappa_m)

    if use_derivative(topography_flag)
        @. ∂ᶜK_∂ᶜuₕ = DiagonalMatrixRow(
            adjoint(CT12(ᶜuₕ)) + adjoint(ᶜinterp(ᶠu₃)) * g³ʰ(ᶜgⁱʲ),
        )
    else
        @. ∂ᶜK_∂ᶜuₕ = DiagonalMatrixRow(adjoint(CT12(ᶜuₕ)))
    end
    @. ∂ᶜK_∂ᶠu₃ =
        ᶜinterp_matrix() ⋅ DiagonalMatrixRow(adjoint(CT3(ᶠu₃))) +
        DiagonalMatrixRow(adjoint(CT3(ᶜuₕ))) ⋅ ᶜinterp_matrix()

    @. ᶠp_grad_matrix = DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) ⋅ ᶠgradᵥ_matrix()

    @. ᶜadvection_matrix =
        -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ)
    @. p.scratch.ᶠbidiagonal_matrix_ct3xct12 =
        ᶠwinterp_matrix(ᶜJ * ᶜρ) ⋅ DiagonalMatrixRow(g³ʰ(ᶜgⁱʲ))
    if use_derivative(topography_flag)
        ∂ᶜρ_err_∂ᶜuₕ = matrix[@name(c.ρ), @name(c.uₕ)]
        @. ∂ᶜρ_err_∂ᶜuₕ =
            dtγ * ᶜadvection_matrix ⋅ p.scratch.ᶠbidiagonal_matrix_ct3xct12
    end
    ∂ᶜρ_err_∂ᶠu₃ = matrix[@name(c.ρ), @name(f.u₃)]
    @. ∂ᶜρ_err_∂ᶠu₃ = dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(g³³(ᶠgⁱʲ))

    tracer_info = (@name(c.ρe_tot), @name(c.ρq_tot))

    MatrixFields.unrolled_foreach(tracer_info) do ρχ_name
        MatrixFields.has_field(Y, ρχ_name) || return
        ᶜχ = ρχ_name === @name(c.ρe_tot) ? ᶜh_tot : (@. lazy(specific(Y.c.ρq_tot, Y.c.ρ)))

        if use_derivative(topography_flag)
            ∂ᶜρχ_err_∂ᶜuₕ = matrix[ρχ_name, @name(c.uₕ)]
            @. ∂ᶜρχ_err_∂ᶜuₕ =
                dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(ᶠinterp(ᶜχ)) ⋅
                p.scratch.ᶠbidiagonal_matrix_ct3xct12
        end

        ∂ᶜρχ_err_∂ᶠu₃ = matrix[ρχ_name, @name(f.u₃)]
        @. ∂ᶜρχ_err_∂ᶠu₃ =
            dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(ᶠinterp(ᶜχ) * g³³(ᶠgⁱʲ))
    end

    ∂ᶠu₃_err_∂ᶜρ = matrix[@name(f.u₃), @name(c.ρ)]
    ∂ᶠu₃_err_∂ᶜρe_tot = matrix[@name(f.u₃), @name(c.ρe_tot)]

    ᶜθ_v = @. lazy(theta_v(thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
    ᶜΠ = @. lazy(TD.exner_given_pressure(thermo_params, ᶜp))
    # Exner-form PGF tendency: grad(Φ) - grad(Φ_r) + cp_d int(θ_v - θ_vr) grad(Π).
    # The reference terms satisfy grad(Φ_r(p)) + cp_d θ_vr(p) grad(Π(p)) ≡ 0
    # pointwise for ANY p field (Φ_r′ = -cp_d θ_vr Π′ by construction), so their
    # state derivative vanishes identically and the exact linearization carries
    # the full θ_v:
    #   d(PGF)·δχ = cp_d (∂θ_v/∂χ δχ) grad(Π)              [thermal buoyancy]
    #             + cp_d θ_v (κ_d Π / p) grad(∂p/∂χ δχ)    [acoustic]
    #             + cp_d θ_v grad(κ_d Π / p) ∂p/∂χ δχ.     [pressure buoyancy]
    # With grad(κ_d Π/p) = -(1 - κ_d) (κ_d Π/p) grad(p)/p and the equation of
    # state ρ = p^(1-κ_d) p₀^κ_d / (R_d θ_v), i.e.
    #   δρ/ρ = (1 - κ_d) δp/p - δθ_v/θ_v,
    # the two buoyancy terms combine into a single term proportional to the
    # thermodynamic density derivative:
    #   thermal + pressure buoyancy = -cp_d θ_v grad(Π) (∂ρ/∂χ δχ) / ρ.
    # It therefore vanishes identically in the columns perturbed at fixed ρ
    # (ρe_tot, the ρq's, and K through the velocities), where thermal and
    # pressure buoyancy are equal and opposite — cancelling them analytically
    # here avoids computing two large opposing terms — and it reduces to
    # grad(p) δρ/ρ² in the ρ column. Since cp_d θ_v κ_d Π/p = R_m T/p = 1/ρ
    # pointwise, the acoustic factor is the familiar -grad(·)/ρ operator
    # (ᶠp_grad_matrix). Sound and gravity waves are thus both treated fully
    # implicitly: acoustics couple ᶠu₃ to every thermodynamic column through
    # ∂p/∂χ, and buoyancy couples it to the ρ column.
    ᶜ∂p∂ρ = @. lazy(
        ᶜkappa_m * (T_0 * cp_d - ᶜK - ᶜΦ) + (R_d - ᶜkappa_m * cv_d) * ᶜT,
    )
    @. ∂ᶠu₃_err_∂ᶜρ =
        dtγ * (
            ᶠp_grad_matrix ⋅ DiagonalMatrixRow(ᶜ∂p∂ρ) +
            DiagonalMatrixRow(cp_d * ᶠinterp(ᶜθ_v) * ᶠgradᵥ(ᶜΠ) / ᶠinterp(ᶜρ)) ⋅
            ᶠinterp_matrix()
        )
    @. ∂ᶠu₃_err_∂ᶜρe_tot = dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(ᶜkappa_m)

    if MatrixFields.has_field(Y, @name(c.ρq_tot))
        ∂ᶠu₃_err_∂ᶜρq_tot = matrix[@name(f.u₃), @name(c.ρq_tot)]
        @. ∂ᶠu₃_err_∂ᶜρq_tot =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(ᶜ∂p∂ρq_tot)
    end

    if p.atmos.microphysics_model isa Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
    }
        for ρq_name in sedimenting_mass_names(Y)
            phase = condensate_phase(ρq_name)
            e_int_q = condensate_e_int_offset(phase, params)
            ∂cv∂q = condensate_cv_difference(phase, params)
            ∂ᶠu₃_err_∂ᶜρq = matrix[@name(f.u₃), center_state_name(ρq_name)]
            # The -R_v T term is ∂p/∂q_c at fixed q_tot: condensate replaces
            # vapor, ∂R_m/∂q_c = -R_v (R_m = (1 - q_tot) R_d + q_vap R_v).
            ᶜ∂p∂ρχ = @. lazy(
                ᶜkappa_m * (e_int_q - ∂cv∂q * (ᶜT - T_0)) - R_v * ᶜT,
            )
            @. ∂ᶠu₃_err_∂ᶜρq =
                dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(ᶜ∂p∂ρχ)
        end
    end

    ∂ᶠu₃_err_∂ᶜuₕ = matrix[@name(f.u₃), @name(c.uₕ)]
    ∂ᶠu₃_err_∂ᶠu₃ = matrix[@name(f.u₃), @name(f.u₃)]
    I_u₃ = DiagonalMatrixRow(one_C3xACT3)
    @. ∂ᶠu₃_err_∂ᶜuₕ =
        dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅ ∂ᶜK_∂ᶜuₕ
    if rs isa RayleighSponge
        @. ∂ᶠu₃_err_∂ᶠu₃ =
            dtγ * (
                ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅
                ∂ᶜK_∂ᶠu₃ +
                DiagonalMatrixRow(-β_rayleigh_u₃(rs, ᶠz, zmax) * (one_C3xACT3,))
            ) - (I_u₃,)
    else
        @. ∂ᶠu₃_err_∂ᶠu₃ =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅
            ∂ᶜK_∂ᶠu₃ - (I_u₃,)
    end
    return nothing
end

"""
    update_sedimentation_jacobian!(matrix, Y, p, dtγ)

Updates the Jacobian blocks for implicit sedimentation of condensate tracers,
including the couplings of sedimenting condensate masses to `ρq_tot` and
`ρe_tot`. Also initializes the `ρe_tot` and `ρq_tot` diagonal blocks (to
`-I`) and their mutual coupling (to zero), which diffusion and SGS mass flux
accumulate into.
"""
function update_sedimentation_jacobian!(matrix, Y, p, dtγ)
    p.atmos.microphysics_model isa DryModel && return nothing
    (; params) = p
    (; ᶜΦ) = p.core
    (; ᶜu, ᶜT) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)

    ᶜρ = Y.c.ρ
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J

    ∂ᶜρe_tot_err_∂ᶜρe_tot = matrix[@name(c.ρe_tot), @name(c.ρe_tot)]
    @. ∂ᶜρe_tot_err_∂ᶜρe_tot = zero(typeof(∂ᶜρe_tot_err_∂ᶜρe_tot)) - (I,)

    ∂ᶜρe_tot_err_∂ᶜρq_tot = matrix[@name(c.ρe_tot), @name(c.ρq_tot)]
    @. ∂ᶜρe_tot_err_∂ᶜρq_tot = zero(typeof(∂ᶜρe_tot_err_∂ᶜρq_tot))

    ∂ᶜρq_tot_err_∂ᶜρq_tot = matrix[@name(c.ρq_tot), @name(c.ρq_tot)]
    @. ∂ᶜρq_tot_err_∂ᶜρq_tot = zero(typeof(∂ᶜρq_tot_err_∂ᶜρq_tot)) - (I,)

    # This scratch variable computation could be skipped if no tracers are present
    @. p.scratch.ᶜbidiagonal_adjoint_matrix_c3 =
        dtγ * (-(ᶜprecipdivᵥ_matrix())) ⋅
        DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ)

    MatrixFields.unrolled_foreach(sedimenting_tracer_names(Y)) do ρχₚ_name
        wₚ_name = sedimentation_velocity_name(ρχₚ_name)
        ρχₚ_state_name = center_state_name(ρχₚ_name)

        ∂ᶜρχₚ_err_∂ᶜρχₚ = matrix[ρχₚ_state_name, ρχₚ_state_name]
        ᶜwₚ = MatrixFields.get_field(p.precomputed, wₚ_name)
        # TODO: come up with read-able names for the intermediate computations...
        @. p.scratch.ᶠband_matrix_wvec =
            ᶠright_bias_matrix() ⋅
            DiagonalMatrixRow(ClimaCore.Geometry.WVector(-(ᶜwₚ) / ᶜρ))
        @. ∂ᶜρχₚ_err_∂ᶜρχₚ =
            p.scratch.ᶜbidiagonal_adjoint_matrix_c3 ⋅
            p.scratch.ᶠband_matrix_wvec - (I,)

        phase = condensate_phase(ρχₚ_name)
        if !isnothing(phase)
            ∂ᶜρq_tot_err_∂ᶜρq = matrix[@name(c.ρq_tot), ρχₚ_state_name]
            @. ∂ᶜρq_tot_err_∂ᶜρq =
                p.scratch.ᶜbidiagonal_adjoint_matrix_c3 ⋅
                p.scratch.ᶠband_matrix_wvec

            # Sedimentation moves (moist air) mass: the same vtt is added
            # to Yₜ.c.ρ and Yₜ.c.ρq_tot in
            # `vertical_advection_of_water_tendency!`.
            ∂ᶜρ_err_∂ᶜρq = matrix[@name(c.ρ), ρχₚ_state_name]
            @. ∂ᶜρ_err_∂ᶜρq =
                p.scratch.ᶜbidiagonal_adjoint_matrix_c3 ⋅
                p.scratch.ᶠband_matrix_wvec

            # This block carries only the grid-mean sedimentation energy flux
            # (specific energy e_int + Φ + Kin). The EDMFX subdomain
            # corrections to the sedimentation energy flux in
            # `vertical_advection_of_water_tendency!` (water_advection.jl),
            # which replace the grid-mean thermodynamic state with
            # updraft/environment states, are treated explicitly and have no
            # Jacobian counterpart — a convergence-rate approximation in EDMF
            # columns with heavy sedimentation.
            ∂ᶜρe_tot_err_∂ᶜρq = matrix[@name(c.ρe_tot), ρχₚ_state_name]
            e_int_func = internal_energy_function(phase)
            @. ∂ᶜρe_tot_err_∂ᶜρq =
                p.scratch.ᶜbidiagonal_adjoint_matrix_c3 ⋅
                p.scratch.ᶠband_matrix_wvec ⋅
                DiagonalMatrixRow(
                    e_int_func(thermo_params, ᶜT) + ᶜΦ + $(Kin(ᶜwₚ, ᶜu)),
                )
        end
    end
    return nothing
end

# Eddy diffusivity and viscosity used by both grid-scale and SGS implicit
# diffusion. May write to ᶜtemp_scalar_3, ᶜtemp_scalar_4, and ᶜtemp_scalar_6.
function eddy_diffusivity_coefficients!(Y, p)
    (; params) = p
    (; turbconv_model, vertical_diffusion, smagorinsky_lilly) = p.atmos
    turbconv_params = CAP.turbconv_params(params)
    (; ᶜp) = p.precomputed
    ᶜK_u = p.scratch.ᶜtemp_scalar_4
    ᶜK_h = p.scratch.ᶜtemp_scalar_6
    if vertical_diffusion isa DecayWithHeightDiffusion
        ᶜK_h .= ᶜcompute_eddy_diffusivity_coefficient(Y.c.ρ, vertical_diffusion)
        ᶜK_u = ᶜK_h
    elseif vertical_diffusion isa VerticalDiffusion
        ᶜK_h .= ᶜcompute_eddy_diffusivity_coefficient(Y.c.uₕ, ᶜp, vertical_diffusion)
        ᶜK_u = ᶜK_h
    elseif is_smagorinsky_vertical(smagorinsky_lilly)
        set_smagorinsky_lilly_precomputed_quantities!(Y, p, smagorinsky_lilly)
        ᶜK_u = p.precomputed.ᶜνₜ_v
        ᶜK_h = p.precomputed.ᶜD_v
    elseif turbconv_model isa AbstractEDMF
        (; ᶜbuoygrad_stab, ᶜstrain_rate_norm) = p.precomputed
        ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
        ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_3
        ᶜmixing_length_field .= ᶜmixing_length(Y, p)
        ᶜK_u = p.scratch.ᶜtemp_scalar_4
        @. ᶜK_u = eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field)
        ᶜprandtl_nvec = @. lazy(
            turbulent_prandtl_number(params, ᶜbuoygrad_stab, ᶜstrain_rate_norm),
        )
        ᶜK_h = p.scratch.ᶜtemp_scalar_6
        @. ᶜK_h = eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec)
    end
    return (; ᶜK_u, ᶜK_h)
end

"""
    update_diffusion_jacobian!(matrix, Y, p, dtγ, diffusion_flag, eddy_diffusivities)

Updates the Jacobian blocks for implicit vertical diffusion of the grid-scale
scalars (including TKE dissipation) and momentum. No-op when diffusion is
treated explicitly.

Reuses `ᶠp_grad_matrix` as scratch space, so it must run after
`update_advection_jacobian!`.
"""
function update_diffusion_jacobian!(
    matrix,
    Y,
    p,
    dtγ,
    diffusion_flag,
    eddy_diffusivities,
)
    use_derivative(diffusion_flag) || return nothing
    (; params) = p
    (; ᶜT) = p.precomputed
    (; ᶜdiffusion_h_matrix, ᶜdiffusion_u_matrix, ᶠp_grad_matrix) = p.scratch
    (; ᶜK_u, ᶜK_h) = eddy_diffusivities
    FT = Spaces.undertype(axes(Y.c))
    T_0 = FT(CAP.T_0(params))
    R_v = FT(CAP.R_v(params))

    ᶜρ = Y.c.ρ
    ᶜkappa_m = ᶜkappa_m_field!(Y, p)
    ᶜ∂p∂ρq_tot = ᶜ∂p∂ρq_tot_field!(Y, p, ᶜkappa_m)

    # In dry configurations, the ρe_tot diagonal is initialized here (moist
    # configurations initialize it in update_sedimentation_jacobian!).
    ∂ᶜρe_tot_err_∂ᶜρe_tot = matrix[@name(c.ρe_tot), @name(c.ρe_tot)]
    if p.atmos.microphysics_model isa DryModel
        @. ∂ᶜρe_tot_err_∂ᶜρe_tot = zero(typeof(∂ᶜρe_tot_err_∂ᶜρe_tot)) - (I,)
    end

    ∂ᶠρχ_dif_flux_∂ᶜχ = ᶠp_grad_matrix
    # Harmonic-mean face interpolation of K, consistent with the diffusive
    # tendencies (see edmfx_sgs_diffusive_flux_tendency! and
    # vertical_diffusion_boundary_layer_tendency!). Smagorinsky tendencies
    # still use arithmetic interpolation, so their Jacobian does too.
    ϵK = eps(FT)
    if is_smagorinsky_vertical(p.atmos.smagorinsky_lilly)
        @. ∂ᶠρχ_dif_flux_∂ᶜχ =
            DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_h)) ⋅ ᶠgradᵥ_matrix()
    else
        @. ∂ᶠρχ_dif_flux_∂ᶜχ =
            DiagonalMatrixRow(ᶠinterp(ᶜρ) / ᶠinterp(1 / max(ᶜK_h, ϵK))) ⋅
            ᶠgradᵥ_matrix()
    end
    @. ᶜdiffusion_h_matrix = ᶜadvdivᵥ_matrix() ⋅ ∂ᶠρχ_dif_flux_∂ᶜχ
    if (
        MatrixFields.has_field(Y, @name(c.ρtke)) ||
        !isnothing(p.atmos.turbconv_model) ||
        !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
    )
        if is_smagorinsky_vertical(p.atmos.smagorinsky_lilly)
            @. ∂ᶠρχ_dif_flux_∂ᶜχ =
                DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_u)) ⋅
                ᶠgradᵥ_matrix()
        else
            @. ∂ᶠρχ_dif_flux_∂ᶜχ =
                DiagonalMatrixRow(ᶠinterp(ᶜρ) / ᶠinterp(1 / max(ᶜK_u, ϵK))) ⋅
                ᶠgradᵥ_matrix()
        end
        @. ᶜdiffusion_u_matrix = ᶜadvdivᵥ_matrix() ⋅ ∂ᶠρχ_dif_flux_∂ᶜχ
    end

    # Jacobian of the decomposed diffusive enthalpy flux
    #   F_h = -K_h ∇s_d + Σ_μ h_tot,μ (-K_h ∇q_μ)
    # (see edmfx_sgs_diffusive_flux_tendency! and
    # vertical_diffusion_boundary_layer_tendency!). The derivatives below hold
    # the h_tot,μ prefactors and the equilibrium condensate partition fixed
    # (consistent with the other approximations in this Jacobian): each block
    # is ∂(flux argument)/∂(prognostic variable), with ∂s_d/∂e_tot = cp_d/cv_m
    # through T, plus the constituent enthalpy carried by the corresponding
    # water-gradient term. The SGS mass-flux enthalpy Jacobian
    # (update_sgs_massflux_jacobian!) is not decomposed: it transports whole
    # parcels at h_tot and so does not incur the dry-air-diffusion artifact.
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜΦ) = p.core
    (; ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    cp_d = FT(CAP.cp_d(params))
    Δcv_v = FT(CAP.cv_v(params)) - FT(CAP.cv_d(params))
    e_int_v0 = FT(CAP.e_int_v0(params))
    ᶜcv_m = @. lazy(TD.cv_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))

    # The (ρe_tot, ρ) and (ρq_tot, ρ) columns are zeroed here and later
    # accumulate only the SGS mass-flux terms (update_sgs_massflux_jacobian!):
    # the diffusive fluxes' ρ-dependence — through χ = ρχ/ρ and the ρ factor
    # in ρ K ∇χ, and through the Yₜ.c.ρ counterpart of the ρq_tot diffusion —
    # is neglected, like the other frozen-coefficient approximations in this
    # Jacobian (convergence-rate impact only; the tendencies are exact).
    ∂ᶜρe_tot_err_∂ᶜρ = matrix[@name(c.ρe_tot), @name(c.ρ)]
    @. ∂ᶜρe_tot_err_∂ᶜρ = zero(typeof(∂ᶜρe_tot_err_∂ᶜρ))
    @. ∂ᶜρe_tot_err_∂ᶜρe_tot +=
        dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(cp_d / (ᶜcv_m * ᶜρ))

    if MatrixFields.has_field(Y, @name(c.ρq_tot))
        ∂ᶜρe_tot_err_∂ᶜρq_tot = matrix[@name(c.ρe_tot), @name(c.ρq_tot)]
        ∂ᶜρq_tot_err_∂ᶜρ = matrix[@name(c.ρq_tot), @name(c.ρ)]
        ∂ᶜρq_tot_err_∂ᶜρq_tot = matrix[@name(c.ρq_tot), @name(c.ρq_tot)]
        # ∂F/∂q_tot: T changes at fixed e_tot (through cv_m and e_int_v0),
        # and the vapor-gradient term carries h_tot,v = h_v + Φ.
        @. ∂ᶜρe_tot_err_∂ᶜρq_tot +=
            dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(
                (
                    TD.enthalpy_vapor(thermo_params, ᶜT) + ᶜΦ -
                    cp_d * (e_int_v0 + Δcv_v * (ᶜT - T_0)) / ᶜcv_m
                ) / ᶜρ,
            )
        @. ∂ᶜρq_tot_err_∂ᶜρ = zero(typeof(∂ᶜρq_tot_err_∂ᶜρ))
        @. ∂ᶜρq_tot_err_∂ᶜρq_tot +=
            dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(1 / ᶜρ)
    end

    if p.atmos.microphysics_model isa Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
    }
        for ρq_name in sedimenting_mass_names(Y)
            phase = condensate_phase(ρq_name)
            e_int_q = condensate_e_int_offset(phase, params)
            ∂cv∂q = condensate_cv_difference(phase, params)
            h_cond_func = enthalpy_function(phase)
            ∂ᶜρe_tot_err_∂ᶜρq =
                matrix[@name(c.ρe_tot), center_state_name(ρq_name)]
            # ∂F/∂q_cond at fixed q_tot: vapor→condensate conversion changes T
            # (latent heating enters s_d) and moves water-gradient enthalpy
            # from h_tot,v to h_tot,cond (the Φ parts cancel).
            @. ∂ᶜρe_tot_err_∂ᶜρq +=
                dtγ * ᶜdiffusion_h_matrix ⋅
                DiagonalMatrixRow(
                    (
                        cp_d * (e_int_q - ∂cv∂q * (ᶜT - T_0)) / ᶜcv_m +
                        h_cond_func(thermo_params, ᶜT) -
                        TD.enthalpy_vapor(thermo_params, ᶜT)
                    ) / ᶜρ,
                )
        end
    end

    # The microphysics tracers carry no (·, ρ) blocks (see
    # `diffusion_jacobian_blocks`), so only their diagonals are updated here.
    α_vert_diff_microphysics = CAP.α_vert_diff_tracer(params)
    MatrixFields.unrolled_foreach(sedimenting_tracer_names(Y)) do ρχ_name
        ρχ_state_name = center_state_name(ρχ_name)
        ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_state_name, ρχ_state_name]
        @. ∂ᶜρχ_err_∂ᶜρχ +=
            dtγ * α_vert_diff_microphysics * ᶜdiffusion_h_matrix ⋅
            DiagonalMatrixRow(1 / ᶜρ)
    end

    # Passive (non-water) grid-scale tracers are diffused with the unscaled
    # K_h (see edmfx_sgs_diffusive_flux_tendency! and
    # vertical_diffusion_boundary_layer_tendency!). Their diagonals receive
    # no other implicit contributions, so they are initialized here.
    MatrixFields.unrolled_foreach(passive_gs_tracer_names(Y)) do ρχ_name
        ρχ_state_name = center_state_name(ρχ_name)
        ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_state_name, ρχ_state_name]
        @. ∂ᶜρχ_err_∂ᶜρχ =
            dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(1 / ᶜρ) - (I,)
    end

    if MatrixFields.has_field(Y, @name(c.ρtke))
        turbconv_params = CAP.turbconv_params(params)
        c_d = CAP.tke_diss_coeff(turbconv_params)
        (; dt) = p
        ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
        ᶜρtke = Y.c.ρtke

        # scratch to prevent GPU Kernel parameter memory error
        ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_3
        ᶜmixing_length_field .= ᶜmixing_length(Y, p)

        # The dissipation derivative below differentiates c_d √tke / l_mix
        # with respect to tke at frozen mixing length, although l_mix itself
        # depends on tke (through l_TKE and l_N). Like the frozen K_h/K_u
        # coefficients above, this omits a ∂l_mix/∂tke chain term — a
        # convergence-rate approximation that is largest in the strongly
        # stable cells where l_N ∝ √tke dominates the mixing length.
        @inline tke_dissipation_rate_tendency(tke, mixing_length) =
            tke >= 0 ? c_d * sqrt(tke) / mixing_length : 1 / typeof(tke)(dt)
        @inline ∂tke_dissipation_rate_tendency_∂tke(tke, mixing_length) =
            tke > 0 ? c_d / (2 * mixing_length * sqrt(tke)) :
            typeof(tke)(0)

        ᶜdissipation_matrix_diagonal = p.scratch.ᶜtemp_scalar
        @. ᶜdissipation_matrix_diagonal =
            ᶜρtke * ∂tke_dissipation_rate_tendency_∂tke(
                ᶜtke,
                ᶜmixing_length_field,
            )

        ∂ᶜρtke_err_∂ᶜρ = matrix[@name(c.ρtke), @name(c.ρ)]
        ∂ᶜρtke_err_∂ᶜρtke =
            matrix[@name(c.ρtke), @name(c.ρtke)]
        @. ∂ᶜρtke_err_∂ᶜρ =
            dtγ * (
                DiagonalMatrixRow(ᶜdissipation_matrix_diagonal)
            ) ⋅ DiagonalMatrixRow(ᶜtke / Y.c.ρ)
        @. ∂ᶜρtke_err_∂ᶜρtke =
            dtγ * (
                (
                    ᶜdiffusion_u_matrix -
                    DiagonalMatrixRow(ᶜdissipation_matrix_diagonal)
                ) ⋅ DiagonalMatrixRow(1 / Y.c.ρ) - DiagonalMatrixRow(
                    tke_dissipation_rate_tendency(
                        ᶜtke,
                        ᶜmixing_length_field,
                    ),
                )
            ) - (I,)
    end

    if (
        !isnothing(p.atmos.turbconv_model) ||
        !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
    )
        ∂ᶜuₕ_err_∂ᶜuₕ = matrix[@name(c.uₕ), @name(c.uₕ)]
        @. ∂ᶜuₕ_err_∂ᶜuₕ =
            dtγ * DiagonalMatrixRow(1 / ᶜρ) ⋅ ᶜdiffusion_u_matrix - (I,)
    end
    return nothing
end

# Upwinding operators and matrices for implicit SGS vertical advection.
# `upwinding` is `Val(:first_order)` or `Val(:third_order)`.
function sgs_upwinding_operators(FT, upwinding)
    is_third_order = upwinding == Val(:third_order)
    ᶠupwind = is_third_order ? ᶠupwind3 : ᶠupwind1
    ᶠset_upwind_bcs = Operators.SetBoundaryOperator(;
        top = Operators.SetValue(zero(CT3{FT})),
        bottom = Operators.SetValue(zero(CT3{FT})),
    ) # Need to wrap ᶠupwind in this for well-defined boundaries.
    UpwindMatrixRowType =
        is_third_order ? QuaddiagonalMatrixRow : BidiagonalMatrixRow
    ᶠupwind_matrix = is_third_order ? ᶠupwind3_matrix : ᶠupwind1_matrix
    ᶠset_upwind_matrix_bcs = Operators.SetBoundaryOperator(;
        top = Operators.SetValue(zero(UpwindMatrixRowType{CT3{FT}})),
        bottom = Operators.SetValue(zero(UpwindMatrixRowType{CT3{FT}})),
    ) # Need to wrap ᶠupwind_matrix in this for well-defined boundaries.
    return (; ᶠupwind, ᶠset_upwind_bcs, ᶠupwind_matrix, ᶠset_upwind_matrix_bcs)
end

"""
    update_sgs_advection_jacobian!(matrix, Y, p, dtγ)

Updates the Jacobian blocks for implicit vertical advection of the updraft
scalars with the updraft velocity, and for implicit sedimentation of the SGS
condensate tracers (including their couplings to the updraft `q_tot`).
"""
function update_sgs_advection_jacobian!(matrix, Y, p, dtγ)
    p.atmos.turbconv_model isa PrognosticEDMFX || return nothing
    (; ᶜρʲs, ᶠu³ʲs) = p.precomputed
    FT = Spaces.undertype(axes(Y.c))
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    (; ᶠsed_tracer_advection, ᶜtridiagonal_matrix_scalar) = p.scratch

    # upwinding options for q_tot and mse
    (; ᶠupwind_matrix, ᶠset_upwind_matrix_bcs) = sgs_upwinding_operators(
        FT,
        p.atmos.numerics.edmfx_mse_q_tot_upwinding,
    )
    # upwinding options for other tracers
    tracer_upwinding_operators = sgs_upwinding_operators(
        FT,
        p.atmos.numerics.edmfx_tracer_upwinding,
    )
    ᶠtracer_upwind_matrix = tracer_upwinding_operators.ᶠupwind_matrix
    ᶠset_tracer_upwind_matrix_bcs =
        tracer_upwinding_operators.ᶠset_upwind_matrix_bcs

    # advection of q_tot and mse
    for χ_name in (@name(q_tot), @name(mse))
        χ_state_name = sgs_state_name(χ_name)
        ∂ᶜχʲ_err_∂ᶜχʲ = matrix[χ_state_name, χ_state_name]
        @. ∂ᶜχʲ_err_∂ᶜχʲ =
            dtγ * (
                DiagonalMatrixRow(ᶜadvdivᵥ(ᶠu³ʲs.:(1))) -
                ᶜadvdivᵥ_matrix() ⋅
                ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1)))
            ) - (I,)
    end

    # advection of passive tracers, e.g. chemistry tracers (no sedimentation)
    MatrixFields.unrolled_foreach(passive_sgs_tracer_names(Y)) do χ_name
        χ_state_name = sgs_state_name(χ_name)
        ∂ᶜχʲ_err_∂ᶜχʲ = matrix[χ_state_name, χ_state_name]
        @. ∂ᶜχʲ_err_∂ᶜχʲ =
            dtγ * (
                DiagonalMatrixRow(ᶜadvdivᵥ(ᶠu³ʲs.:(1))) -
                ᶜadvdivᵥ_matrix() ⋅
                ᶠset_tracer_upwind_matrix_bcs(
                    ᶠtracer_upwind_matrix(ᶠu³ʲs.:(1)),
                )
            ) - (I,)
    end

    # advection and sedimentation of microphysics tracers
    if p.atmos.microphysics_model isa Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
    }
        ᶜa = (@. lazy(draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1))))
        ᶜ∂a∂z = p.scratch.ᶜtemp_scalar_7
        @. ᶜ∂a∂z = ᶜprecipdivᵥ(ᶠinterp(ᶜJ) / ᶠJ * ᶠright_bias(Geometry.WVector(ᶜa)))
        ᶜinv_ρ̂ = (@. lazy(
            specific(
                FT(1),
                Y.c.sgsʲs.:(1).ρa,
                FT(0),
                ᶜρʲs.:(1),
                p.atmos.turbconv_model,
            ),
        ))
        MatrixFields.unrolled_foreach(
            sedimenting_sgs_tracer_names(Y),
        ) do χ_name
            wʲ_name = sgs_sedimentation_velocity_name(χ_name)
            χ_state_name = sgs_state_name(χ_name)
            ᶜwʲ = MatrixFields.get_field(p.precomputed, wʲ_name)

            # advection
            ∂ᶜχʲ_err_∂ᶜχʲ = matrix[χ_state_name, χ_state_name]
            @. ∂ᶜχʲ_err_∂ᶜχʲ =
                dtγ * (
                    DiagonalMatrixRow(ᶜadvdivᵥ(ᶠu³ʲs.:(1))) -
                    ᶜadvdivᵥ_matrix() ⋅
                    ᶠset_tracer_upwind_matrix_bcs(
                        ᶠtracer_upwind_matrix(ᶠu³ʲs.:(1)),
                    )
                ) - (I,)

            # sedimentation
            # (pull out common subexpression for performance)
            @. ᶠsed_tracer_advection =
                DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) ⋅
                ᶠright_bias_matrix() ⋅
                DiagonalMatrixRow(-Geometry.WVector(ᶜwʲ))
            @. ᶜtridiagonal_matrix_scalar =
                dtγ * ifelse(ᶜ∂a∂z < 0,
                    -(ᶜprecipdivᵥ_matrix()) ⋅ ᶠsed_tracer_advection *
                    DiagonalMatrixRow(ᶜa),
                    -DiagonalMatrixRow(ᶜa) ⋅ ᶜprecipdivᵥ_matrix() ⋅
                    ᶠsed_tracer_advection,
                )

            @. ∂ᶜχʲ_err_∂ᶜχʲ +=
                DiagonalMatrixRow(ᶜinv_ρ̂) ⋅ ᶜtridiagonal_matrix_scalar

            if !isnothing(condensate_phase(χ_name))
                ∂ᶜq_totʲ_err_∂ᶜχʲ =
                    matrix[@name(c.sgsʲs.:(1).q_tot), χ_state_name]
                @. ∂ᶜq_totʲ_err_∂ᶜχʲ =
                    DiagonalMatrixRow(ᶜinv_ρ̂) ⋅ ᶜtridiagonal_matrix_scalar
            end
        end
    end
    return nothing
end

"""
    update_sgs_diffusion_jacobian!(matrix, Y, p, dtγ, diffusion_flag, eddy_diffusivities)

Updates the Jacobian blocks for implicit vertical diffusion of the updraft
scalars. No-op when diffusion is treated explicitly.

Reuses `ᶜdiffusion_h_matrix` as scratch space, so it must run after
`update_diffusion_jacobian!`.
"""
function update_sgs_diffusion_jacobian!(
    matrix,
    Y,
    p,
    dtγ,
    diffusion_flag,
    eddy_diffusivities,
)
    p.atmos.turbconv_model isa PrognosticEDMFX || return nothing
    use_derivative(diffusion_flag) || return nothing
    # Mirror the gate of the tendency this linearizes
    # (edmfx_vertical_diffusion_tendency!): without it, the updraft scalar
    # diagonals would carry diffusion terms that have no tendency counterpart.
    p.atmos.edmfx_model.vertical_diffusion isa Val{true} || return nothing
    (; params) = p
    (; ᶜρʲs) = p.precomputed
    (; ᶜdiffusion_h_matrix) = p.scratch
    (; ᶜK_h) = eddy_diffusivities
    FT = Spaces.undertype(axes(Y.c))

    α_vert_diff_microphysics = CAP.α_vert_diff_tracer(params)
    # Harmonic-mean face K, consistent with
    # edmfx_vertical_diffusion_tendency!
    ϵK = eps(FT)
    @. ᶜdiffusion_h_matrix =
        ᶜadvdivᵥ_matrix() ⋅
        DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1)) / ᶠinterp(1 / max(ᶜK_h, ϵK))) ⋅
        ᶠgradᵥ_matrix()

    ∂ᶜmseʲ_err_∂ᶜmseʲ =
        matrix[@name(c.sgsʲs.:(1).mse), @name(c.sgsʲs.:(1).mse)]
    ∂ᶜq_totʲ_err_∂ᶜq_totʲ =
        matrix[@name(c.sgsʲs.:(1).q_tot), @name(c.sgsʲs.:(1).q_tot)]
    @. ∂ᶜmseʲ_err_∂ᶜmseʲ +=
        dtγ * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅ ᶜdiffusion_h_matrix
    @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ +=
        dtγ * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅ ᶜdiffusion_h_matrix

    if p.atmos.microphysics_model isa Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
    }
        MatrixFields.unrolled_foreach(
            sedimenting_sgs_tracer_names(Y),
        ) do χ_name
            χ_state_name = sgs_state_name(χ_name)
            ∂ᶜχʲ_err_∂ᶜχʲ = matrix[χ_state_name, χ_state_name]
            @. ∂ᶜχʲ_err_∂ᶜχʲ +=
                dtγ * α_vert_diff_microphysics *
                DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅
                ᶜdiffusion_h_matrix
        end
    end

    # Passive SGS tracers are diffused with the unscaled K_h (see
    # edmfx_vertical_diffusion_tendency!); their diagonals are initialized by
    # update_sgs_advection_jacobian!, so the diffusion term is accumulated.
    MatrixFields.unrolled_foreach(passive_sgs_tracer_names(Y)) do χ_name
        χ_state_name = sgs_state_name(χ_name)
        ∂ᶜχʲ_err_∂ᶜχʲ = matrix[χ_state_name, χ_state_name]
        @. ∂ᶜχʲ_err_∂ᶜχʲ +=
            dtγ * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅ ᶜdiffusion_h_matrix
    end
    return nothing
end

"""
    update_sgs_entr_detr_jacobian!(matrix, Y, p, dtγ)

Updates the Jacobian blocks for implicit entrainment of the updraft scalars
(entrainment and detrainment rates are treated explicitly).
"""
function update_sgs_entr_detr_jacobian!(matrix, Y, p, dtγ)
    p.atmos.turbconv_model isa PrognosticEDMFX || return nothing
    (; ᶜturb_entrʲs, ᶜentr_vel_scaleʲs, ᶜarea_bounding_entr_detrʲs, ᶜuʲs) =
        p.precomputed
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜentrʲ = @. lazy(
        compute_entrainment(
            ᶜentr_vel_scaleʲs.:(1),
            ᶜarea_bounding_entr_detrʲs.:(1),
            get_physical_w(ᶜuʲs.:(1), ᶜlg),
        ),
    )

    # Entrainment relaxation of updraft scalars: the implicit tendency
    # (edmfx_entr_detr_tendency!) applies (ε + ε_turb) * (χ⁰ - χʲ) to each
    # updraft scalar χʲ. The diagonal includes both the direct dependence,
    # ∂/∂χʲ = -(ε + ε_turb), and the feedback through the relaxation target,
    # ∂χ⁰/∂χʲ = -w ρaʲ/ρa⁰ (the exact derivative of the regularized `specific`
    # that diagnoses χ⁰ from the domain decomposition), which scales the
    # diagonal by (1 + w ρaʲ/ρa⁰). Only the entrainment rates themselves are
    # treated explicitly.
    turbconv_model = p.atmos.turbconv_model
    ᶜrelax_rateʲ = @. lazy(
        (ᶜentrʲ + ᶜturb_entrʲs.:(1)) * (
            1 + env_relaxation_feedback(
                Y.c.sgsʲs.:(1).ρa,
                ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model),
                Y.c.ρ,
                turbconv_model,
            )
        ),
    )

    # All advected updraft scalars are entrained the same way: q_tot, mse,
    # sedimenting tracers (masses and number concentrations), and passive
    # tracers (e.g. chemistry tracers).
    MatrixFields.unrolled_foreach(advected_sgs_scalar_names(Y)) do χ_name
        χ_state_name = sgs_state_name(χ_name)
        ∂ᶜχʲ_err_∂ᶜχʲ = matrix[χ_state_name, χ_state_name]
        @. ∂ᶜχʲ_err_∂ᶜχʲ -= dtγ * DiagonalMatrixRow(ᶜrelax_rateʲ)
    end
    return nothing
end

"""
    update_sgs_boundary_condition_jacobian!(matrix, Y, p, dtγ)

Updates the Jacobian blocks for the surface mass-flux boundary condition at
the first interior level.

The boundary condition contributes
`∂F_BC/∂mse[1] = ∂F_BC/∂q_tot[1] = -mass_flux_source/ρa_floor`, where
`ρa_floor = max(ρa, ρ·a_min)`. We build a level-1-only rate field (zero
elsewhere) and add it as a diagonal.
"""
function update_sgs_boundary_condition_jacobian!(matrix, Y, p, dtγ)
    p.atmos.turbconv_model isa PrognosticEDMFX || return nothing
    (; params) = p
    (; ᶜρʲs) = p.precomputed
    FT = Spaces.undertype(axes(Y.c))

    turbconv_params = CAP.turbconv_params(params)
    a_min = CAP.min_area(turbconv_params)
    ᶜsfc_bc_rate = p.scratch.ᶜtemp_scalar
    @. ᶜsfc_bc_rate = FT(0)
    ᶜsfc_bc_rate_first =
        Fields.field_values(Fields.level(ᶜsfc_bc_rate, 1))
    ρʲ_int_val =
        Fields.field_values(Fields.level(ᶜρʲs.:(1), 1))
    ρaʲ_int_val = Fields.field_values(
        Fields.level(Y.c.sgsʲs.:(1).ρa, 1),
    )
    mass_flux_source_val = Fields.field_values(
        Fields.level(p.precomputed.sfc_mass_flux_sourceʲs.:(1), 1),
    )
    @. ᶜsfc_bc_rate_first =
        mass_flux_source_val /
        max(ρaʲ_int_val, ρʲ_int_val * FT(a_min))

    ∂ᶜmseʲ_err_∂ᶜmseʲ =
        matrix[@name(c.sgsʲs.:(1).mse), @name(c.sgsʲs.:(1).mse)]
    ∂ᶜq_totʲ_err_∂ᶜq_totʲ =
        matrix[@name(c.sgsʲs.:(1).q_tot), @name(c.sgsʲs.:(1).q_tot)]
    @. ∂ᶜmseʲ_err_∂ᶜmseʲ -=
        dtγ * DiagonalMatrixRow(ᶜsfc_bc_rate)
    @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ -=
        dtγ * DiagonalMatrixRow(ᶜsfc_bc_rate)
    return nothing
end

"""
    update_sgs_massflux_jacobian!(matrix, Y, p, dtγ, diffusion_flag)

Updates the Jacobian blocks for the contributions of the SGS mass flux to the
grid-mean scalars.
"""
function update_sgs_massflux_jacobian!(matrix, Y, p, dtγ, diffusion_flag)
    (
        p.atmos.turbconv_model isa PrognosticEDMFX &&
        p.atmos.edmfx_model.sgs_mass_flux isa Val{true}
    ) || return nothing
    (; params) = p
    (; ᶜΦ) = p.core
    (; ᶜρʲs, ᶠu³ʲs, ᶜKʲs) = p.precomputed
    (; ᶠu³, ᶜK, ᶜT, ᶜh_tot) = p.precomputed
    (; ᶠbidiagonal_matrix_ct3) = p.scratch

    FT = Spaces.undertype(axes(Y.c))
    cv_d = FT(CAP.cv_d(params))
    T_0 = FT(CAP.T_0(params))
    R_d = FT(CAP.R_d(params))
    R_v = FT(CAP.R_v(params))
    cp_d = FT(CAP.cp_d(params))

    ᶜρ = Y.c.ρ
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶠgⁱʲ = Fields.local_geometry_field(Y.f).gⁱʲ

    # If diffusion is explicit, zero-initialize (ρe_tot, ρ) and
    # (ρq_tot, ρ) here so both blocks can safely use +=.
    ∂ᶜρe_tot_err_∂ᶜρ = matrix[@name(c.ρe_tot), @name(c.ρ)]
    ∂ᶜρq_tot_err_∂ᶜρ = matrix[@name(c.ρq_tot), @name(c.ρ)]
    if !use_derivative(diffusion_flag)
        @. ∂ᶜρe_tot_err_∂ᶜρ = zero(typeof(∂ᶜρe_tot_err_∂ᶜρ))
        @. ∂ᶜρq_tot_err_∂ᶜρ = zero(typeof(∂ᶜρq_tot_err_∂ᶜρ))
    end

    ∂ᶜupdraft_mass_flux_∂ᶜscalar = ᶠbidiagonal_matrix_ct3
    @. ∂ᶜupdraft_mass_flux_∂ᶜscalar =
        DiagonalMatrixRow(
            (ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) * (ᶠu³ʲs.:(1) - ᶠu³),
        ) ⋅ ᶠinterp_matrix() ⋅
        DiagonalMatrixRow(Y.c.sgsʲs.:(1).ρa / ᶜρʲs.:(1))
    @. p.scratch.ᶜtridiagonal_matrix_scalar =
        dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar

    # Derivative of total energy tendency with respect to updraft MSE
    ## grid-mean ρe_tot
    ᶜkappa_m = ᶜkappa_m_field!(Y, p)
    ᶜ∂p∂ρq_tot = ᶜ∂p∂ρq_tot_field!(Y, p, ᶜkappa_m)

    ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
    @. ∂ᶜρe_tot_err_∂ᶜρ +=
        p.scratch.ᶜtridiagonal_matrix_scalar ⋅
        DiagonalMatrixRow(
            (
                -(ᶜh_tot) +
                ᶜkappa_m * (T_0 * cp_d - ᶜK - ᶜΦ) +
                (R_d - ᶜkappa_m * cv_d) * ᶜT
            ) / ᶜρ,
        )

    ∂ᶜρe_tot_err_∂ᶜρq_tot = matrix[@name(c.ρe_tot), @name(c.ρq_tot)]
    @. ∂ᶜρe_tot_err_∂ᶜρq_tot +=
        p.scratch.ᶜtridiagonal_matrix_scalar ⋅
        DiagonalMatrixRow(ᶜ∂p∂ρq_tot / ᶜρ)

    if p.atmos.microphysics_model isa Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
    }
        for ρq_name in sedimenting_mass_names(Y)
            phase = condensate_phase(ρq_name)
            e_int_q = condensate_e_int_offset(phase, params)
            ∂cv∂q = condensate_cv_difference(phase, params)
            ∂ᶜρe_tot_err_∂ᶜρq =
                matrix[@name(c.ρe_tot), center_state_name(ρq_name)]
            @. ∂ᶜρe_tot_err_∂ᶜρq +=
                p.scratch.ᶜtridiagonal_matrix_scalar ⋅
                DiagonalMatrixRow(
                    (ᶜkappa_m * (e_int_q - ∂cv∂q * (ᶜT - T_0)) - R_v * ᶜT) / ᶜρ,
                )
        end
    end

    ∂ᶜρe_tot_err_∂ᶜρe_tot = matrix[@name(c.ρe_tot), @name(c.ρe_tot)]
    @. ∂ᶜρe_tot_err_∂ᶜρe_tot +=
        p.scratch.ᶜtridiagonal_matrix_scalar ⋅
        DiagonalMatrixRow((1 + ᶜkappa_m) / ᶜρ)

    ∂ᶜρe_tot_err_∂ᶜmseʲ =
        matrix[@name(c.ρe_tot), @name(c.sgsʲs.:(1).mse)]
    @. ∂ᶜρe_tot_err_∂ᶜmseʲ =
        -(p.scratch.ᶜtridiagonal_matrix_scalar)

    ## grid-mean ρq_tot
    @. ∂ᶜρq_tot_err_∂ᶜρ +=
        p.scratch.ᶜtridiagonal_matrix_scalar ⋅
        DiagonalMatrixRow(-(ᶜq_tot) / ᶜρ)

    ∂ᶜρq_tot_err_∂ᶜρq_tot = matrix[@name(c.ρq_tot), @name(c.ρq_tot)]
    @. ∂ᶜρq_tot_err_∂ᶜρq_tot +=
        p.scratch.ᶜtridiagonal_matrix_scalar ⋅
        DiagonalMatrixRow(1 / ᶜρ)

    ∂ᶜρq_tot_err_∂ᶜq_totʲ =
        matrix[@name(c.ρq_tot), @name(c.sgsʲs.:(1).q_tot)]
    @. ∂ᶜρq_tot_err_∂ᶜq_totʲ =
        -(p.scratch.ᶜtridiagonal_matrix_scalar)

    # grid-mean ∂/∂(u₃ʲ)
    ∂ᶜρe_tot_err_∂ᶠu₃ = matrix[@name(c.ρe_tot), @name(f.u₃)]
    @. ∂ᶜρe_tot_err_∂ᶠu₃ +=
        dtγ * ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(
            ᶠinterp(
                (Y.c.sgsʲs.:(1).mse + ᶜKʲs.:(1) - ᶜh_tot) *
                ᶜρʲs.:(1) *
                ᶜJ *
                draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
            ) / ᶠJ * (g³³(ᶠgⁱʲ)),
        )

    ∂ᶜρq_tot_err_∂ᶠu₃ = matrix[@name(c.ρq_tot), @name(f.u₃)]
    @. ∂ᶜρq_tot_err_∂ᶠu₃ +=
        dtγ * ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(
            ᶠinterp(
                (Y.c.sgsʲs.:(1).q_tot - ᶜq_tot) *
                ᶜρʲs.:(1) *
                ᶜJ *
                draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
            ) / ᶠJ * (g³³(ᶠgⁱʲ)),
        )

    # grid-mean tracers
    # The implicit SGS tracer fluxes are difference-form
    # corrections ρᵏaᵏ(u³ᵏ - u³)(χᵏ - χ) (the grid-mean advection
    # -∇·(ρ u³ χ) is handled explicitly). As for mse and q_tot,
    # the derivatives are linearized with the central interpolant
    # (exact for the default :none upwinding), and the environment
    # contributions are neglected: (χ⁰ - χ) and (u³⁰ - u³) are each
    # O(aʲ), so every environment Jacobian entry is O(aʲ²) while
    # the updraft entries are O(aʲ). The updraft entries reuse
    # ᶜtridiagonal_matrix_scalar, which is the same for every scalar
    # transported by the updraft flux.
    if p.atmos.microphysics_model isa Union{
        NonEquilibriumMicrophysics1M,
        NonEquilibriumMicrophysics2M,
    }
        MatrixFields.unrolled_foreach(
            sedimenting_tracer_names(Y),
        ) do ρχ_name
            ρχ_state_name = center_state_name(ρχ_name)
            χʲ_name = get_χʲ_name_from_ρχ_name(ρχ_state_name)
            MatrixFields.has_field(Y, χʲ_name) || return
            ᶜρχ = MatrixFields.get_field(Y, ρχ_state_name)
            ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)

            ∂ᶜρχ_err_∂ᶜχʲ = matrix[ρχ_state_name, χʲ_name]
            @. ∂ᶜρχ_err_∂ᶜχʲ =
                -(p.scratch.ᶜtridiagonal_matrix_scalar)

            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_state_name, ρχ_state_name]
            @. ∂ᶜρχ_err_∂ᶜρχ +=
                p.scratch.ᶜtridiagonal_matrix_scalar ⋅
                DiagonalMatrixRow(1 / ᶜρ)

            ∂ᶜρχ_err_∂ᶠu₃ = matrix[ρχ_state_name, @name(f.u₃)]
            @. ∂ᶜρχ_err_∂ᶠu₃ =
                dtγ * ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(
                    ᶠinterp(
                        (ᶜχʲ - specific(ᶜρχ, Y.c.ρ)) *
                        ᶜρʲs.:(1) *
                        ᶜJ *
                        draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                    ) / ᶠJ * (g³³(ᶠgⁱʲ)),
                )
        end
    end
    return nothing
end

function update_jacobian!(alg::ManualSparseJacobian, cache, Y, p, dtγ, t)
    (; topography_flag, diffusion_flag) = cache.derivative_flags
    (; matrix) = cache

    # Ordering contract between the process updates:
    #   - update_advection_jacobian! fills ᶠp_grad_matrix, which
    #     update_diffusion_jacobian! reuses as scratch space.
    #   - The ρe_tot and ρq_tot diagonal blocks and their couplings to ρ and
    #     to each other are initialized by update_sedimentation_jacobian!
    #     (moist), update_diffusion_jacobian! (dry, and couplings to ρ), or
    #     update_sgs_massflux_jacobian! (couplings to ρ with explicit
    #     diffusion), and accumulated into by the updates that follow.
    #   - The updraft mse and q_tot diagonal blocks are set by
    #     update_sgs_advection_jacobian! and accumulated into by the SGS
    #     diffusion, entrainment, and boundary condition updates.
    #   - The eddy diffusivities are computed once and shared between the
    #     grid-scale and SGS diffusion updates.
    update_advection_jacobian!(matrix, Y, p, dtγ, topography_flag)
    update_sedimentation_jacobian!(matrix, Y, p, dtγ)
    eddy_diffusivities =
        use_derivative(diffusion_flag) ? eddy_diffusivity_coefficients!(Y, p) :
        nothing
    update_diffusion_jacobian!(
        matrix,
        Y,
        p,
        dtγ,
        diffusion_flag,
        eddy_diffusivities,
    )
    update_sgs_advection_jacobian!(matrix, Y, p, dtγ)
    update_sgs_diffusion_jacobian!(
        matrix,
        Y,
        p,
        dtγ,
        diffusion_flag,
        eddy_diffusivities,
    )
    update_sgs_entr_detr_jacobian!(matrix, Y, p, dtγ)
    update_sgs_boundary_condition_jacobian!(matrix, Y, p, dtγ)
    update_sgs_massflux_jacobian!(matrix, Y, p, dtγ, diffusion_flag)

    # NOTE: All velocity tendency derivatives should be set BEFORE this call.
    zero_velocity_jacobian!(matrix, Y, p, t)
    return nothing
end

invert_jacobian!(::ManualSparseJacobian, cache, ΔY, R) =
    LinearAlgebra.ldiv!(ΔY, cache.matrix, R)
