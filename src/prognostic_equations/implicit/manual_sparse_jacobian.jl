import LinearAlgebra: I, Adjoint

using ClimaCore.MatrixFields
import ClimaCore.MatrixFields: @name

abstract type DerivativeFlag end
struct UseDerivative <: DerivativeFlag end
struct IgnoreDerivative <: DerivativeFlag end

DerivativeFlag(value) = value ? UseDerivative() : IgnoreDerivative()
DerivativeFlag(mode::AbstractTimesteppingMode) =
    DerivativeFlag(mode == Implicit())

use_derivative(::UseDerivative) = true
use_derivative(::IgnoreDerivative) = false

"""
    ManualSparseJacobian(
        topography_flag,
        diffusion_flag,
        sgs_advection_flag,
        sgs_entr_detr_flag,
        sgs_mass_flux_flag,
        sgs_nh_pressure_flag,
        sgs_vertdiff_flag,
        acoustic_diagonal_flag,
        approximate_solve_iters,
    )

A [`JacobianAlgorithm`](@ref) that approximates the Jacobian using analytically
derived tendency derivatives and inverts it using a specialized nested linear
solver. Certain groups of derivatives can be toggled on or off by setting their
`DerivativeFlag`s to either `UseDerivative` or `IgnoreDerivative`.

# Arguments

- `topography_flag::DerivativeFlag`: whether the derivative of vertical
  contravariant velocity with respect to horizontal covariant velocity should be
  computed
- `diffusion_flag::DerivativeFlag`: whether the derivatives of the grid-scale
  diffusion tendency should be computed
- `sgs_advection_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale advection tendency should be computed
- `sgs_entr_detr_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale entrainment and detrainment tendencies should be computed
- `sgs_mass_flux_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale mass flux tendency should be computed
- `sgs_nh_pressure_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale non-hydrostatic pressure drag tendency should be computed
- `sgs_vertdiff_flag::DerivativeFlag`: whether the derivatives of the
  subgrid-scale vertical diffusion tendency should be computed
- `acoustic_diagonal_flag::DerivativeFlag`: whether to add a diagonal
  approximation of the horizontal acoustic Schur complement to the `(ρ,ρ)`,
  `(uₕ,uₕ)`, and `(ρe_tot,ρe_tot)` blocks, improving convergence for fully
  implicit solves where horizontal acoustic/gravity wave stiffness dominates
- `approximate_solve_iters::Int`: number of iterations to take for the
  approximate linear solve required when the `diffusion_flag` is `UseDerivative`
"""
struct ManualSparseJacobian{F1, F2, F3, F4, F5, F6, F7, F8} <: SparseJacobian
    topography_flag::F1
    diffusion_flag::F2
    sgs_advection_flag::F3
    sgs_entr_detr_flag::F4
    sgs_mass_flux_flag::F5
    sgs_nh_pressure_flag::F6
    sgs_vertdiff_flag::F7
    acoustic_diagonal_flag::F8
    approximate_solve_iters::Int
end

"""
    HelmholtzPreconditionerState

Stores state needed by `helmholtz_correction!` during `invert_jacobian!`.
These quantities are captured from `update_jacobian!` so that the Helmholtz
solve in the preconditioner has access to the current Jacobian parameters.
"""
mutable struct HelmholtzPreconditionerState{FT, F, GB}
    dtγ::FT
    ᶜα_acoustic::F       # diagonal: 1 + dtγ²·cs²·2π²/Δx²
    ᶜcs²::F              # sound speed squared: γ_d·ᶜp/ᶜρ
    ᶜρ::F                # density
    ᶜe_tot::F            # specific total energy (for ρe_tot correction)
    ghost_buffer_c::GB   # DSS buffer for center vector fields (uₕ)
    n_helmholtz_iters::Int
end

function jacobian_cache(alg::ManualSparseJacobian, Y, atmos)
    (;
        topography_flag,
        diffusion_flag,
        sgs_advection_flag,
        sgs_mass_flux_flag,
        acoustic_diagonal_flag,
        approximate_solve_iters,
    ) = alg
    FT = Spaces.undertype(axes(Y.c))

    DiagonalRow = DiagonalMatrixRow{FT}
    TridiagonalRow = TridiagonalMatrixRow{FT}
    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    TridiagonalRow_ACT12 = TridiagonalMatrixRow{Adjoint{FT, CT12{FT}}}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}
    BidiagonalRow_C3xACT12 =
        BidiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CT12{FT})')}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CT3{FT})')}

    is_in_Y(name) = MatrixFields.has_field(Y, name)

    ρq_tot_if_available = is_in_Y(@name(c.ρq_tot)) ? (@name(c.ρq_tot),) : ()
    ρtke_if_available =
        is_in_Y(@name(c.ρtke)) ? (@name(c.ρtke),) : ()
    sfc_if_available = is_in_Y(@name(sfc)) ? (@name(sfc),) : ()

    condensate_mass_names = (
        @name(c.ρq_liq),
        @name(c.ρq_ice),
        @name(c.ρq_rai),
        @name(c.ρq_sno),
    )
    available_condensate_mass_names =
        MatrixFields.unrolled_filter(is_in_Y, condensate_mass_names)
    condensate_names = (
        condensate_mass_names...,
        @name(c.ρn_liq),
        @name(c.ρn_rai),
        # P3 frozen
        @name(c.ρn_ice), @name(c.ρq_rim), @name(c.ρb_rim),
    )
    available_condensate_names =
        MatrixFields.unrolled_filter(is_in_Y, condensate_names)
    available_tracer_names =
        (ρq_tot_if_available..., available_condensate_names...)

    # we define the list of condensate masses separately because ρa and q_tot
    # depend on the masses via sedimentation
    sgs_condensate_mass_names = (
        @name(c.sgsʲs.:(1).q_liq),
        @name(c.sgsʲs.:(1).q_ice),
        @name(c.sgsʲs.:(1).q_rai),
        @name(c.sgsʲs.:(1).q_sno),
    )
    available_sgs_condensate_mass_names =
        MatrixFields.unrolled_filter(is_in_Y, sgs_condensate_mass_names)

    sgs_condensate_names =
        (sgs_condensate_mass_names..., @name(c.sgsʲs.:(1).n_liq), @name(c.sgsʲs.:(1).n_rai))
    available_sgs_condensate_names =
        MatrixFields.unrolled_filter(is_in_Y, sgs_condensate_names)

    sgs_scalar_names =
        (
            sgs_condensate_names...,
            @name(c.sgsʲs.:(1).q_tot),
            @name(c.sgsʲs.:(1).mse),
            @name(c.sgsʲs.:(1).ρa)
        )
    available_sgs_scalar_names =
        MatrixFields.unrolled_filter(is_in_Y, sgs_scalar_names)

    sgs_u³_if_available =
        is_in_Y(@name(f.sgsʲs.:(1).u₃)) ? (@name(f.sgsʲs.:(1).u₃),) : ()

    # Note: We have to use FT(-1) * I instead of -I because inv(-1) == -1.0,
    # which means that multiplying inv(-1) by a Float32 will yield a Float64.
    identity_blocks = if use_derivative(acoustic_diagonal_flag)
        (
            (@name(c.ρ), @name(c.ρ)) => similar(Y.c, DiagonalRow),
            MatrixFields.unrolled_map(
                name -> (name, name) => FT(-1) * I,
                (sfc_if_available...,),
            )...,
        )
    else
        MatrixFields.unrolled_map(
            name -> (name, name) => FT(-1) * I,
            (@name(c.ρ), sfc_if_available...),
        )
    end

    active_scalar_names = (@name(c.ρ), @name(c.ρe_tot), ρq_tot_if_available...)
    advection_blocks = (
        (
            use_derivative(topography_flag) ?
            MatrixFields.unrolled_map(
                name ->
                    (name, @name(c.uₕ)) =>
                        similar(Y.c, TridiagonalRow_ACT12),
                active_scalar_names,
            ) : ()
        )...,
        MatrixFields.unrolled_map(
            name -> (name, @name(f.u₃)) => similar(Y.c, BidiagonalRow_ACT3),
            active_scalar_names,
        )...,
        MatrixFields.unrolled_map(
            name -> (@name(f.u₃), name) => similar(Y.f, BidiagonalRow_C3),
            active_scalar_names,
        )...,
        MatrixFields.unrolled_map(
            name -> (@name(f.u₃), name) => similar(Y.f, BidiagonalRow_C3),
            available_condensate_mass_names,
        )...,
        (@name(f.u₃), @name(c.uₕ)) => similar(Y.f, BidiagonalRow_C3xACT12),
        (@name(f.u₃), @name(f.u₃)) => similar(Y.f, TridiagonalRow_C3xACT3),
    )

    diffused_scalar_names = (@name(c.ρe_tot), available_tracer_names...)
    diffusion_blocks = if use_derivative(diffusion_flag)
        (
            MatrixFields.unrolled_map(
                name -> (name, @name(c.ρ)) => similar(Y.c, TridiagonalRow),
                (diffused_scalar_names..., ρtke_if_available...),
            )...,
            MatrixFields.unrolled_map(
                name -> (name, name) => similar(Y.c, TridiagonalRow),
                (diffused_scalar_names..., ρtke_if_available...),
            )...,
            (
                is_in_Y(@name(c.ρq_tot)) ?
                (
                    (@name(c.ρe_tot), @name(c.ρq_tot)) =>
                        similar(Y.c, TridiagonalRow),
                ) : ()
            )...,
            MatrixFields.unrolled_map(
                name -> (@name(c.ρe_tot), name) => similar(Y.c, TridiagonalRow),
                available_condensate_mass_names,
            )...,
            MatrixFields.unrolled_map(
                name -> (@name(c.ρq_tot), name) => similar(Y.c, TridiagonalRow),
                available_condensate_mass_names,
            )...,
            (@name(c.uₕ), @name(c.uₕ)) =>
                !isnothing(atmos.turbconv_model) ||
                    !disable_momentum_vertical_diffusion(
                        atmos.vertical_diffusion,
                    ) ? similar(Y.c, TridiagonalRow) :
                use_derivative(acoustic_diagonal_flag) ?
                    similar(Y.c, DiagonalRow) : FT(-1) * I,
        )
    elseif atmos.microphysics_model isa DryModel
        if use_derivative(acoustic_diagonal_flag)
            (
                (@name(c.ρe_tot), @name(c.ρe_tot)) => similar(Y.c, DiagonalRow),
                MatrixFields.unrolled_map(
                    name -> (name, name) => FT(-1) * I,
                    (ρtke_if_available...,),
                )...,
                (@name(c.uₕ), @name(c.uₕ)) => similar(Y.c, DiagonalRow),
            )
        else
            MatrixFields.unrolled_map(
                name -> (name, name) => FT(-1) * I,
                (diffused_scalar_names..., ρtke_if_available..., @name(c.uₕ)),
            )
        end
    else
        (
            MatrixFields.unrolled_map(
                name -> (name, name) => similar(Y.c, TridiagonalRow),
                diffused_scalar_names,
            )...,
            MatrixFields.unrolled_map(
                name -> (@name(c.ρe_tot), name) => similar(Y.c, TridiagonalRow),
                available_condensate_mass_names,
            )...,
            MatrixFields.unrolled_map(
                name -> (@name(c.ρq_tot), name) => similar(Y.c, TridiagonalRow),
                available_condensate_mass_names,
            )...,
            (@name(c.ρe_tot), @name(c.ρq_tot)) =>
                similar(Y.c, TridiagonalRow),
            MatrixFields.unrolled_map(
                name -> (name, name) => FT(-1) * I,
                (ρtke_if_available...,),
            )...,
            (@name(c.uₕ), @name(c.uₕ)) =>
                use_derivative(acoustic_diagonal_flag) ?
                    similar(Y.c, DiagonalRow) : FT(-1) * I,
        )
    end

    sgs_advection_blocks = if atmos.turbconv_model isa PrognosticEDMFX
        if use_derivative(sgs_advection_flag)
            (
                MatrixFields.unrolled_map(
                    name -> (name, name) => similar(Y.c, TridiagonalRow),
                    available_sgs_scalar_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (@name(c.sgsʲs.:(1).q_tot), name) =>
                            similar(Y.c, TridiagonalRow),
                    available_sgs_condensate_mass_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (@name(c.sgsʲs.:(1).ρa), name) => similar(Y.c, TridiagonalRow),
                    available_sgs_condensate_mass_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (@name(c.sgsʲs.:(1).mse), name) => similar(Y.c, DiagonalRow),
                    available_sgs_condensate_mass_names,
                )...,
                (@name(c.sgsʲs.:(1).mse), @name(c.sgsʲs.:(1).q_tot)) =>
                    similar(Y.c, DiagonalRow),
                (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).q_tot)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).mse)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) => FT(-1) * I,
            )
        else
            # When implicit microphysics is active, some SGS scalar entries
            # need a DiagonalRow so that update_microphysics_jacobian! can
            # increment them.  UniformScaling is not incrementable in-place.
            needs_implicit_micro =
                atmos.microphysics_tendency_timestepping == Implicit()
            # 0M EDMF writes to q_tot and ρa; 1M EDMF writes to
            # condensate species (q_liq, q_ice, q_rai, q_sno).
            sgs_micro_names =
                needs_implicit_micro ?
                (
                    (
                        atmos.microphysics_model isa EquilibriumMicrophysics0M ?
                        (
                            @name(c.sgsʲs.:(1).q_tot),
                            @name(c.sgsʲs.:(1).ρa),
                        ) : ()
                    )...,
                    (
                        atmos.microphysics_model isa NonEquilibriumMicrophysics ?
                        sgs_condensate_mass_names : ()
                    )...,
                ) : ()
            (
                MatrixFields.unrolled_map(
                    name ->
                        (name, name) =>
                            name in sgs_micro_names ?
                            similar(Y.c, DiagonalRow) : FT(-1) * I,
                    available_sgs_scalar_names,
                )...,
                (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) => FT(-1) * I,
            )
        end
    else
        ()
    end

    sgs_massflux_blocks = if atmos.turbconv_model isa PrognosticEDMFX
        if use_derivative(sgs_mass_flux_flag)
            (
                MatrixFields.unrolled_map(
                    name ->
                        (name, get_χʲ_name_from_ρχ_name(name)) =>
                            similar(Y.c, TridiagonalRow),
                    available_tracer_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (name, @name(c.sgsʲs.:(1).ρa)) =>
                            similar(Y.c, TridiagonalRow),
                    available_tracer_names,
                )...,
                MatrixFields.unrolled_map(
                    name ->
                        (name, @name(f.u₃)) =>
                            similar(Y.c, BidiagonalRow_ACT3),
                    available_condensate_names,
                )...,
                (@name(c.ρe_tot), @name(c.sgsʲs.:(1).mse)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)) =>
                    similar(Y.c, TridiagonalRow),
            )
        else
            ()
        end
    else
        ()
    end

    matrix = MatrixFields.FieldMatrix(
        identity_blocks...,
        sgs_advection_blocks...,
        advection_blocks...,
        diffusion_blocks...,
        sgs_massflux_blocks...,
    )

    mass_and_surface_names = (@name(c.ρ), sfc_if_available...)
    available_scalar_names = (
        mass_and_surface_names...,
        available_tracer_names...,
        @name(c.ρe_tot),
        ρtke_if_available...,
        available_sgs_scalar_names...,
    )

    velocity_alg = MatrixFields.BlockLowerTriangularSolve(
        @name(c.uₕ),
        sgs_u³_if_available...,
    )
    full_alg =
        if use_derivative(diffusion_flag) ||
           use_derivative(sgs_advection_flag) ||
           !(atmos.microphysics_model isa DryModel)
            gs_scalar_subalg = if !(atmos.microphysics_model isa DryModel)
                MatrixFields.BlockLowerTriangularSolve(
                    available_condensate_mass_names...,
                    alg₂ = MatrixFields.BlockLowerTriangularSolve(
                        @name(c.ρq_tot),
                    ),
                )
            else
                MatrixFields.BlockDiagonalSolve()
            end
            scalar_subalg =
                if atmos.turbconv_model isa PrognosticEDMFX &&
                   use_derivative(sgs_advection_flag)
                    MatrixFields.BlockLowerTriangularSolve(
                        available_sgs_condensate_names...;
                        alg₂ = MatrixFields.BlockLowerTriangularSolve(
                            @name(c.sgsʲs.:(1).q_tot);
                            alg₂ = MatrixFields.BlockLowerTriangularSolve(
                                @name(c.sgsʲs.:(1).mse);
                                alg₂ = MatrixFields.BlockLowerTriangularSolve(
                                    @name(c.sgsʲs.:(1).ρa);
                                    alg₂ = gs_scalar_subalg,
                                ),
                            ),
                        ),
                    )
                else
                    gs_scalar_subalg
                end
            scalar_alg = MatrixFields.BlockLowerTriangularSolve(
                mass_and_surface_names...;
                alg₂ = scalar_subalg,
            )
            MatrixFields.ApproximateBlockArrowheadIterativeSolve(
                available_scalar_names...;
                alg₁ = scalar_alg,
                alg₂ = velocity_alg,
                P_alg₁ = MatrixFields.MainDiagonalPreconditioner(),
                n_iters = approximate_solve_iters,
            )
        else
            MatrixFields.BlockArrowheadSolve(
                available_scalar_names...;
                alg₂ = velocity_alg,
            )
        end

    matrix_cache = MatrixFields.FieldMatrixWithSolver(matrix, Y, full_alg)

    # Helmholtz preconditioner state and scratch fields
    if use_derivative(acoustic_diagonal_flag)
        ᶜscalar_field() = similar(Y.c, FT)
        helmholtz_state = HelmholtzPreconditionerState(
            FT(0),                  # dtγ (updated in update_jacobian!)
            ᶜscalar_field(),        # ᶜα_acoustic
            ᶜscalar_field(),        # ᶜcs²
            ᶜscalar_field(),        # ᶜρ
            ᶜscalar_field(),        # ᶜe_tot
            nothing,                # ghost_buffer_c (set in update_jacobian!)
            10,                     # n_helmholtz_iters
        )
        helmholtz_scratch = (;
            ᶜhelmholtz_ρ = ᶜscalar_field(),
            ᶜhelmholtz_rhs = ᶜscalar_field(),
            ᶜhelmholtz_laplacian = ᶜscalar_field(),
            ᶜhelmholtz_dss_buffer = Spaces.create_dss_buffer(
                ᶜscalar_field(),
            ),
        )
        return (;
            matrix = matrix_cache,
            helmholtz_state,
            helmholtz_scratch,
        )
    else
        return (;
            matrix = matrix_cache,
            helmholtz_state = nothing,
            helmholtz_scratch = nothing,
        )
    end
end

# TODO: There are a few for loops in this function. This is because
# using unrolled_foreach allocates (breaks the flame tests)
function update_jacobian!(alg::ManualSparseJacobian, cache, Y, p, dtγ, t)
    (;
        topography_flag,
        diffusion_flag,
        sgs_advection_flag,
        sgs_entr_detr_flag,
        sgs_mass_flux_flag,
        sgs_vertdiff_flag,
        acoustic_diagonal_flag,
    ) = alg
    (; matrix) = cache
    (; params) = p
    (; ᶜΦ) = p.core
    (; ᶜu, ᶠu³, ᶜK, ᶜp, ᶜT, ᶜh_tot) = p.precomputed
    (; ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) = p.precomputed
    (;
        ∂ᶜK_∂ᶜuₕ,
        ∂ᶜK_∂ᶠu₃,
        ᶠp_grad_matrix,
        ᶜadvection_matrix,
        ᶜdiffusion_h_matrix,
        ᶜdiffusion_u_matrix,
        ᶜtridiagonal_matrix_scalar,
        ᶠbidiagonal_matrix_ct3,
        ᶠbidiagonal_matrix_ct3_2,
        ᶠsed_tracer_advection,
        ᶜtracer_advection_matrix,
        ᶜtridiagonal_matrix,
    ) = p.scratch
    rs = p.atmos.rayleigh_sponge

    FT = Spaces.undertype(axes(Y.c))
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'

    cv_d = FT(CAP.cv_d(params))
    Δcv_v = FT(CAP.cv_v(params)) - cv_d
    T_0 = FT(CAP.T_0(params))
    R_d = FT(CAP.R_d(params))
    R_v = FT(CAP.R_v(params))
    ΔR_v = R_v - R_d
    cp_d = FT(CAP.cp_d(params))
    Δcp_v = FT(CAP.cp_v(params)) - cp_d
    e_int_v0 = FT(CAP.e_int_v0(params))
    LH_v0 = FT(CAP.LH_v0(params))
    LH_s0 = FT(CAP.LH_s0(params))
    Δcp_l = FT(CAP.cp_l(params) - CAP.cp_v(params))
    Δcp_i = FT(CAP.cp_i(params) - CAP.cp_v(params))
    Δcv_l = FT(CAP.cp_l(params) - CAP.cv_v(params))
    Δcv_i = FT(CAP.cp_i(params) - CAP.cv_v(params))
    e_int_v0 = FT(CAP.e_int_v0(params))
    e_int_s0 = FT(CAP.e_int_i0(params)) + e_int_v0
    thermo_params = CAP.thermodynamics_params(params)

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    ᶜgⁱʲ = Fields.local_geometry_field(Y.c).gⁱʲ
    ᶠgⁱʲ = Fields.local_geometry_field(Y.f).gⁱʲ
    ᶠz = Fields.coordinate_field(Y.f).z
    zmax = z_max(axes(Y.f))

    ᶜkappa_m = p.scratch.ᶜtemp_scalar
    @. ᶜkappa_m =
        TD.gas_constant_air(thermo_params, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno) /
        TD.cv_m(thermo_params, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno)

    ᶜ∂p∂ρq_tot = p.scratch.ᶜtemp_scalar_2
    @. ᶜ∂p∂ρq_tot = ᶜkappa_m * (-e_int_v0 - R_d * T_0 - Δcv_v * (ᶜT - T_0)) + ΔR_v * ᶜT

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

    ᶜθ_v = p.scratch.ᶜtemp_scalar_3
    @. ᶜθ_v = theta_v(thermo_params, ᶜT, ᶜp, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno)
    ᶜΠ = @. lazy(TD.exner_given_pressure(thermo_params, ᶜp))
    # In implicit tendency, we use the new pressure-gradient formulation (PGF) and gravitational acceleration:
    #              grad(p) / ρ + grad(Φ)  =  cp_d * θ_v * grad(Π) + grad(Φ).
    # Here below, we use the old formulation of (grad(Φ) + grad(p) / ρ).
    # This is because the new formulation would require computing the derivative of θ_v.
    # The only exception is:
    # We are rewriting grad(p) / ρ from the expansion of ∂ᶠu₃_err_∂ᶜρ with the new PGF.
    @. ∂ᶠu₃_err_∂ᶜρ =
        dtγ * (
            ᶠp_grad_matrix ⋅
            DiagonalMatrixRow(
                ᶜkappa_m * (T_0 * cp_d - ᶜK - ᶜΦ) + (R_d - ᶜkappa_m * cv_d) * ᶜT,
            ) +
            DiagonalMatrixRow(cp_d * ᶠinterp(ᶜθ_v) * ᶠgradᵥ(ᶜΠ) / ᶠinterp(ᶜρ)) ⋅
            ᶠinterp_matrix()
        )
    @. ∂ᶠu₃_err_∂ᶜρe_tot = dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(ᶜkappa_m)

    if MatrixFields.has_field(Y, @name(c.ρq_tot))
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
        ∂ᶠu₃_err_∂ᶜρq_tot = matrix[@name(f.u₃), @name(c.ρq_tot)]
        @. ∂ᶠu₃_err_∂ᶜρq_tot =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(ᶜ∂p∂ρq_tot)
    end

    microphysics_tracers =
        p.atmos.microphysics_model isa Union{
            NonEquilibriumMicrophysics1M,
            NonEquilibriumMicrophysics2M,
        } ?
        (
            (@name(c.ρq_liq), e_int_v0, Δcv_l),
            (@name(c.ρq_ice), e_int_s0, Δcv_i),
            (@name(c.ρq_rai), e_int_v0, Δcv_l),
            (@name(c.ρq_sno), e_int_s0, Δcv_i),
        ) : (;)

    for (q_name, e_int_q, ∂cv∂q) in microphysics_tracers
        MatrixFields.has_field(Y, q_name) || continue
        ∂ᶠu₃_err_∂ᶜρq = matrix[@name(f.u₃), q_name]
        @. ∂ᶠu₃_err_∂ᶜρq =
            dtγ * ᶠp_grad_matrix ⋅
            DiagonalMatrixRow(ᶜkappa_m * (e_int_q - ∂cv∂q * (ᶜT - T_0)) - R_v * ᶜT)
    end

    ∂ᶠu₃_err_∂ᶜuₕ = matrix[@name(f.u₃), @name(c.uₕ)]
    ∂ᶠu₃_err_∂ᶠu₃ = matrix[@name(f.u₃), @name(f.u₃)]
    I_u₃ = DiagonalMatrixRow(one_C3xACT3)
    if rs isa RayleighSponge
        @. ∂ᶠu₃_err_∂ᶜuₕ =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅ ∂ᶜK_∂ᶜuₕ
        @. ∂ᶠu₃_err_∂ᶠu₃ =
            dtγ * (
                ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅
                ∂ᶜK_∂ᶠu₃ +
                DiagonalMatrixRow(-β_rayleigh_u₃(rs, ᶠz, zmax) * (one_C3xACT3,))
            ) - (I_u₃,)
    else
        @. ∂ᶠu₃_err_∂ᶜuₕ =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅ ∂ᶜK_∂ᶜuₕ
        @. ∂ᶠu₃_err_∂ᶠu₃ =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅
            ∂ᶜK_∂ᶠu₃ - (I_u₃,)
    end

    α_vert_diff_tracer = CAP.α_vert_diff_tracer(params)
    tracer_info = (
        (@name(c.ρq_liq), @name(ᶜwₗ), FT(1)),
        (@name(c.ρq_ice), @name(ᶜwᵢ), FT(1)),
        (@name(c.ρq_rai), @name(ᶜwᵣ), α_vert_diff_tracer),
        (@name(c.ρq_sno), @name(ᶜwₛ), α_vert_diff_tracer),
        (@name(c.ρn_liq), @name(ᶜwₙₗ), FT(1)),
        (@name(c.ρn_rai), @name(ᶜwₙᵣ), α_vert_diff_tracer),
        (@name(c.ρn_ice), @name(ᶜwnᵢ), FT(1)),
        (@name(c.ρq_rim), @name(ᶜwᵢ), FT(1)),
        (@name(c.ρb_rim), @name(ᶜwᵢ), FT(1)),
    )
    internal_energy_func(name) =
        (name == @name(c.ρq_liq) || name == @name(c.ρq_rai)) ? TD.internal_energy_liquid :
        (name == @name(c.ρq_ice) || name == @name(c.ρq_sno)) ? TD.internal_energy_ice :
        nothing
    if !(p.atmos.microphysics_model isa DryModel) || use_derivative(diffusion_flag)
        ∂ᶜρe_tot_err_∂ᶜρe_tot = matrix[@name(c.ρe_tot), @name(c.ρe_tot)]
        @. ∂ᶜρe_tot_err_∂ᶜρe_tot = zero(typeof(∂ᶜρe_tot_err_∂ᶜρe_tot)) - (I,)
    end

    if !(p.atmos.microphysics_model isa DryModel)
        ∂ᶜρe_tot_err_∂ᶜρq_tot = matrix[@name(c.ρe_tot), @name(c.ρq_tot)]
        @. ∂ᶜρe_tot_err_∂ᶜρq_tot = zero(typeof(∂ᶜρe_tot_err_∂ᶜρq_tot))

        ∂ᶜρq_tot_err_∂ᶜρq_tot = matrix[@name(c.ρq_tot), @name(c.ρq_tot)]
        @. ∂ᶜρq_tot_err_∂ᶜρq_tot = zero(typeof(∂ᶜρq_tot_err_∂ᶜρq_tot)) - (I,)

        # This scratch variable computation could be skipped if no tracers are present
        @. p.scratch.ᶜbidiagonal_adjoint_matrix_c3 =
            dtγ * (-ClimaAtmos.ᶜprecipdivᵥ_matrix()) ⋅
            DiagonalMatrixRow(ClimaAtmos.ᶠinterp(ᶜρ * ᶜJ) / ᶠJ)

        MatrixFields.unrolled_foreach(tracer_info) do (ρχₚ_name, wₚ_name, _)
            MatrixFields.has_field(Y, ρχₚ_name) || return

            ∂ᶜρχₚ_err_∂ᶜρχₚ = matrix[ρχₚ_name, ρχₚ_name]
            ᶜwₚ = MatrixFields.get_field(p.precomputed, wₚ_name)
            # TODO: come up with read-able names for the intermediate computations...
            @. p.scratch.ᶠband_matrix_wvec =
                ClimaAtmos.ᶠright_bias_matrix() ⋅
                DiagonalMatrixRow(ClimaCore.Geometry.WVector(-(ᶜwₚ) / ᶜρ))
            @. ∂ᶜρχₚ_err_∂ᶜρχₚ =
                p.scratch.ᶜbidiagonal_adjoint_matrix_c3 ⋅
                p.scratch.ᶠband_matrix_wvec - (I,)

            if ρχₚ_name in
               (@name(c.ρq_liq), @name(c.ρq_ice), @name(c.ρq_rai), @name(c.ρq_sno))
                ∂ᶜρq_tot_err_∂ᶜρq = matrix[@name(c.ρq_tot), ρχₚ_name]
                @. ∂ᶜρq_tot_err_∂ᶜρq =
                    p.scratch.ᶜbidiagonal_adjoint_matrix_c3 ⋅
                    p.scratch.ᶠband_matrix_wvec

                ∂ᶜρe_tot_err_∂ᶜρq = matrix[@name(c.ρe_tot), ρχₚ_name]
                e_int_func = internal_energy_func(ρχₚ_name)
                @. ∂ᶜρe_tot_err_∂ᶜρq =
                    p.scratch.ᶜbidiagonal_adjoint_matrix_c3 ⋅
                    p.scratch.ᶠband_matrix_wvec ⋅
                    DiagonalMatrixRow(
                        e_int_func(thermo_params, ᶜT) + ᶜΦ + $(Kin(ᶜwₚ, ᶜu)),
                    )
            end
        end

    end

    if use_derivative(diffusion_flag)
        (; turbconv_model) = p.atmos
        turbconv_params = CAP.turbconv_params(params)
        FT = eltype(params)
        (; vertical_diffusion, smagorinsky_lilly) = p.atmos
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
            (; ᶜlinear_buoygrad, ᶜstrain_rate_norm) = p.precomputed
            ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
            ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_3
            ᶜmixing_length_field .= ᶜmixing_length(Y, p)
            ᶜK_u = p.scratch.ᶜtemp_scalar_4
            @. ᶜK_u = eddy_viscosity(turbconv_params, ᶜtke, ᶜmixing_length_field)
            ᶜprandtl_nvec = @. lazy(
                turbulent_prandtl_number(params, ᶜlinear_buoygrad, ᶜstrain_rate_norm),
            )
            ᶜK_h = p.scratch.ᶜtemp_scalar_6
            @. ᶜK_h = eddy_diffusivity(ᶜK_u, ᶜprandtl_nvec)
        end

        ∂ᶠρχ_dif_flux_∂ᶜχ = ᶠp_grad_matrix
        @. ∂ᶠρχ_dif_flux_∂ᶜχ =
            DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_h)) ⋅ ᶠgradᵥ_matrix()
        @. ᶜdiffusion_h_matrix = ᶜadvdivᵥ_matrix() ⋅ ∂ᶠρχ_dif_flux_∂ᶜχ
        if (
            MatrixFields.has_field(Y, @name(c.ρtke)) ||
            !isnothing(p.atmos.turbconv_model) ||
            !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
        )
            @. ∂ᶠρχ_dif_flux_∂ᶜχ =
                DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_u)) ⋅ ᶠgradᵥ_matrix()
            @. ᶜdiffusion_u_matrix = ᶜadvdivᵥ_matrix() ⋅ ∂ᶠρχ_dif_flux_∂ᶜχ
        end

        ∂ᶜρe_tot_err_∂ᶜρ = matrix[@name(c.ρe_tot), @name(c.ρ)]
        @. ∂ᶜρe_tot_err_∂ᶜρ = zero(typeof(∂ᶜρe_tot_err_∂ᶜρ))
        @. ∂ᶜρe_tot_err_∂ᶜρe_tot +=
            dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow((1 + ᶜkappa_m) / ᶜρ)

        if MatrixFields.has_field(Y, @name(c.ρq_tot))
            ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
            ∂ᶜρe_tot_err_∂ᶜρq_tot = matrix[@name(c.ρe_tot), @name(c.ρq_tot)]
            ∂ᶜρq_tot_err_∂ᶜρ = matrix[@name(c.ρq_tot), @name(c.ρ)]
            @. ∂ᶜρe_tot_err_∂ᶜρq_tot +=
                dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(ᶜ∂p∂ρq_tot / ᶜρ)
            @. ∂ᶜρq_tot_err_∂ᶜρ = zero(typeof(∂ᶜρq_tot_err_∂ᶜρ))
            @. ∂ᶜρq_tot_err_∂ᶜρq_tot +=
                dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(1 / ᶜρ)
        end

        for (q_name, e_int_q, ∂cv∂q) in microphysics_tracers
            MatrixFields.has_field(Y, q_name) || continue
            ∂ᶜρe_tot_err_∂ᶜρq = matrix[@name(c.ρe_tot), q_name]
            @. ∂ᶜρe_tot_err_∂ᶜρq +=
                dtγ * ᶜdiffusion_h_matrix ⋅
                DiagonalMatrixRow(
                    (ᶜkappa_m * (e_int_q - ∂cv∂q * (ᶜT - T_0)) - R_v * ᶜT) / ᶜρ,
                )
        end

        MatrixFields.unrolled_foreach(tracer_info) do (ρχ_name, _, α)
            MatrixFields.has_field(Y, ρχ_name) || return
            ∂ᶜρχ_err_∂ᶜρ = matrix[ρχ_name, @name(c.ρ)]
            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_name, ρχ_name]
            @. ∂ᶜρχ_err_∂ᶜρ = zero(typeof(∂ᶜρχ_err_∂ᶜρ))
            @. ∂ᶜρχ_err_∂ᶜρχ +=
                dtγ * α * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(1 / ᶜρ)
        end

        if MatrixFields.has_field(Y, @name(c.ρtke))
            turbconv_params = CAP.turbconv_params(params)
            c_d = CAP.tke_diss_coeff(turbconv_params)
            (; dt) = p
            turbconv_model = p.atmos.turbconv_model
            ᶜtke = @. lazy(specific(Y.c.ρtke, Y.c.ρ))
            ᶜρtke = Y.c.ρtke

            # scratch to prevent GPU Kernel parameter memory error
            ᶜmixing_length_field = p.scratch.ᶜtemp_scalar_3
            ᶜmixing_length_field .= ᶜmixing_length(Y, p)

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

    end

    # Acoustic diagonal shift: approximate the Schur complement of the
    # coupled (ρ, uₕ) horizontal acoustic system. The Schur complement
    # gives a Helmholtz operator -I - dtγ² c_s² ∇²ₕ on the diagonal.
    # We approximate the diagonal of -∇²ₕ as 2π²/Δx² (2D spectral element).
    if use_derivative(acoustic_diagonal_flag)
        hspace = Spaces.horizontal_space(axes(Y.c))
        Δx = FT(Spaces.node_horizontal_length_scale(hspace))
        γ_d = cp_d / cv_d
        ᶜα_acoustic = p.scratch.ᶜtemp_scalar_2
        @. ᶜα_acoustic = FT(dtγ)^2 * γ_d * ᶜp / ᶜρ * FT(2 * π^2) / Δx^2

        ∂ᶜρ_err_∂ᶜρ = matrix[@name(c.ρ), @name(c.ρ)]
        @. ∂ᶜρ_err_∂ᶜρ = DiagonalMatrixRow(-(FT(1) + ᶜα_acoustic))

        ∂ᶜρe_tot_err_∂ᶜρe_tot = matrix[@name(c.ρe_tot), @name(c.ρe_tot)]
        ρe_tot_already_initialized =
            !(p.atmos.microphysics_model isa DryModel) || use_derivative(diffusion_flag)
        if ρe_tot_already_initialized
            @. ∂ᶜρe_tot_err_∂ᶜρe_tot += DiagonalMatrixRow(FT(-1) * ᶜα_acoustic)
        else
            @. ∂ᶜρe_tot_err_∂ᶜρe_tot = DiagonalMatrixRow(-(FT(1) + ᶜα_acoustic))
        end

        uₕ_already_initialized = use_derivative(diffusion_flag) && (
            !isnothing(p.atmos.turbconv_model) ||
            !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
        )
        ∂ᶜuₕ_err_∂ᶜuₕ = matrix[@name(c.uₕ), @name(c.uₕ)]
        if uₕ_already_initialized
            @. ∂ᶜuₕ_err_∂ᶜuₕ += DiagonalMatrixRow(FT(-1) * ᶜα_acoustic)
        else
            @. ∂ᶜuₕ_err_∂ᶜuₕ = DiagonalMatrixRow(-(FT(1) + ᶜα_acoustic))
        end

        # Store state for Helmholtz solve in invert_jacobian!
        cache.helmholtz_state.dtγ = dtγ
        @. cache.helmholtz_state.ᶜα_acoustic = ᶜα_acoustic
        @. cache.helmholtz_state.ᶜcs² = γ_d * ᶜp / ᶜρ
        @. cache.helmholtz_state.ᶜρ = ᶜρ
        @. cache.helmholtz_state.ᶜe_tot = Y.c.ρe_tot / ᶜρ
        if do_dss(axes(Y.c))
            cache.helmholtz_state.ghost_buffer_c = p.ghost_buffer.c
        end
    end

    if p.atmos.turbconv_model isa PrognosticEDMFX
        if use_derivative(sgs_advection_flag)
            (; ᶜgradᵥ_ᶠΦ) = p.core
            (;
                ᶜρʲs,
                ᶠu³ʲs,
                ᶜTʲs,
                ᶜq_tot_safeʲs,
                ᶜq_liq_raiʲs,
                ᶜq_ice_snoʲs,
                ᶜKʲs,
            ) = p.precomputed

            # upwinding options for q_tot and mse
            is_third_order =
                p.atmos.numerics.edmfx_mse_q_tot_upwinding == Val(:third_order)
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

            # upwinding options for other tracers
            is_tracer_upwinding_third_order =
                p.atmos.numerics.edmfx_tracer_upwinding == Val(:third_order)
            ᶠtracer_upwind = is_tracer_upwinding_third_order ? ᶠupwind3 : ᶠupwind1
            ᶠset_tracer_upwind_bcs = Operators.SetBoundaryOperator(;
                top = Operators.SetValue(zero(CT3{FT})),
                bottom = Operators.SetValue(zero(CT3{FT})),
            ) # Need to wrap ᶠtracer_upwind in this for well-defined boundaries.
            TracerUpwindMatrixRowType =
                is_tracer_upwinding_third_order ? QuaddiagonalMatrixRow :
                BidiagonalMatrixRow
            ᶠtracer_upwind_matrix =
                is_tracer_upwinding_third_order ? ᶠupwind3_matrix : ᶠupwind1_matrix
            ᶠset_tracer_upwind_matrix_bcs = Operators.SetBoundaryOperator(;
                top = Operators.SetValue(zero(TracerUpwindMatrixRowType{CT3{FT}})),
                bottom = Operators.SetValue(zero(TracerUpwindMatrixRowType{CT3{FT}})),
            ) # Need to wrap ᶠtracer_upwind_matrix in this for well-defined boundaries.

            ᶠu³ʲ_data = ᶠu³ʲs.:(1).components.data.:1

            ᶜkappa_mʲ = p.scratch.ᶜtemp_scalar
            @. ᶜkappa_mʲ =
                TD.gas_constant_air(
                    thermo_params,
                    ᶜq_tot_safeʲs.:(1),
                    ᶜq_liq_raiʲs.:(1),
                    ᶜq_ice_snoʲs.:(1),
                ) /
                TD.cv_m(
                    thermo_params,
                    ᶜq_tot_safeʲs.:(1),
                    ᶜq_liq_raiʲs.:(1),
                    ᶜq_ice_snoʲs.:(1),
                )

            ∂ᶜq_totʲ_err_∂ᶜq_totʲ =
                matrix[@name(c.sgsʲs.:(1).q_tot), @name(c.sgsʲs.:(1).q_tot)]
            @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ =
                dtγ * (
                    DiagonalMatrixRow(ᶜadvdivᵥ(ᶠu³ʲs.:(1))) -
                    ᶜadvdivᵥ_matrix() ⋅
                    ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1)))
                ) - (I,)

            ∂ᶜmseʲ_err_∂ᶜmseʲ =
                matrix[@name(c.sgsʲs.:(1).mse), @name(c.sgsʲs.:(1).mse)]
            @. ∂ᶜmseʲ_err_∂ᶜmseʲ =
                dtγ * (
                    DiagonalMatrixRow(ᶜadvdivᵥ(ᶠu³ʲs.:(1))) -
                    ᶜadvdivᵥ_matrix() ⋅
                    ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) -
                    DiagonalMatrixRow(
                        adjoint(ᶜinterp(ᶠu³ʲs.:(1))) *
                        ᶜgradᵥ_ᶠΦ *
                        Y.c.ρ *
                        ᶜkappa_mʲ / ((ᶜkappa_mʲ + 1) * ᶜp),
                    )
                ) - (I,)

            ∂ᶜρaʲ_err_∂ᶜρaʲ =
                matrix[@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).ρa)]
            @. ᶜadvection_matrix =
                -(ᶜadvdivᵥ_matrix()) ⋅
                DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ)
            @. ∂ᶜρaʲ_err_∂ᶜρaʲ =
                dtγ * ᶜadvection_matrix ⋅
                ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) ⋅
                DiagonalMatrixRow(1 / ᶜρʲs.:(1)) - (I,)

            # contribution of ρʲ variations in vertical transport of ρa and updraft buoyancy eq
            ∂ᶜρaʲ_err_∂ᶜmseʲ =
                matrix[@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).mse)]
            @. ᶠbidiagonal_matrix_ct3 =
                DiagonalMatrixRow(
                    ᶠset_upwind_bcs(
                        ᶠupwind(
                            ᶠu³ʲs.:(1),
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ),
                    ) / ᶠJ,
                ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(
                    ᶜJ * ᶜkappa_mʲ * (ᶜρʲs.:(1))^2 / ((ᶜkappa_mʲ + 1) * ᶜp),
                )
            @. ᶠbidiagonal_matrix_ct3_2 =
                DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) ⋅
                ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) ⋅
                DiagonalMatrixRow(
                    Y.c.sgsʲs.:(1).ρa * ᶜkappa_mʲ / ((ᶜkappa_mʲ + 1) * ᶜp),
                )
            @. ∂ᶜρaʲ_err_∂ᶜmseʲ =
                dtγ * ᶜadvdivᵥ_matrix() ⋅
                (ᶠbidiagonal_matrix_ct3 - ᶠbidiagonal_matrix_ct3_2)

            turbconv_params = CAP.turbconv_params(params)
            α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
            ᶜ∂RmT∂qʲ = p.scratch.ᶜtemp_scalar_2
            sgs_microphysics_tracers =
                p.atmos.microphysics_model isa Union{
                    NonEquilibriumMicrophysics1M,
                    NonEquilibriumMicrophysics2M,
                } ?
                (
                    (@name(c.sgsʲs.:(1).q_tot), -LH_v0, Δcp_v, ΔR_v),
                    (@name(c.sgsʲs.:(1).q_liq), LH_v0, Δcp_l, -R_v),
                    (@name(c.sgsʲs.:(1).q_ice), LH_s0, Δcp_i, -R_v),
                    (@name(c.sgsʲs.:(1).q_rai), LH_v0, Δcp_l, -R_v),
                    (@name(c.sgsʲs.:(1).q_sno), LH_s0, Δcp_i, -R_v),
                ) : (
                    (@name(c.sgsʲs.:(1).q_tot), -LH_v0, Δcp_v, ΔR_v),
                )

            for (qʲ_name, LH, ∂cp∂q, ∂Rm∂q) in sgs_microphysics_tracers
                MatrixFields.has_field(Y, qʲ_name) || continue

                @. ᶜ∂RmT∂qʲ =
                    ᶜkappa_mʲ / (ᶜkappa_mʲ + 1) * (LH - ∂cp∂q * (ᶜTʲs.:(1) - T_0)) +
                    ∂Rm∂q * ᶜTʲs.:(1)

                # ∂ᶜρaʲ_err_∂ᶜqʲ through ρʲ variations in vertical transport of ρa
                ∂ᶜρaʲ_err_∂ᶜqʲ = matrix[@name(c.sgsʲs.:(1).ρa), qʲ_name]
                @. ᶠbidiagonal_matrix_ct3 =
                    DiagonalMatrixRow(
                        ᶠset_upwind_bcs(
                            ᶠupwind(
                                ᶠu³ʲs.:(1),
                                draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                            ),
                        ) / ᶠJ,
                    ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(
                        ᶜJ * (ᶜρʲs.:(1))^2 / ᶜp * ᶜ∂RmT∂qʲ,
                    )
                @. ᶠbidiagonal_matrix_ct3_2 =
                    DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) ⋅
                    ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) ⋅
                    DiagonalMatrixRow(
                        Y.c.sgsʲs.:(1).ρa / ᶜp * ᶜ∂RmT∂qʲ,
                    )
                @. ∂ᶜρaʲ_err_∂ᶜqʲ =
                    dtγ * ᶜadvdivᵥ_matrix() ⋅
                    (ᶠbidiagonal_matrix_ct3 - ᶠbidiagonal_matrix_ct3_2)

                # ∂ᶜmseʲ_err_∂ᶜqʲ through ρʲ variations in buoyancy term in mse eq
                ∂ᶜmseʲ_err_∂ᶜqʲ = matrix[@name(c.sgsʲs.:(1).mse), qʲ_name]
                @. ∂ᶜmseʲ_err_∂ᶜqʲ =
                    dtγ * (
                        -DiagonalMatrixRow(
                            adjoint(ᶜinterp(ᶠu³ʲs.:(1))) * ᶜgradᵥ_ᶠΦ * Y.c.ρ / ᶜp *
                            ᶜ∂RmT∂qʲ,
                        )
                    )
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
                sgs_microphysics_tracers = (
                    (@name(c.sgsʲs.:(1).q_liq), @name(ᶜwₗʲs.:(1))),
                    (@name(c.sgsʲs.:(1).q_ice), @name(ᶜwᵢʲs.:(1))),
                    (@name(c.sgsʲs.:(1).q_rai), @name(ᶜwᵣʲs.:(1))),
                    (@name(c.sgsʲs.:(1).q_sno), @name(ᶜwₛʲs.:(1))),
                    (@name(c.sgsʲs.:(1).n_liq), @name(ᶜwₙₗʲs.:(1))),
                    (@name(c.sgsʲs.:(1).n_rai), @name(ᶜwₙᵣʲs.:(1))),
                )
                MatrixFields.unrolled_foreach(
                    sgs_microphysics_tracers,
                ) do (χʲ_name, wʲ_name)
                    MatrixFields.has_field(Y, χʲ_name) || return
                    ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                    ᶜwʲ = MatrixFields.get_field(p.precomputed, wʲ_name)

                    # advection
                    ∂ᶜχʲ_err_∂ᶜχʲ = matrix[χʲ_name, χʲ_name]
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

                    if χʲ_name in (
                        @name(c.sgsʲs.:(1).q_liq),
                        @name(c.sgsʲs.:(1).q_ice),
                        @name(c.sgsʲs.:(1).q_rai),
                        @name(c.sgsʲs.:(1).q_sno),
                    )
                        ∂ᶜq_totʲ_err_∂ᶜχʲ =
                            matrix[@name(c.sgsʲs.:(1).q_tot), χʲ_name]
                        @. ∂ᶜq_totʲ_err_∂ᶜχʲ =
                            DiagonalMatrixRow(ᶜinv_ρ̂) ⋅ ᶜtridiagonal_matrix_scalar
                    end

                end
            end

            # vertical diffusion of updrafts
            if use_derivative(sgs_vertdiff_flag)
                α_vert_diff_tracer = CAP.α_vert_diff_tracer(params)
                @. p.scratch.ᶜbidiagonal_adjoint_matrix_c3 =
                    ᶜadvdivᵥ_matrix() ⋅
                    DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1)) * ᶠinterp(ᶜK_h))
                @. ᶜdiffusion_h_matrix =
                    p.scratch.ᶜbidiagonal_adjoint_matrix_c3 ⋅ ᶠgradᵥ_matrix()

                @. ∂ᶜmseʲ_err_∂ᶜmseʲ +=
                    dtγ * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅ ᶜdiffusion_h_matrix
                @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ +=
                    dtγ * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅ ᶜdiffusion_h_matrix
                @. ∂ᶜρaʲ_err_∂ᶜρaʲ +=
                    dtγ * DiagonalMatrixRow(1 / (1 - Y.c.sgsʲs.:(1).q_tot) / ᶜρʲs.:(1)) ⋅
                    ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(Y.c.sgsʲs.:(1).q_tot)
                ∂ᶜρaʲ_err_∂ᶜq_totʲ =
                    matrix[@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).q_tot)]
                @. ∂ᶜρaʲ_err_∂ᶜq_totʲ +=
                    dtγ * DiagonalMatrixRow(
                        Y.c.sgsʲs.:(1).ρa / (1 - Y.c.sgsʲs.:(1).q_tot) / ᶜρʲs.:(1),
                    ) ⋅
                    ᶜdiffusion_h_matrix
                @. ∂ᶜρaʲ_err_∂ᶜq_totʲ +=
                    dtγ * DiagonalMatrixRow(
                        Y.c.sgsʲs.:(1).ρa / (1 - Y.c.sgsʲs.:(1).q_tot)^2 / ᶜρʲs.:(1),
                    ) ⋅
                    ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(Y.c.sgsʲs.:(1).q_tot)
                if p.atmos.microphysics_model isa Union{
                    NonEquilibriumMicrophysics1M,
                    NonEquilibriumMicrophysics2M,
                }
                    sgs_microphysics_tracers = (
                        (@name(c.sgsʲs.:(1).q_liq), FT(1)),
                        (@name(c.sgsʲs.:(1).q_ice), FT(1)),
                        (@name(c.sgsʲs.:(1).q_rai), α_vert_diff_tracer),
                        (@name(c.sgsʲs.:(1).q_sno), α_vert_diff_tracer),
                        (@name(c.sgsʲs.:(1).n_liq), FT(1)),
                        (@name(c.sgsʲs.:(1).n_rai), α_vert_diff_tracer),
                    )
                    MatrixFields.unrolled_foreach(
                        sgs_microphysics_tracers,
                    ) do (χʲ_name, α)
                        MatrixFields.has_field(Y, χʲ_name) || return
                        ∂ᶜχʲ_err_∂ᶜχʲ = matrix[χʲ_name, χʲ_name]
                        @. ∂ᶜχʲ_err_∂ᶜχʲ +=
                            dtγ * α * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅
                            ᶜdiffusion_h_matrix
                    end
                end
            end
            # entrainment and detrainment (rates are treated explicitly)
            if use_derivative(sgs_entr_detr_flag)
                (; ᶜentrʲs, ᶜdetrʲs, ᶜturb_entrʲs) = p.precomputed
                @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ -=
                    dtγ * DiagonalMatrixRow(ᶜentrʲs.:(1) + ᶜturb_entrʲs.:(1))
                @. ∂ᶜmseʲ_err_∂ᶜmseʲ -=
                    dtγ * DiagonalMatrixRow(ᶜentrʲs.:(1) + ᶜturb_entrʲs.:(1))
                @. ∂ᶜρaʲ_err_∂ᶜρaʲ +=
                    dtγ * DiagonalMatrixRow(ᶜentrʲs.:(1) - ᶜdetrʲs.:(1))
                if p.atmos.microphysics_model isa Union{
                    NonEquilibriumMicrophysics1M,
                    NonEquilibriumMicrophysics2M,
                }
                    sgs_microphysics_tracers = (
                        (@name(c.sgsʲs.:(1).q_liq)),
                        (@name(c.sgsʲs.:(1).q_ice)),
                        (@name(c.sgsʲs.:(1).q_rai)),
                        (@name(c.sgsʲs.:(1).q_sno)),
                    )
                    MatrixFields.unrolled_foreach(
                        sgs_microphysics_tracers,
                    ) do (qʲ_name)
                        MatrixFields.has_field(Y, qʲ_name) || return

                        ∂ᶜqʲ_err_∂ᶜqʲ = matrix[qʲ_name, qʲ_name]
                        @. ∂ᶜqʲ_err_∂ᶜqʲ -=
                            dtγ * DiagonalMatrixRow(ᶜentrʲs.:(1) + ᶜturb_entrʲs.:(1))
                    end
                end
            end

            # add updraft mass flux contributions to grid-mean
            if use_derivative(sgs_mass_flux_flag)
                # Jacobian contributions of updraft massflux to grid-mean
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
                ᶜkappa_m = p.scratch.ᶜtemp_scalar
                @. ᶜkappa_m =
                    TD.gas_constant_air(
                        thermo_params,
                        ᶜq_tot_safe,
                        ᶜq_liq_rai,
                        ᶜq_ice_sno,
                    ) /
                    TD.cv_m(thermo_params, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno)


                ᶜ∂p∂ρq_tot = p.scratch.ᶜtemp_scalar_2
                @. ᶜ∂p∂ρq_tot =
                    ᶜkappa_m * (-e_int_v0 - R_d * T_0 - Δcv_v * (ᶜT - T_0)) + ΔR_v * ᶜT

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

                @. ∂ᶜρe_tot_err_∂ᶜρq_tot +=
                    p.scratch.ᶜtridiagonal_matrix_scalar ⋅
                    DiagonalMatrixRow(ᶜ∂p∂ρq_tot / ᶜρ)

                for (q_name, e_int_q, ∂cv∂q) in microphysics_tracers
                    MatrixFields.has_field(Y, q_name) || continue
                    ∂ᶜρe_tot_err_∂ᶜρq = matrix[@name(c.ρe_tot), q_name]
                    @. ∂ᶜρe_tot_err_∂ᶜρq +=
                        p.scratch.ᶜtridiagonal_matrix_scalar ⋅
                        DiagonalMatrixRow(
                            (ᶜkappa_m * (e_int_q - ∂cv∂q * (ᶜT - T_0)) - R_v * ᶜT) / ᶜρ,
                        )
                end

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

                @. p.scratch.ᶠdiagonal_matrix_ct3xct3 = DiagonalMatrixRow(
                    ᶠinterp(
                        (Y.c.sgsʲs.:(1).q_tot - ᶜq_tot) *
                        ᶜρʲs.:(1) *
                        ᶜJ *
                        draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                    ) / ᶠJ * (g³³(ᶠgⁱʲ)),
                )

                ∂ᶜρq_tot_err_∂ᶠu₃ = matrix[@name(c.ρq_tot), @name(f.u₃)]
                @. ∂ᶜρq_tot_err_∂ᶠu₃ +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ p.scratch.ᶠdiagonal_matrix_ct3xct3

                # grid-mean ∂/∂(rho*a)
                ∂ᶜρe_tot_err_∂ᶜρa =
                    matrix[@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)]
                @. p.scratch.ᶠtemp_CT3_2 =
                    (ᶠu³ʲs.:(1) - ᶠu³) *
                    ᶠinterp((Y.c.sgsʲs.:(1).mse + ᶜKʲs.:(1) - ᶜh_tot)) / ᶠJ
                @. p.scratch.ᶜbidiagonal_matrix_scalar =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(p.scratch.ᶠtemp_CT3_2)
                @. ∂ᶜρe_tot_err_∂ᶜρa =
                    p.scratch.ᶜbidiagonal_matrix_scalar ⋅ ᶠinterp_matrix() ⋅
                    DiagonalMatrixRow(ᶜJ)

                ∂ᶜρq_tot_err_∂ᶜρa =
                    matrix[@name(c.ρq_tot), @name(c.sgsʲs.:(1).ρa)]
                @. p.scratch.ᶠtemp_CT3_2 =
                    (ᶠu³ʲs.:(1) - ᶠu³) *
                    ᶠinterp((Y.c.sgsʲs.:(1).q_tot - ᶜq_tot)) / ᶠJ
                @. p.scratch.ᶜbidiagonal_matrix_scalar =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(p.scratch.ᶠtemp_CT3_2)
                @. ∂ᶜρq_tot_err_∂ᶜρa =
                    p.scratch.ᶜbidiagonal_matrix_scalar ⋅ ᶠinterp_matrix() ⋅
                    DiagonalMatrixRow(ᶜJ)

                # grid-mean tracers
                if p.atmos.microphysics_model isa Union{
                    NonEquilibriumMicrophysics1M,
                    NonEquilibriumMicrophysics2M,
                }

                    microphysics_tracers = (
                        (@name(c.ρq_liq), @name(c.sgsʲs.:(1).q_liq), @name(q_liq)),
                        (@name(c.ρq_ice), @name(c.sgsʲs.:(1).q_ice), @name(q_ice)),
                        (@name(c.ρq_rai), @name(c.sgsʲs.:(1).q_rai), @name(q_rai)),
                        (@name(c.ρq_sno), @name(c.sgsʲs.:(1).q_sno), @name(q_sno)),
                        (@name(c.ρn_liq), @name(c.sgsʲs.:(1).n_liq), @name(n_liq)),
                        (@name(c.ρn_rai), @name(c.sgsʲs.:(1).n_rai), @name(n_rai)),
                    )

                    # add updraft contributions
                    # pull common subexpressions that don't depend on which
                    # tracer out of the tracer loop for performance
                    @. ᶜtracer_advection_matrix =
                        -(ᶜadvdivᵥ_matrix()) ⋅
                        DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ)
                    @. ᶜtridiagonal_matrix =
                        ᶜtracer_advection_matrix ⋅ ᶠset_tracer_upwind_matrix_bcs(
                            ᶠtracer_upwind_matrix(ᶠu³ʲs.:(1)),
                        )
                    MatrixFields.unrolled_foreach(
                        microphysics_tracers,
                    ) do (ρχ_name, χʲ_name, χ_name)
                        MatrixFields.has_field(Y, ρχ_name) || return
                        ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)

                        ∂ᶜρχ_err_∂ᶜχʲ =
                            matrix[ρχ_name, χʲ_name]
                        @. ∂ᶜρχ_err_∂ᶜχʲ =
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)))

                        ∂ᶜρχ_err_∂ᶜρa =
                            matrix[ρχ_name, @name(c.sgsʲs.:(1).ρa)]
                        @. ∂ᶜρχ_err_∂ᶜρa =
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(ᶜχʲ / ᶜρʲs.:(1))

                    end

                    # add env flux contributions
                    (; ᶜp) = p.precomputed
                    (; ᶠu³⁰, ᶜT⁰, ᶜq_tot_safe⁰, ᶜq_liq_rai⁰, ᶜq_ice_sno⁰) = p.precomputed
                    ᶜρ⁰ = @. lazy(
                        TD.air_density(
                            thermo_params,
                            ᶜT⁰,
                            ᶜp,
                            ᶜq_tot_safe⁰,
                            ᶜq_liq_rai⁰,
                            ᶜq_ice_sno⁰,
                        ),
                    )
                    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, turbconv_model))
                    ᶠu³⁰_data = ᶠu³⁰.components.data.:1

                    # pull common subexpressions that don't depend on which
                    # tracer out of the tracer loop for performance
                    @. ᶜtracer_advection_matrix =
                        -(ᶜadvdivᵥ_matrix()) ⋅
                        DiagonalMatrixRow(ᶠinterp(ᶜρ⁰ * ᶜJ) / ᶠJ)
                    @. ᶜtridiagonal_matrix =
                        ᶜtracer_advection_matrix ⋅ ᶠset_tracer_upwind_matrix_bcs(
                            ᶠtracer_upwind_matrix(ᶠu³⁰),
                        )
                    MatrixFields.unrolled_foreach(
                        microphysics_tracers,
                    ) do (ρχ_name, χʲ_name, χ_name)
                        MatrixFields.has_field(Y, ρχ_name) || return
                        ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)
                        ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)

                        ∂ᶜρχ_err_∂ᶜχʲ =
                            matrix[ρχ_name, χʲ_name]
                        @. ∂ᶜρχ_err_∂ᶜχʲ +=
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(-1 * Y.c.sgsʲs.:(1).ρa / ᶜρ⁰)

                        ∂ᶜρχ_err_∂ᶜρa =
                            matrix[ρχ_name, @name(c.sgsʲs.:(1).ρa)]
                        # pull out and store for kernel performance
                        @. ᶠbidiagonal_matrix_ct3_2 =
                            ᶠset_tracer_upwind_matrix_bcs(
                                ᶠtracer_upwind_matrix(CT3(sign(ᶠu³⁰_data))),
                            ) ⋅ DiagonalMatrixRow(ᶜχ⁰ * draft_area(ᶜρa⁰, ᶜρ⁰))
                        @. ∂ᶜρχ_err_∂ᶜρa +=
                            dtγ *
                            ᶜtracer_advection_matrix ⋅
                            DiagonalMatrixRow(
                                (ᶠu³⁰_data - ᶠu³ʲ_data) / ᶠinterp(ᶜρa⁰),
                            ) ⋅ ᶠbidiagonal_matrix_ct3_2

                        @. ∂ᶜρχ_err_∂ᶜρa +=
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(-1 * ᶜχʲ / ᶜρ⁰)

                        ∂ᶜρχ_err_∂ᶜρχ =
                            matrix[ρχ_name, ρχ_name]
                        @. ∂ᶜρχ_err_∂ᶜρχ +=
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(1 / ᶜρ⁰)

                        ∂ᶜρχ_err_∂ᶠu₃ =
                            matrix[ρχ_name, @name(f.u₃)]
                        @. ∂ᶜρχ_err_∂ᶠu₃ =
                            dtγ * ᶜtracer_advection_matrix ⋅
                            DiagonalMatrixRow(
                                ᶠset_tracer_upwind_bcs(
                                    ᶠtracer_upwind(CT3(sign(ᶠu³⁰_data)),
                                        ᶜχ⁰ * draft_area(ᶜρa⁰, ᶜρ⁰),
                                    ),
                                ) * adjoint(C3(sign(ᶠu³⁰_data))) *
                                ᶠinterp(Y.c.ρ / ᶜρa⁰) * g³³(ᶠgⁱʲ),
                            )
                    end
                end
            end
        end
    end

    update_microphysics_jacobian!(matrix, Y, p, dtγ, sgs_advection_flag)

    # NOTE: All velocity tendency derivatives should be set BEFORE this call.
    zero_velocity_jacobian!(matrix, Y, p, t)
end

"""
    update_microphysics_jacobian!(matrix, Y, p, dtγ, sgs_advection_flag)

Add diagonal Jacobian entries for implicit microphysics tendencies (0M, 1M, 2M,
and EDMF updraft species).

Extracted from `update_jacobian!` to keep the parent function below Julia's
optimization threshold — large functions cause the compiler to miss inlining
opportunities in broadcast expressions, resulting in heap allocations.
"""
function update_microphysics_jacobian!(matrix, Y, p, dtγ, sgs_advection_flag)
    p.atmos.microphysics_tendency_timestepping == Implicit() || return nothing

    ᶜρ = Y.c.ρ
    # TODO - do we need a corresponding term for ρe_tot?

    # 0M microphysics: diagonal entry for ρq_tot
    if p.atmos.microphysics_model isa EquilibriumMicrophysics0M
        if MatrixFields.has_field(Y, @name(c.ρq_tot))
            (; ᶜρ_dq_tot_dt) = p.precomputed
            ∂ᶜρq_tot_err_∂ᶜρq_tot = matrix[@name(c.ρq_tot), @name(c.ρq_tot)]
            @. ∂ᶜρq_tot_err_∂ᶜρq_tot +=
                dtγ * DiagonalMatrixRow(_jac_coeff(
                    ᶜρ_dq_tot_dt, Y.c.ρq_tot,
                ))
        end
    end

    # 1M microphysics: diagonal entries for ρq_liq, ρq_ice, ρq_rai, ρq_sno
    if p.atmos.microphysics_model isa NonEquilibriumMicrophysics1M
        (; ᶜmp_derivative) = p.precomputed

        # Cloud condensate (q_lcl, q_icl): use BMT grid-mean derivatives
        # (dominated by the condensation/deposition term -1/τ_relax, which
        # is independent of the SGS distribution)
        cloud_1m_deriv_tracers = (
            (@name(c.ρq_liq), ᶜmp_derivative.∂tendency_∂q_lcl),
            (@name(c.ρq_ice), ᶜmp_derivative.∂tendency_∂q_icl),
        )
        MatrixFields.unrolled_foreach(
            cloud_1m_deriv_tracers,
        ) do (ρχ_name, ᶜ∂S∂q)
            MatrixFields.has_field(Y, ρχ_name) || return
            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_name, ρχ_name]
            @. ∂ᶜρχ_err_∂ᶜρχ += dtγ * DiagonalMatrixRow(ᶜ∂S∂q)
        end

        # Precipitation (q_rai, q_sno): use S/q from quadrature-integrated
        # tendencies. This makes the Jacobian consistent with the SGS quadrature
        # used in the implicit tendency, preventing Newton solver divergence
        # when the SGS distribution differs from the grid mean.
        if p.atmos.turbconv_model isa PrognosticEDMFX
            # Environment quadrature tendencies
            (; ᶜmp_tendency⁰) = p.precomputed
            precip_1m_sq_tracers = (
                (@name(c.ρq_rai), ᶜmp_tendency⁰.dq_rai_dt, Y.c.ρq_rai),
                (@name(c.ρq_sno), ᶜmp_tendency⁰.dq_sno_dt, Y.c.ρq_sno),
            )
        else
            # Grid-mean quadrature tendencies
            (; ᶜmp_tendency) = p.precomputed
            precip_1m_sq_tracers = (
                (@name(c.ρq_rai), ᶜmp_tendency.dq_rai_dt, Y.c.ρq_rai),
                (@name(c.ρq_sno), ᶜmp_tendency.dq_sno_dt, Y.c.ρq_sno),
            )
        end
        MatrixFields.unrolled_foreach(
            precip_1m_sq_tracers,
        ) do (ρχ_name, ᶜS, ᶜρχ)
            MatrixFields.has_field(Y, ρχ_name) || return
            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_name, ρχ_name]
            # S/q approximation: ∂(dq/dt)/∂q ≈ (dq/dt) / q
            # Uses the full derivative (including source terms) for an accurate
            # Newton linearization consistent with the quadrature tendencies.
            @. ∂ᶜρχ_err_∂ᶜρχ += dtγ * DiagonalMatrixRow(
                _jac_coeff_from_ratio(ᶜS, ᶜρχ, ᶜρ),
            )
        end
    end

    # 2M microphysics: diagonal entries for ρq_liq, ρq_rai, ρn_liq, ρn_rai
    if p.atmos.microphysics_model isa NonEquilibriumMicrophysics2M
        (; ᶜmp_derivative) = p.precomputed

        # Cloud fields: use BMT grid-mean derivatives
        cloud_2m_deriv_tracers = (
            (@name(c.ρq_liq), ᶜmp_derivative.∂tendency_∂q_lcl),
            (@name(c.ρn_liq), ᶜmp_derivative.∂tendency_∂n_lcl),
        )
        MatrixFields.unrolled_foreach(
            cloud_2m_deriv_tracers,
        ) do (ρχ_name, ᶜ∂S∂q)
            MatrixFields.has_field(Y, ρχ_name) || return
            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_name, ρχ_name]
            @. ∂ᶜρχ_err_∂ᶜρχ += dtγ * DiagonalMatrixRow(ᶜ∂S∂q)
        end

        # Precipitation: use S/q from quadrature-integrated tendencies
        # _jac_coeff_from_ratio safely returns zero when |q| < ε
        (; ᶜmp_tendency) = p.precomputed
        precip_2m_sq_tracers = (
            (@name(c.ρq_rai), ᶜmp_tendency.dq_rai_dt, Y.c.ρq_rai),
            (@name(c.ρn_rai), ᶜmp_tendency.dn_rai_dt, Y.c.ρn_rai),
        )
        MatrixFields.unrolled_foreach(
            precip_2m_sq_tracers,
        ) do (ρχ_name, ᶜS, ᶜρχ)
            MatrixFields.has_field(Y, ρχ_name) || return
            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_name, ρχ_name]
            @. ∂ᶜρχ_err_∂ᶜρχ += dtγ * DiagonalMatrixRow(
                _jac_coeff_from_ratio(ᶜS, ᶜρχ, ᶜρ),
            )
        end
    end

    # EDMF microphysics: diagonal entries for updraft variables
    if p.atmos.turbconv_model isa PrognosticEDMFX
        # 0M EDMF
        if p.atmos.microphysics_model isa EquilibriumMicrophysics0M
            if hasproperty(p.precomputed, :ᶜmp_tendencyʲs)
                (; ᶜmp_tendencyʲs) = p.precomputed
                ᶜSq_tot = ᶜmp_tendency.:(1).dq_tot_dt

                q_name = @name(c.sgsʲs.:(1).q_tot)
                if MatrixFields.has_field(Y, q_name)
                    ∂ᶜq_err_∂ᶜq = matrix[q_name, q_name]
                    if !use_derivative(sgs_advection_flag)
                        @. ∂ᶜq_err_∂ᶜq =
                            zero(typeof(∂ᶜq_err_∂ᶜq)) - (I,)
                    end
                    add_microphysics_jacobian_entry!(
                        ∂ᶜq_err_∂ᶜq, dtγ, ᶜSq_tot, Y.c.sgsʲs.:(1).q_tot,
                    )
                end

                ρa_name = @name(c.sgsʲs.:(1).ρa)
                if MatrixFields.has_field(Y, ρa_name)
                    ∂ᶜρa_err_∂ᶜρa = matrix[ρa_name, ρa_name]
                    if !use_derivative(sgs_advection_flag)
                        @. ∂ᶜρa_err_∂ᶜρa =
                            zero(typeof(∂ᶜρa_err_∂ᶜρa)) - (I,)
                    end
                    @. ∂ᶜρa_err_∂ᶜρa += dtγ * DiagonalMatrixRow(ᶜSq_tot)
                end
            end
        end

        # 1M EDMF: diagonal entries for individual condensate species.
        if p.atmos.microphysics_model isa NonEquilibriumMicrophysics1M
            # Cloud (q_liq, q_ice): BMT analytical derivatives precomputed per
            # updraft.  Same pattern as grid-mean (dominated by −1/τ_relax).
            (; ᶜmp_derivativeʲs) = p.precomputed
            ᶜ∂Sq_liq = ᶜmp_derivativeʲs.:(1).∂tendency_∂q_lcl
            ᶜ∂Sq_ice = ᶜmp_derivativeʲs.:(1).∂tendency_∂q_icl
            sgs_cloud_deriv_tracers = (
                (@name(c.sgsʲs.:(1).q_liq), ᶜ∂Sq_liq),
                (@name(c.sgsʲs.:(1).q_ice), ᶜ∂Sq_ice),
            )
            MatrixFields.unrolled_foreach(
                sgs_cloud_deriv_tracers,
            ) do (q_name, ᶜ∂S∂q)
                MatrixFields.has_field(Y, q_name) || return
                ∂ᶜq_err_∂ᶜq = matrix[q_name, q_name]
                if !use_derivative(sgs_advection_flag)
                    @. ∂ᶜq_err_∂ᶜq =
                        zero(typeof(∂ᶜq_err_∂ᶜq)) - (I,)
                end
                @. ∂ᶜq_err_∂ᶜq += dtγ * DiagonalMatrixRow(ᶜ∂S∂q)
            end

            # Precipitation (q_rai, q_sno): S/q computed inline using frozen
            # tendencies and the current iterate.  Matches grid-mean treatment.
            (; ᶜmp_tendencyʲs) = p.precomputed

            sgs_precip_sq_tracers = (
                (
                    @name(c.sgsʲs.:(1).q_rai),
                    ᶜmp_tendencyʲs.:(1).dq_rai_dt,
                    Y.c.sgsʲs.:(1).q_rai,
                ),
                (
                    @name(c.sgsʲs.:(1).q_sno),
                    ᶜmp_tendencyʲs.:(1).dq_sno_dt,
                    Y.c.sgsʲs.:(1).q_sno,
                ),
            )
            MatrixFields.unrolled_foreach(
                sgs_precip_sq_tracers,
            ) do (q_name, ᶜS, ᶜq)
                MatrixFields.has_field(Y, q_name) || return
                ∂ᶜq_err_∂ᶜq = matrix[q_name, q_name]
                if !use_derivative(sgs_advection_flag)
                    @. ∂ᶜq_err_∂ᶜq =
                        zero(typeof(∂ᶜq_err_∂ᶜq)) - (I,)
                end
                @. ∂ᶜq_err_∂ᶜq += dtγ * DiagonalMatrixRow(_jac_coeff(ᶜS, ᶜq))
            end
        end

        # TODO: 2M EDMF updraft Jacobian entries remain to be implemented.
        # This requires extending the Jacobian sparsity pattern to include
        # diagonal blocks for updraft n_liq and n_rai species.
        # Without these entries, 2M microphysics should use explicit
        # timestepping for stability.

    end
    return nothing
end

function invert_jacobian!(alg::ManualSparseJacobian, cache, ΔY, R)
    # Step 1: Column-local solve
    LinearAlgebra.ldiv!(ΔY, cache.matrix, R)

    # Step 2: Horizontal Helmholtz correction (if enabled)
    if use_derivative(alg.acoustic_diagonal_flag) &&
       !isnothing(cache.helmholtz_state)
        helmholtz_correction!(cache, ΔY)
    end
end

"""
    helmholtz_correction!(cache, ΔY)

Apply multiplicative horizontal Helmholtz correction after the column-local
solve. This solves the horizontal acoustic Schur complement:
    (I - dtγ²·cs²·∇²h)·Δρ = z.ρ - dtγ·wdivₕ(ρ·z.uₕ)
via Jacobi-preconditioned Richardson iteration, then back-substitutes to
correct uₕ and ρe_tot.
"""
function helmholtz_correction!(cache, ΔY)
    (; helmholtz_state, helmholtz_scratch) = cache
    (; dtγ, ᶜα_acoustic, ᶜcs², ᶜρ, ᶜe_tot, n_helmholtz_iters) = helmholtz_state
    (; ᶜhelmholtz_ρ, ᶜhelmholtz_rhs, ᶜhelmholtz_laplacian, ᶜhelmholtz_dss_buffer) =
        helmholtz_scratch

    FT = eltype(dtγ)
    α = FT(dtγ)^2  # cs² factor already stored in ᶜcs²

    # Step 2a: Form Helmholtz RHS = z.ρ - dtγ·wdivₕ(ρ·z.uₕ)
    @. ᶜhelmholtz_rhs = wdivₕ(ᶜρ * ΔY.c.uₕ)
    Spaces.weighted_dss!(ᶜhelmholtz_rhs => ᶜhelmholtz_dss_buffer)
    @. ᶜhelmholtz_rhs = ΔY.c.ρ - FT(dtγ) * ᶜhelmholtz_rhs

    # Step 2b: Jacobi-preconditioned Richardson iteration
    # Solve (I - α·cs²·∇²h)Δρ = rhs, where D = 1 + ᶜα_acoustic
    @. ᶜhelmholtz_ρ = ᶜhelmholtz_rhs  # initial guess
    for _ in 1:n_helmholtz_iters
        @. ᶜhelmholtz_laplacian = wdivₕ(gradₕ(ᶜhelmholtz_ρ))
        Spaces.weighted_dss!(ᶜhelmholtz_laplacian => ᶜhelmholtz_dss_buffer)
        @. ᶜhelmholtz_ρ +=
            (ᶜhelmholtz_rhs - ᶜhelmholtz_ρ + α * ᶜcs² * ᶜhelmholtz_laplacian) /
            (FT(1) + ᶜα_acoustic)
    end

    # Step 2c: Save old z.ρ (still in ΔY.c.ρ) before overwriting
    @. ᶜhelmholtz_laplacian = ΔY.c.ρ
    ΔY.c.ρ .= ᶜhelmholtz_ρ

    # Step 2d: Back-substitute uₕ correction
    # Note: no DSS on ΔY.c.uₕ here — this is a preconditioner approximation;
    # GMRES applies the full operator (including DSS) in the next iteration.
    @. ΔY.c.uₕ -= C12(FT(dtγ) * (ᶜcs² / ᶜρ) * gradₕ(ᶜhelmholtz_ρ))

    # Step 2e: Correct ρe_tot (isentropic approximation)
    # Δρe_tot += e_tot · (Δρ_new - z.ρ_old)
    @. ΔY.c.ρe_tot += ᶜe_tot * (ᶜhelmholtz_ρ - ᶜhelmholtz_laplacian)
end
