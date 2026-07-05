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

function jacobian_cache(alg::ManualSparseJacobian, Y, atmos)
    derivative_flags = _derivative_flags(atmos, Y)
    (; topography_flag, diffusion_flag) = derivative_flags
    approximate_solve_iters = alg.approximate_solve_iters
    FT = Spaces.undertype(axes(Y.c))

    DiagonalRow = DiagonalMatrixRow{FT}
    TridiagonalRow = TridiagonalMatrixRow{FT}
    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    TridiagonalRow_ACT12 = TridiagonalMatrixRow{typeof(CT12(FT(0), FT(0))')}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{typeof(CT3(FT(0))')}
    BidiagonalRow_C3xACT12 =
        BidiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CT12{FT})')}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CT3{FT})')}

    is_in_Y(name) = MatrixFields.has_field(Y, name)

    ρq_tot_if_available = is_in_Y(@name(c.ρq_tot)) ? (@name(c.ρq_tot),) : ()
    ρtke_if_available =
        is_in_Y(@name(c.ρtke)) ? (@name(c.ρtke),) : ()
    sfc_if_available = is_in_Y(@name(sfc)) ? (@name(sfc),) : ()
    ρq_gas_A_if_available = is_in_Y(@name(c.ρq_gas_A)) ? (@name(c.ρq_gas_A),) : ()
    sgs_q_gas_A_if_available =
        is_in_Y(@name(c.sgsʲs.:(1).q_gas_A)) ? (@name(c.sgsʲs.:(1).q_gas_A),) : ()

    condensate_mass_names = (
        @name(c.ρq_lcl),
        @name(c.ρq_icl),
        @name(c.ρq_rai),
        @name(c.ρq_sno),
    )
    available_condensate_mass_names =
        filter(is_in_Y, condensate_mass_names)
    condensate_names = (
        condensate_mass_names...,
        @name(c.ρn_lcl),
        @name(c.ρn_rai),
        # P3 frozen
        @name(c.ρn_ice), @name(c.ρq_rim), @name(c.ρb_rim),
    )
    available_condensate_names =
        filter(is_in_Y, condensate_names)
    available_tracer_names =
        (ρq_tot_if_available..., available_condensate_names...)

    # we define the list of condensate masses separately because ρa and q_tot
    # depend on the masses via sedimentation
    sgs_condensate_mass_names = (
        @name(c.sgsʲs.:(1).q_lcl),
        @name(c.sgsʲs.:(1).q_icl),
        @name(c.sgsʲs.:(1).q_rai),
        @name(c.sgsʲs.:(1).q_sno),
    )
    available_sgs_condensate_mass_names =
        filter(is_in_Y, sgs_condensate_mass_names)

    sgs_condensate_names =
        (sgs_condensate_mass_names..., @name(c.sgsʲs.:(1).n_lcl), @name(c.sgsʲs.:(1).n_rai))
    available_sgs_condensate_names =
        filter(is_in_Y, sgs_condensate_names)

    sgs_scalar_names =
        (
            sgs_condensate_names...,
            @name(c.sgsʲs.:(1).q_tot),
            @name(c.sgsʲs.:(1).mse),
            sgs_q_gas_A_if_available...,
        )
    available_sgs_scalar_names =
        filter(is_in_Y, sgs_scalar_names)

    sgs_u³_if_available =
        is_in_Y(@name(f.sgsʲs.:(1).u₃)) ? (@name(f.sgsʲs.:(1).u₃),) : ()

    # Note: We have to use FT(-1) * I instead of -I because inv(-1) == -1.0,
    # which means that multiplying inv(-1) by a Float32 will yield a Float64.
    identity_blocks = map(
        name -> (name, name) => FT(-1) * I,
        (@name(c.ρ), sfc_if_available..., ρq_gas_A_if_available...),
    )

    active_scalar_names = (@name(c.ρ), @name(c.ρe_tot), ρq_tot_if_available...)
    advection_blocks = (
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
            available_condensate_mass_names,
        )...,
        (@name(f.u₃), @name(c.uₕ)) => similar(Y.f, BidiagonalRow_C3xACT12),
        (@name(f.u₃), @name(f.u₃)) => similar(Y.f, TridiagonalRow_C3xACT3),
    )

    diffused_scalar_names = (@name(c.ρe_tot), available_tracer_names...)
    diffusion_blocks = if use_derivative(diffusion_flag)
        (
            map(
                name -> (name, @name(c.ρ)) => similar(Y.c, TridiagonalRow),
                (diffused_scalar_names..., ρtke_if_available...),
            )...,
            map(
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
            map(
                name -> (@name(c.ρe_tot), name) => similar(Y.c, TridiagonalRow),
                available_condensate_mass_names,
            )...,
            # TODO should we check is_in_Y(@name(c.ρq_tot)) here
            map(
                name -> (@name(c.ρq_tot), name) => similar(Y.c, TridiagonalRow),
                available_condensate_mass_names,
            )...,
            (@name(c.uₕ), @name(c.uₕ)) =>
                !isnothing(atmos.turbconv_model) ||
                    !disable_momentum_vertical_diffusion(
                        atmos.vertical_diffusion,
                    ) ? similar(Y.c, TridiagonalRow) : FT(-1) * I,
        )
    elseif atmos.microphysics_model isa DryModel
        map(
            name -> (name, name) => FT(-1) * I,
            (diffused_scalar_names..., ρtke_if_available..., @name(c.uₕ)),
        )
    else
        (
            map(
                name -> (name, name) => similar(Y.c, TridiagonalRow),
                diffused_scalar_names,
            )...,
            map(
                name -> (@name(c.ρe_tot), name) => similar(Y.c, TridiagonalRow),
                available_condensate_mass_names,
            )...,
            map(
                name -> (@name(c.ρq_tot), name) => similar(Y.c, TridiagonalRow),
                available_condensate_mass_names,
            )...,
            (@name(c.ρe_tot), @name(c.ρq_tot)) =>
                similar(Y.c, TridiagonalRow),
            map(
                name -> (name, name) => FT(-1) * I,
                (ρtke_if_available..., @name(c.uₕ)),
            )...,
        )
    end

    sgs_advection_blocks = if atmos.turbconv_model isa PrognosticEDMFX
        (
            map(
                name -> (name, name) => similar(Y.c, TridiagonalRow),
                available_sgs_scalar_names,
            )...,
            map(
                name ->
                    (@name(c.sgsʲs.:(1).q_tot), name) =>
                        similar(Y.c, TridiagonalRow),
                available_sgs_condensate_mass_names,
            )...,
            (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).ρa)) => FT(-1) * I,
            (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) => FT(-1) * I,
        )
    else
        ()
    end

    sgs_massflux_blocks =
        if atmos.turbconv_model isa PrognosticEDMFX &&
           atmos.edmfx_model.sgs_mass_flux isa Val{true}
            (
                map(
                    name ->
                        (name, get_χʲ_name_from_ρχ_name(name)) =>
                            similar(Y.c, TridiagonalRow),
                    available_tracer_names,
                )...,
                map(
                    name ->
                        (name, @name(f.u₃)) =>
                            similar(Y.c, BidiagonalRow_ACT3),
                    available_condensate_names,
                )...,
                (@name(c.ρe_tot), @name(c.sgsʲs.:(1).mse)) =>
                    similar(Y.c, TridiagonalRow),
                # (ρe_tot, ρ) and (ρq_tot, ρ) are needed for the mass flux Jacobian.
                # When diffusion is implicit they already appear in diffusion_blocks;
                # add them here only when diffusion is explicit to avoid duplicates.
                (
                    use_derivative(diffusion_flag) ? () :
                    (
                        (@name(c.ρe_tot), @name(c.ρ)) => similar(Y.c, TridiagonalRow),
                        (@name(c.ρq_tot), @name(c.ρ)) => similar(Y.c, TridiagonalRow),
                    )
                )...,
            )
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
    sgs_ρa_if_available =
        is_in_Y(@name(c.sgsʲs.:(1).ρa)) ? (@name(c.sgsʲs.:(1).ρa),) : ()
    available_scalar_names = (
        mass_and_surface_names...,
        available_tracer_names...,
        @name(c.ρe_tot),
        ρtke_if_available...,
        available_sgs_scalar_names...,
        sgs_ρa_if_available...,
    )

    velocity_alg = MatrixFields.BlockLowerTriangularSolve(
        @name(c.uₕ),
        sgs_u³_if_available...,
    )
    full_alg =
        if use_derivative(diffusion_flag) ||
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
                if atmos.turbconv_model isa PrognosticEDMFX
                    MatrixFields.BlockLowerTriangularSolve(
                        available_sgs_condensate_names...;
                        alg₂ = MatrixFields.BlockLowerTriangularSolve(
                            @name(c.sgsʲs.:(1).q_tot),
                            @name(c.sgsʲs.:(1).mse);
                            alg₂ = gs_scalar_subalg,
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

    return (;
        matrix = MatrixFields.FieldMatrixWithSolver(matrix, Y, full_alg),
        derivative_flags,
    )
end

# TODO: There are a few for loops in this function. This is because
# using unrolled_foreach allocates (breaks the flame tests)
function update_jacobian!(alg::ManualSparseJacobian, cache, Y, p, dtγ, t)
    (; topography_flag, diffusion_flag) = cache.derivative_flags
    (; matrix) = cache
    (; params) = p
    (; ᶜΦ) = p.core
    (; ᶜu, ᶠu³, ᶜK, ᶜp, ᶜT, ᶜh_tot) = p.precomputed
    (; ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
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
    zmax = Spaces.z_max(axes(Y.f))

    ᶜkappa_m = p.scratch.ᶜtemp_scalar
    @. ᶜkappa_m =
        TD.gas_constant_air(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) /
        TD.cv_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)

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

    ᶜθ_v = @. lazy(theta_v(thermo_params, ᶜT, ᶜp, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
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
            (@name(c.ρq_lcl), e_int_v0, Δcv_l),
            (@name(c.ρq_icl), e_int_s0, Δcv_i),
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

    α_vert_diff_microphysics = CAP.α_vert_diff_tracer(params)
    tracer_info = (
        (@name(c.ρq_lcl), @name(ᶜwₗ)),
        (@name(c.ρq_icl), @name(ᶜwᵢ)),
        (@name(c.ρq_rai), @name(ᶜwᵣ)),
        (@name(c.ρq_sno), @name(ᶜwₛ)),
        (@name(c.ρn_lcl), @name(ᶜwₙₗ)),
        (@name(c.ρn_rai), @name(ᶜwₙᵣ)),
        (@name(c.ρn_ice), @name(ᶜwnᵢ)),
        (@name(c.ρq_rim), @name(ᶜwᵢ)),
        (@name(c.ρb_rim), @name(ᶜwᵢ)),
    )
    internal_energy_func(name) =
        (name == @name(c.ρq_lcl) || name == @name(c.ρq_rai)) ? TD.internal_energy_liquid :
        (name == @name(c.ρq_icl) || name == @name(c.ρq_sno)) ? TD.internal_energy_ice :
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
            dtγ * (-(ᶜprecipdivᵥ_matrix())) ⋅
            DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ)

        MatrixFields.unrolled_foreach(tracer_info) do (ρχₚ_name, wₚ_name)
            MatrixFields.has_field(Y, ρχₚ_name) || return

            ∂ᶜρχₚ_err_∂ᶜρχₚ = matrix[ρχₚ_name, ρχₚ_name]
            ᶜwₚ = MatrixFields.get_field(p.precomputed, wₚ_name)
            # TODO: come up with read-able names for the intermediate computations...
            @. p.scratch.ᶠband_matrix_wvec =
                ᶠright_bias_matrix() ⋅
                DiagonalMatrixRow(ClimaCore.Geometry.WVector(-(ᶜwₚ) / ᶜρ))
            @. ∂ᶜρχₚ_err_∂ᶜρχₚ =
                p.scratch.ᶜbidiagonal_adjoint_matrix_c3 ⋅
                p.scratch.ᶠband_matrix_wvec - (I,)

            if ρχₚ_name in
               (@name(c.ρq_lcl), @name(c.ρq_icl), @name(c.ρq_rai), @name(c.ρq_sno))
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

        ∂ᶠρχ_dif_flux_∂ᶜχ = ᶠp_grad_matrix
        # Harmonic-mean face interpolation of K, consistent with the diffusive
        # tendencies (see edmfx_sgs_diffusive_flux_tendency! and
        # vertical_diffusion_boundary_layer_tendency!). Smagorinsky tendencies
        # still use arithmetic interpolation, so their Jacobian does too.
        ϵK = eps(FT)
        if is_smagorinsky_vertical(smagorinsky_lilly)
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
            if is_smagorinsky_vertical(smagorinsky_lilly)
                @. ∂ᶠρχ_dif_flux_∂ᶜχ =
                    DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_u)) ⋅ ᶠgradᵥ_matrix()
            else
                @. ∂ᶠρχ_dif_flux_∂ᶜχ =
                    DiagonalMatrixRow(ᶠinterp(ᶜρ) / ᶠinterp(1 / max(ᶜK_u, ϵK))) ⋅
                    ᶠgradᵥ_matrix()
            end
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

        MatrixFields.unrolled_foreach(tracer_info) do (ρχ_name, _)
            MatrixFields.has_field(Y, ρχ_name) || return
            ∂ᶜρχ_err_∂ᶜρ = matrix[ρχ_name, @name(c.ρ)]
            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_name, ρχ_name]
            @. ∂ᶜρχ_err_∂ᶜρ = zero(typeof(∂ᶜρχ_err_∂ᶜρ))
            @. ∂ᶜρχ_err_∂ᶜρχ +=
                dtγ * α_vert_diff_microphysics * ᶜdiffusion_h_matrix ⋅
                DiagonalMatrixRow(1 / ᶜρ)
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

    if p.atmos.turbconv_model isa PrognosticEDMFX
        begin # sgs_adv always implicit
            (;
                ᶜρʲs,
                ᶠu³ʲs,
                ᶜTʲs,
                ᶜq_tot_nonnegʲs,
                ᶜq_liqʲs,
                ᶜq_iceʲs,
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
                    ᶜq_tot_nonnegʲs.:(1),
                    ᶜq_liqʲs.:(1),
                    ᶜq_iceʲs.:(1),
                ) /
                TD.cv_m(
                    thermo_params,
                    ᶜq_tot_nonnegʲs.:(1),
                    ᶜq_liqʲs.:(1),
                    ᶜq_iceʲs.:(1),
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
                    ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1)))
                ) - (I,)

            # advection of passive chemistry tracers (no sedimentation)
            if MatrixFields.has_field(Y, @name(c.sgsʲs.:(1).q_gas_A))
                ∂ᶜq_gas_Aʲ_err_∂ᶜq_gas_Aʲ =
                    matrix[@name(c.sgsʲs.:(1).q_gas_A), @name(c.sgsʲs.:(1).q_gas_A)]
                @. ∂ᶜq_gas_Aʲ_err_∂ᶜq_gas_Aʲ =
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
                sgs_microphysics_tracers = (
                    (@name(c.sgsʲs.:(1).q_lcl), @name(ᶜwₗʲs.:(1))),
                    (@name(c.sgsʲs.:(1).q_icl), @name(ᶜwᵢʲs.:(1))),
                    (@name(c.sgsʲs.:(1).q_rai), @name(ᶜwᵣʲs.:(1))),
                    (@name(c.sgsʲs.:(1).q_sno), @name(ᶜwₛʲs.:(1))),
                    (@name(c.sgsʲs.:(1).n_lcl), @name(ᶜwₙₗʲs.:(1))),
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
                        @name(c.sgsʲs.:(1).q_lcl),
                        @name(c.sgsʲs.:(1).q_icl),
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

            # vertical diffusion of updrafts — uses ᶜK_h computed in diffusion block
            if use_derivative(diffusion_flag) # sgs_vertdiff always implicit
                α_vert_diff_microphysics = CAP.α_vert_diff_tracer(params)
                # Harmonic-mean face K, consistent with
                # edmfx_vertical_diffusion_tendency!
                ϵK = eps(FT)
                @. ᶜdiffusion_h_matrix =
                    ᶜadvdivᵥ_matrix() ⋅
                    DiagonalMatrixRow(
                        ᶠinterp(ᶜρʲs.:(1)) / ᶠinterp(1 / max(ᶜK_h, ϵK)),
                    ) ⋅ ᶠgradᵥ_matrix()

                @. ∂ᶜmseʲ_err_∂ᶜmseʲ +=
                    dtγ * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅ ᶜdiffusion_h_matrix
                @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ +=
                    dtγ * DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅ ᶜdiffusion_h_matrix
                if p.atmos.microphysics_model isa Union{
                    NonEquilibriumMicrophysics1M,
                    NonEquilibriumMicrophysics2M,
                }
                    sgs_microphysics_tracers = (
                        (@name(c.sgsʲs.:(1).q_lcl)), (@name(c.sgsʲs.:(1).q_icl)),
                        (@name(c.sgsʲs.:(1).q_rai)), (@name(c.sgsʲs.:(1).q_sno)),
                        (@name(c.sgsʲs.:(1).n_lcl)), (@name(c.sgsʲs.:(1).n_rai)),
                    )
                    MatrixFields.unrolled_foreach(
                        sgs_microphysics_tracers,
                    ) do (χʲ_name)
                        MatrixFields.has_field(Y, χʲ_name) || return
                        ∂ᶜχʲ_err_∂ᶜχʲ = matrix[χʲ_name, χʲ_name]
                        @. ∂ᶜχʲ_err_∂ᶜχʲ +=
                            dtγ * α_vert_diff_microphysics *
                            DiagonalMatrixRow(1 / ᶜρʲs.:(1)) ⋅
                            ᶜdiffusion_h_matrix
                    end
                end
            end
            # entrainment and detrainment (rates are treated explicitly)
            begin # sgs_entr_detr always implicit
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
                @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ -=
                    dtγ * DiagonalMatrixRow(ᶜentrʲ + ᶜturb_entrʲs.:(1))
                @. ∂ᶜmseʲ_err_∂ᶜmseʲ -=
                    dtγ * DiagonalMatrixRow(ᶜentrʲ + ᶜturb_entrʲs.:(1))
                if p.atmos.microphysics_model isa Union{
                    NonEquilibriumMicrophysics1M,
                    NonEquilibriumMicrophysics2M,
                }
                    sgs_microphysics_tracers = (
                        (@name(c.sgsʲs.:(1).q_lcl)),
                        (@name(c.sgsʲs.:(1).q_icl)),
                        (@name(c.sgsʲs.:(1).q_rai)),
                        (@name(c.sgsʲs.:(1).q_sno)),
                    )
                    MatrixFields.unrolled_foreach(
                        sgs_microphysics_tracers,
                    ) do (qʲ_name)
                        MatrixFields.has_field(Y, qʲ_name) || return

                        ∂ᶜqʲ_err_∂ᶜqʲ = matrix[qʲ_name, qʲ_name]
                        @. ∂ᶜqʲ_err_∂ᶜqʲ -=
                            dtγ * DiagonalMatrixRow(ᶜentrʲ + ᶜturb_entrʲs.:(1))
                    end
                end
                # passive chemistry tracers
                if MatrixFields.has_field(Y, @name(c.sgsʲs.:(1).q_gas_A))
                    @. ∂ᶜq_gas_Aʲ_err_∂ᶜq_gas_Aʲ -=
                        dtγ * DiagonalMatrixRow(ᶜentrʲ + ᶜturb_entrʲs.:(1))
                end
            end

            # Surface mass-flux BC at level 1 contributes
            #   ∂F_BC/∂mse[1] = ∂F_BC/∂q_tot[1] = -mass_flux_source/ρa_floor
            # where ρa_floor = max(ρa, ρ·a_min). Build a level-1-only
            # rate field (zero elsewhere) and add as a diagonal.
            begin
                turbconv_params_local = CAP.turbconv_params(params)
                a_min = CAP.min_area(turbconv_params_local)
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
                @. ∂ᶜmseʲ_err_∂ᶜmseʲ -=
                    dtγ * DiagonalMatrixRow(ᶜsfc_bc_rate)
                @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ -=
                    dtγ * DiagonalMatrixRow(ᶜsfc_bc_rate)
            end

            # add updraft mass flux contributions to grid-mean
            if p.atmos.edmfx_model.sgs_mass_flux isa Val{true}

                # If diffusion is explicit, zero-initialize (ρe_tot, ρ) and
                # (ρq_tot, ρ) here so both blocks can safely use +=.
                if !use_derivative(diffusion_flag)
                    ∂ᶜρe_tot_err_∂ᶜρ = matrix[@name(c.ρe_tot), @name(c.ρ)]
                    @. ∂ᶜρe_tot_err_∂ᶜρ = zero(typeof(∂ᶜρe_tot_err_∂ᶜρ))
                    ∂ᶜρq_tot_err_∂ᶜρ = matrix[@name(c.ρq_tot), @name(c.ρ)]
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
                ᶜkappa_m = p.scratch.ᶜtemp_scalar
                @. ᶜkappa_m =
                    TD.gas_constant_air(
                        thermo_params,
                        ᶜq_tot_nonneg,
                        ᶜq_liq,
                        ᶜq_ice,
                    ) /
                    TD.cv_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)


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
                # -∇·(ρ u³ χ) is handled explicitly), so all derivatives below
                # are of these difference-form fluxes.
                if p.atmos.microphysics_model isa Union{
                    NonEquilibriumMicrophysics1M,
                    NonEquilibriumMicrophysics2M,
                }

                    microphysics_tracers = (
                        (@name(c.ρq_lcl), @name(c.sgsʲs.:(1).q_lcl), @name(q_lcl)),
                        (@name(c.ρq_icl), @name(c.sgsʲs.:(1).q_icl), @name(q_icl)),
                        (@name(c.ρq_rai), @name(c.sgsʲs.:(1).q_rai), @name(q_rai)),
                        (@name(c.ρq_sno), @name(c.sgsʲs.:(1).q_sno), @name(q_sno)),
                        (@name(c.ρn_lcl), @name(c.sgsʲs.:(1).n_lcl), @name(n_lcl)),
                        (@name(c.ρn_rai), @name(c.sgsʲs.:(1).n_rai), @name(n_rai)),
                    )

                    ᶠu³_data = ᶠu³.components.data.:1

                    # add updraft contributions
                    # pull common subexpressions that don't depend on which
                    # tracer out of the tracer loop for performance
                    @. ᶜtracer_advection_matrix =
                        -(ᶜadvdivᵥ_matrix()) ⋅
                        DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ)
                    @. ᶜtridiagonal_matrix =
                        ᶜtracer_advection_matrix ⋅ ᶠset_tracer_upwind_matrix_bcs(
                            ᶠtracer_upwind_matrix(ᶠu³ʲs.:(1) - ᶠu³),
                        )
                    MatrixFields.unrolled_foreach(
                        microphysics_tracers,
                    ) do (ρχ_name, χʲ_name, χ_name)
                        MatrixFields.has_field(Y, ρχ_name) || return
                        ᶜρχ = MatrixFields.get_field(Y, ρχ_name)
                        ᶜχʲ = MatrixFields.get_field(Y, χʲ_name)

                        ∂ᶜρχ_err_∂ᶜχʲ =
                            matrix[ρχ_name, χʲ_name]
                        @. ∂ᶜρχ_err_∂ᶜχʲ =
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)))

                        ∂ᶜρχ_err_∂ᶜρχ =
                            matrix[ρχ_name, ρχ_name]
                        @. ∂ᶜρχ_err_∂ᶜρχ +=
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(
                                -1 * draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)) /
                                Y.c.ρ,
                            )

                        ∂ᶜρχ_err_∂ᶠu₃ =
                            matrix[ρχ_name, @name(f.u₃)]
                        @. ∂ᶜρχ_err_∂ᶠu₃ =
                            -dtγ * ᶜtracer_advection_matrix ⋅
                            DiagonalMatrixRow(
                                ᶠset_tracer_upwind_bcs(
                                    ᶠtracer_upwind(
                                        CT3(sign(ᶠu³ʲ_data - ᶠu³_data)),
                                        (ᶜχʲ - specific(ᶜρχ, Y.c.ρ)) *
                                        draft_area(
                                            Y.c.sgsʲs.:(1).ρa,
                                            ᶜρʲs.:(1),
                                        ),
                                    ),
                                ) * adjoint(C3(sign(ᶠu³ʲ_data - ᶠu³_data))) *
                                g³³(ᶠgⁱʲ),
                            )
                    end

                    # add env flux contributions
                    (; ᶜp) = p.precomputed
                    (; ᶠu³⁰, ᶜT⁰, ᶜq_tot_nonneg⁰, ᶜq_liq⁰, ᶜq_ice⁰) =
                        p.precomputed
                    ᶜρ⁰ = @. lazy(
                        TD.air_density(
                            thermo_params,
                            ᶜT⁰,
                            ᶜp,
                            ᶜq_tot_nonneg⁰,
                            ᶜq_liq⁰,
                            ᶜq_ice⁰,
                        ),
                    )
                    ᶜρa⁰ = @. lazy(ρa⁰(Y.c.ρ, Y.c.sgsʲs, p.atmos.turbconv_model))
                    ᶠu³⁰_data = ᶠu³⁰.components.data.:1

                    # pull common subexpressions that don't depend on which
                    # tracer out of the tracer loop for performance
                    @. ᶜtracer_advection_matrix =
                        -(ᶜadvdivᵥ_matrix()) ⋅
                        DiagonalMatrixRow(ᶠinterp(ᶜρ⁰ * ᶜJ) / ᶠJ)
                    @. ᶜtridiagonal_matrix =
                        ᶜtracer_advection_matrix ⋅ ᶠset_tracer_upwind_matrix_bcs(
                            ᶠtracer_upwind_matrix(ᶠu³⁰ - ᶠu³),
                        )
                    MatrixFields.unrolled_foreach(
                        microphysics_tracers,
                    ) do (ρχ_name, χʲ_name, χ_name)
                        MatrixFields.has_field(Y, ρχ_name) || return
                        ᶜρχ = MatrixFields.get_field(Y, ρχ_name)
                        ᶜχ⁰ = ᶜspecific_env_value(χ_name, Y, p)

                        ∂ᶜρχ_err_∂ᶜχʲ =
                            matrix[ρχ_name, χʲ_name]
                        @. ∂ᶜρχ_err_∂ᶜχʲ +=
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(-1 * Y.c.sgsʲs.:(1).ρa / ᶜρ⁰)

                        ∂ᶜρχ_err_∂ᶜρχ =
                            matrix[ρχ_name, ρχ_name]
                        @. ∂ᶜρχ_err_∂ᶜρχ +=
                            dtγ *
                            ᶜtridiagonal_matrix ⋅
                            DiagonalMatrixRow(
                                1 / ᶜρ⁰ - draft_area(ᶜρa⁰, ᶜρ⁰) / Y.c.ρ,
                            )

                        ∂ᶜρχ_err_∂ᶠu₃ =
                            matrix[ρχ_name, @name(f.u₃)]
                        @. ∂ᶜρχ_err_∂ᶠu₃ +=
                            dtγ * ᶜtracer_advection_matrix ⋅
                            DiagonalMatrixRow(
                                ᶠset_tracer_upwind_bcs(
                                    ᶠtracer_upwind(
                                        CT3(sign(ᶠu³⁰_data - ᶠu³_data)),
                                        (ᶜχ⁰ - specific(ᶜρχ, Y.c.ρ)) *
                                        draft_area(ᶜρa⁰, ᶜρ⁰),
                                    ),
                                ) * adjoint(C3(sign(ᶠu³⁰_data - ᶠu³_data))) *
                                (ᶠinterp(Y.c.ρ / ᶜρa⁰) - 1) * g³³(ᶠgⁱʲ),
                            )
                    end
                end
            end
        end
    end

    # NOTE: All velocity tendency derivatives should be set BEFORE this call.
    zero_velocity_jacobian!(matrix, Y, p, t)
end

invert_jacobian!(::ManualSparseJacobian, cache, ΔY, R) =
    LinearAlgebra.ldiv!(ΔY, cache.matrix, R)
