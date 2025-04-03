using ClimaCore.MatrixFields
import LinearAlgebra: I, Adjoint

abstract type DerivativeFlag end
struct UseDerivative <: DerivativeFlag end
struct IgnoreDerivative <: DerivativeFlag end

DerivativeFlag(value) = value ? UseDerivative() : IgnoreDerivative()
DerivativeFlag(mode::AbstractTimesteppingMode) =
    DerivativeFlag(mode == Implicit())

use_derivative(::UseDerivative) = true
use_derivative(::IgnoreDerivative) = false

"""
    ApproxJacobian(
        topography_flag,
        diffusion_flag,
        sgs_advection_flag,
        sgs_entr_detr_flag,
        sgs_mass_flux_flag,
        sgs_nh_pressure_flag,
        approximate_solve_iters,
    )

A `JacobianAlgorithm` that approximates the `ImplicitEquationJacobian` using
analytically derived tendency derivatives and inverts it using a specialized
nested linear solver. Certain groups of derivatives can be toggled on or off by
setting their `DerivativeFlag`s to either `UseDerivative` or `IgnoreDerivative`.

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
- `approximate_solve_iters::Int`: number of iterations to take for the
  approximate linear solve required when the `diffusion_flag` is `UseDerivative`
"""
struct ApproxJacobian{F1, F2, F3, F4, F5, F6} <: JacobianAlgorithm
    topography_flag::F1
    diffusion_flag::F2
    sgs_advection_flag::F3
    sgs_entr_detr_flag::F4
    sgs_mass_flux_flag::F5
    sgs_nh_pressure_flag::F6
    approximate_solve_iters::Int
end

function jacobian_cache(alg::ApproxJacobian, Y, atmos)
    (;
        topography_flag,
        diffusion_flag,
        sgs_advection_flag,
        sgs_mass_flux_flag,
    ) = alg
    FT = Spaces.undertype(axes(Y.c))
    CTh = CTh_vector_type(axes(Y.c))

    DiagonalRow = DiagonalMatrixRow{FT}
    TridiagonalRow = TridiagonalMatrixRow{FT}
    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    TridiagonalRow_ACTh = TridiagonalMatrixRow{Adjoint{FT, CTh{FT}}}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}
    BidiagonalRow_C3xACTh =
        BidiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CTh{FT})')}
    DiagonalRow_C3xACT3 =
        DiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CT3{FT})')}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CT3{FT})')}

    is_in_Y(name) = MatrixFields.has_field(Y, name)

    ρq_tot_if_available = is_in_Y(@name(c.ρq_tot)) ? (@name(c.ρq_tot),) : ()
    ρatke_if_available =
        is_in_Y(@name(c.sgs⁰.ρatke)) ? (@name(c.sgs⁰.ρatke),) : ()
    sfc_if_available = is_in_Y(@name(sfc)) ? (@name(sfc),) : ()

    tracer_names = (
        @name(c.ρq_tot),
        @name(c.ρq_liq),
        @name(c.ρq_ice),
        @name(c.ρq_rai),
        @name(c.ρq_sno),
    )
    available_tracer_names = MatrixFields.unrolled_filter(is_in_Y, tracer_names)

    sgs_tracer_names = (
        @name(c.sgsʲs.:(1).q_tot),
        @name(c.sgsʲs.:(1).q_liq),
        @name(c.sgsʲs.:(1).q_ice),
        @name(c.sgsʲs.:(1).q_rai),
        @name(c.sgsʲs.:(1).q_sno),
    )
    available_sgs_tracer_names =
        MatrixFields.unrolled_filter(is_in_Y, sgs_tracer_names)

    sgs_scalar_names =
        (sgs_tracer_names..., @name(c.sgsʲs.:(1).mse), @name(c.sgsʲs.:(1).ρa))
    available_sgs_scalar_names =
        MatrixFields.unrolled_filter(is_in_Y, sgs_scalar_names)

    sgs_u³_if_available =
        is_in_Y(@name(f.sgsʲs.:(1).u₃)) ? (@name(f.sgsʲs.:(1).u₃),) : ()

    # Note: We have to use FT(-1) * I instead of -I because inv(-1) == -1.0,
    # which means that multiplying inv(-1) by a Float32 will yield a Float64.
    identity_blocks = MatrixFields.unrolled_map(
        name -> (name, name) => FT(-1) * I,
        (@name(c.ρ), sfc_if_available...),
    )

    active_scalar_names = (@name(c.ρ), @name(c.ρe_tot), ρq_tot_if_available...)
    advection_blocks = (
        (
            use_derivative(topography_flag) ?
            MatrixFields.unrolled_map(
                name ->
                    (name, @name(c.uₕ)) =>
                        similar(Y.c, TridiagonalRow_ACTh),
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
        (@name(f.u₃), @name(c.uₕ)) => similar(Y.f, BidiagonalRow_C3xACTh),
        (@name(f.u₃), @name(f.u₃)) => similar(Y.f, TridiagonalRow_C3xACT3),
    )

    diffused_scalar_names = (@name(c.ρe_tot), available_tracer_names...)
    diffusion_blocks = if use_derivative(diffusion_flag)
        (
            MatrixFields.unrolled_map(
                name -> (name, @name(c.ρ)) => similar(Y.c, TridiagonalRow),
                (diffused_scalar_names..., ρatke_if_available...),
            )...,
            MatrixFields.unrolled_map(
                name -> (name, name) => similar(Y.c, TridiagonalRow),
                (diffused_scalar_names..., ρatke_if_available...),
            )...,
            (
                is_in_Y(@name(c.ρq_tot)) ?
                (
                    (@name(c.ρe_tot), @name(c.ρq_tot)) =>
                        similar(Y.c, TridiagonalRow),
                ) : ()
            )...,
            (@name(c.uₕ), @name(c.uₕ)) =>
                !isnothing(atmos.turbconv_model) ||
                    !disable_momentum_vertical_diffusion(atmos.vert_diff) ?
                similar(Y.c, TridiagonalRow) : FT(-1) * I,
        )
    elseif atmos.moisture_model isa DryModel
        MatrixFields.unrolled_map(
            name -> (name, name) => FT(-1) * I,
            (diffused_scalar_names..., ρatke_if_available..., @name(c.uₕ)),
        )
    else
        (
            MatrixFields.unrolled_map(
                name -> (name, name) => similar(Y.c, TridiagonalRow),
                diffused_scalar_names,
            )...,
            (@name(c.ρe_tot), @name(c.ρq_tot)) =>
                similar(Y.c, TridiagonalRow),
            MatrixFields.unrolled_map(
                name -> (name, name) => FT(-1) * I,
                (ρatke_if_available..., @name(c.uₕ)),
            )...,
        )

    end

    sgs_advection_blocks = if atmos.turbconv_model isa PrognosticEDMFX
        @assert n_prognostic_mass_flux_subdomains(atmos.turbconv_model) == 1
        if use_derivative(sgs_advection_flag)
            (
                MatrixFields.unrolled_map(
                    name -> (name, name) => similar(Y.c, TridiagonalRow),
                    available_sgs_scalar_names,
                )...,
                (@name(c.sgsʲs.:(1).mse), @name(c.sgsʲs.:(1).q_tot)) =>
                    similar(Y.c, DiagonalRow),
                (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).q_tot)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).mse)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(c.sgsʲs.:(1).ρa), @name(f.sgsʲs.:(1).u₃)) =>
                    similar(Y.c, BidiagonalRow_ACT3),
                (@name(c.sgsʲs.:(1).mse), @name(f.sgsʲs.:(1).u₃)) =>
                    similar(Y.c, BidiagonalRow_ACT3),
                (@name(c.sgsʲs.:(1).q_tot), @name(f.sgsʲs.:(1).u₃)) =>
                    similar(Y.c, BidiagonalRow_ACT3),
                (@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).q_tot)) =>
                    similar(Y.f, BidiagonalRow_C3),
                (@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).mse)) =>
                    similar(Y.f, BidiagonalRow_C3),
                (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) =>
                    similar(Y.f, TridiagonalRow_C3xACT3),
            )
        else
            (
                MatrixFields.unrolled_map(
                    name -> (name, name) => FT(-1) * I,
                    available_sgs_scalar_names,
                )...,
                (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) =>
                    !isnothing(atmos.rayleigh_sponge) ?
                    similar(Y.f, DiagonalRow_C3xACT3) : FT(-1) * I,
            )
        end
    else
        ()
    end

    sgs_massflux_blocks = if atmos.turbconv_model isa PrognosticEDMFX
        @assert n_prognostic_mass_flux_subdomains(atmos.turbconv_model) == 1
        if use_derivative(sgs_mass_flux_flag)
            (
                (@name(c.ρe_tot), @name(c.sgsʲs.:(1).mse)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(c.ρq_tot), @name(c.sgsʲs.:(1).q_tot)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(c.ρe_tot), @name(f.sgsʲs.:(1).u₃)) =>
                    similar(Y.c, BidiagonalRow_ACT3),
                (@name(c.ρq_tot), @name(f.sgsʲs.:(1).u₃)) =>
                    similar(Y.c, BidiagonalRow_ACT3),
                (@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(c.ρq_tot), @name(c.sgsʲs.:(1).ρa)) =>
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

    names₁_group₁ = (@name(c.ρ), sfc_if_available...)
    names₁_group₂ = (available_tracer_names..., ρatke_if_available...)
    names₁_group₃ = (@name(c.ρe_tot),)
    names₁ = (
        names₁_group₁...,
        names₁_group₂...,
        names₁_group₃...,
        available_sgs_scalar_names...,
    )

    alg₂ = MatrixFields.BlockLowerTriangularSolve(
        @name(c.uₕ),
        sgs_u³_if_available...,
    )
    solver_alg =
        if use_derivative(diffusion_flag) ||
           use_derivative(sgs_advection_flag) ||
           !(atmos.moisture_model isa DryModel)
            alg₁_subalg₂ =
                if atmos.turbconv_model isa PrognosticEDMFX &&
                   use_derivative(sgs_advection_flag)
                    diff_subalg =
                        use_derivative(diffusion_flag) ?
                        (;
                            alg₂ = MatrixFields.BlockLowerTriangularSolve(
                                names₁_group₂...,
                            )
                        ) : (;)
                    (;
                        alg₂ = MatrixFields.BlockLowerTriangularSolve(
                            available_sgs_tracer_names...;
                            alg₂ = MatrixFields.BlockLowerTriangularSolve(
                                @name(c.sgsʲs.:(1).mse);
                                alg₂ = MatrixFields.BlockLowerTriangularSolve(
                                    @name(c.sgsʲs.:(1).ρa);
                                    diff_subalg...,
                                ),
                            ),
                        )
                    )
                else
                    is_in_Y(@name(c.ρq_tot)) ?
                    (;
                        alg₂ = MatrixFields.BlockLowerTriangularSolve(
                            names₁_group₂...,
                        )
                    ) : (;)
                end
            alg₁ = MatrixFields.BlockLowerTriangularSolve(
                names₁_group₁...;
                alg₁_subalg₂...,
            )
            MatrixFields.ApproximateBlockArrowheadIterativeSolve(
                names₁...;
                alg₁,
                alg₂,
                P_alg₁ = MatrixFields.MainDiagonalPreconditioner(),
                n_iters = alg.approximate_solve_iters,
            )
        else
            MatrixFields.BlockArrowheadSolve(names₁...; alg₂)
        end

    # TODO: Fix bug in ClimaCore's column function, so that we can use
    # lazy.((matrix .+ identity_matrix(Y)) ./ dtγ) instead of preallocating.
    temp_matrix = (matrix .+ identity_matrix(matrix, Y)) ./ FT(1)

    return (;
        matrix = MatrixFields.FieldMatrixWithSolver(matrix, Y, solver_alg),
        temp_matrix,
    )
end

# TODO: Replace some scalar matrix entries with tensor entries so that we can
# use MatrixFields.identity_field_matrix(Y) instead of identity_matrix(Y).
function identity_matrix(matrix, Y)
    I_matrix = MatrixFields.identity_field_matrix(Y)
    new_pairs = MatrixFields.unrolled_map(pairs(I_matrix)) do (key, value)
        replace_tensor_value_with_scalar_value =
            key == (@name(c.uₕ), @name(c.uₕ)) || (
                key == (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) &&
                matrix[key] isa LinearAlgebra.UniformScaling
            )
        key => (replace_tensor_value_with_scalar_value ? I : value)
    end
    return MatrixFields.replace_name_tree(
        MatrixFields.FieldMatrix(new_pairs...),
        MatrixFields.FieldNameTree(Y),
    )
end

function update_jacobian!(alg::ApproxJacobian, cache, Y, p, dtγ, t)
    (;
        topography_flag,
        diffusion_flag,
        sgs_advection_flag,
        sgs_entr_detr_flag,
        sgs_nh_pressure_flag,
        sgs_mass_flux_flag,
    ) = alg
    (; matrix) = cache
    (; params) = p
    (; ᶜΦ, ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜspecific, ᶠu³, ᶜK, ᶜts, ᶜp, ᶜh_tot) = p.precomputed
    (;
        ∂ᶜK_∂ᶜuₕ,
        ∂ᶜK_∂ᶠu₃,
        ᶠp_grad_matrix,
        ᶜadvection_matrix,
        ᶜdiffusion_h_matrix,
        ᶜdiffusion_h_matrix_scaled,
        ᶜdiffusion_u_matrix,
        ᶠbidiagonal_matrix_ct3,
        ᶠbidiagonal_matrix_ct3_2,
        ᶠtridiagonal_matrix_c3,
    ) = p.scratch
    rs = p.atmos.rayleigh_sponge

    FT = Spaces.undertype(axes(Y.c))
    CTh = CTh_vector_type(axes(Y.c))
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'

    cv_d = FT(CAP.cv_d(params))
    Δcv_v = FT(CAP.cv_v(params)) - cv_d
    T_0 = FT(CAP.T_0(params))
    R_d = FT(CAP.R_d(params))
    cp_d = FT(CAP.cp_d(params))
    # This term appears a few times in the Jacobian, and is technically
    # minus ∂e_int_∂q_tot
    ∂e_int_∂q_tot = T_0 * (Δcv_v - R_d) - FT(CAP.e_int_v0(params))
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
        TD.gas_constant_air(thermo_params, ᶜts) / TD.cv_m(thermo_params, ᶜts)

    if use_derivative(topography_flag)
        @. ∂ᶜK_∂ᶜuₕ = DiagonalMatrixRow(
            adjoint(CTh(ᶜuₕ)) + adjoint(ᶜinterp(ᶠu₃)) * g³ʰ(ᶜgⁱʲ),
        )
    else
        @. ∂ᶜK_∂ᶜuₕ = DiagonalMatrixRow(adjoint(CTh(ᶜuₕ)))
    end
    @. ∂ᶜK_∂ᶠu₃ =
        ᶜinterp_matrix() ⋅ DiagonalMatrixRow(adjoint(CT3(ᶠu₃))) +
        DiagonalMatrixRow(adjoint(CT3(ᶜuₕ))) ⋅ ᶜinterp_matrix()

    @. ᶠp_grad_matrix = DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) ⋅ ᶠgradᵥ_matrix()

    @. ᶜadvection_matrix =
        -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ)

    if use_derivative(topography_flag)
        ∂ᶜρ_err_∂ᶜuₕ = matrix[@name(c.ρ), @name(c.uₕ)]
        @. ∂ᶜρ_err_∂ᶜuₕ =
            dtγ * ᶜadvection_matrix ⋅ ᶠwinterp_matrix(ᶜJ * ᶜρ) ⋅
            DiagonalMatrixRow(g³ʰ(ᶜgⁱʲ))
    end
    ∂ᶜρ_err_∂ᶠu₃ = matrix[@name(c.ρ), @name(f.u₃)]
    @. ∂ᶜρ_err_∂ᶠu₃ = dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(g³³(ᶠgⁱʲ))

    tracer_info = (
        (@name(c.ρe_tot), @name(ᶜh_tot)),
        (@name(c.ρq_tot), @name(ᶜspecific.q_tot)),
    )
    MatrixFields.unrolled_foreach(tracer_info) do (ρχ_name, χ_name)
        MatrixFields.has_field(Y, ρχ_name) || return
        ᶜχ = MatrixFields.get_field(p.precomputed, χ_name)
        if use_derivative(topography_flag)
            ∂ᶜρχ_err_∂ᶜuₕ = matrix[ρχ_name, @name(c.uₕ)]
        end
        ∂ᶜρχ_err_∂ᶠu₃ = matrix[ρχ_name, @name(f.u₃)]
        use_derivative(topography_flag) && @. ∂ᶜρχ_err_∂ᶜuₕ =
            dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(ᶠinterp(ᶜχ)) ⋅
            ᶠwinterp_matrix(ᶜJ * ᶜρ) ⋅ DiagonalMatrixRow(g³ʰ(ᶜgⁱʲ))
        @. ∂ᶜρχ_err_∂ᶠu₃ =
            dtγ * ᶜadvection_matrix ⋅ DiagonalMatrixRow(ᶠinterp(ᶜχ) * g³³(ᶠgⁱʲ))
    end

    ∂ᶠu₃_err_∂ᶜρ = matrix[@name(f.u₃), @name(c.ρ)]
    ∂ᶠu₃_err_∂ᶜρe_tot = matrix[@name(f.u₃), @name(c.ρe_tot)]
    @. ∂ᶠu₃_err_∂ᶜρ =
        dtγ * (
            ᶠp_grad_matrix ⋅
            DiagonalMatrixRow(ᶜkappa_m * (T_0 * cp_d - ᶜK - ᶜΦ)) +
            DiagonalMatrixRow(ᶠgradᵥ(ᶜp) / abs2(ᶠinterp(ᶜρ))) ⋅
            ᶠinterp_matrix()
        )
    @. ∂ᶠu₃_err_∂ᶜρe_tot = dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(ᶜkappa_m)
    if MatrixFields.has_field(Y, @name(c.ρq_tot))
        ∂ᶠu₃_err_∂ᶜρq_tot = matrix[@name(f.u₃), @name(c.ρq_tot)]
        @. ∂ᶠu₃_err_∂ᶜρq_tot =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(ᶜkappa_m * ∂e_int_∂q_tot)
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
                DiagonalMatrixRow(-β_rayleigh_w(rs, ᶠz, zmax) * (one_C3xACT3,))
            ) - (I_u₃,)
    else
        @. ∂ᶠu₃_err_∂ᶠu₃ =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅
            ∂ᶜK_∂ᶠu₃ - (I_u₃,)
    end

    tracer_info = (
        (@name(c.ρq_liq), @name(q_liq), @name(ᶜwₗ)),
        (@name(c.ρq_ice), @name(q_ice), @name(ᶜwᵢ)),
        (@name(c.ρq_rai), @name(q_rai), @name(ᶜwᵣ)),
        (@name(c.ρq_sno), @name(q_sno), @name(ᶜwₛ)),
    )
    if !(p.atmos.moisture_model isa DryModel) || use_derivative(diffusion_flag)
        ∂ᶜρe_tot_err_∂ᶜρe_tot = matrix[@name(c.ρe_tot), @name(c.ρe_tot)]
        @. ∂ᶜρe_tot_err_∂ᶜρe_tot = zero(typeof(∂ᶜρe_tot_err_∂ᶜρe_tot)) - (I,)
    end

    if !(p.atmos.moisture_model isa DryModel)
        #TODO: tetsing explicit vs implicit
        #@. ∂ᶜρe_tot_err_∂ᶜρe_tot +=
        #    dtγ * -(ᶜprecipdivᵥ_matrix()) ⋅
        #    DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ) ⋅ ᶠright_bias_matrix() ⋅
        #    DiagonalMatrixRow(
        #        -(1 + ᶜkappa_m) / ᶜρ * ifelse(
        #            ᶜh_tot == 0,
        #            (Geometry.WVector(FT(0)),),
        #            p.precomputed.ᶜwₕhₜ / ᶜh_tot,
        #        ),
        #    )

        ∂ᶜρe_tot_err_∂ᶜρq_tot = matrix[@name(c.ρe_tot), @name(c.ρq_tot)]
        @. ∂ᶜρe_tot_err_∂ᶜρq_tot = zero(typeof(∂ᶜρe_tot_err_∂ᶜρq_tot))
        #TODO: tetsing explicit vs implicit
        #@. ∂ᶜρe_tot_err_∂ᶜρq_tot =
        #    dtγ * -(ᶜprecipdivᵥ_matrix()) ⋅
        #    DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ) ⋅ ᶠright_bias_matrix() ⋅
        #    DiagonalMatrixRow(
        #        -(ᶜkappa_m) * ∂e_int_∂q_tot / ᶜρ * ifelse(
        #            ᶜh_tot == 0,
        #            (Geometry.WVector(FT(0)),),
        #            p.precomputed.ᶜwₕhₜ / ᶜh_tot,
        #        ),
        #    )

        ∂ᶜρq_tot_err_∂ᶜρq_tot = matrix[@name(c.ρq_tot), @name(c.ρq_tot)]
        @. ∂ᶜρq_tot_err_∂ᶜρq_tot = zero(typeof(∂ᶜρq_tot_err_∂ᶜρq_tot)) - (I,)
        #TODO: tetsing explicit vs implicit
        #@. ∂ᶜρq_tot_err_∂ᶜρq_tot =
        #    dtγ * -(ᶜprecipdivᵥ_matrix()) ⋅
        #    DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ) ⋅ ᶠright_bias_matrix() ⋅
        #    DiagonalMatrixRow(
        #        -1 / ᶜρ * ifelse(
        #            ᶜspecific.q_tot == 0,
        #            (Geometry.WVector(FT(0)),),
        #            p.precomputed.ᶜwₜqₜ / ᶜspecific.q_tot,
        #        ),
        #    ) - (I,)

        MatrixFields.unrolled_foreach(tracer_info) do (ρqₚ_name, _, wₚ_name)
            MatrixFields.has_field(Y, ρqₚ_name) || return
            ∂ᶜρqₚ_err_∂ᶜρqₚ = matrix[ρqₚ_name, ρqₚ_name]
            ᶜwₚ = MatrixFields.get_field(p.precomputed, wₚ_name)
            @. ∂ᶜρqₚ_err_∂ᶜρqₚ =
                dtγ * -(ᶜprecipdivᵥ_matrix()) ⋅
                DiagonalMatrixRow(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ) ⋅
                ᶠright_bias_matrix() ⋅
                DiagonalMatrixRow(-Geometry.WVector(ᶜwₚ) / ᶜρ) - (I,)
        end

    end

    if use_derivative(diffusion_flag)
        α_vert_diff_tracer = CAP.α_vert_diff_tracer(params)
        (; ᶜK_h, ᶜK_u) = p.precomputed
        @. ᶜdiffusion_h_matrix =
            ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_h)) ⋅
            ᶠgradᵥ_matrix()
        @. ᶜdiffusion_h_matrix_scaled =
            ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(
                ᶠinterp(ᶜρ) * ᶠinterp(α_vert_diff_tracer * ᶜK_h),
            ) ⋅ ᶠgradᵥ_matrix()
        if (
            MatrixFields.has_field(Y, @name(c.sgs⁰.ρatke)) ||
            !isnothing(p.atmos.turbconv_model) ||
            !disable_momentum_vertical_diffusion(p.atmos.vert_diff)
        )
            @. ᶜdiffusion_u_matrix =
                ᶜadvdivᵥ_matrix() ⋅
                DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_u)) ⋅ ᶠgradᵥ_matrix()
        end

        ∂ᶜρe_tot_err_∂ᶜρ = matrix[@name(c.ρe_tot), @name(c.ρ)]
        @. ∂ᶜρe_tot_err_∂ᶜρ =
            dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(
                (
                    -(1 + ᶜkappa_m) * ᶜspecific.e_tot -
                    ᶜkappa_m * ∂e_int_∂q_tot * ᶜspecific.q_tot
                ) / ᶜρ,
            )
        @. ∂ᶜρe_tot_err_∂ᶜρe_tot +=
            dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow((1 + ᶜkappa_m) / ᶜρ)

        if MatrixFields.has_field(Y, @name(c.ρq_tot))
            ∂ᶜρe_tot_err_∂ᶜρq_tot = matrix[@name(c.ρe_tot), @name(c.ρq_tot)]
            ∂ᶜρq_tot_err_∂ᶜρ = matrix[@name(c.ρq_tot), @name(c.ρ)]
            @. ∂ᶜρe_tot_err_∂ᶜρq_tot +=
                dtγ * ᶜdiffusion_h_matrix ⋅
                DiagonalMatrixRow(ᶜkappa_m * ∂e_int_∂q_tot / ᶜρ)
            @. ∂ᶜρq_tot_err_∂ᶜρ =
                dtγ * ᶜdiffusion_h_matrix ⋅
                DiagonalMatrixRow(-(ᶜspecific.q_tot) / ᶜρ)
            @. ∂ᶜρq_tot_err_∂ᶜρq_tot +=
                dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(1 / ᶜρ)
        end

        MatrixFields.unrolled_foreach(tracer_info) do (ρq_name, q_name, _)
            MatrixFields.has_field(Y, ρq_name) || return
            ᶜq = MatrixFields.get_field(ᶜspecific, q_name)
            ∂ᶜρq_err_∂ᶜρ = matrix[ρq_name, @name(c.ρ)]
            ∂ᶜρq_err_∂ᶜρq = matrix[ρq_name, ρq_name]
            ᶜtridiagonal_matrix_scalar = ifelse(
                q_name in (@name(q_rai), @name(q_sno)),
                ᶜdiffusion_h_matrix_scaled,
                ᶜdiffusion_h_matrix,
            )
            @. ∂ᶜρq_err_∂ᶜρ =
                dtγ * ᶜtridiagonal_matrix_scalar ⋅ DiagonalMatrixRow(-(ᶜq) / ᶜρ)
            @. ∂ᶜρq_err_∂ᶜρq +=
                dtγ * ᶜtridiagonal_matrix_scalar ⋅ DiagonalMatrixRow(1 / ᶜρ)
        end

        if MatrixFields.has_field(Y, @name(c.sgs⁰.ρatke))
            turbconv_params = CAP.turbconv_params(params)
            c_d = CAP.tke_diss_coeff(turbconv_params)
            (; ᶜtke⁰, ᶜmixing_length) = p.precomputed
            (; dt) = p
            ᶜρa⁰ =
                p.atmos.turbconv_model isa PrognosticEDMFX ?
                p.precomputed.ᶜρa⁰ : ᶜρ
            ᶜρatke⁰ = Y.c.sgs⁰.ρatke

            @inline dissipation_rate(tke⁰, mixing_length) =
                tke⁰ >= 0 ? c_d * sqrt(tke⁰) / max(mixing_length, 1) :
                1 / float(dt)
            @inline ∂dissipation_rate_∂tke⁰(tke⁰, mixing_length) =
                tke⁰ > 0 ? c_d / (2 * max(mixing_length, 1) * sqrt(tke⁰)) :
                typeof(tke⁰)(0)

            ᶜdissipation_matrix_diagonal = p.scratch.ᶜtemp_scalar
            @. ᶜdissipation_matrix_diagonal =
                ᶜρatke⁰ * ∂dissipation_rate_∂tke⁰(ᶜtke⁰, ᶜmixing_length)

            ∂ᶜρatke⁰_err_∂ᶜρ = matrix[@name(c.sgs⁰.ρatke), @name(c.ρ)]
            ∂ᶜρatke⁰_err_∂ᶜρatke⁰ =
                matrix[@name(c.sgs⁰.ρatke), @name(c.sgs⁰.ρatke)]
            @. ∂ᶜρatke⁰_err_∂ᶜρ =
                dtγ * (
                    ᶜdiffusion_u_matrix -
                    DiagonalMatrixRow(ᶜdissipation_matrix_diagonal)
                ) ⋅ DiagonalMatrixRow(-(ᶜtke⁰) / ᶜρa⁰)
            @. ∂ᶜρatke⁰_err_∂ᶜρatke⁰ =
                dtγ * (
                    (
                        ᶜdiffusion_u_matrix -
                        DiagonalMatrixRow(ᶜdissipation_matrix_diagonal)
                    ) ⋅ DiagonalMatrixRow(1 / ᶜρa⁰) -
                    DiagonalMatrixRow(dissipation_rate(ᶜtke⁰, ᶜmixing_length))
                ) - (I,)
        end

        if (
            !isnothing(p.atmos.turbconv_model) ||
            !disable_momentum_vertical_diffusion(p.atmos.vert_diff)
        )
            ∂ᶜuₕ_err_∂ᶜuₕ = matrix[@name(c.uₕ), @name(c.uₕ)]
            @. ∂ᶜuₕ_err_∂ᶜuₕ =
                dtγ * DiagonalMatrixRow(1 / ᶜρ) ⋅ ᶜdiffusion_u_matrix - (I,)
        end

    end

    if p.atmos.turbconv_model isa PrognosticEDMFX
        if use_derivative(sgs_advection_flag)
            (; ᶜgradᵥ_ᶠΦ) = p.core
            (; ᶜρʲs, ᶠu³ʲs, ᶜtsʲs, ᶜKʲs, bdmr_l, bdmr_r, bdmr) = p.precomputed
            is_third_order =
                p.atmos.numerics.edmfx_upwinding == Val(:third_order)
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

            ᶠu³ʲ_data = ᶠu³ʲs.:(1).components.data.:1

            ᶜkappa_mʲ = p.scratch.ᶜtemp_scalar
            @. ᶜkappa_mʲ =
                TD.gas_constant_air(thermo_params, ᶜtsʲs.:(1)) /
                TD.cv_m(thermo_params, ᶜtsʲs.:(1))

            ∂ᶜq_totʲ_err_∂ᶜq_totʲ =
                matrix[@name(c.sgsʲs.:(1).q_tot), @name(c.sgsʲs.:(1).q_tot)]
            @. ∂ᶜq_totʲ_err_∂ᶜq_totʲ =
                dtγ * (
                    DiagonalMatrixRow(ᶜadvdivᵥ(ᶠu³ʲs.:(1))) -
                    ᶜadvdivᵥ_matrix() ⋅
                    ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1)))
                ) - (I,)
            ∂ᶜq_totʲ_err_∂ᶠu₃ʲ =
                matrix[@name(c.sgsʲs.:(1).q_tot), @name(f.sgsʲs.:(1).u₃)]
            @. ∂ᶜq_totʲ_err_∂ᶠu₃ʲ =
                dtγ * (
                    -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        ᶠset_upwind_bcs(
                            ᶠupwind(CT3(sign(ᶠu³ʲ_data)), Y.c.sgsʲs.:(1).q_tot),
                        ) * adjoint(C3(sign(ᶠu³ʲ_data))),
                    ) +
                    DiagonalMatrixRow(Y.c.sgsʲs.:(1).q_tot) ⋅ ᶜadvdivᵥ_matrix()
                ) ⋅ DiagonalMatrixRow(g³³(ᶠgⁱʲ))

            ∂ᶜmseʲ_err_∂ᶜq_totʲ =
                matrix[@name(c.sgsʲs.:(1).mse), @name(c.sgsʲs.:(1).q_tot)]
            @. ∂ᶜmseʲ_err_∂ᶜq_totʲ =
                dtγ * (
                    -DiagonalMatrixRow(
                        adjoint(ᶜinterp(ᶠu³ʲs.:(1))) *
                        ᶜgradᵥ_ᶠΦ *
                        Y.c.ρ *
                        ᶜkappa_mʲ / ((ᶜkappa_mʲ + 1) * ᶜp) * ∂e_int_∂q_tot,
                    )
                )
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
            ∂ᶜmseʲ_err_∂ᶠu₃ʲ =
                matrix[@name(c.sgsʲs.:(1).mse), @name(f.sgsʲs.:(1).u₃)]
            @. ∂ᶜmseʲ_err_∂ᶠu₃ʲ =
                dtγ * (
                    -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        ᶠset_upwind_bcs(
                            ᶠupwind(CT3(sign(ᶠu³ʲ_data)), Y.c.sgsʲs.:(1).mse),
                        ) * adjoint(C3(sign(ᶠu³ʲ_data))),
                    ) +
                    DiagonalMatrixRow(Y.c.sgsʲs.:(1).mse) ⋅ ᶜadvdivᵥ_matrix()
                ) ⋅ DiagonalMatrixRow(g³³(ᶠgⁱʲ))

            ∂ᶜρaʲ_err_∂ᶜq_totʲ =
                matrix[@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).q_tot)]
            @. ᶠbidiagonal_matrix_ct3 =
                DiagonalMatrixRow(
                    ᶠset_upwind_bcs(
                        ᶠupwind(
                            ᶠu³ʲs.:(1),
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ),
                    ) / ᶠJ,
                ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(
                    ᶜJ * ᶜkappa_mʲ * (ᶜρʲs.:(1))^2 / ((ᶜkappa_mʲ + 1) * ᶜp) *
                    ∂e_int_∂q_tot,
                )
            @. ᶠbidiagonal_matrix_ct3_2 =
                DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ) ⋅
                ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) ⋅
                DiagonalMatrixRow(
                    Y.c.sgsʲs.:(1).ρa * ᶜkappa_mʲ / ((ᶜkappa_mʲ + 1) * ᶜp) *
                    ∂e_int_∂q_tot,
                )
            @. ∂ᶜρaʲ_err_∂ᶜq_totʲ =
                dtγ * ᶜadvdivᵥ_matrix() ⋅
                (ᶠbidiagonal_matrix_ct3 - ᶠbidiagonal_matrix_ct3_2)

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

            ∂ᶜρaʲ_err_∂ᶜρaʲ =
                matrix[@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).ρa)]
            @. ᶜadvection_matrix =
                -(ᶜadvdivᵥ_matrix()) ⋅
                DiagonalMatrixRow(ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ)
            @. ∂ᶜρaʲ_err_∂ᶜρaʲ =
                dtγ * ᶜadvection_matrix ⋅
                ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) ⋅
                DiagonalMatrixRow(1 / ᶜρʲs.:(1)) - (I,)

            ∂ᶜρaʲ_err_∂ᶠu₃ʲ =
                matrix[@name(c.sgsʲs.:(1).ρa), @name(f.sgsʲs.:(1).u₃)]
            @. ∂ᶜρaʲ_err_∂ᶠu₃ʲ =
                dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                    ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ *
                    ᶠset_upwind_bcs(
                        ᶠupwind(
                            CT3(sign(ᶠu³ʲ_data)),
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ),
                    ) *
                    adjoint(C3(sign(ᶠu³ʲ_data))) *
                    g³³(ᶠgⁱʲ),
                )

            ∂ᶠu₃ʲ_err_∂ᶜq_totʲ =
                matrix[@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).q_tot)]
            @. ∂ᶠu₃ʲ_err_∂ᶜq_totʲ =
                dtγ * DiagonalMatrixRow(
                    ᶠgradᵥ_ᶜΦ * ᶠinterp(Y.c.ρ) / (ᶠinterp(ᶜρʲs.:(1)))^2,
                ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(
                    ᶜkappa_mʲ * (ᶜρʲs.:(1))^2 / ((ᶜkappa_mʲ + 1) * ᶜp) *
                    ∂e_int_∂q_tot,
                )
            ∂ᶠu₃ʲ_err_∂ᶜmseʲ =
                matrix[@name(f.sgsʲs.:(1).u₃), @name(c.sgsʲs.:(1).mse)]
            @. ∂ᶠu₃ʲ_err_∂ᶜmseʲ =
                dtγ * DiagonalMatrixRow(
                    ᶠgradᵥ_ᶜΦ * ᶠinterp(Y.c.ρ) / (ᶠinterp(ᶜρʲs.:(1)))^2,
                ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(
                    ᶜkappa_mʲ * (ᶜρʲs.:(1))^2 / ((ᶜkappa_mʲ + 1) * ᶜp),
                )
            ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                matrix[@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)]
            ᶜu₃ʲ = p.scratch.ᶜtemp_C3
            @. ᶜu₃ʲ = ᶜinterp(Y.f.sgsʲs.:(1).u₃)

            @. bdmr_l = convert(BidiagonalMatrixRow{FT}, ᶜleft_bias_matrix())
            @. bdmr_r = convert(BidiagonalMatrixRow{FT}, ᶜright_bias_matrix())
            @. bdmr = ifelse(ᶜu₃ʲ.components.data.:1 > 0, bdmr_l, bdmr_r)

            @. ᶠtridiagonal_matrix_c3 = -(ᶠgradᵥ_matrix()) ⋅ bdmr
            if rs isa RayleighSponge
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                    dtγ * (
                        ᶠtridiagonal_matrix_c3 ⋅
                        DiagonalMatrixRow(adjoint(CT3(Y.f.sgsʲs.:(1).u₃))) -
                        DiagonalMatrixRow(
                            β_rayleigh_w(rs, ᶠz, zmax) * (one_C3xACT3,),
                        )
                    ) - (I_u₃,)
            else
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                    dtγ * ᶠtridiagonal_matrix_c3 ⋅
                    DiagonalMatrixRow(adjoint(CT3(Y.f.sgsʲs.:(1).u₃))) - (I_u₃,)
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
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ -=
                    dtγ * (DiagonalMatrixRow(
                        (ᶠinterp(ᶜentrʲs.:(1) + ᶜturb_entrʲs.:(1))) *
                        (one_C3xACT3,),
                    ))
            end

            # non-hydrostatic pressure drag
            # (quadratic drag term treated implicitly, buoyancy term explicitly)
            if use_derivative(sgs_nh_pressure_flag)
                (; ᶠu₃⁰) = p.precomputed
                turbconv_params = CAP.turbconv_params(params)
                α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)
                scale_height =
                    CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
                H_up_min = CAP.min_updraft_top(turbconv_params)
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ -=
                    dtγ * (DiagonalMatrixRow(
                        2 * α_d * norm(Y.f.sgsʲs.:(1).u₃ - ᶠu₃⁰) /
                        max(scale_height, H_up_min) * (one_C3xACT3,),
                    ))
            end

            # add updraft mass flux contributions to grid-mean
            if use_derivative(sgs_mass_flux_flag)
                # Jacobian contributions of updraft massflux to grid-mean
                ∂ᶜupdraft_mass_flux_∂ᶜscalar = ᶠbidiagonal_matrix_ct3
                @. ∂ᶜupdraft_mass_flux_∂ᶜscalar =
                    DiagonalMatrixRow(
                        (ᶠinterp(ᶜρ * ᶜJ) / ᶠJ) * (ᶠu³ʲs.:(1) - ᶠu³),
                    ) ⋅ ᶠinterp_matrix() ⋅
                    DiagonalMatrixRow(Y.c.sgsʲs.:(1).ρa / ᶜρʲs.:(1))

                # Derivative of total energy tendency with respect to updraft MSE
                ## grid-mean ρe_tot
                @. ∂ᶜρe_tot_err_∂ᶜρ +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar ⋅
                    DiagonalMatrixRow(
                        (
                            -(1 + ᶜkappa_m) * ᶜspecific.e_tot -
                            ᶜkappa_m * ∂e_int_∂q_tot * ᶜspecific.q_tot
                        ) / ᶜρ,
                    )

                @. ∂ᶜρe_tot_err_∂ᶜρq_tot +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar ⋅
                    DiagonalMatrixRow(ᶜkappa_m * ∂e_int_∂q_tot / ᶜρ)

                @. ∂ᶜρe_tot_err_∂ᶜρe_tot +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar ⋅
                    DiagonalMatrixRow((1 + ᶜkappa_m) / ᶜρ)

                ∂ᶜρe_tot_err_∂ᶜmseʲ =
                    matrix[@name(c.ρe_tot), @name(c.sgsʲs.:(1).mse)]
                @. ∂ᶜρe_tot_err_∂ᶜmseʲ =
                    -(dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar)

                ## grid-mean ρq_tot
                ∂ᶜρq_tot_err_∂ᶜρ = matrix[@name(c.ρq_tot), @name(c.ρ)]
                @. ∂ᶜρq_tot_err_∂ᶜρ +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar ⋅
                    DiagonalMatrixRow(-(ᶜspecific.q_tot) / ᶜρ)

                ∂ᶜρq_tot_err_∂ᶜρq_tot = matrix[@name(c.ρq_tot), @name(c.ρq_tot)]
                @. ∂ᶜρq_tot_err_∂ᶜρq_tot +=
                    dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar ⋅
                    DiagonalMatrixRow(1 / ᶜρ)

                ∂ᶜρq_tot_err_∂ᶜq_totʲ =
                    matrix[@name(c.ρq_tot), @name(c.sgsʲs.:(1).q_tot)]
                @. ∂ᶜρq_tot_err_∂ᶜq_totʲ =
                    -(dtγ * ᶜadvdivᵥ_matrix() ⋅ ∂ᶜupdraft_mass_flux_∂ᶜscalar)

                # grid-mean ∂/∂(u₃ʲ)
                ∂ᶜρe_tot_err_∂ᶠu₃ʲ =
                    matrix[@name(c.ρe_tot), @name(f.sgsʲs.:(1).u₃)]
                @. ∂ᶜρe_tot_err_∂ᶠu₃ʲ =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        ᶠinterp(
                            (Y.c.sgsʲs.:(1).mse + ᶜKʲs.:(1) - ᶜh_tot) *
                            ᶜρʲs.:(1) *
                            ᶜJ *
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ) / ᶠJ * (g³³(ᶠgⁱʲ)),
                    )


                ∂ᶜρq_tot_err_∂ᶠu₃ʲ =
                    matrix[@name(c.ρq_tot), @name(f.sgsʲs.:(1).u₃)]
                @. ∂ᶜρq_tot_err_∂ᶠu₃ʲ =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        ᶠinterp(
                            (Y.c.sgsʲs.:(1).q_tot - ᶜspecific.q_tot) *
                            ᶜρʲs.:(1) *
                            ᶜJ *
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ) / ᶠJ * (g³³(ᶠgⁱʲ)),
                    )

                # grid-mean ∂/∂(rho*a)
                ∂ᶜρe_tot_err_∂ᶜρa =
                    matrix[@name(c.ρe_tot), @name(c.sgsʲs.:(1).ρa)]
                @. ∂ᶜρe_tot_err_∂ᶜρa =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        (ᶠu³ʲs.:(1) - ᶠu³) *
                        ᶠinterp((Y.c.sgsʲs.:(1).mse + ᶜKʲs.:(1) - ᶜh_tot)) / ᶠJ,
                    ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(ᶜJ)

                ∂ᶜρq_tot_err_∂ᶜρa =
                    matrix[@name(c.ρq_tot), @name(c.sgsʲs.:(1).ρa)]
                @. ∂ᶜρq_tot_err_∂ᶜρa =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        (ᶠu³ʲs.:(1) - ᶠu³) *
                        ᶠinterp((Y.c.sgsʲs.:(1).q_tot - ᶜspecific.q_tot)) / ᶠJ,
                    ) ⋅ ᶠinterp_matrix() ⋅ DiagonalMatrixRow(ᶜJ)
            end
        elseif rs isa RayleighSponge
            ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                matrix[@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)]
            @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                dtγ *
                -DiagonalMatrixRow(
                    β_rayleigh_w(rs, ᶠz, zmax) * (one_C3xACT3,),
                ) - (I_u₃,)
        end
    end

    # NOTE: All velocity tendency derivatives should be set BEFORE this call.
    zero_velocity_jacobian!(matrix, Y, p, t)
end

invert_jacobian!(::ApproxJacobian, cache, ΔY, R) =
    LinearAlgebra.ldiv!(ΔY, cache.matrix, R)

# TODO: Rewrite the plotting infrastructure to handle `FieldMatrix`, so that we
# can avoid inefficiently converting the approximate Jacobian to a dense matrix.
function save_jacobian(alg::ApproxJacobian, cache, Y, dtγ, t)
    (; matrix, temp_matrix, column_matrix) = cache
    temp_matrix .= (matrix .+ identity_matrix(matrix, Y)) ./ dtγ
    one_column = length(column_iterator(Y)) == 1

    field_matrix_to_dense_matrix!(column_matrix, temp_matrix, Y, :first)
    file_name = "approx_jacobian" * (one_column ? "" : "_first")
    description =
        "Approximate ∂Yₜ/∂Y matrix" *
        (one_column ? "" : " at $(first_column_coordinate_string(Y))")
    save_column_matrix(cache, file_name, description, Y, t)

    if !one_column
        field_matrix_to_dense_matrix!(column_matrix, temp_matrix, Y, :abs_max)
        file_name = "approx_jacobian_max"
        description = "Maximum of approximate ∂Yₜ/∂Y matrix over all columns"
        save_column_matrix(cache, file_name, description, Y, t)

        field_matrix_to_dense_matrix!(column_matrix, temp_matrix, Y, :abs_avg)
        file_name = "approx_jacobian_avg"
        description = "Average of approximate ∂Yₜ/∂Y matrix over all columns"
        save_column_matrix(cache, file_name, description, Y, t)
    end
end

# TODO: Remove all of the following code after extending ClimaCore.MatrixFields.

tensor_axes_tuple(::Type{T}) where {T} =
    T <: Geometry.AxisTensor ?
    map(axis -> typeof(axis).parameters[1], axes(T)) : ()

primitive_value_at_index(value, (row_axes, col_axes)) =
    if isprimitivetype(typeof(value)) # same as a LinearAlgebra.UniformScaling
        row_axes == col_axes ? value : zero(value)
    elseif value isa Geometry.AxisVector
        @assert isprimitivetype(eltype(value))
        @assert length(row_axes) == 1 && length(col_axes) == 0
        value_axes = tensor_axes_tuple(typeof(value))
        row_axis_index = findfirst(==(row_axes[1]), value_axes[1])
        isnothing(row_axis_index) ? zero(eltype(value)) : value[row_axis_index]
    elseif value isa Geometry.AxisTensor
        @assert isprimitivetype(eltype(value))
        @assert length(row_axes) == 1 && length(col_axes) == 1
        value_axes = tensor_axes_tuple(typeof(value))
        row_axis_index = findfirst(==(row_axes[1]), value_axes[1])
        col_axis_index = findfirst(==(col_axes[1]), value_axes[2])
        isnothing(row_axis_index) || isnothing(col_axis_index) ?
        zero(eltype(value)) : value[row_axis_index, col_axis_index]
    elseif value isa LinearAlgebra.Adjoint
        primitive_value_at_index(parent(value), (col_axes, row_axes))
    else
        sub_names = fieldnames(typeof(value))
        sub_values =
            MatrixFields.unrolled_map(Base.Fix1(getfield, value), sub_names)
        nonempty_sub_values =
            MatrixFields.unrolled_filter(x -> sizeof(x) > 0, sub_values)
        @assert length(nonempty_sub_values) == 1
        primitive_value_at_index(nonempty_sub_values[1], (row_axes, col_axes))
    end

@static if hasfield(Method, :recursion_relation)
    for method in methods(primitive_value_at_index)
        method.recursion_relation = Returns(true)
    end
end

function field_matrix_to_dense_matrix!(out, matrix, Y, column_reduce_flag)
    device = ClimaComms.device(Y.c) # ClimaComms.device(Y)
    field_names = scalar_field_names(Y)
    index_ranges = scalar_field_index_ranges(Y)
    out .= 0

    for ((block_row, block_col), matrix_block) in matrix
        is_child_name_of_row = Base.Fix2(MatrixFields.is_child_name, block_row)
        is_child_name_of_col = Base.Fix2(MatrixFields.is_child_name, block_col)
        subblock_row_indices = findall(is_child_name_of_row, field_names)
        subblock_col_indices = findall(is_child_name_of_col, field_names)
        block_row_field = MatrixFields.get_field(Y, block_row)
        block_col_field = MatrixFields.get_field(Y, block_col)

        for (sub_row, subblock_row_index) in enumerate(subblock_row_indices)
            for (sub_col, subblock_col_index) in enumerate(subblock_col_indices)
                row_index_range = index_ranges[subblock_row_index]
                col_index_range = index_ranges[subblock_col_index]
                out_subblock = view(out, row_index_range, col_index_range)

                if matrix_block isa LinearAlgebra.UniformScaling
                    view(out_subblock, LinearAlgebra.diagind(out_subblock)) .=
                        sub_row == sub_col ? matrix_block.λ :
                        zero(matrix_block.λ)
                else
                    subblock_row_axes = map(
                        Base.Fix2(getindex, sub_row),
                        tensor_axes_tuple(eltype(block_row_field)),
                    )
                    subblock_col_axes = map(
                        Base.Fix2(getindex, sub_col),
                        tensor_axes_tuple(eltype(block_col_field)),
                    )
                    @assert length(subblock_row_axes) in (0, 1)
                    @assert length(subblock_col_axes) in (0, 1)
                    value_in_subblock = Base.Fix2(
                        primitive_value_at_index,
                        (subblock_row_axes, subblock_col_axes),
                    )

                    column_blocks = column_iterator(matrix_block)
                    if column_reduce_flag == :first
                        column_subblock =
                            map.(value_in_subblock, first(column_blocks))
                    else
                        column_subblock =
                            map.(abs ∘ value_in_subblock, first(column_blocks))
                        column_subblock_data = Fields.todata(column_subblock)
                        if column_reduce_flag == :abs_max
                            for new_column_block in Base.rest(column_blocks)
                                new_column_block_data =
                                    Fields.todata(new_column_block)
                                column_subblock_data .=
                                    map.(
                                        max,
                                        map.(abs, column_subblock_data),
                                        map.(
                                            abs ∘ value_in_subblock,
                                            new_column_block_data,
                                        ),
                                    )
                            end
                        else
                            @assert column_reduce_flag == :abs_avg
                            for new_column_block in Base.rest(column_blocks)
                                new_column_block_data =
                                    Fields.todata(new_column_block)
                                column_subblock_data .+=
                                    map.(
                                        abs ∘ value_in_subblock,
                                        new_column_block_data,
                                    )
                            end
                            column_subblock ./= length(column_blocks)
                        end
                    end

                    ClimaComms.allowscalar(
                        copyto!,
                        device,
                        out_subblock,
                        MatrixFields.column_field2array_view(column_subblock),
                    ) # BandedMatrices.jl does not properly support CuArrays.
                end
            end
        end
    end
end
