import LinearAlgebra: I, Adjoint

abstract type DerivativeFlag end
struct UseDerivative <: DerivativeFlag end
struct IgnoreDerivative <: DerivativeFlag end
use_derivative(::UseDerivative) = true
use_derivative(::IgnoreDerivative) = false

"""
    ApproxJacobian(;
        [diffusion_flag],
        [topography_flag],
        [sgs_advection_flag],
        [approximate_solve_iters],
    )

An approximation of the exact `ImplicitEquationJacobian`, which is updated using
analytically derived tendency derivatives and inverted using a specialized
nested linear solver.

# Keyword Arguments

- `diffusion_flag::DerivativeFlag`: whether the derivative of the
  diffusion tendency with respect to the quantities being diffused should be
  computed or approximated as 0; must be either `UseDerivative()` or
  `Ignoreerivative()` instead of a `Bool` to ensure type-stability
- `topography_flag::DerivativeFlag`: whether the derivative of vertical
  contravariant velocity with respect to horizontal covariant velocity should be
  computed or approximated as 0; must be either `UseDerivative()` or
  `IgnoreDerivative()` instead of a `Bool` to ensure type-stability
- `sgs_advection_flag::DerivativeFlag`: whether the derivative of the
  subgrid-scale advection tendency with respect to the updraft quantities should
  be computed or approximated as 0; must be either `UseDerivative()` or
  `IgnoreDerivative()` instead of a `Bool` to ensure type-stability
- `approximate_solve_iters::Int`: number of iterations to take for the
  approximate linear solve required when `diffusion_flag = UseDerivative()`
"""
@kwdef struct ApproxJacobian{F1, F2, F3} <: JacobianAlgorithm
    diffusion_flag::F1 = nothing
    topography_flag::F2 = nothing
    sgs_advection_flag::F3 = nothing
    approximate_solve_iters::Int = 1
end

function jacobian_cache(alg::ApproxJacobian, Y, atmos)
    diffusion_flag = if isnothing(alg.diffusion_flag)
        atmos.diff_mode == Implicit() ? UseDerivative() : IgnoreDerivative()
    else
        alg.diffusion_flag
    end
    topography_flag = if isnothing(alg.topography_flag)
        has_topography(axes(Y.c)) ? UseDerivative() : IgnoreDerivative()
    else
        alg.topography_flag
    end
    sgs_advection_flag = if isnothing(alg.sgs_advection_flag)
        atmos.sgs_adv_mode == Implicit() ? UseDerivative() : IgnoreDerivative()
    else
        alg.sgs_advection_flag
    end

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

    diffused_scalar_names =
        (@name(c.ρe_tot), available_tracer_names..., ρatke_if_available...)
    diffusion_blocks = if use_derivative(diffusion_flag)
        (
            MatrixFields.unrolled_map(
                name -> (name, @name(c.ρ)) => similar(Y.c, TridiagonalRow),
                diffused_scalar_names,
            )...,
            MatrixFields.unrolled_map(
                name -> (name, name) => similar(Y.c, TridiagonalRow),
                diffused_scalar_names,
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
                    diffuse_momentum(atmos.vert_diff) ?
                similar(Y.c, TridiagonalRow) : FT(-1) * I,
        )
    else
        MatrixFields.unrolled_map(
            name -> (name, name) => FT(-1) * I,
            (diffused_scalar_names..., @name(c.uₕ)),
        )
    end

    sgs_advection_blocks = if atmos.turbconv_model isa PrognosticEDMFX
        @assert n_prognostic_mass_flux_subdomains(atmos.turbconv_model) == 1
        sgs_scalar_names = (
            @name(c.sgsʲs.:(1).q_tot),
            @name(c.sgsʲs.:(1).mse),
            @name(c.sgsʲs.:(1).ρa)
        )
        if use_derivative(sgs_advection_flag)
            (
                MatrixFields.unrolled_map(
                    name -> (name, name) => similar(Y.c, TridiagonalRow),
                    sgs_scalar_names,
                )...,
                (@name(c.sgsʲs.:(1).mse), @name(c.ρ)) =>
                    similar(Y.c, DiagonalRow),
                (@name(c.sgsʲs.:(1).mse), @name(c.sgsʲs.:(1).q_tot)) =>
                    similar(Y.c, DiagonalRow),
                (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).q_tot)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).mse)) =>
                    similar(Y.c, TridiagonalRow),
                (@name(f.sgsʲs.:(1).u₃), @name(c.ρ)) =>
                    similar(Y.f, BidiagonalRow_C3),
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
                    sgs_scalar_names,
                )...,
                (@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)) =>
                    !isnothing(atmos.rayleigh_sponge) ?
                    similar(Y.f, DiagonalRow_C3xACT3) : FT(-1) * I,
            )
        end
    else
        ()
    end

    matrix = MatrixFields.FieldMatrix(
        identity_blocks...,
        sgs_advection_blocks...,
        advection_blocks...,
        diffusion_blocks...,
    )

    sgs_names_if_available = if atmos.turbconv_model isa PrognosticEDMFX
        (
            @name(c.sgsʲs.:(1).q_tot),
            @name(c.sgsʲs.:(1).mse),
            @name(c.sgsʲs.:(1).ρa),
            @name(f.sgsʲs.:(1).u₃),
        )
    else
        ()
    end
    names₁_group₁ = (@name(c.ρ), sfc_if_available...)
    names₁_group₂ = (available_tracer_names..., ρatke_if_available...)
    names₁_group₃ = (@name(c.ρe_tot),)
    names₁ = (
        names₁_group₁...,
        sgs_names_if_available...,
        names₁_group₂...,
        names₁_group₃...,
    )

    alg₂ = MatrixFields.BlockLowerTriangularSolve(@name(c.uₕ))
    alg =
        if use_derivative(diffusion_flag) || use_derivative(sgs_advection_flag)
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
                            @name(c.sgsʲs.:(1).q_tot);
                            alg₂ = MatrixFields.BlockLowerTriangularSolve(
                                @name(c.sgsʲs.:(1).mse);
                                alg₂ = MatrixFields.BlockLowerTriangularSolve(
                                    @name(c.sgsʲs.:(1).ρa),
                                    @name(f.sgsʲs.:(1).u₃);
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

    return (;
        diffusion_flag,
        topography_flag,
        sgs_advection_flag,
        matrix = MatrixFields.FieldMatrixWithSolver(matrix, Y, alg),
    )
end

always_update_exact_jacobian(::ApproxJacobian) = false

factorize_exact_jacobian!(::ApproxJacobian, _, _, _, _, _) = nothing

function approximate_jacobian!(::ApproxJacobian, cache, Y, p, dtγ, t)
    (; diffusion_flag, topography_flag, sgs_advection_flag, matrix) = cache
    (; params) = p
    (; edmfx_upwinding) = p.atmos.numerics
    (; ᶜΦ, ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜspecific, ᶜK, ᶜts, ᶜp) = p.precomputed
    (;
        ∂ᶜK_∂ᶜuₕ,
        ∂ᶜK_∂ᶠu₃,
        ᶠp_grad_matrix,
        ᶜadvection_matrix,
        ᶜdiffusion_h_matrix,
        ᶜdiffusion_u_matrix,
        ᶠbidiagonal_matrix_ct3,
        ᶠbidiagonal_matrix_ct3_2,
        ᶠtridiagonal_matrix_c3,
    ) = p.scratch
    if p.atmos.rayleigh_sponge isa RayleighSponge
        (; ᶠβ_rayleigh_w) = p.rayleigh_sponge
    end

    FT = Spaces.undertype(axes(Y.c))
    CTh = CTh_vector_type(axes(Y.c))
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'

    thermo_params = CAP.thermodynamics_params(params)
    cv_d = FT(CAP.cv_d(params))
    Δcv_v = FT(CAP.cv_v(params)) - cv_d
    T_0 = FT(CAP.T_0(params))
    R_d = FT(CAP.R_d(params))
    cp_d = FT(CAP.cp_d(params))
    e_int_v0 = FT(CAP.e_int_v0(params))
    ∂e_int_∂q_tot = T_0 * (Δcv_v - R_d) - e_int_v0 # technically -∂e_int_∂q_tot

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶜgⁱʲ = Fields.local_geometry_field(Y.c).gⁱʲ
    ᶠgⁱʲ = Fields.local_geometry_field(Y.f).gⁱʲ

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
        -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(ᶠwinterp(ᶜJ, ᶜρ))

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

    if p.atmos.rayleigh_sponge isa RayleighSponge
        @. ∂ᶠu₃_err_∂ᶠu₃ =
            dtγ * (
                ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅
                ∂ᶜK_∂ᶠu₃ + DiagonalMatrixRow(-(ᶠβ_rayleigh_w) * (one_C3xACT3,))
            ) - (I_u₃,)
    else
        @. ∂ᶠu₃_err_∂ᶠu₃ =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅
            ∂ᶜK_∂ᶠu₃ - (I_u₃,)
    end

    if use_derivative(diffusion_flag)
        (; ᶜK_h, ᶜK_u) = p.precomputed
        @. ᶜdiffusion_h_matrix =
            ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_h)) ⋅
            ᶠgradᵥ_matrix()
        if (
            MatrixFields.has_field(Y, @name(c.sgs⁰.ρatke)) ||
            !isnothing(p.atmos.turbconv_model) ||
            diffuse_momentum(p.atmos.vert_diff)
        )
            @. ᶜdiffusion_u_matrix =
                ᶜadvdivᵥ_matrix() ⋅
                DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_u)) ⋅ ᶠgradᵥ_matrix()
        end

        ∂ᶜρe_tot_err_∂ᶜρ = matrix[@name(c.ρe_tot), @name(c.ρ)]
        ∂ᶜρe_tot_err_∂ᶜρe_tot = matrix[@name(c.ρe_tot), @name(c.ρe_tot)]
        @. ∂ᶜρe_tot_err_∂ᶜρ =
            dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(
                (
                    -(1 + ᶜkappa_m) * ᶜspecific.e_tot -
                    ᶜkappa_m * ∂e_int_∂q_tot * ᶜspecific.q_tot
                ) / ᶜρ,
            )
        @. ∂ᶜρe_tot_err_∂ᶜρe_tot =
            dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow((1 + ᶜkappa_m) / ᶜρ) -
            (I,)
        if MatrixFields.has_field(Y, @name(c.ρq_tot))
            ∂ᶜρe_tot_err_∂ᶜρq_tot = matrix[@name(c.ρe_tot), @name(c.ρq_tot)]
            @. ∂ᶜρe_tot_err_∂ᶜρq_tot =
                dtγ * ᶜdiffusion_h_matrix ⋅
                DiagonalMatrixRow(ᶜkappa_m * ∂e_int_∂q_tot / ᶜρ)
        end

        tracer_info = (
            (@name(c.ρq_tot), @name(q_tot)),
            (@name(c.ρq_liq), @name(q_liq)),
            (@name(c.ρq_ice), @name(q_ice)),
            (@name(c.ρq_rai), @name(q_rai)),
            (@name(c.ρq_sno), @name(q_sno)),
        )
        MatrixFields.unrolled_foreach(tracer_info) do (ρq_name, q_name)
            MatrixFields.has_field(Y, ρq_name) || return
            ᶜq = MatrixFields.get_field(ᶜspecific, q_name)
            ∂ᶜρq_err_∂ᶜρ = matrix[ρq_name, @name(c.ρ)]
            ∂ᶜρq_err_∂ᶜρq = matrix[ρq_name, ρq_name]
            @. ∂ᶜρq_err_∂ᶜρ =
                dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(-(ᶜq) / ᶜρ)
            @. ∂ᶜρq_err_∂ᶜρq =
                dtγ * ᶜdiffusion_h_matrix ⋅ DiagonalMatrixRow(1 / ᶜρ) - (I,)
        end

        if MatrixFields.has_field(Y, @name(c.sgs⁰.ρatke))
            turbconv_params = CAP.turbconv_params(params)
            c_d = CAP.tke_diss_coeff(turbconv_params)
            (; ᶜtke⁰, ᶜmixing_length) = p.precomputed
            ᶜρa⁰ =
                p.atmos.turbconv_model isa PrognosticEDMFX ?
                p.precomputed.ᶜρa⁰ : ᶜρ
            ᶜρatke⁰ = Y.c.sgs⁰.ρatke

            @inline dissipation_rate(tke⁰, mixing_length) =
                tke⁰ >= 0 ? c_d * sqrt(tke⁰) / max(mixing_length, 1) : 1 / p.dt
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
            diffuse_momentum(p.atmos.vert_diff)
        )
            ∂ᶜuₕ_err_∂ᶜuₕ = matrix[@name(c.uₕ), @name(c.uₕ)]
            @. ∂ᶜuₕ_err_∂ᶜuₕ =
                dtγ * DiagonalMatrixRow(1 / ᶜρ) ⋅ ᶜdiffusion_u_matrix - (I,)
        end

        ᶠlg = Fields.local_geometry_field(Y.f)
        precip_info =
            ((@name(c.ρq_rai), @name(ᶜwᵣ)), (@name(c.ρq_sno), @name(ᶜwₛ)))
        MatrixFields.unrolled_foreach(precip_info) do (ρqₚ_name, wₚ_name)
            MatrixFields.has_field(Y, ρqₚ_name) || return
            ∂ᶜρqₚ_err_∂ᶜρqₚ = matrix[ρqₚ_name, ρqₚ_name]
            ᶜwₚ = MatrixFields.get_field(p.precomputed, wₚ_name)
            ᶠtmp = p.scratch.ᶠtemp_CT3
            @. ᶠtmp = CT3(unit_basis_vector_data(CT3, ᶠlg)) * ᶠwinterp(ᶜJ, ᶜρ)
            @. ∂ᶜρqₚ_err_∂ᶜρqₚ +=
                dtγ * -(ᶜprecipdivᵥ_matrix()) ⋅ DiagonalMatrixRow(ᶠtmp) ⋅
                ᶠright_bias_matrix() ⋅ DiagonalMatrixRow(-(ᶜwₚ) / ᶜρ)
        end
    end

    if p.atmos.turbconv_model isa PrognosticEDMFX
        if use_derivative(sgs_advection_flag)
            (; ᶜgradᵥ_ᶠΦ) = p.core
            (; ᶜρʲs, ᶠu³ʲs, ᶜtsʲs, bdmr_l, bdmr_r, bdmr) = p.precomputed
            is_third_order = edmfx_upwinding == Val(:third_order)
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
            ∂ᶜmseʲ_err_∂ᶜρ = matrix[@name(c.sgsʲs.:(1).mse), @name(c.ρ)]
            @. ∂ᶜmseʲ_err_∂ᶜρ =
                dtγ * (
                    -DiagonalMatrixRow(
                        adjoint(ᶜinterp(ᶠu³ʲs.:(1))) * ᶜgradᵥ_ᶠΦ / ᶜρʲs.:(1),
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

            ∂ᶜρaʲ_err_∂ᶜq_totʲ =
                matrix[@name(c.sgsʲs.:(1).ρa), @name(c.sgsʲs.:(1).q_tot)]
            @. ᶠbidiagonal_matrix_ct3 =
                DiagonalMatrixRow(
                    ᶠset_upwind_bcs(
                        ᶠupwind(
                            ᶠu³ʲs.:(1),
                            draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)),
                        ),
                    ),
                ) ⋅ ᶠwinterp_matrix(ᶜJ) ⋅ DiagonalMatrixRow(
                    ᶜkappa_mʲ * (ᶜρʲs.:(1))^2 / ((ᶜkappa_mʲ + 1) * ᶜp) *
                    ∂e_int_∂q_tot,
                )
            @. ᶠbidiagonal_matrix_ct3_2 =
                DiagonalMatrixRow(ᶠwinterp(ᶜJ, ᶜρʲs.:(1))) ⋅
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
                    ),
                ) ⋅ ᶠwinterp_matrix(ᶜJ) ⋅ DiagonalMatrixRow(
                    ᶜkappa_mʲ * (ᶜρʲs.:(1))^2 / ((ᶜkappa_mʲ + 1) * ᶜp),
                )
            @. ᶠbidiagonal_matrix_ct3_2 =
                DiagonalMatrixRow(ᶠwinterp(ᶜJ, ᶜρʲs.:(1))) ⋅
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
                DiagonalMatrixRow(ᶠwinterp(ᶜJ, ᶜρʲs.:(1)))
            @. ∂ᶜρaʲ_err_∂ᶜρaʲ =
                dtγ * ᶜadvection_matrix ⋅
                ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³ʲs.:(1))) ⋅
                DiagonalMatrixRow(1 / ᶜρʲs.:(1)) - (I,)

            ∂ᶠu₃ʲ_err_∂ᶜρ = matrix[@name(f.sgsʲs.:(1).u₃), @name(c.ρ)]
            @. ∂ᶠu₃ʲ_err_∂ᶜρ =
                dtγ * DiagonalMatrixRow(ᶠgradᵥ_ᶜΦ / ᶠinterp(ᶜρʲs.:(1))) ⋅
                ᶠinterp_matrix()
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
            if p.atmos.rayleigh_sponge isa RayleighSponge
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                    dtγ * (
                        ᶠtridiagonal_matrix_c3 ⋅
                        DiagonalMatrixRow(adjoint(CT3(Y.f.sgsʲs.:(1).u₃))) -
                        DiagonalMatrixRow(ᶠβ_rayleigh_w * (one_C3xACT3,))
                    ) - (I_u₃,)
            else
                @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                    dtγ * ᶠtridiagonal_matrix_c3 ⋅
                    DiagonalMatrixRow(adjoint(CT3(Y.f.sgsʲs.:(1).u₃))) - (I_u₃,)
            end
        elseif p.atmos.rayleigh_sponge isa RayleighSponge
            ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                matrix[@name(f.sgsʲs.:(1).u₃), @name(f.sgsʲs.:(1).u₃)]
            @. ∂ᶠu₃ʲ_err_∂ᶠu₃ʲ =
                dtγ * -DiagonalMatrixRow(ᶠβ_rayleigh_w * (one_C3xACT3,)) -
                (I_u₃,)
        end
    end

    zero_jacobian!(matrix, p.atmos.tendency_model, p.atmos.turbconv_model)
end

invert_jacobian!(::ApproxJacobian, cache, x, b) =
    LinearAlgebra.ldiv!(x, cache.matrix, b)
