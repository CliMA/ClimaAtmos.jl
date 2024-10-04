import LinearAlgebra: I, Adjoint, ldiv!
import ClimaCore.MatrixFields: @name
using ClimaCore.MatrixFields

abstract type DerivativeFlag end
struct UseDerivative <: DerivativeFlag end
struct IgnoreDerivative <: DerivativeFlag end
use_derivative(::UseDerivative) = true
use_derivative(::IgnoreDerivative) = false

"""
    ImplicitEquationJacobian(
        Y, atmos;
        approximate_solve_iters, diffusion_flag, topography_flag, sgs_advection_flag, transform_flag
    )

A wrapper for the matrix ``∂E/∂Y``, where ``E(Y)`` is the "error" of the
implicit step with the state ``Y``.

# Background

When we use an implicit or split implicit-explicit (IMEX) timestepping scheme,
we end up with a nonlinear equation of the form ``E(Y) = 0``, where
```math
    E(Y) = Y_{imp}(Y) - Y = \\hat{Y} + Δt * T_{imp}(Y) - Y.
```
In this expression, ``Y_{imp}(Y)`` denotes the state at some time ``t + Δt``.
This can be expressed as the sum of ``\\hat{Y}``, the contribution from the
state at time ``t`` (and possibly also at earlier times, depending on the order
of the timestepping scheme), and ``Δt * T_{imp}(Y)``, the contribution from the
implicit tendency ``T_{imp}`` between times ``t`` and ``t + Δt``. The new state
at the end of each implicit step in the timestepping scheme is the value of
``Y`` that solves this equation, i.e., the value of ``Y`` that is consistent
with the state ``Y_{imp}(Y)`` predicted by the implicit step.

Note: When we use a higher-order timestepping scheme, the full step ``Δt`` is
divided into several sub-steps or "stages", where the duration of stage ``i`` is
``Δt * γ_i`` for some constant ``γ_i`` between 0 and 1.

In order to solve this equation using Newton's method, we must specify the
derivative ``∂E/∂Y``. Since ``\\hat{Y}`` does not depend on ``Y`` (it is only a
function of the state at or before time ``t``), this derivative is
```math
    E'(Y) = Δt * T_{imp}'(Y) - I.
```
In addition, we must specify how to divide ``E(Y)`` by this derivative, i.e.,
how to solve the linear equation
```math
    E'(Y) * ΔY = E(Y).
```

Note: This equation comes from assuming that there is some ``ΔY`` such that
``E(Y - ΔY) = 0`` and making the first-order approximation
```math
    E(Y - ΔY) \\approx E(Y) - E'(Y) * ΔY.
```

After initializing ``Y`` to ``Y[0] = \\hat{Y}``, Newton's method executes the
following steps:
- Compute the derivative ``E'(Y[0])``.
- Compute the implicit tendency ``T_{imp}(Y[0])`` and use it to get ``E(Y[0])``.
- Solve the linear equation ``E'(Y[0]) * ΔY[0] = E(Y[0])`` for ``ΔY[0]``.
- Update ``Y`` to ``Y[1] = Y[0] - ΔY[0]``.
If the number of Newton iterations is limited to 1, this new value of ``Y`` is
taken to be the solution of the implicit equation. Otherwise, this sequence of
steps is repeated, i.e., ``ΔY[1]`` is computed and used to update ``Y`` to
``Y[2] = Y[1] - ΔY[1]``, then ``ΔY[2]`` is computed and used to update ``Y`` to
``Y[3] = Y[2] - ΔY[2]``, and so on. The iterative process is terminated either
when the error ``E(Y)`` is sufficiently close to 0 (according to the convergence
condition passed to Newton's method), or when the maximum number of iterations
is reached.

# Arguments

- `Y::FieldVector`: the state of the simulation
- `atmos::AtmosModel`: the model configuration
- `approximate_solve_iters::Int`: number of iterations to take for the
  approximate linear solve required when `diffusion_flag = UseDerivative()`
- `diffusion_flag::DerivativeFlag`: whether the derivative of the
  diffusion tendency with respect to the quantities being diffused should be
  computed or approximated as 0; must be either `UseDerivative()` or
  `IgnoreDerivative()` instead of a `Bool` to ensure type-stability
- `topography_flag::DerivativeFlag`: whether the derivative of vertical
  contravariant velocity with respect to horizontal covariant velocity should be
  computed or approximated as 0; must be either `UseDerivative()` or
  `IgnoreDerivative()` instead of a `Bool` to ensure type-stability
- `sgs_advection_flag::DerivativeFlag`: whether the derivative of the
  subgrid-scale advection tendency with respect to the updraft quantities should be
  computed or approximated as 0; must be either `UseDerivative()` or
  `IgnoreDerivative()` instead of a `Bool` to ensure type-stability
- `transform_flag::Bool`: whether the error should be transformed from ``E(Y)``
  to ``E(Y)/Δt``, which is required for non-Rosenbrock timestepping schemes from
  OrdinaryDiffEq.jl
"""
struct ImplicitEquationJacobian{
    M <: MatrixFields.FieldMatrix,
    S <: MatrixFields.FieldMatrixSolver,
    F1 <: DerivativeFlag,
    F2 <: DerivativeFlag,
    F3 <: DerivativeFlag,
    T <: Fields.FieldVector,
    R <: Base.RefValue,
}
    # stores the matrix E'(Y) = Δt * T_imp'(Y) - I
    matrix::M

    # solves the linear equation E'(Y) * ΔY = E(Y) for ΔY
    solver::S

    # flags that determine how E'(Y) is approximated
    diffusion_flag::F1
    topography_flag::F2
    sgs_advection_flag::F3

    # required by Krylov.jl to evaluate ldiv! with AbstractVector inputs
    temp_b::T
    temp_x::T

    # required by OrdinaryDiffEq.jl to run non-Rosenbrock timestepping schemes
    transform_flag::Bool
    dtγ_ref::R
end

function ImplicitEquationJacobian(
    Y,
    atmos;
    approximate_solve_iters = 1,
    diffusion_flag = atmos.diff_mode == Implicit() ? UseDerivative() :
                     IgnoreDerivative(),
    topography_flag = has_topography(axes(Y.c)) ? UseDerivative() :
                      IgnoreDerivative(),
    sgs_advection_flag = atmos.sgs_adv_mode == Implicit() ? UseDerivative() :
                         IgnoreDerivative(),
    transform_flag = false,
)
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
        @name(c.ρn_liq),
        @name(c.ρn_rai),
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
                n_iters = approximate_solve_iters,
            )
        else
            MatrixFields.BlockArrowheadSolve(names₁...; alg₂)
        end

    return ImplicitEquationJacobian(
        matrix,
        MatrixFields.FieldMatrixSolver(alg, matrix, Y),
        diffusion_flag,
        topography_flag,
        sgs_advection_flag,
        similar(Y),
        similar(Y),
        transform_flag,
        Ref{FT}(),
    )
end

# We only use A, but ClimaTimeSteppers.jl require us to
# pass jac_prototype and then call similar(jac_prototype) to
# obtain A. This is a workaround to avoid unnecessary allocations.
Base.similar(A::ImplicitEquationJacobian) = A

# This method specifies how to solve the equation E'(Y) * ΔY = E(Y) for ΔY.
NVTX.@annotate function ldiv!(
    x::Fields.FieldVector,
    A::ImplicitEquationJacobian,
    b::Fields.FieldVector,
)
    MatrixFields.field_matrix_solve!(A.solver, x, A.matrix, b)
    if A.transform_flag
        @. x *= -A.dtγ_ref[]
    end
end

# This method for ldiv! is called by Krylov.jl from inside ClimaTimeSteppers.jl.
# See https://github.com/JuliaSmoothOptimizers/Krylov.jl/issues/605 for a
# related issue that requires the same workaround.
NVTX.@annotate function ldiv!(
    x::AbstractVector,
    A::ImplicitEquationJacobian,
    b::AbstractVector,
)
    A.temp_b .= b
    ldiv!(A.temp_x, A, A.temp_b)
    x .= A.temp_x
end

# This function is used by DiffEqBase.jl instead of ldiv!.
linsolve!(::Type{Val{:init}}, f, u0; kwargs...) = _linsolve!
_linsolve!(x, A, b, update_matrix = false; kwargs...) = ldiv!(x, A, b)

# This method specifies how to compute E'(Y), which is referred to as "Wfact" in
# DiffEqBase.jl.
NVTX.@annotate function Wfact!(A, Y, p, dtγ, t)
    # Remove unnecessary values from p to avoid allocations in bycolumn.
    p′ = (;
        p.precomputed.ᶜspecific,
        p.precomputed.ᶜK,
        p.precomputed.ᶜts,
        p.precomputed.ᶜp,
        (
            p.atmos.precip_model isa Microphysics1Moment ?
            (; p.precomputed.ᶜwᵣ, p.precomputed.ᶜwₛ) : (;)
        )...,
        p.precomputed.ᶜh_tot,
        (
            use_derivative(A.diffusion_flag) ?
            (; p.precomputed.ᶜK_u, p.precomputed.ᶜK_h) : (;)
        )...,
        (
            use_derivative(A.diffusion_flag) &&
            p.atmos.turbconv_model isa AbstractEDMF ?
            (; p.precomputed.ᶜtke⁰, p.precomputed.ᶜmixing_length) : (;)
        )...,
        (
            use_derivative(A.diffusion_flag) &&
            p.atmos.turbconv_model isa PrognosticEDMFX ?
            (; p.precomputed.ᶜρa⁰) : (;)
        )...,
        (
            use_derivative(A.sgs_advection_flag) &&
            p.atmos.turbconv_model isa PrognosticEDMFX ?
            (;
                p.core.ᶜgradᵥ_ᶠΦ,
                p.precomputed.ᶜρʲs,
                p.precomputed.ᶠu³ʲs,
                p.precomputed.ᶜtsʲs,
                p.precomputed.bdmr_l,
                p.precomputed.bdmr_r,
                p.precomputed.bdmr,
            ) : (;)
        )...,
        p.core.ᶜΦ,
        p.core.ᶠgradᵥ_ᶜΦ,
        p.scratch.ᶜtemp_scalar,
        p.scratch.ᶜtemp_C3,
        p.scratch.ᶠtemp_CT3,
        p.scratch.∂ᶜK_∂ᶜuₕ,
        p.scratch.∂ᶜK_∂ᶠu₃,
        p.scratch.ᶠp_grad_matrix,
        p.scratch.ᶜadvection_matrix,
        p.scratch.ᶜdiffusion_h_matrix,
        p.scratch.ᶜdiffusion_u_matrix,
        p.scratch.ᶠbidiagonal_matrix_ct3,
        p.scratch.ᶠbidiagonal_matrix_ct3_2,
        p.scratch.ᶠtridiagonal_matrix_c3,
        p.dt,
        p.params,
        p.atmos,
        (
            p.atmos.rayleigh_sponge isa RayleighSponge ?
            (; p.rayleigh_sponge.ᶠβ_rayleigh_w) : (;)
        )...,
    )

    # Convert dtγ from a Float64 to an FT.
    FT = Spaces.undertype(axes(Y.c))
    dtγ′ = FT(dtγ)

    A.dtγ_ref[] = dtγ′
    update_implicit_equation_jacobian!(A, Y, p′, dtγ′)
end

function update_implicit_equation_jacobian!(A, Y, p, dtγ)
    (; matrix, diffusion_flag, sgs_advection_flag, topography_flag) = A
    (; ᶜspecific, ᶜK, ᶜts, ᶜp, ᶜΦ, ᶠgradᵥ_ᶜΦ) = p
    (;
        ᶜtemp_C3,
        ∂ᶜK_∂ᶜuₕ,
        ∂ᶜK_∂ᶠu₃,
        ᶠp_grad_matrix,
        ᶜadvection_matrix,
        ᶠbidiagonal_matrix_ct3,
        ᶠbidiagonal_matrix_ct3_2,
        ᶠtridiagonal_matrix_c3,
    ) = p
    (; ᶜdiffusion_h_matrix, ᶜdiffusion_u_matrix, params) = p
    (; edmfx_upwinding) = p.atmos.numerics

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
    ᶜgⁱʲ = Fields.local_geometry_field(Y.c).gⁱʲ
    ᶠgⁱʲ = Fields.local_geometry_field(Y.f).gⁱʲ

    ᶜkappa_m = p.ᶜtemp_scalar
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
        ᶜχ = MatrixFields.get_field(p, χ_name)
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
                ∂ᶜK_∂ᶠu₃ + DiagonalMatrixRow(-p.ᶠβ_rayleigh_w * (one_C3xACT3,))
            ) - (I_u₃,)
    else
        @. ∂ᶠu₃_err_∂ᶠu₃ =
            dtγ * ᶠp_grad_matrix ⋅ DiagonalMatrixRow(-(ᶜkappa_m) * ᶜρ) ⋅
            ∂ᶜK_∂ᶠu₃ - (I_u₃,)
    end

    if use_derivative(diffusion_flag)
        (; ᶜK_h, ᶜK_u) = p
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
            (@name(c.ρn_liq), @name(n_liq)),
            (@name(c.ρn_rai), @name(n_rai)),
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
            (; ᶜtke⁰, ᶜmixing_length, dt) = p
            ᶜρa⁰ = p.atmos.turbconv_model isa PrognosticEDMFX ? p.ᶜρa⁰ : ᶜρ
            ᶜρatke⁰ = Y.c.sgs⁰.ρatke

            @inline dissipation_rate(tke⁰, mixing_length) =
                tke⁰ >= 0 ? c_d * sqrt(tke⁰) / max(mixing_length, 1) : 1 / dt
            @inline ∂dissipation_rate_∂tke⁰(tke⁰, mixing_length) =
                tke⁰ > 0 ? c_d / (2 * max(mixing_length, 1) * sqrt(tke⁰)) :
                typeof(tke⁰)(0)

            ᶜdissipation_matrix_diagonal = p.ᶜtemp_scalar
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
            ((@name(c.ρq_rai), @name(ᶜwᵣ)),
             (@name(c.ρq_sno), @name(ᶜwₛ)),
             (@name(c.ρn_rai), @name(ᶜw_nᵣ)),
            )
        MatrixFields.unrolled_foreach(precip_info) do (ρqₚ_name, wₚ_name)
            MatrixFields.has_field(Y, ρqₚ_name) || return
            ∂ᶜρqₚ_err_∂ᶜρqₚ = matrix[ρqₚ_name, ρqₚ_name]
            ᶜwₚ = MatrixFields.get_field(p, wₚ_name)
            ᶠtmp = p.ᶠtemp_CT3
            @. ᶠtmp = CT3(unit_basis_vector_data(CT3, ᶠlg)) * ᶠwinterp(ᶜJ, ᶜρ)
            @. ∂ᶜρqₚ_err_∂ᶜρqₚ +=
                dtγ * -(ᶜprecipdivᵥ_matrix()) ⋅ DiagonalMatrixRow(ᶠtmp) ⋅
                ᶠright_bias_matrix() ⋅ DiagonalMatrixRow(-(ᶜwₚ) / ᶜρ)
        end
    end

    if p.atmos.turbconv_model isa PrognosticEDMFX
        if use_derivative(sgs_advection_flag)
            (; ᶜgradᵥ_ᶠΦ, ᶜρʲs, ᶠu³ʲs, ᶜtsʲs) = p
            (; bdmr_l, bdmr_r, bdmr) = p
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
            ᶜkappa_mʲ = p.ᶜtemp_scalar
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
            ᶜu₃ʲ = ᶜtemp_C3
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
                        DiagonalMatrixRow(p.ᶠβ_rayleigh_w * (one_C3xACT3,))
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
                dtγ * -DiagonalMatrixRow(p.ᶠβ_rayleigh_w * (one_C3xACT3,)) -
                (I_u₃,)
        end
    end

end
