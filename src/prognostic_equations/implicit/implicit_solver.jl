import LinearAlgebra: I, Adjoint, ldiv!
import ClimaCore.MatrixFields: @name
using ClimaCore.MatrixFields

abstract type DerivativeFlag end
struct UseDerivative <: DerivativeFlag end
struct IgnoreDerivative <: DerivativeFlag end
use_derivative(::UseDerivative) = true
use_derivative(::IgnoreDerivative) = false

"""
    ImplicitEquationJacobian(Y, atmos; approximate_solve_iters, diffusion_flag, topography_flag, transform_flag)

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
  `Ignoreerivative()` instead of a `Bool` to ensure type-stability
- `topography_flag::DerivativeFlag`: whether the derivative of vertical
  contravariant velocity with respect to horizontal covariant velocity should be
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
    transform_flag = false,
)
    FT = Spaces.undertype(axes(Y.c))
    CTh = CTh_vector_type(axes(Y.c))

    TridiagonalRow = TridiagonalMatrixRow{FT}
    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    TridiagonalRow_ACTh = TridiagonalMatrixRow{Adjoint{FT, CTh{FT}}}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}
    BidiagonalRow_C3xACTh =
        BidiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CTh{FT})')}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(zero(C3{FT}) * zero(CT3{FT})')}

    is_in_Y(name) = MatrixFields.has_field(Y, name)

    ρq_tot_if_available = is_in_Y(@name(c.ρq_tot)) ? (@name(c.ρq_tot),) : ()
    ρatke_if_available =
        is_in_Y(@name(c.sgs⁰.ρatke)) ? (@name(c.sgs⁰.ρatke),) : ()

    tracer_names = (
        @name(c.ρq_tot),
        @name(c.ρq_liq),
        @name(c.ρq_ice),
        @name(c.ρq_rai),
        @name(c.ρq_sno),
    )
    available_tracer_names = MatrixFields.unrolled_filter(is_in_Y, tracer_names)
    other_names = (@name(c.sgsʲs), @name(f.sgsʲs), @name(sfc))
    other_available_names = MatrixFields.unrolled_filter(is_in_Y, other_names)

    # Note: We have to use FT(-1) * I instead of -I because inv(-1) == -1.0,
    # which means that multiplying inv(-1) by a Float32 will yield a Float64.
    identity_blocks = MatrixFields.unrolled_map(
        name -> (name, name) => FT(-1) * I,
        (@name(c.ρ), other_available_names...),
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

    matrix = MatrixFields.FieldMatrix(
        identity_blocks...,
        advection_blocks...,
        diffusion_blocks...,
    )

    names₁_group₁ = (@name(c.ρ), other_available_names...)
    names₁_group₂ = (available_tracer_names..., ρatke_if_available...)
    names₁_group₃ = (@name(c.ρe_tot),)
    names₁ = (names₁_group₁..., names₁_group₂..., names₁_group₃...)

    alg₂ = MatrixFields.BlockLowerTriangularSolve(@name(c.uₕ))
    alg = if use_derivative(diffusion_flag)
        alg₁_subalg₂ =
            is_in_Y(@name(c.ρq_tot)) ?
            (;
                alg₂ = MatrixFields.BlockLowerTriangularSolve(names₁_group₂...)
            ) : (;)
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
function ldiv!(
    x::Fields.FieldVector,
    A::ImplicitEquationJacobian,
    b::Fields.FieldVector,
)
    MatrixFields.field_matrix_solve!(A.solver, x, A.matrix, b)
    if A.transform_flag
        @. x *= A.dtγ_ref[]
    end
end

# This method for ldiv! is called by Krylov.jl from inside ClimaTimeSteppers.jl.
# See https://github.com/JuliaSmoothOptimizers/Krylov.jl/issues/605 for a
# related issue that requires the same workaround.
function ldiv!(
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
function Wfact!(A, Y, p, dtγ, t)
    NVTX.@range "Wfact!" color = colorant"green" begin
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
            p.core.ᶜΦ,
            p.core.ᶠgradᵥ_ᶜΦ,
            p.core.ᶜρ_ref,
            p.core.ᶜp_ref,
            p.scratch.ᶜtemp_scalar,
            p.scratch.∂ᶜK_∂ᶜuₕ,
            p.scratch.∂ᶜK_∂ᶠu₃,
            p.scratch.ᶠp_grad_matrix,
            p.scratch.ᶜadvection_matrix,
            p.scratch.ᶜdiffusion_h_matrix,
            p.scratch.ᶜdiffusion_u_matrix,
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
        Fields.bycolumn(axes(Y.c)) do colidx
            update_implicit_equation_jacobian!(A, Y, p′, dtγ′, colidx)
        end
    end
end

function update_implicit_equation_jacobian!(A, Y, p, dtγ, colidx)
    (; matrix, diffusion_flag, topography_flag) = A
    (; ᶜspecific, ᶜK, ᶜts, ᶜp, ᶜΦ, ᶠgradᵥ_ᶜΦ, ᶜρ_ref, ᶜp_ref) = p
    (; ∂ᶜK_∂ᶜuₕ, ∂ᶜK_∂ᶠu₃, ᶠp_grad_matrix, ᶜadvection_matrix) = p
    (; ᶜdiffusion_h_matrix, ᶜdiffusion_u_matrix, params) = p

    FT = Spaces.undertype(axes(Y.c))
    CTh = CTh_vector_type(axes(Y.c))
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'

    cv_d = FT(CAP.cv_d(params))
    Δcv_v = FT(CAP.cv_v(params)) - cv_d
    T_tri = FT(CAP.T_triple(params))
    e_int_v0 = T_tri * Δcv_v - FT(CAP.e_int_v0(params))
    thermo_params = CAP.thermodynamics_params(params)

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶜgⁱʲ = Fields.local_geometry_field(Y.c).gⁱʲ
    ᶠgⁱʲ = Fields.local_geometry_field(Y.f).gⁱʲ

    ᶜkappa_m = p.ᶜtemp_scalar
    @. ᶜkappa_m[colidx] =
        TD.gas_constant_air(thermo_params, ᶜts[colidx]) /
        TD.cv_m(thermo_params, ᶜts[colidx])

    if use_derivative(topography_flag)
        @. ∂ᶜK_∂ᶜuₕ[colidx] = DiagonalMatrixRow(
            adjoint(CTh(ᶜuₕ[colidx])) +
            adjoint(ᶜinterp(ᶠu₃[colidx])) * g³ʰ(ᶜgⁱʲ[colidx]),
        )
    else
        @. ∂ᶜK_∂ᶜuₕ[colidx] = DiagonalMatrixRow(adjoint(CTh(ᶜuₕ[colidx])))
    end
    @. ∂ᶜK_∂ᶠu₃[colidx] =
        ᶜinterp_matrix() ⋅ DiagonalMatrixRow(adjoint(CT3(ᶠu₃[colidx]))) +
        DiagonalMatrixRow(adjoint(CT3(ᶜuₕ[colidx]))) ⋅ ᶜinterp_matrix()

    @. ᶠp_grad_matrix[colidx] =
        DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ[colidx])) ⋅ ᶠgradᵥ_matrix()

    @. ᶜadvection_matrix[colidx] =
        -(ᶜadvdivᵥ_matrix()) ⋅
        DiagonalMatrixRow(ᶠwinterp(ᶜJ[colidx], ᶜρ[colidx]))

    if use_derivative(topography_flag)
        ∂ᶜρ_err_∂ᶜuₕ = matrix[@name(c.ρ), @name(c.uₕ)]
        @. ∂ᶜρ_err_∂ᶜuₕ[colidx] =
            dtγ * ᶜadvection_matrix[colidx] ⋅
            ᶠwinterp_matrix(ᶜJ[colidx] * ᶜρ[colidx]) ⋅
            DiagonalMatrixRow(g³ʰ(ᶜgⁱʲ[colidx]))
    end
    ∂ᶜρ_err_∂ᶠu₃ = matrix[@name(c.ρ), @name(f.u₃)]
    @. ∂ᶜρ_err_∂ᶠu₃[colidx] =
        dtγ * ᶜadvection_matrix[colidx] ⋅ DiagonalMatrixRow(g³³(ᶠgⁱʲ[colidx]))

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
        use_derivative(topography_flag) && @. ∂ᶜρχ_err_∂ᶜuₕ[colidx] =
            dtγ * ᶜadvection_matrix[colidx] ⋅
            DiagonalMatrixRow(ᶠinterp(ᶜχ[colidx])) ⋅
            ᶠwinterp_matrix(ᶜJ[colidx] * ᶜρ[colidx]) ⋅
            DiagonalMatrixRow(g³ʰ(ᶜgⁱʲ[colidx]))
        @. ∂ᶜρχ_err_∂ᶠu₃[colidx] =
            dtγ * ᶜadvection_matrix[colidx] ⋅
            DiagonalMatrixRow(ᶠinterp(ᶜχ[colidx]) * g³³(ᶠgⁱʲ[colidx]))
    end

    ∂ᶠu₃_err_∂ᶜρ = matrix[@name(f.u₃), @name(c.ρ)]
    ∂ᶠu₃_err_∂ᶜρe_tot = matrix[@name(f.u₃), @name(c.ρe_tot)]
    @. ∂ᶠu₃_err_∂ᶜρ[colidx] =
        dtγ * (
            ᶠp_grad_matrix[colidx] ⋅ DiagonalMatrixRow(
                ᶜkappa_m[colidx] * (T_tri * cv_d - ᶜK[colidx] - ᶜΦ[colidx]),
            ) +
            DiagonalMatrixRow(
                (
                    ᶠgradᵥ(ᶜp[colidx] - ᶜp_ref[colidx]) -
                    ᶠinterp(ᶜρ_ref[colidx]) * ᶠgradᵥ_ᶜΦ[colidx]
                ) / abs2(ᶠinterp(ᶜρ[colidx])),
            ) ⋅ ᶠinterp_matrix()
        )
    @. ∂ᶠu₃_err_∂ᶜρe_tot[colidx] =
        dtγ * ᶠp_grad_matrix[colidx] ⋅ DiagonalMatrixRow(ᶜkappa_m[colidx])
    if MatrixFields.has_field(Y, @name(c.ρq_tot))
        ∂ᶠu₃_err_∂ᶜρq_tot = matrix[@name(f.u₃), @name(c.ρq_tot)]
        @. ∂ᶠu₃_err_∂ᶜρq_tot[colidx] =
            dtγ * ᶠp_grad_matrix[colidx] ⋅
            DiagonalMatrixRow(ᶜkappa_m[colidx] * e_int_v0)
    end

    ∂ᶠu₃_err_∂ᶜuₕ = matrix[@name(f.u₃), @name(c.uₕ)]
    ∂ᶠu₃_err_∂ᶠu₃ = matrix[@name(f.u₃), @name(f.u₃)]
    I_u₃ = DiagonalMatrixRow(one_C3xACT3)
    @. ∂ᶠu₃_err_∂ᶜuₕ[colidx] =
        dtγ * ᶠp_grad_matrix[colidx] ⋅
        DiagonalMatrixRow(-(ᶜkappa_m[colidx]) * ᶜρ[colidx]) ⋅ ∂ᶜK_∂ᶜuₕ[colidx]
    if p.atmos.rayleigh_sponge isa RayleighSponge
        @. ∂ᶠu₃_err_∂ᶠu₃[colidx] =
            dtγ * (
                ᶠp_grad_matrix[colidx] ⋅
                DiagonalMatrixRow(-(ᶜkappa_m[colidx]) * ᶜρ[colidx]) ⋅
                ∂ᶜK_∂ᶠu₃[colidx] +
                DiagonalMatrixRow(-p.ᶠβ_rayleigh_w[colidx] * (one_C3xACT3,))
            ) - (I_u₃,)
    else
        @. ∂ᶠu₃_err_∂ᶠu₃[colidx] =
            dtγ * ᶠp_grad_matrix[colidx] ⋅
            DiagonalMatrixRow(-(ᶜkappa_m[colidx]) * ᶜρ[colidx]) ⋅
            ∂ᶜK_∂ᶠu₃[colidx] - (I_u₃,)
    end

    if use_derivative(diffusion_flag)
        (; ᶜK_h, ᶜK_u) = p
        @. ᶜdiffusion_h_matrix[colidx] =
            ᶜadvdivᵥ_matrix() ⋅
            DiagonalMatrixRow(ᶠinterp(ᶜρ[colidx]) * ᶠinterp(ᶜK_h[colidx])) ⋅
            ᶠgradᵥ_matrix()
        if (
            MatrixFields.has_field(Y, @name(c.sgs⁰.ρatke)) ||
            !isnothing(p.atmos.turbconv_model) ||
            diffuse_momentum(p.atmos.vert_diff)
        )
            @. ᶜdiffusion_u_matrix[colidx] =
                ᶜadvdivᵥ_matrix() ⋅
                DiagonalMatrixRow(ᶠinterp(ᶜρ[colidx]) * ᶠinterp(ᶜK_u[colidx])) ⋅
                ᶠgradᵥ_matrix()
        end

        ∂ᶜρe_tot_err_∂ᶜρ = matrix[@name(c.ρe_tot), @name(c.ρ)]
        ∂ᶜρe_tot_err_∂ᶜρe_tot = matrix[@name(c.ρe_tot), @name(c.ρe_tot)]
        @. ∂ᶜρe_tot_err_∂ᶜρ[colidx] =
            dtγ * ᶜdiffusion_h_matrix[colidx] ⋅ DiagonalMatrixRow(
                (
                    -(1 + ᶜkappa_m[colidx]) * ᶜspecific.e_tot[colidx] -
                    ᶜkappa_m[colidx] * e_int_v0 * ᶜspecific.q_tot[colidx]
                ) / ᶜρ[colidx],
            )
        @. ∂ᶜρe_tot_err_∂ᶜρe_tot[colidx] =
            dtγ * ᶜdiffusion_h_matrix[colidx] ⋅
            DiagonalMatrixRow((1 + ᶜkappa_m[colidx]) / ᶜρ[colidx]) - (I,)
        if MatrixFields.has_field(Y, @name(c.ρq_tot))
            ∂ᶜρe_tot_err_∂ᶜρq_tot = matrix[@name(c.ρe_tot), @name(c.ρq_tot)]
            @. ∂ᶜρe_tot_err_∂ᶜρq_tot[colidx] =
                dtγ * ᶜdiffusion_h_matrix[colidx] ⋅
                DiagonalMatrixRow(ᶜkappa_m[colidx] * e_int_v0 / ᶜρ[colidx])
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
            @. ∂ᶜρq_err_∂ᶜρ[colidx] =
                dtγ * ᶜdiffusion_h_matrix[colidx] ⋅
                DiagonalMatrixRow(-(ᶜq[colidx]) / ᶜρ[colidx])
            @. ∂ᶜρq_err_∂ᶜρq[colidx] =
                dtγ * ᶜdiffusion_h_matrix[colidx] ⋅
                DiagonalMatrixRow(1 / ᶜρ[colidx]) - (I,)
        end

        if MatrixFields.has_field(Y, @name(c.sgs⁰.ρatke))
            turbconv_params = CAP.turbconv_params(params)
            c_d = CAP.tke_diss_coeff(turbconv_params)
            (; ᶜtke⁰, ᶜmixing_length, dt) = p
            ᶜρa⁰ = p.atmos.turbconv_model isa PrognosticEDMFX ? p.ᶜρa⁰ : ᶜρ
            ᶜρatke⁰ = Y.c.sgs⁰.ρatke

            dissipation_rate(tke⁰, mixing_length) =
                tke⁰ >= 0 ? c_d * sqrt(tke⁰) / max(mixing_length, 1) : 1 / dt
            ∂dissipation_rate_∂tke⁰(tke⁰, mixing_length) =
                tke⁰ > 0 ? c_d / (2 * max(mixing_length, 1) * sqrt(tke⁰)) : 0

            ᶜdissipation_matrix_diagonal = p.ᶜtemp_scalar
            @. ᶜdissipation_matrix_diagonal[colidx] =
                ᶜρatke⁰[colidx] *
                ∂dissipation_rate_∂tke⁰(ᶜtke⁰[colidx], ᶜmixing_length[colidx])

            ∂ᶜρatke⁰_err_∂ᶜρ = matrix[@name(c.sgs⁰.ρatke), @name(c.ρ)]
            ∂ᶜρatke⁰_err_∂ᶜρatke⁰ =
                matrix[@name(c.sgs⁰.ρatke), @name(c.sgs⁰.ρatke)]
            @. ∂ᶜρatke⁰_err_∂ᶜρ[colidx] =
                dtγ * (
                    ᶜdiffusion_u_matrix[colidx] -
                    DiagonalMatrixRow(ᶜdissipation_matrix_diagonal[colidx])
                ) ⋅ DiagonalMatrixRow(-(ᶜtke⁰[colidx]) / ᶜρa⁰[colidx])
            @. ∂ᶜρatke⁰_err_∂ᶜρatke⁰[colidx] =
                dtγ * (
                    (
                        ᶜdiffusion_u_matrix[colidx] -
                        DiagonalMatrixRow(ᶜdissipation_matrix_diagonal[colidx])
                    ) ⋅ DiagonalMatrixRow(1 / ᶜρa⁰[colidx]) -
                    DiagonalMatrixRow(
                        dissipation_rate(ᶜtke⁰[colidx], ᶜmixing_length[colidx]),
                    )
                ) - (I,)
        end

        if (
            !isnothing(p.atmos.turbconv_model) ||
            diffuse_momentum(p.atmos.vert_diff)
        )
            ∂ᶜuₕ_err_∂ᶜuₕ = matrix[@name(c.uₕ), @name(c.uₕ)]
            @. ∂ᶜuₕ_err_∂ᶜuₕ[colidx] =
                dtγ * DiagonalMatrixRow(1 / ᶜρ[colidx]) ⋅
                ᶜdiffusion_u_matrix[colidx] - (I,)
        end

        ᶠlg = Fields.local_geometry_field(Y.f)
        precip_info =
            ((@name(c.ρq_rai), @name(ᶜwᵣ)), (@name(c.ρq_sno), @name(ᶜwₛ)))
        MatrixFields.unrolled_foreach(precip_info) do (ρqₚ_name, wₚ_name)
            MatrixFields.has_field(Y, ρqₚ_name) || return
            ∂ᶜρqₚ_err_∂ᶜρqₚ = matrix[ρqₚ_name, ρqₚ_name]
            ᶜwₚ = MatrixFields.get_field(p, wₚ_name)
            @. ∂ᶜρqₚ_err_∂ᶜρqₚ[colidx] +=
                dtγ * -(ᶜprecipdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                    CT3(unit_basis_vector_data(CT3, ᶠlg[colidx])) *
                    ᶠwinterp(ᶜJ[colidx], ᶜρ[colidx]),
                ) ⋅ ᶠright_bias_matrix() ⋅
                DiagonalMatrixRow(-(ᶜwₚ[colidx]) / ᶜρ[colidx])
        end
    end
end
