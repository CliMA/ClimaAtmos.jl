import LinearAlgebra: I, Adjoint, ldiv!
import ClimaCore.MatrixFields: @name
using ClimaCore.MatrixFields

abstract type DiffusionDerivativeFlag end
struct UseDiffusionDerivative <: DiffusionDerivativeFlag end
struct IgnoreDiffusionDerivative <: DiffusionDerivativeFlag end
use_derivative(flag::DiffusionDerivativeFlag) = flag == UseDiffusionDerivative()

abstract type EnthalpyDerivativeFlag end
struct UseEnthalpyDerivative <: EnthalpyDerivativeFlag end
struct IgnoreEnthalpyDerivative <: EnthalpyDerivativeFlag end
use_derivative(flag::EnthalpyDerivativeFlag) = flag == UseEnthalpyDerivative()

"""
    ImplicitEquationJacobian(Y, atmos, diffusion_flag, enthalpy_flag, transform_flag, approximate_solve_iters)

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
- `diffusion_flag::DiffusionDerivativeFlag`: whether the derivative of the
  diffusion tendency with respect to the quantities being diffused should be
  computed or approximated as 0; must be either `UseDiffusionDerivative()` or
  `IgnoreDiffusionDerivative()` instead of a `Bool` to ensure type-stability
- `enthalpy_flag::EnthalpyDerivativeFlag`: whether the derivative of total
  enthalpy with respect to vertical velocity should be computed or approximated
  as 0; must be either `UseEnthalpyDerivative()` or `IgnoreEnthalpyDerivative()`
  instead of a `Bool` to ensure type-stability
- `transform_flag::Bool`: whether the error should be transformed from ``E(Y)``
  to ``E(Y)/Δt``, which is required for non-Rosenbrock timestepping schemes from
  OrdinaryDiffEq.jl
- `approximate_solve_iters::Int`: number of iterations to take for the
  approximate linear solve required by `UseDiffusionDerivative()`
"""
struct ImplicitEquationJacobian{
    M <: MatrixFields.FieldMatrix,
    S <: MatrixFields.FieldMatrixSolver,
    D <: DiffusionDerivativeFlag,
    E <: EnthalpyDerivativeFlag,
    T <: Fields.FieldVector,
    R <: Base.RefValue,
}
    # stores the matrix E'(Y) = Δt * T_imp'(Y) - I
    matrix::M

    # solves the linear equation E'(Y) * ΔY = E(Y) for ΔY
    solver::S

    # flags that determine how E'(Y) is approximated
    diffusion_flag::D
    enthalpy_flag::E

    # required by Krylov.jl to evaluate ldiv! with AbstractVector inputs
    temp_b::T
    temp_x::T

    # required by OrdinaryDiffEq.jl to run non-Rosenbrock timestepping schemes
    transform_flag::Bool
    dtγ_ref::R
end

function ImplicitEquationJacobian(
    Y,
    atmos,
    diffusion_flag,
    enthalpy_flag,
    transform_flag,
    approximate_solve_iters,
)
    FT = Spaces.undertype(axes(Y.c))
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'
    TridiagonalRow = TridiagonalMatrixRow{FT}
    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}
    QuaddiagonalRow_ACT3 = QuaddiagonalMatrixRow{Adjoint{FT, CT3{FT}}}
    TridiagonalRow_C3xACT3 = TridiagonalMatrixRow{typeof(one_C3xACT3)}

    is_in_Y(name) = MatrixFields.has_field(Y, name)

    tracer_names = (
        @name(c.ρq_tot),
        @name(c.ρq_liq),
        @name(c.ρq_ice),
        @name(c.ρq_rai),
        @name(c.ρq_sno),
        @name(c.sgs⁰.ρatke),
    )
    available_tracer_names = MatrixFields.unrolled_filter(is_in_Y, tracer_names)
    other_names = (
        @name(c.sgsʲs),
        @name(f.sgsʲs),
        @name(c.turbconv),
        @name(f.turbconv),
        @name(sfc),
    )
    other_available_names = MatrixFields.unrolled_filter(is_in_Y, other_names)

    # Note: We have to use FT(-1) * I instead of -I because inv(-1) == -1.0,
    # which means that multiplying inv(-1) by a Float32 will yield a Float64.
    matrix = MatrixFields.FieldMatrix(
        (@name(c.ρ), @name(c.ρ)) => FT(-1) * I,
        MatrixFields.unrolled_map(
            name ->
                (name, name) =>
                    use_derivative(diffusion_flag) ?
                    similar(Y.c, TridiagonalRow) : FT(-1) * I,
            (@name(c.ρe_tot), available_tracer_names...),
        )...,
        (@name(c.uₕ), @name(c.uₕ)) =>
            use_derivative(diffusion_flag) && (
                !isnothing(atmos.turbconv_model) ||
                diffuse_momentum(atmos.vert_diff)
            ) ? similar(Y.c, TridiagonalRow) : FT(-1) * I,
        MatrixFields.unrolled_map(
            name -> (name, name) => FT(-1) * I,
            other_available_names,
        )...,
        (@name(c.ρ), @name(f.u₃)) => similar(Y.c, BidiagonalRow_ACT3),
        (@name(c.ρe_tot), @name(f.u₃)) =>
            use_derivative(enthalpy_flag) ?
            similar(Y.c, QuaddiagonalRow_ACT3) :
            similar(Y.c, BidiagonalRow_ACT3),
        MatrixFields.unrolled_map(
            name -> (name, @name(f.u₃)) => similar(Y.c, BidiagonalRow_ACT3),
            available_tracer_names,
        )...,
        (@name(f.u₃), @name(c.ρ)) => similar(Y.f, BidiagonalRow_C3),
        (@name(f.u₃), @name(c.ρe_tot)) => similar(Y.f, BidiagonalRow_C3),
        (@name(f.u₃), @name(f.u₃)) => similar(Y.f, TridiagonalRow_C3xACT3),
    )

    alg =
        use_derivative(diffusion_flag) ?
        MatrixFields.ApproximateBlockArrowheadIterativeSolve(
            @name(c);
            n_iters = approximate_solve_iters,
        ) : MatrixFields.BlockArrowheadSolve(@name(c))

    # By default, the ApproximateBlockArrowheadIterativeSolve takes 1 iteration
    # and approximates the A[c, c] block with a MainDiagonalPreconditioner.

    return ImplicitEquationJacobian(
        matrix,
        MatrixFields.FieldMatrixSolver(alg, matrix, Y),
        diffusion_flag,
        enthalpy_flag,
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
            p.precomputed.ᶠu³,
            p.precomputed.ᶜK,
            p.precomputed.ᶜp,
            p.precomputed.ᶜh_tot,
            (
                use_derivative(A.diffusion_flag) ?
                (; p.precomputed.ᶜK_u, p.precomputed.ᶜK_h) : (;)
            )...,
            (
                use_derivative(A.diffusion_flag) &&
                p.atmos.turbconv_model isa PrognosticEDMFX ?
                (; p.precomputed.ᶜρa⁰) : (;)
            )...,
            p.core.∂ᶜK_∂ᶠu₃,
            p.core.ᶜΦ,
            p.core.ᶠgradᵥ_ᶜΦ,
            p.core.ᶜρ_ref,
            p.core.ᶜp_ref,
            p.scratch.ᶜtemp_scalar,
            p.scratch.ᶠtemp_scalar,
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
    (; matrix, diffusion_flag, enthalpy_flag) = A
    (; ᶜspecific, ᶠu³, ᶜK, ᶜp, ∂ᶜK_∂ᶠu₃) = p
    (; ᶜΦ, ᶠgradᵥ_ᶜΦ, ᶜρ_ref, ᶜp_ref, params) = p
    (; energy_upwinding, density_upwinding) = p.atmos.numerics
    (; tracer_upwinding, precip_upwinding) = p.atmos.numerics

    FT = Spaces.undertype(axes(Y.c))
    one_ATC3 = CT3(FT(1))'
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'
    one_CT3xACT3 = CT3(FT(1)) * CT3(FT(1))'
    R_d = FT(CAP.R_d(params))
    cv_d = FT(CAP.cv_d(params))
    T_tri = FT(CAP.T_triple(params))

    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠg³³ = g³³_field(Y.f)

    # We can express the kinetic energy as
    # ᶜK =
    #     adjoint(CT12(ᶜuₕ)) * ᶜuₕ / 2 +
    #     ᶜinterp(adjoint(CT3(ᶠu₃)) * ᶠu₃) / 2 +
    #     adjoint(CT3(ᶜuₕ)) * ᶜinterp(ᶠu₃).
    # This means that
    # ∂(ᶜK)/∂(ᶠu₃) =
    #     ᶜinterp_matrix() ⋅ DiagonalMatrixRow(adjoint(CT3(ᶠu₃))) +
    #     DiagonalMatrixRow(adjoint(CT3(ᶜuₕ))) ⋅ ᶜinterp_matrix().
    @. ∂ᶜK_∂ᶠu₃[colidx] =
        ᶜinterp_matrix() ⋅ DiagonalMatrixRow(adjoint(CT3(ᶠu₃[colidx]))) +
        DiagonalMatrixRow(adjoint(CT3(ᶜuₕ[colidx]))) ⋅ ᶜinterp_matrix()

    # We can express the pressure as
    # ᶜp = R_d * (ᶜρe_tot / cv_d + ᶜρ * (T_tri - (ᶜK + ᶜΦ) / cv_d)) + O(ᶜq_tot).
    # we can ignore all O(ᶜq_tot) terms to approximate
    # ∂(ᶜp)/∂(ᶜρ) ≈ DiagonalMatrixRow(R_d * (T_tri - (ᶜK + ᶜΦ) / cv_d)),
    # ∂(ᶜp)/∂(ᶜK) ≈ DiagonalMatrixRow(-R_d * ᶜρ / cv_d), and
    # ∂(ᶜp)/∂(ᶜρe_tot) ≈ DiagonalMatrixRow(R_d / cv_d).

    # We can express the implicit advection tendency for scalars as either
    # ᶜρχₜ = -(ᶜadvdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠu³ * ᶠinterp(ᶜχ))) or
    # ᶜρχₜ = -(ᶜadvdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind(ᶠu³, ᶜχ))).
    # The implicit advection tendency for density is computed with ᶜχ = 1.
    # This means that either
    # ∂(ᶜρχₜ)/∂(ᶠu₃) =
    #     -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(ᶠwinterp(ᶜJ, ᶜρ)) ⋅ (
    #         ∂(ᶠu³)/∂(ᶠu₃) ⋅ DiagonalMatrixRow(ᶠinterp(ᶜχ)) +
    #         DiagonalMatrixRow(ᶠu³) ⋅ ᶠinterp_matrix() ⋅ ∂(ᶜχ)/∂(ᶠu₃)
    #     ) or
    # ∂(ᶜρχₜ)/∂(ᶠu₃) =
    #     -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(ᶠwinterp(ᶜJ, ᶜρ)) ⋅ (
    #         ∂(ᶠupwind(ᶠu³, ᶜχ))/∂(ᶠu³) ⋅ ∂(ᶠu³)/∂(ᶠu₃) +
    #         ᶠupwind_matrix(ᶠu³) ⋅ ∂(ᶜχ)/∂(ᶠu₃)
    #     ).
    # For simplicity, we approximate the value of ∂(ᶜρχₜ)/∂(ᶠu₃) for FCT (both
    # Boris-Book and Zalesak) using the value for first-order upwinding.
    # In addition, we have that
    # ∂(ᶠu³)/∂(ᶠu₃) = DiagonalMatrixRow(ᶠg³³ * one_CT3xACT3) and
    # ∂(ᶠupwind(ᶠu³, ᶜχ))/∂(ᶠu³) =
    #     DiagonalMatrixRow(
    #         u³_sign(ᶠu³) * ᶠupwind(CT3(u³_sign(ᶠu³)), ᶜχ) * (one_AC3,)
    #     ).
    # Since one_AC3 * one_CT3xACT3 = one_ACT3, we can simplify the product of
    # these derivatives to
    # ∂(ᶠupwind(ᶠu³, ᶜχ))/∂(ᶠu³) ⋅ ∂(ᶠu³)/∂(ᶠu₃) =
    #     DiagonalMatrixRow(
    #         u³_sign(ᶠu³) * ᶠupwind(CT3(u³_sign(ᶠu³)), ᶜχ) * ᶠg³³ * (one_ATC3,)
    #     ).
    # In general, ∂(ᶜχ)/∂(ᶠu₃) = 0I, except for the case
    # ∂(ᶜh_tot)/∂(ᶠu₃) =
    #     ∂((ᶜρe_tot + ᶜp) / ᶜρ)/∂(ᶜK) ⋅ ∂(ᶜK)/∂(ᶠu₃) =
    #     ∂(ᶜp)/∂(ᶜK) * DiagonalMatrixRow(1 / ᶜρ) ⋅ ∂(ᶜK)/∂(ᶠu₃).
    u³_sign(u³) = sign(u³.u³) # sign of the scalar value stored inside u³
    scalar_info = (
        (@name(c.ρ), @name(ᶜ1), density_upwinding),
        (@name(c.ρe_tot), @name(ᶜh_tot), energy_upwinding),
        (@name(c.ρq_tot), @name(ᶜspecific.q_tot), tracer_upwinding),
        (@name(c.ρq_liq), @name(ᶜspecific.q_liq), tracer_upwinding),
        (@name(c.ρq_ice), @name(ᶜspecific.q_ice), tracer_upwinding),
        (@name(c.ρq_rai), @name(ᶜspecific.q_rai), tracer_upwinding),
        (@name(c.ρq_sno), @name(ᶜspecific.q_sno), tracer_upwinding),
        (@name(c.ρq_rai), @name(ᶜspecific.q_rai), precip_upwinding),
        (@name(c.ρq_sno), @name(ᶜspecific.q_sno), precip_upwinding),
    )
    MatrixFields.unrolled_foreach(scalar_info) do (ρχ_name, χ_name, upwinding)
        MatrixFields.has_field(Y, ρχ_name) || return
        ∂ᶜρχ_err_∂ᶠu₃ = matrix[ρχ_name, @name(f.u₃)]

        ᶜχ = if χ_name == @name(ᶜ1)
            @. p.ᶜtemp_scalar[colidx] = 1
            p.ᶜtemp_scalar
        else
            MatrixFields.get_field(p, χ_name)
        end

        if upwinding == Val(:none)
            if use_derivative(enthalpy_flag) && ρχ_name == @name(c.ρe_tot)
                @. ∂ᶜρχ_err_∂ᶠu₃[colidx] =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅
                    DiagonalMatrixRow(ᶠwinterp(ᶜJ[colidx], ᶜρ[colidx])) ⋅ (
                        DiagonalMatrixRow(
                            ᶠg³³[colidx] *
                            (one_CT3xACT3,) *
                            ᶠinterp(ᶜχ[colidx]),
                        ) +
                        DiagonalMatrixRow(ᶠu³[colidx]) ⋅ ᶠinterp_matrix() ⋅
                        ∂ᶜK_∂ᶠu₃[colidx] * (-R_d / cv_d)
                    )
            else
                @. ∂ᶜρχ_err_∂ᶠu₃[colidx] =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        ᶠwinterp(ᶜJ[colidx], ᶜρ[colidx]) *
                        ᶠg³³[colidx] *
                        (one_CT3xACT3,) *
                        ᶠinterp(ᶜχ[colidx]),
                    )
            end
        else
            is_third_order = upwinding == Val(:third_order)
            ᶠupwind = is_third_order ? ᶠupwind3 : ᶠupwind1
            ᶠupwind_matrix = is_third_order ? ᶠupwind3_matrix : ᶠupwind1_matrix
            UpwindMatrixRowType =
                is_third_order ? QuaddiagonalMatrixRow : BidiagonalMatrixRow

            ᶠset_upwind_bcs = Operators.SetBoundaryOperator(;
                top = Operators.SetValue(zero(CT3{FT})),
                bottom = Operators.SetValue(zero(CT3{FT})),
            ) # Need to wrap ᶠupwind in this to give it well-defined boundaries.
            ᶠset_upwind_matrix_bcs = Operators.SetBoundaryOperator(;
                top = Operators.SetValue(zero(UpwindMatrixRowType{CT3{FT}})),
                bottom = Operators.SetValue(zero(UpwindMatrixRowType{CT3{FT}})),
            ) # Need to wrap ᶠupwind_matrix in this for the same reason.

            if use_derivative(enthalpy_flag) && ρχ_name == @name(c.ρe_tot)
                @. ∂ᶜρχ_err_∂ᶠu₃[colidx] =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅
                    DiagonalMatrixRow(ᶠwinterp(ᶜJ[colidx], ᶜρ[colidx])) ⋅ (
                        DiagonalMatrixRow(
                            u³_sign(ᶠu³[colidx]) *
                            ᶠset_upwind_bcs(
                                ᶠupwind(CT3(u³_sign(ᶠu³[colidx])), ᶜχ[colidx]),
                            ) *
                            ᶠg³³[colidx] *
                            (one_ATC3,),
                        ) +
                        ᶠset_upwind_matrix_bcs(ᶠupwind_matrix(ᶠu³[colidx])) ⋅
                        ∂ᶜK_∂ᶠu₃[colidx] * (-R_d / cv_d)
                    )
            else
                @. ∂ᶜρχ_err_∂ᶠu₃[colidx] =
                    dtγ * -(ᶜadvdivᵥ_matrix()) ⋅ DiagonalMatrixRow(
                        ᶠwinterp(ᶜJ[colidx], ᶜρ[colidx]) *
                        u³_sign(ᶠu³[colidx]) *
                        ᶠset_upwind_bcs(
                            ᶠupwind(CT3(u³_sign(ᶠu³[colidx])), ᶜχ[colidx]),
                        ) *
                        ᶠg³³[colidx] *
                        (one_ATC3,),
                    )
            end
        end
    end

    # TODO: Move the vertical advection of ρatke into the implicit tendency.
    if MatrixFields.has_field(Y, @name(c.sgs⁰.ρatke))
        ∂ᶜρatke_err_∂ᶠu₃ = matrix[@name(c.sgs⁰.ρatke), @name(f.u₃)]
        ∂ᶜρatke_err_∂ᶠu₃[colidx] .= (zero(eltype(∂ᶜρatke_err_∂ᶠu₃)),)
    end

    # We can express the implicit tendency for vertical velocity as
    # ᶠu₃ₜ =
    #     -(ᶠgradᵥ(ᶜp - ᶜp_ref) + ᶠinterp(ᶜρ - ᶜρ_ref) * ᶠgradᵥ_ᶜΦ) /
    #     ᶠinterp(ᶜρ).
    # The derivative of this expression with respect to density is
    # ∂(ᶠu₃ₜ)/∂(ᶜρ) =
    #     ∂(ᶠu₃ₜ)/∂(ᶠgradᵥ(ᶜp - ᶜp_ref)) ⋅ ∂(ᶠgradᵥ(ᶜp - ᶜp_ref))/∂(ᶜρ) +
    #     ∂(ᶠu₃ₜ)/∂(ᶠinterp(ᶜρ - ᶜρ_ref)) ⋅ ∂(ᶠinterp(ᶜρ - ᶜρ_ref))/∂(ᶜρ) +
    #     ∂(ᶠu₃ₜ)/∂(ᶠinterp(ᶜρ)) ⋅ ∂(ᶠinterp(ᶜρ))/∂(ᶜρ) =
    #     DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) ⋅ ᶠgradᵥ_matrix() ⋅ ∂(ᶜp)/∂(ᶜρ) +
    #     DiagonalMatrixRow(-(ᶠgradᵥ_ᶜΦ) / ᶠinterp(ᶜρ)) ⋅ ᶠinterp_matrix() +
    #     DiagonalMatrixRow(
    #         (ᶠgradᵥ(ᶜp - ᶜp_ref) + ᶠinterp(ᶜρ - ᶜρ_ref) * ᶠgradᵥ_ᶜΦ) /
    #         ᶠinterp(ᶜρ)^2
    #     ) ⋅ ᶠinterp_matrix() =
    #     DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) ⋅ ᶠgradᵥ_matrix() ⋅ ∂(ᶜp)/∂(ᶜρ) +
    #     DiagonalMatrixRow(
    #         (ᶠgradᵥ(ᶜp - ᶜp_ref) - ᶠinterp(ᶜρ_ref) * ᶠgradᵥ_ᶜΦ) /
    #         ᶠinterp(ᶜρ)^2
    #     ) ⋅ ᶠinterp_matrix().
    # The pressure is computed using total energy, therefore
    # ∂(ᶠu₃ₜ)/∂(ᶜρe_tot) =
    #     ∂(ᶠu₃ₜ)/∂(ᶠgradᵥ(ᶜp - ᶜp_ref)) ⋅ ∂(ᶠgradᵥ(ᶜp - ᶜp_ref))/∂(ᶜρe_tot) =
    #     DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) ⋅ ᶠgradᵥ_matrix() ⋅
    #     ∂(ᶜp)/∂(ᶜρe_tot) and
    # ∂(ᶠu₃ₜ)/∂(ᶠu₃) =
    #     ∂(ᶠu₃ₜ)/∂(ᶠgradᵥ(ᶜp - ᶜp_ref)) ⋅ ∂(ᶠgradᵥ(ᶜp - ᶜp_ref))/∂(ᶠu₃) =
    #     DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) ⋅ ᶠgradᵥ_matrix() ⋅ ∂(ᶜp)/∂(ᶠu₃) =
    #     DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) ⋅ ᶠgradᵥ_matrix() ⋅ ∂(ᶜp)/∂(ᶜK) ⋅
    #     ∂(ᶜK)/∂(ᶠu₃).
    # In addition, we sometimes have the Rayleigh sponge tendency modification
    # ᶠu₃ₜ += -p.ᶠβ_rayleigh_w * ᶠu₃,
    # which translates to
    # ∂(ᶠu₃ₜ)/∂(ᶠu₃) += DiagonalMatrixRow(-p.ᶠβ_rayleigh_w).
    # Note: Because ∂(u₃)/∂(u₃) is actually the C3 identity tensor (which we
    # denote by one_C3xACT3), rather than the scalar 1, we need to replace I
    # with I_u₃ = DiagonalMatrixRow(one_C3xACT3). We cannot use I * one_C3xACT3
    # because UniformScaling only supports Numbers as scale factors.
    ∂ᶠu₃_err_∂ᶜρ = matrix[@name(f.u₃), @name(c.ρ)]
    ∂ᶠu₃_err_∂ᶠu₃ = matrix[@name(f.u₃), @name(f.u₃)]
    I_u₃ = DiagonalMatrixRow(one_C3xACT3)
    ∂ᶠu₃_err_∂ᶜρe_tot = matrix[@name(f.u₃), @name(c.ρe_tot)]
    @. ∂ᶠu₃_err_∂ᶜρ[colidx] =
        dtγ * (
            DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ[colidx])) ⋅ ᶠgradᵥ_matrix() ⋅
            DiagonalMatrixRow(
                R_d * (T_tri - (ᶜK[colidx] + ᶜΦ[colidx]) / cv_d),
            ) +
            DiagonalMatrixRow(
                (
                    ᶠgradᵥ(ᶜp[colidx] - ᶜp_ref[colidx]) -
                    ᶠinterp(ᶜρ_ref[colidx]) * ᶠgradᵥ_ᶜΦ[colidx]
                ) / abs2(ᶠinterp(ᶜρ[colidx])),
            ) ⋅ ᶠinterp_matrix()
        )
    @. ∂ᶠu₃_err_∂ᶜρe_tot[colidx] =
        dtγ * DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ[colidx])) ⋅ ᶠgradᵥ_matrix() *
        R_d / cv_d
    if p.atmos.rayleigh_sponge isa RayleighSponge
        @. ∂ᶠu₃_err_∂ᶠu₃[colidx] =
            dtγ * (
                DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ[colidx])) ⋅ ᶠgradᵥ_matrix() ⋅
                DiagonalMatrixRow(-R_d * ᶜρ[colidx] / cv_d) ⋅ ∂ᶜK_∂ᶠu₃[colidx] +
                DiagonalMatrixRow(-p.ᶠβ_rayleigh_w[colidx] * (one_C3xACT3,))
            ) - (I_u₃,)
    else
        @. ∂ᶠu₃_err_∂ᶠu₃[colidx] =
            dtγ * DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ[colidx])) ⋅
            ᶠgradᵥ_matrix() ⋅ DiagonalMatrixRow(-R_d * ᶜρ[colidx] / cv_d) ⋅
            ∂ᶜK_∂ᶠu₃[colidx] - (I_u₃,)
    end

    # We can express the implicit diffusion tendency for horizontal velocity and
    # scalars as
    # ᶜuₕₜ = ᶜadvdivᵥ(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_u) * ᶠgradᵥ(ᶜuₕ)) / ᶜρ and
    # ᶜρχₜ = ᶜadvdivᵥ(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_h) * ᶠgradᵥ(ᶜχ)).
    # The implicit diffusion tendency for horizontal velocity actually uses
    # 2 * ᶠstrain_rate instead of ᶠgradᵥ(ᶜuₕ), but these are roughly equivalent.
    # The implicit diffusion tendency for density is the sum of the ᶜρχₜ's, but
    # we approximate the derivative of this sum with respect to ᶜρ as 0.
    # This means that
    # ∂(ᶜuₕₜ)/∂(ᶜuₕ) =
    #     DiagonalMatrixRow(1 / ᶜρ) ⋅ ᶜadvdivᵥ_matrix() ⋅
    #     DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_u)) ⋅ ᶠgradᵥ_matrix() and
    # ∂(ᶜρχₜ)/∂(ᶜρχ) =
    #     ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(ᶠinterp(ᶜρ) * ᶠinterp(ᶜK_h)) ⋅
    #     ᶠgradᵥ_matrix() ⋅ ∂(ᶜχ)/∂(ᶜρχ).
    # In general, ∂(ᶜχ)/∂(ᶜρχ) = DiagonalMatrixRow(1 / ᶜρ), except for the case
    # ∂(ᶜh_tot)/∂(ᶜρe_tot) =
    #     ∂((ᶜρe_tot + ᶜp) / ᶜρ)/∂(ᶜρe_tot) =
    #     (I + ∂(ᶜp)/∂(ᶜρe_tot)) ⋅ DiagonalMatrixRow(1 / ᶜρ) ≈
    #     DiagonalMatrixRow((1 + R_d / cv_d) / ᶜρ).
    if use_derivative(diffusion_flag)
        if (
            !isnothing(p.atmos.turbconv_model) ||
            diffuse_momentum(p.atmos.vert_diff)
        )
            ∂ᶜuₕ_err_∂ᶜuₕ = matrix[@name(c.uₕ), @name(c.uₕ)]
            @. ∂ᶜuₕ_err_∂ᶜuₕ[colidx] =
                dtγ * DiagonalMatrixRow(1 / ᶜρ[colidx]) ⋅ ᶜadvdivᵥ_matrix() ⋅
                DiagonalMatrixRow(
                    ᶠinterp(ᶜρ[colidx]) * ᶠinterp(p.ᶜK_u[colidx]),
                ) ⋅ ᶠgradᵥ_matrix() - (I,)
        end

        ᶜρ⁰ = p.atmos.turbconv_model isa PrognosticEDMFX ? p.ᶜρa⁰ : ᶜρ
        scalar_info = (
            (@name(c.ρe_tot), ᶜρ, p.ᶜK_h, 1 + R_d / cv_d),
            (@name(c.ρq_tot), ᶜρ, p.ᶜK_h, 1),
            (@name(c.ρq_liq), ᶜρ, p.ᶜK_h, 1),
            (@name(c.ρq_ice), ᶜρ, p.ᶜK_h, 1),
            (@name(c.ρq_rai), ᶜρ, p.ᶜK_h, 1),
            (@name(c.ρq_sno), ᶜρ, p.ᶜK_h, 1),
            (@name(c.sgs⁰.ρatke), ᶜρ⁰, p.ᶜK_u, 1),
        )
        MatrixFields.unrolled_foreach(scalar_info) do (ρχ_name, ᶜρ_χ, ᶜK_χ, s_χ)
            MatrixFields.has_field(Y, ρχ_name) || return
            ∂ᶜρχ_err_∂ᶜρχ = matrix[ρχ_name, ρχ_name]
            @. ∂ᶜρχ_err_∂ᶜρχ[colidx] =
                dtγ * ᶜadvdivᵥ_matrix() ⋅ DiagonalMatrixRow(
                    ᶠinterp(ᶜρ_χ[colidx]) * ᶠinterp(ᶜK_χ[colidx]),
                ) ⋅ ᶠgradᵥ_matrix() ⋅ DiagonalMatrixRow(s_χ / ᶜρ_χ[colidx]) -
                (I,)
        end
    end
end
