import LinearAlgebra: I, Adjoint, ldiv!
import ClimaCore.MatrixFields: @name
using ClimaCore.MatrixFields

abstract type EnthalpyDerivativeFlag end
struct IgnoreEnthalpyDerivative <: EnthalpyDerivativeFlag end
struct UseEnthalpyDerivative <: EnthalpyDerivativeFlag end
use_enthalpy_derivative(::IgnoreEnthalpyDerivative, _) = false
use_enthalpy_derivative(::UseEnthalpyDerivative, name) = name == @name(c.ρe_tot)

"""
    ImplicitEquationJacobian(Y; [enthalpy_flag], [transform])

A wrapper for the matrix ``∂E/∂Y``, where ``E(Y)`` is the "error" of the
implicit step with the state ``Y`` (see below for more details on this). The
`enthalpy_flag`, which can be set to either `UseEnthalpyDerivative()` or
`IgnoreEnthalpyDerivative()`, specifies whether the derivative of total enthalpy
with respect to vertical velocity should be computed or approximated as 0. The
`transform` flag, which can be set to `true` or `false`, specifies whether or
not the error should be transformed from ``E(Y)`` to ``E(Y)/Δt``, which is
required for non-Rosenbrock timestepping schemes from OrdinaryDiffEq.jl. The
default values for these flags are `IgnoreEnthalpyDerivative()` and `false`,
respectively.

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
"""
struct ImplicitEquationJacobian{
    M <: MatrixFields.FieldMatrix,
    S <: MatrixFields.FieldMatrixSolver,
    E <: EnthalpyDerivativeFlag,
    T <: Fields.FieldVector,
    R <: Base.RefValue,
}
    # stores the matrix E'(Y) = Δt * T_imp'(Y) - I
    matrix::M

    # solves the linear equation E'(Y) * ΔY = E(Y) for ΔY
    solver::S

    # whether to compute ∂(ᶜh_tot)/∂(ᶠu₃) or to approximate it as 0
    enthalpy_flag::E

    # required by Krylov.jl to evaluate ldiv! with AbstractVector inputs
    temp_b::T
    temp_x::T

    # required by OrdinaryDiffEq.jl to run non-Rosenbrock timestepping schemes
    transform::Bool
    dtγ_ref::R
end

function ImplicitEquationJacobian(
    Y;
    enthalpy_flag = IgnoreEnthalpyDerivative(),
    transform = false,
)
    FT = Spaces.undertype(axes(Y.c))
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'
    Bidiagonal_C3 = BidiagonalMatrixRow{C3{FT}}
    Bidiagonal_ACT3 = BidiagonalMatrixRow{Adjoint{FT, CT3{FT}}}
    Quaddiagonal_ACT3 = QuaddiagonalMatrixRow{Adjoint{FT, CT3{FT}}}
    Tridiagonal_C3xACT3 = TridiagonalMatrixRow{typeof(one_C3xACT3)}

    is_in_Y(name) = MatrixFields.has_field(Y, name)

    energy_name =
        MatrixFields.unrolled_findonly(is_in_Y, (@name(c.ρe_tot), @name(c.ρθ)))
    tracer_names = (
        @name(c.ρq_tot),
        @name(c.ρq_liq),
        @name(c.ρq_ice),
        @name(c.ρq_rai),
        @name(c.ρq_sno),
    )
    available_tracer_names = MatrixFields.unrolled_filter(is_in_Y, tracer_names)
    other_names = (
        @name(c.sgs⁰.ρatke),
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
        (@name(c.uₕ), @name(c.uₕ)) => FT(-1) * I,
        MatrixFields.unrolled_map(
            name -> (name, name) => FT(-1) * I,
            (energy_name, available_tracer_names..., other_available_names...),
        )...,
        (@name(c.ρ), @name(f.u₃)) => similar(Y.c, Bidiagonal_ACT3),
        (energy_name, @name(f.u₃)) =>
            use_enthalpy_derivative(enthalpy_flag, energy_name) ?
            similar(Y.c, Quaddiagonal_ACT3) : similar(Y.c, Bidiagonal_ACT3),
        MatrixFields.unrolled_map(
            name -> (name, @name(f.u₃)) => similar(Y.c, Bidiagonal_ACT3),
            available_tracer_names,
        )...,
        (@name(f.u₃), @name(c.ρ)) => similar(Y.f, Bidiagonal_C3),
        (@name(f.u₃), energy_name) => similar(Y.f, Bidiagonal_C3),
        (@name(f.u₃), @name(f.u₃)) => similar(Y.f, Tridiagonal_C3xACT3),
    )

    alg = MatrixFields.SchurComplementSolve(@name(f))

    return ImplicitEquationJacobian(
        matrix,
        MatrixFields.FieldMatrixSolver(alg, matrix, Y),
        enthalpy_flag,
        similar(Y),
        similar(Y),
        transform,
        Ref{FT}(),
    )
end

# We only use A, but OrdinaryDiffEq.jl and ClimaTimeSteppers.jl require us to
# pass jac_prototype and then call similar(jac_prototype) to obtain A. This is a
# workaround to avoid unnecessary allocations.
Base.similar(A::ImplicitEquationJacobian) = A

# This method specifies how to solve the equation E'(Y) * ΔY = E(Y) for ΔY.
function ldiv!(
    x::Fields.FieldVector,
    A::ImplicitEquationJacobian,
    b::Fields.FieldVector,
)
    MatrixFields.field_matrix_solve!(A.solver, x, A.matrix, b)
    if A.transform
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

# This function is used by OrdinaryDiffEq.jl instead of ldiv!.
linsolve!(::Type{Val{:init}}, f, u0; kwargs...) = _linsolve!
_linsolve!(x, A, b, update_matrix = false; kwargs...) = ldiv!(x, A, b)

# This method specifies how to compute E'(Y), which is referred to as "Wfact" in
# OrdinaryDiffEq.jl.
function Wfact!(A, Y, p, dtγ, t)
    NVTX.@range "Wfact!" color = colorant"green" begin
        # Remove unnecessary values from p to avoid allocations in bycolumn.
        (; energy_form, rayleigh_sponge) = p.atmos
        p′ = (;
            p.ᶜspecific,
            p.ᶠu³,
            p.ᶜK,
            p.ᶜp,
            p.∂ᶜK_∂ᶠu₃,
            p.ᶜΦ,
            p.ᶠgradᵥ_ᶜΦ,
            p.ᶜρ_ref,
            p.ᶜp_ref,
            p.ᶜtemp_scalar,
            p.params,
            p.energy_upwinding,
            p.tracer_upwinding,
            p.density_upwinding,
            p.atmos,
            (energy_form isa TotalEnergy ? (; p.ᶜh_tot) : (;))...,
            (rayleigh_sponge isa RayleighSponge ? (; p.ᶠβ_rayleigh_w) : (;))...,
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
    (; matrix, enthalpy_flag) = A
    (; ᶜspecific, ᶠu³, ᶜK, ᶜp, ∂ᶜK_∂ᶠu₃) = p
    (; ᶜΦ, ᶠgradᵥ_ᶜΦ, ᶜρ_ref, ᶜp_ref, params) = p
    (; energy_upwinding, tracer_upwinding, density_upwinding) = p

    FT = Spaces.undertype(axes(Y.c))
    one_ATC3 = CT3(FT(1))'
    one_C3xACT3 = C3(FT(1)) * CT3(FT(1))'
    one_CT3xACT3 = CT3(FT(1)) * CT3(FT(1))'
    R_d = FT(CAP.R_d(params))
    cv_d = FT(CAP.cv_d(params))
    κ_d = FT(CAP.kappa_d(params))
    p_ref_theta = FT(CAP.p_ref_theta(params))
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

    # We can express the pressure as either
    # ᶜp = p_ref_theta * (ᶜρθ * R_d / p_ref_theta)^(1 / (1 - κ_d)) or
    # ᶜp = R_d * (ᶜρe_tot / cv_d + ᶜρ * (T_tri - (ᶜK + ᶜΦ) / cv_d)) + O(ᶜq_tot).
    # In the first case, we find that
    # ∂(ᶜp)/∂(ᶜρ) = 0I,
    # ∂(ᶜp)/∂(ᶜK) = 0I, and
    # ∂(ᶜp)/∂(ᶜρθ) =
    #     DiagonalMatrixRow(
    #         R_d / (1 - κ_d) * (ᶜρθ * R_d / p_ref_theta)^(κ_d / (1 - κ_d))
    #     ).
    # In the second case, we can ignore all O(ᶜq_tot) terms to approximate
    # ∂(ᶜp)/∂(ᶜρ) ≈ DiagonalMatrixRow(R_d * (T_tri - (ᶜK + ᶜΦ) / cv_d)),
    # ∂(ᶜp)/∂(ᶜK) ≈ DiagonalMatrixRow(-R_d * ᶜρ / cv_d), and
    # ∂(ᶜp)/∂(ᶜρe_tot) ≈ DiagonalMatrixRow(R_d / cv_d).

    # We can express the implicit tendency for tracers (and density) as either
    # ᶜρχₜ = -(ᶜadvdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠu³ * ᶠinterp(ᶜχ))) or
    # ᶜρχₜ = -(ᶜadvdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind(ᶠu³, ᶜχ))).
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
    tracer_info = (
        (@name(c.ρ), @name(ᶜ1), density_upwinding),
        (@name(c.ρθ), @name(ᶜspecific.θ), energy_upwinding),
        (@name(c.ρe_tot), @name(ᶜh_tot), energy_upwinding),
        (@name(c.ρq_tot), @name(ᶜspecific.q_tot), tracer_upwinding),
        (@name(c.ρq_liq), @name(ᶜspecific.q_liq), tracer_upwinding),
        (@name(c.ρq_ice), @name(ᶜspecific.q_ice), tracer_upwinding),
        (@name(c.ρq_rai), @name(ᶜspecific.q_rai), tracer_upwinding),
        (@name(c.ρq_sno), @name(ᶜspecific.q_sno), tracer_upwinding),
    )
    available_tracer_info =
        MatrixFields.unrolled_filter(tracer_info) do (ρχ_name, _, _)
            MatrixFields.has_field(Y, ρχ_name)
        end
    MatrixFields.unrolled_foreach(
        available_tracer_info,
    ) do (ρχ_name, χ_name, upwinding)
        ∂ᶜρχ_err_∂ᶠu₃ = matrix[ρχ_name, @name(f.u₃)]

        ᶜχ = if χ_name == @name(ᶜ1)
            @. p.ᶜtemp_scalar[colidx] = 1
            p.ᶜtemp_scalar
        else
            MatrixFields.get_field(p, χ_name)
        end

        if upwinding == Val(:none)
            if use_enthalpy_derivative(enthalpy_flag, ρχ_name)
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

            if use_enthalpy_derivative(enthalpy_flag, ρχ_name)
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
    # If the pressure is computed using potential temperature, then
    # ∂(ᶠu₃ₜ)/∂(ᶜρθ) =
    #     ∂(ᶠu₃ₜ)/∂(ᶠgradᵥ(ᶜp - ᶜp_ref)) ⋅ ∂(ᶠgradᵥ(ᶜp - ᶜp_ref))/∂(ᶜρθ) =
    #     DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ)) ⋅ ᶠgradᵥ_matrix() ⋅
    #     ∂(ᶜp)/∂(ᶜρθ) and
    # ∂(ᶠu₃ₜ)/∂(ᶠu₃) = 0I.
    # If the pressure is computed using total energy, then
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
    if MatrixFields.has_field(Y, @name(c.ρθ))
        ᶜρθ = Y.c.ρθ
        ∂ᶠu₃_err_∂ᶜρθ = matrix[@name(f.u₃), @name(c.ρθ)]
        @. ∂ᶠu₃_err_∂ᶜρ[colidx] =
            dtγ * DiagonalMatrixRow(
                (
                    ᶠgradᵥ(ᶜp[colidx] - ᶜp_ref[colidx]) -
                    ᶠinterp(ᶜρ_ref[colidx]) * ᶠgradᵥ_ᶜΦ[colidx]
                ) / abs2(ᶠinterp(ᶜρ[colidx])),
            ) ⋅ ᶠinterp_matrix()
        @. ∂ᶠu₃_err_∂ᶜρθ[colidx] =
            dtγ * DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ[colidx])) ⋅
            ᶠgradᵥ_matrix() ⋅ DiagonalMatrixRow(
                R_d / (1 - κ_d) *
                (ᶜρθ[colidx] * R_d / p_ref_theta)^(κ_d / (1 - κ_d)),
            )
        if p.atmos.rayleigh_sponge isa RayleighSponge
            @. ∂ᶠu₃_err_∂ᶠu₃[colidx] =
                dtγ *
                DiagonalMatrixRow(-p.ᶠβ_rayleigh_w[colidx] * (one_C3xACT3,)) -
                (I_u₃,)
        else
            ∂ᶠu₃_err_∂ᶠu₃[colidx] .= (-I_u₃,)
        end
    elseif MatrixFields.has_field(Y, @name(c.ρe_tot))
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
            dtγ * DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ[colidx])) ⋅
            ᶠgradᵥ_matrix() * R_d / cv_d
        if p.atmos.rayleigh_sponge isa RayleighSponge
            @. ∂ᶠu₃_err_∂ᶠu₃[colidx] =
                dtγ * (
                    DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ[colidx])) ⋅
                    ᶠgradᵥ_matrix() ⋅
                    DiagonalMatrixRow(-R_d * ᶜρ[colidx] / cv_d) ⋅
                    ∂ᶜK_∂ᶠu₃[colidx] + DiagonalMatrixRow(
                        -p.ᶠβ_rayleigh_w[colidx] * (one_C3xACT3,),
                    )
                ) - (I_u₃,)
        else
            @. ∂ᶠu₃_err_∂ᶠu₃[colidx] =
                dtγ * DiagonalMatrixRow(-1 / ᶠinterp(ᶜρ[colidx])) ⋅
                ᶠgradᵥ_matrix() ⋅ DiagonalMatrixRow(-R_d * ᶜρ[colidx] / cv_d) ⋅
                ∂ᶜK_∂ᶠu₃[colidx] - (I_u₃,)
        end
    end
end
