#=
Column-wise analytic Jacobian of the HEVI implicit tendency for the
flux-form FDDG core — port of fddg_fluxform_jacobian.jl (+ the operator
matrices of sphere_dg_fd_jacobian.jl), de-globalized: the constructor takes
the DGModel and the Wfact receives it as the integrator parameter `p`.

State names follow the ClimaAtmos convention: @name(c.ρ), @name(c.ρe),
@name(f.ρw).

Implicit tendency (vertical acoustic subsystem; everything else explicit):
    ᶜρₜ  = −ᶜdivᵥ(ᶠρw)                       [linear in ρw]
    ᶜρeₜ = −ᶜdivᵥ(ᶠρw · ᶠinterp(ᶜh_tot))     [central; VanLeer corr. explicit]
    ᶠρwₜ = −ᶠgradᵥ(ᶜp) − ᶠinterp(ᶜρ)·ᶠgradᵥ(ᶜΦ)
    ρu_c: no implicit tendency.

Nonzero blocks (h_tot and K frozen — the validated :no_∂ᶜp∂ᶜK analog;
∂ᶠρwₜ/∂ᶠρw ≡ 0 under frozen K):
    ∂ᶜρₜ/∂ᶠρw  = −ᶜdivᵥ_matrix ⋅ Diag(g³³)
    ∂ᶜρeₜ/∂ᶠρw = −ᶜdivᵥ_matrix ⋅ Diag(ᶠinterp(ᶜh_tot)·g³³)
    ∂ᶠρwₜ/∂ᶜρe = −ᶠgradᵥ_matrix ⋅ (R_d/cv_d)
    ∂ᶠρwₜ/∂ᶜρ  = −ᶠgradᵥ_matrix ⋅ Diag(R_d(−(K+Φ)/cv_d + T_tri))
                 − Diag(ᶠgradᵥ(ᶜΦ)) ⋅ ᶠinterp_matrix

Residual convention (ClimaTimeSteppers, transform = false):
    R(Y) = Yᵖʳᵉᵛ + δtγ·Yₜ(Y) − Y,   ∂R/∂Y = δtγ·∂Yₜ/∂Y − I
=#

const ᶜρ_name = @name(c.ρ)
const ᶜ𝔼_name = @name(c.ρe)
const ᶠ𝕄F_name = @name(f.ρw)

struct FDDGImplicitEquationJacobian{TJ, RJ}
    ∂Yₜ∂Y::TJ
    ∂R∂Y::RJ
end

function FDDGImplicitEquationJacobian(Y, m::DGModel{FT}) where {FT}
    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{typeof(CT3(FT(0))')}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(C3(FT(0)) * CT3(FT(0))')}
    ∂Yₜ∂Y = MatrixFields.FieldMatrix(
        (ᶜρ_name, ᶠ𝕄F_name) => zeros(BidiagonalRow_ACT3, axes(Y.c)),
        (ᶜ𝔼_name, ᶠ𝕄F_name) => zeros(BidiagonalRow_ACT3, axes(Y.c)),
        (ᶠ𝕄F_name, ᶜρ_name) => zeros(BidiagonalRow_C3, axes(Y.f)),
        (ᶠ𝕄F_name, ᶜ𝔼_name) => zeros(BidiagonalRow_C3, axes(Y.f)),
        # kept (at zero) so the arrowhead structure matches the solver
        (ᶠ𝕄F_name, ᶠ𝕄F_name) =>
            zeros(TridiagonalRow_C3xACT3, axes(Y.f)),
    )
    I = MatrixFields.identity_field_matrix(Y)
    ∂R∂Y = FT(1) .* ∂Yₜ∂Y .- I
    alg = MatrixFields.BlockArrowheadSolve(ᶜρ_name, ᶜ𝔼_name)
    return FDDGImplicitEquationJacobian(
        ∂Yₜ∂Y,
        FieldMatrixWithSolver(∂R∂Y, Y, alg),
    )
end

Base.similar(j::FDDGImplicitEquationJacobian) =
    FDDGImplicitEquationJacobian(similar(j.∂Yₜ∂Y), similar(j.∂R∂Y))
Base.zero(j::FDDGImplicitEquationJacobian) =
    FDDGImplicitEquationJacobian(zero(j.∂Yₜ∂Y), zero(j.∂R∂Y))

ldiv!(
    δY::Fields.FieldVector,
    j::FDDGImplicitEquationJacobian,
    R::Fields.FieldVector,
) = ldiv!(δY, j.∂R∂Y, R)

function fddg_implicit_equation_jacobian!(
    j::FDDGImplicitEquationJacobian,
    Y,
    m::DGModel{FT},
    δtγ,
    t,
) where {FT}
    c = m.c
    (; Ic, If, ᶠgradᵥ) = m.ops
    (; ᶜdivᵥ_matrix, ᶠgradᵥ_matrix, ᶠinterp_matrix) = m.opmats
    (; ᶜΦ, eE1, eE2, eE3, eN1, eN2, eN3) = m.fields
    (; ∂Yₜ∂Y, ∂R∂Y) = j
    ρ = Y.c.ρ
    ρe = Y.c.ρe

    ∂ᶜρₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜρ_name, ᶠ𝕄F_name]
    ∂ᶜ𝔼ₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜ𝔼_name, ᶠ𝕄F_name]
    ∂ᶠ𝕄ₜ∂ᶜρ = ∂Yₜ∂Y[ᶠ𝕄F_name, ᶜρ_name]
    ∂ᶠ𝕄ₜ∂ᶜ𝔼 = ∂Yₜ∂Y[ᶠ𝕄F_name, ᶜ𝔼_name]

    uE = @. (Y.c.ρu1 * eE1 + Y.c.ρu2 * eE2 + Y.c.ρu3 * eE3) / ρ
    uN = @. (Y.c.ρu1 * eN1 + Y.c.ρu2 * eN2 + Y.c.ρu3 * eN3) / ρ
    w_c = @. Ic(Geometry.WVector(Y.f.ρw)).components.data.:1 / ρ
    K = @. (uE^2 + uN^2 + w_c^2) / 2
    p_thermo = @. pres_ρe(c, ρe, K, ᶜΦ, ρ)
    h_tot = @. (ρe + p_thermo) / ρ

    ᶠgⁱʲ = Fields.local_geometry_field(Y.f.ρw).gⁱʲ
    g³³(gⁱʲ) = reshape(
        gⁱʲ,
        Geometry.Contravariant3Axis(),
        Geometry.Contravariant3Axis(),
    )

    # ᶜρₜ = −ᶜdivᵥ(ᶠρw)
    @. ∂ᶜρₜ∂ᶠ𝕄 = -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(g³³(ᶠgⁱʲ))

    # ᶜρeₜ = −ᶜdivᵥ(ᶠρw · ᶠinterp(ᶜh_tot)); h_tot frozen
    @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
        -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(If(h_tot) * g³³(ᶠgⁱʲ))

    # ᶠρwₜ = −ᶠgradᵥ(ᶜp) − ᶠinterp(ᶜρ)·ᶠgradᵥ(ᶜΦ)
    @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 = -(ᶠgradᵥ_matrix() * c.R_d / c.cv_d)
    @. ∂ᶠ𝕄ₜ∂ᶜρ =
        -(ᶠgradᵥ_matrix()) *
        DiagonalMatrixRow(c.R_d * (-(K + ᶜΦ) / c.cv_d + c.T_tri)) -
        DiagonalMatrixRow(ᶠgradᵥ(ᶜΦ)) * ᶠinterp_matrix()

    I = one(∂R∂Y)
    @. ∂R∂Y = FT(δtγ) * ∂Yₜ∂Y - I
end

# ---------------------------------------------------------------------------
# Vector-invariant core: port of sphere_dg_fd_jacobian.jl (w faces carry
# VELOCITY, so ∂ᶠwₜ/∂ᶠw is nonzero through K). Default flags of the
# validated configuration: ∂ᶜ𝔼ₜ∂ᶠ𝕄 :no_∂ᶜp∂ᶜK, ∂ᶠ𝕄ₜ∂ᶜρ :exact.
# ---------------------------------------------------------------------------

const ᶠ𝕄V_name = @name(f.w)

struct VIImplicitEquationJacobian{TJ, RJ}
    ∂Yₜ∂Y::TJ
    ∂R∂Y::RJ
end

function VIImplicitEquationJacobian(Y, m::DGModel{FT}) where {FT}
    BidiagonalRow_C3 = BidiagonalMatrixRow{C3{FT}}
    BidiagonalRow_ACT3 = BidiagonalMatrixRow{typeof(CT3(FT(0))')}
    TridiagonalRow_C3xACT3 =
        TridiagonalMatrixRow{typeof(C3(FT(0)) * CT3(FT(0))')}
    ∂Yₜ∂Y = MatrixFields.FieldMatrix(
        (ᶜρ_name, ᶠ𝕄V_name) => zeros(BidiagonalRow_ACT3, axes(Y.c)),
        (ᶜ𝔼_name, ᶠ𝕄V_name) => zeros(BidiagonalRow_ACT3, axes(Y.c)),
        (ᶠ𝕄V_name, ᶜρ_name) => zeros(BidiagonalRow_C3, axes(Y.f)),
        (ᶠ𝕄V_name, ᶜ𝔼_name) => zeros(BidiagonalRow_C3, axes(Y.f)),
        (ᶠ𝕄V_name, ᶠ𝕄V_name) =>
            zeros(TridiagonalRow_C3xACT3, axes(Y.f)),
    )
    I = MatrixFields.identity_field_matrix(Y)
    ∂R∂Y = FT(1) .* ∂Yₜ∂Y .- I
    alg = MatrixFields.BlockArrowheadSolve(ᶜρ_name, ᶜ𝔼_name)
    return VIImplicitEquationJacobian(
        ∂Yₜ∂Y,
        FieldMatrixWithSolver(∂R∂Y, Y, alg),
    )
end

Base.similar(j::VIImplicitEquationJacobian) =
    VIImplicitEquationJacobian(similar(j.∂Yₜ∂Y), similar(j.∂R∂Y))
Base.zero(j::VIImplicitEquationJacobian) =
    VIImplicitEquationJacobian(zero(j.∂Yₜ∂Y), zero(j.∂R∂Y))

ldiv!(
    δY::Fields.FieldVector,
    j::VIImplicitEquationJacobian,
    R::Fields.FieldVector,
) = ldiv!(δY, j.∂R∂Y, R)

function vi_implicit_equation_jacobian!(
    j::VIImplicitEquationJacobian,
    Y,
    m::DGModel{FT},
    δtγ,
    t,
) where {FT}
    c = m.c
    (; Ic, If, ᶠgradᵥ) = m.ops
    (; ᶜdivᵥ_matrix, ᶠgradᵥ_matrix, ᶠinterp_matrix, ᶜinterp_matrix) =
        m.opmats
    (; ᶜΦ) = m.fields
    (; ∂Yₜ∂Y, ∂R∂Y) = j
    ρ = Y.c.ρ
    ρe = Y.c.ρe
    uₕ = Y.c.uₕ
    w = Y.f.w

    ∂ᶜρₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜρ_name, ᶠ𝕄V_name]
    ∂ᶜ𝔼ₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶜ𝔼_name, ᶠ𝕄V_name]
    ∂ᶠ𝕄ₜ∂ᶜρ = ∂Yₜ∂Y[ᶠ𝕄V_name, ᶜρ_name]
    ∂ᶠ𝕄ₜ∂ᶜ𝔼 = ∂Yₜ∂Y[ᶠ𝕄V_name, ᶜ𝔼_name]
    ∂ᶠ𝕄ₜ∂ᶠ𝕄 = ∂Yₜ∂Y[ᶠ𝕄V_name, ᶠ𝕄V_name]

    uv = @. Geometry.UVVector(uₕ)
    w_c = @. Ic(Geometry.WVector(w))
    K = @. (norm_sqr(uv) + norm_sqr(w_c)) / 2
    p_thermo = @. pres_ρe(c, ρe, K, ᶜΦ, ρ)

    ᶠgⁱʲ = Fields.local_geometry_field(w).gⁱʲ
    g³³(gⁱʲ) = reshape(
        gⁱʲ,
        Geometry.Contravariant3Axis(),
        Geometry.Contravariant3Axis(),
    )

    h_tot = @. (ρe + p_thermo) / ρ
    ∂ᶜK∂ᶠw = @. DiagonalMatrixRow(adjoint(CT3(Ic(w)))) * ᶜinterp_matrix()

    @. ∂ᶜρₜ∂ᶠ𝕄 = -(ᶜdivᵥ_matrix()) * DiagonalMatrixRow(If(ρ) * g³³(ᶠgⁱʲ))
    # :no_∂ᶜp∂ᶜK (ClimaAtmos default)
    @. ∂ᶜ𝔼ₜ∂ᶠ𝕄 =
        -(ᶜdivᵥ_matrix()) *
        DiagonalMatrixRow(If(ρ) * If(h_tot) * g³³(ᶠgⁱʲ))
    @. ∂ᶠ𝕄ₜ∂ᶜ𝔼 =
        -DiagonalMatrixRow(1 / If(ρ)) * (ᶠgradᵥ_matrix() * c.R_d / c.cv_d)
    # :exact
    @. ∂ᶠ𝕄ₜ∂ᶜρ =
        -DiagonalMatrixRow(1 / If(ρ)) *
        ᶠgradᵥ_matrix() *
        DiagonalMatrixRow(c.R_d * (-(K + ᶜΦ) / c.cv_d + c.T_tri)) +
        DiagonalMatrixRow(ᶠgradᵥ(p_thermo) / If(ρ)^2) * ᶠinterp_matrix()
    @. ∂ᶠ𝕄ₜ∂ᶠ𝕄 =
        -(
            DiagonalMatrixRow(1 / If(ρ)) *
            ᶠgradᵥ_matrix() *
            DiagonalMatrixRow(-(ρ * c.R_d / c.cv_d)) + ᶠgradᵥ_matrix()
        ) * ∂ᶜK∂ᶠw

    I = one(∂R∂Y)
    @. ∂R∂Y = FT(δtγ) * ∂Yₜ∂Y - I
end
