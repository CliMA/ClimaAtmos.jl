using LinearAlgebra

using ClimaCore: Spaces, Fields, Operators
using ClimaCore.Utilities: half

const compose = Operators.ComposeStencils()
const apply = Operators.ApplyStencil()

struct SchurComplementW{F, FT, J1, J2, J3, J4, S, A}
    # whether this struct is used to compute Wfact_t or Wfact
    transform::Bool

    # flags for computing the Jacobian
    flags::F

    # reference to dtγ, which is specified by the ODE solver
    dtγ_ref::FT

    # nonzero blocks of the Jacobian
    ∂ᶜρₜ∂ᶠ𝕄::J1
    ∂ᶜ𝔼ₜ∂ᶠ𝕄::J2
    ∂ᶠ𝕄ₜ∂ᶜ𝔼::J3
    ∂ᶠ𝕄ₜ∂ᶜρ::J3
    ∂ᶠ𝕄ₜ∂ᶠ𝕄::J4

    # cache for the Schur complement linear solve
    S::S
    S_column_array::A

    # whether to test the Jacobian and linear solver
    test::Bool
end

function SchurComplementW(Y, transform, flags, test = false)
    FT = eltype(Y)
    dtγ_ref = Ref(zero(FT))
    center_space = axes(Y.c)
    face_space = axes(Y.f)

    # TODO: Automate this.
    J_eltype1 = Operators.StencilCoefs{-half, half, NTuple{2, FT}}
    J_eltype2 = flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact && :ρe in propertynames(Y.c) ?
        Operators.StencilCoefs{-(1 + half), 1 + half, NTuple{4, FT}} :
        J_eltype1
    J_eltype3 = Operators.StencilCoefs{-1, 1, NTuple{3, FT}}
    ∂ᶜρₜ∂ᶠ𝕄 = Fields.Field(J_eltype1, center_space)
    ∂ᶜ𝔼ₜ∂ᶠ𝕄 = Fields.Field(J_eltype2, center_space)
    ∂ᶠ𝕄ₜ∂ᶜ𝔼 = Fields.Field(J_eltype1, face_space)
    ∂ᶠ𝕄ₜ∂ᶜρ = Fields.Field(J_eltype1, face_space)
    ∂ᶠ𝕄ₜ∂ᶠ𝕄 = Fields.Field(J_eltype3, face_space)

    # TODO: Automate this.
    S_eltype = Operators.StencilCoefs{-1, 1, NTuple{3, FT}}
    S = Fields.Field(S_eltype, face_space)
    N = Spaces.nlevels(face_space)
    S_column_array = Tridiagonal(
        Array{FT}(undef, N - 1),
        Array{FT}(undef, N),
        Array{FT}(undef, N - 1),
    )

    SchurComplementW{
        typeof(flags),
        typeof(dtγ_ref),
        typeof(∂ᶜρₜ∂ᶠ𝕄),
        typeof(∂ᶜ𝔼ₜ∂ᶠ𝕄),
        typeof(∂ᶠ𝕄ₜ∂ᶜρ),
        typeof(∂ᶠ𝕄ₜ∂ᶠ𝕄),
        typeof(S),
        typeof(S_column_array),
    }(
        transform,
        flags,
        dtγ_ref,
        ∂ᶜρₜ∂ᶠ𝕄,
        ∂ᶜ𝔼ₜ∂ᶠ𝕄,
        ∂ᶠ𝕄ₜ∂ᶜ𝔼,
        ∂ᶠ𝕄ₜ∂ᶜρ,
        ∂ᶠ𝕄ₜ∂ᶠ𝕄,
        S,
        S_column_array,
        test,
    )
end

# We only use Wfact, but the implicit/IMEX solvers require us to pass
# jac_prototype, then call similar(jac_prototype) to obtain J and Wfact. Here
# is a temporary workaround to avoid unnecessary allocations.
Base.similar(w::SchurComplementW) = w

#=
A = [-I           0            dtγ ∂ᶜρₜ∂ᶠ𝕄    ;
     0            -I           dtγ ∂ᶜ𝔼ₜ∂ᶠ𝕄    ;
     dtγ ∂ᶠ𝕄ₜ∂ᶜρ  dtγ ∂ᶠ𝕄ₜ∂ᶜ𝔼  dtγ ∂ᶠ𝕄ₜ∂ᶠ𝕄 - I] =
    [-I   0    A13    ;
     0    -I   A23    ;
     A31  A32  A33 - I]
b = [b1; b2; b3]
x = [x1; x2; x3]
Solving A x = b:
    -x1 + A13 x3 = b1 ==> x1 = -b1 + A13 x3  (1)
    -x2 + A23 x3 = b2 ==> x2 = -b2 + A23 x3  (2)
    A31 x1 + A32 x2 + (A33 - I) x3 = b3  (3)
Substitute (1) and (2) into (3):
    A31 (-b1 + A13 x3) + A32 (-b2 + A23 x3) + (A33 - I) x3 = b3 ==>
    (A31 A13 + A32 A23 + A33 - I) x3 = b3 + A31 b1 + A32 b2 ==>
    x3 = (A31 A13 + A32 A23 + A33 - I) \ (b3 + A31 b1 + A32 b2)
Finally, use (1) and (2) to get x1 and x2.
Note: The matrix S = A31 A13 + A32 A23 + A33 - I is the "Schur complement" of
[-I 0; 0 -I] (the top-left 4 blocks) in A.
=#
function linsolve!(::Type{Val{:init}}, f, u0; kwargs...)
    function _linsolve!(x, A, b, update_matrix = false; kwargs...)
        # Initialize x as if the Jacobian is 0. This properly set the values for
        # Fields that are not included in the Jacobian (e.g, uₕ and tracers).
        @. x = -b

        (; dtγ_ref, ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶠ𝕄ₜ∂ᶠ𝕄) = A
        (; S, S_column_array) = A
        dtγ = dtγ_ref[]

        xᶜρ = x.c.ρ
        bᶜρ = b.c.ρ
        if :ρθ in propertynames(x.c)
            xᶜ𝔼 = x.c.ρθ
            bᶜ𝔼 = b.c.ρθ
        elseif :ρe in propertynames(x.c)
            xᶜ𝔼 = x.c.ρe
            bᶜ𝔼 = b.c.ρe
        elseif :ρe_int in propertynames(x.c)
            xᶜ𝔼 = x.c.ρe_int
            bᶜ𝔼 = b.c.ρe_int
        end
        if :ρw in propertynames(x.f)
            xᶠ𝕄 = x.f.ρw.components.data.:1
            bᶠ𝕄 = b.f.ρw.components.data.:1
        elseif :w in propertynames(x.f)
            xᶠ𝕄 = x.f.w.components.data.:1
            bᶠ𝕄 = b.f.w.components.data.:1
        end

        # TODO: Extend LinearAlgebra.I to work with stencil fields.
        FT = eltype(eltype(S))
        I = Ref(Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT))))
        if Operators.bandwidths(eltype(∂ᶜ𝔼ₜ∂ᶠ𝕄)) != (-half, half)
            str = "The linear solver cannot yet be run with the given ∂ᶜ𝔼ₜ/∂ᶠ𝕄 \
                block, since it has more than 2 diagonals. So, ∂ᶜ𝔼ₜ/∂ᶠ𝕄 will \
                be set to 0 for the Schur complement computation. Consider \
                changing the ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode or the energy variable."
            @warn str maxlog = 1
            @. S = dtγ^2 * compose(∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄) + dtγ * ∂ᶠ𝕄ₜ∂ᶠ𝕄 - I
        else
            @. S =
                dtγ^2 * compose(∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄) +
                dtγ^2 * compose(∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶜ𝔼ₜ∂ᶠ𝕄) +
                dtγ * ∂ᶠ𝕄ₜ∂ᶠ𝕄 - I
        end

        @. xᶠ𝕄 = bᶠ𝕄 + dtγ * (apply(∂ᶠ𝕄ₜ∂ᶜρ, bᶜρ) + apply(∂ᶠ𝕄ₜ∂ᶜ𝔼, bᶜ𝔼))

        # TODO: Do this with stencil_solve!.
        Ni, Nj, _, _, Nh = size(Spaces.local_geometry_data(axes(xᶜρ)))
        for h in 1:Nh, j in 1:Nj, i in 1:Ni
            xᶠ𝕄_column_view = parent(Spaces.column(xᶠ𝕄, i, j, h))
            S_column = Spaces.column(S, i, j, h)
            @views S_column_array.dl .= parent(S_column.coefs.:1)[2:end]
            S_column_array.d .= parent(S_column.coefs.:2)
            @views S_column_array.du .= parent(S_column.coefs.:3)[1:(end - 1)]
            ldiv!(lu!(S_column_array), xᶠ𝕄_column_view)
        end

        @. xᶜρ = -bᶜρ + dtγ * apply(∂ᶜρₜ∂ᶠ𝕄, xᶠ𝕄)
        @. xᶜ𝔼 = -bᶜ𝔼 + dtγ * apply(∂ᶜ𝔼ₜ∂ᶠ𝕄, xᶠ𝕄)

        if A.test && Operators.bandwidths(eltype(∂ᶜ𝔼ₜ∂ᶠ𝕄)) == (-half, half)
            Ni, Nj, _, Nv, Nh = size(Spaces.local_geometry_data(axes(xᶜρ)))
            ∂Yₜ∂Y = Array{FT}(undef, 3 * Nv + 1, 3 * Nv + 1)
            ΔY = Array{FT}(undef, 3 * Nv + 1)
            ΔΔY = Array{FT}(undef, 3 * Nv + 1)
            for h in 1:Nh, j in 1:Nj, i in 1:Ni
                ∂Yₜ∂Y .= zero(FT)
                ∂Yₜ∂Y[1:Nv, (2 * Nv + 1):(3 * Nv + 1)] .=
                    matrix_column(∂ᶜρₜ∂ᶠ𝕄, axes(x.f), i, j, h)
                ∂Yₜ∂Y[(Nv + 1):(2 * Nv), (2 * Nv + 1):(3 * Nv + 1)] .=
                    matrix_column(∂ᶜ𝔼ₜ∂ᶠ𝕄, axes(x.f), i, j, h)
                ∂Yₜ∂Y[(2 * Nv + 1):(3 * Nv + 1), 1:Nv] .=
                    matrix_column(∂ᶠ𝕄ₜ∂ᶜρ, axes(x.c), i, j, h)
                ∂Yₜ∂Y[(2 * Nv + 1):(3 * Nv + 1), (Nv + 1):(2 * Nv)] .=
                    matrix_column(∂ᶠ𝕄ₜ∂ᶜ𝔼, axes(x.c), i, j, h)
                ∂Yₜ∂Y[(2 * Nv + 1):(3 * Nv + 1), (2 * Nv + 1):(3 * Nv + 1)] .=
                    matrix_column(∂ᶠ𝕄ₜ∂ᶠ𝕄, axes(x.f), i, j, h)
                ΔY[1:Nv] .= vector_column(xᶜρ, i, j, h)
                ΔY[(Nv + 1):(2 * Nv)] .= vector_column(xᶜ𝔼, i, j, h)
                ΔY[(2 * Nv + 1):(3 * Nv + 1)] .= vector_column(xᶠ𝕄, i, j, h)
                ΔΔY[1:Nv] .= vector_column(bᶜρ, i, j, h)
                ΔΔY[(Nv + 1):(2 * Nv)] .= vector_column(bᶜ𝔼, i, j, h)
                ΔΔY[(2 * Nv + 1):(3 * Nv + 1)] .= vector_column(bᶠ𝕄, i, j, h)
                @assert (-LinearAlgebra.I + dtγ * ∂Yₜ∂Y) * ΔY ≈ ΔΔY
            end
        end

        if A.transform
            x .*= dtγ
        end
    end
end
