#####
##### Schur Complement for wfact
#####

using LinearAlgebra

import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators
using ClimaCore.Utilities: half

struct SchurComplementW{F, FT, J1, J2, J3, J4, J5, S, A, T}
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
    ∂ᶜ𝕋ₜ∂ᶠ𝕄_field::J5

    # cache for the Schur complement linear solve
    S::S
    S_column_arrays::A

    # whether to test the Jacobian and linear solver
    test::Bool

    # cache that is used to evaluate ldiv!
    temp1::T
    temp2::T
end

function tracer_variables(::Type{FT}, ᶜ𝕋_names) where {FT}
    (; zip(ᶜ𝕋_names, bidiag_ntuple(FT, Val(length(ᶜ𝕋_names))))...)
end

function bidiag_ntuple(::Type{FT}, ::Val{N}) where {FT, N}
    ntuple(
        i -> Operators.StencilCoefs{-half, half, NTuple{2, FT}}((FT(0), FT(0))),
        Val(N),
    )
end

# TODO: remove this
function _FieldFromNamedTuple(space, nt::NamedTuple)
    cmv(z) = nt
    return cmv.(Fields.coordinate_field(space))
end

function SchurComplementW(Y, transform, flags, test = false)
    @assert length(filter(isequal(:ρ), propertynames(Y.c))) == 1
    @assert length(filter(is_energy_var, propertynames(Y.c))) == 1
    @assert length(filter(is_momentum_var, propertynames(Y.c))) == 1
    @assert length(filter(is_momentum_var, propertynames(Y.f))) == 1

    FT = Spaces.undertype(axes(Y.c))
    dtγ_ref = Ref(zero(FT))

    bidiag_type = Operators.StencilCoefs{-half, half, NTuple{2, FT}}
    tridiag_type = Operators.StencilCoefs{-1, 1, NTuple{3, FT}}
    quaddiag_type = Operators.StencilCoefs{-(1 + half), 1 + half, NTuple{4, FT}}

    ∂ᶜ𝔼ₜ∂ᶠ𝕄_type =
        flags.∂ᶜ𝔼ₜ∂ᶠ𝕄_mode == :exact && :ρe_tot in propertynames(Y.c) ?
        quaddiag_type : bidiag_type
    ∂ᶜρₜ∂ᶠ𝕄 = Fields.Field(bidiag_type, axes(Y.c))
    ∂ᶜ𝔼ₜ∂ᶠ𝕄 = Fields.Field(∂ᶜ𝔼ₜ∂ᶠ𝕄_type, axes(Y.c))
    ∂ᶠ𝕄ₜ∂ᶜ𝔼 = Fields.Field(bidiag_type, axes(Y.f))
    ∂ᶠ𝕄ₜ∂ᶜρ = Fields.Field(bidiag_type, axes(Y.f))
    ∂ᶠ𝕄ₜ∂ᶠ𝕄 = Fields.Field(tridiag_type, axes(Y.f))
    ᶜ𝕋_names = filter(is_tracer_var, propertynames(Y.c))

    # TODO: can we make this work instead?
    # cf = Fields.coordinate_field(axes(Y.c))
    # named_tuple_field(z) = tracer_variables(FT, ᶜ𝕋_names)
    # ∂ᶜ𝕋ₜ∂ᶠ𝕄_field = named_tuple_field.(cf)
    ∂ᶜ𝕋ₜ∂ᶠ𝕄_field =
        _FieldFromNamedTuple(axes(Y.c), tracer_variables(FT, ᶜ𝕋_names))

    S = Fields.Field(tridiag_type, axes(Y.f))
    N = Spaces.nlevels(axes(Y.f))
    S_column_arrays = [
        Tridiagonal(
            Array{FT}(undef, N - 1),
            Array{FT}(undef, N),
            Array{FT}(undef, N - 1),
        ) for _ in 1:Threads.nthreads()
    ]

    SchurComplementW{
        typeof(flags),
        typeof(dtγ_ref),
        typeof(∂ᶜρₜ∂ᶠ𝕄),
        typeof(∂ᶜ𝔼ₜ∂ᶠ𝕄),
        typeof(∂ᶠ𝕄ₜ∂ᶜρ),
        typeof(∂ᶠ𝕄ₜ∂ᶠ𝕄),
        typeof(∂ᶜ𝕋ₜ∂ᶠ𝕄_field),
        typeof(S),
        typeof(S_column_arrays),
        typeof(Y),
    }(
        transform,
        flags,
        dtγ_ref,
        ∂ᶜρₜ∂ᶠ𝕄,
        ∂ᶜ𝔼ₜ∂ᶠ𝕄,
        ∂ᶠ𝕄ₜ∂ᶜ𝔼,
        ∂ᶠ𝕄ₜ∂ᶜρ,
        ∂ᶠ𝕄ₜ∂ᶠ𝕄,
        ∂ᶜ𝕋ₜ∂ᶠ𝕄_field,
        S,
        S_column_arrays,
        test,
        similar(Y),
        similar(Y),
    )
end

# We only use Wfact, but the implicit/IMEX solvers require us to pass
# jac_prototype, then call similar(jac_prototype) to obtain J and Wfact. Here
# is a temporary workaround to avoid unnecessary allocations.
Base.similar(w::SchurComplementW) = w

#=
x = [xᶜρ
     xᶜ𝔼
     xᶜ𝕄
     ⋮
     xᶜ𝕋[i]
     ⋮
     xᶠ𝕄],
b = [bᶜρ
     bᶜ𝔼
     bᶜ𝕄
     ⋮
     bᶜ𝕋[i]
     ⋮
     bᶠ𝕄], and
A = -I + dtγ J =
    [    -I            0       0  ⋯  0  ⋯  dtγ ∂ᶜρₜ∂ᶠ𝕄
          0           -I       0  ⋯  0  ⋯  dtγ ∂ᶜ𝔼ₜ∂ᶠ𝕄
          0            0      -I  ⋯  0  ⋯       0
          ⋮            ⋮        ⋮  ⋱  ⋮          ⋮
          0            0       0  ⋯ -I  ⋯  dtγ ∂ᶜ𝕋[i]ₜ∂ᶠ𝕄
          ⋮            ⋮        ⋮     ⋮  ⋱       ⋮
     dtγ ∂ᶠ𝕄ₜ∂ᶜρ  dtγ ∂ᶠ𝕄ₜ∂ᶜ𝔼  0  ⋯  0  ⋯  dtγ ∂ᶠ𝕄ₜ∂ᶠ𝕄 - I].

To simplify our notation, let us denote
A = [-I    0    0  ⋯  0  ⋯  Aρ𝕄
      0   -I    0  ⋯  0  ⋯  A𝔼𝕄
      0    0   -I  ⋯  0  ⋯   0
      ⋮    ⋮     ⋮  ⋱  ⋮      ⋮
      0    0    0  ⋯ -I  ⋯  A𝕋𝕄[i]
      ⋮    ⋮     ⋮     ⋮  ⋱   ⋮
     A𝕄ρ A𝕄𝔼   0  ⋯  0  ⋯  A𝕄𝕄 - I]

If A x = b, then
    -xᶜρ + Aρ𝕄 xᶠ𝕄 = bᶜρ ==> xᶜρ = -bᶜρ + Aρ𝕄 xᶠ𝕄                   (1)
    -xᶜ𝔼 + A𝔼𝕄 xᶠ𝕄 = bᶜ𝔼 ==> xᶜ𝔼 = -bᶜ𝔼 + A𝔼𝕄 xᶠ𝕄                   (2)
    -xᶜ𝕄 = bᶜ𝕄 ==> xᶜ𝕄 = -bᶜ𝕄                                       (3)
    -xᶜ𝕋[i] + A𝕋𝕄[i] xᶠ𝕄 = bᶜ𝕋[i] ==> xᶜ𝕋[i] = -bᶜ𝕋[i] + A𝕋𝕄[i] xᶠ𝕄  (4)
    A𝕄ρ xᶜρ + A𝕄𝔼 xᶜ𝔼 + (A𝕄𝕄 - I) xᶠ𝕄 = bᶠ𝕄                        (5)

Substituting (1) and (2) into (5) gives us
    A𝕄ρ (-bᶜρ + Aρ𝕄 xᶠ𝕄) + A𝕄𝔼 (-bᶜ𝔼 + A𝔼𝕄 xᶠ𝕄) + (A𝕄𝕄 - I) xᶠ𝕄 = bᶠ𝕄 ==>
    (A𝕄ρ Aρ𝕄 + A𝕄𝔼 A𝔼𝕄 + A𝕄𝕄 - I) xᶠ𝕄 = bᶠ𝕄 + A𝕄ρ bᶜρ + A𝕄𝔼 bᶜ𝔼 ==>
    xᶠ𝕄 = (A𝕄ρ Aρ𝕄 + A𝕄𝔼 A𝔼𝕄 + A𝕄𝕄 - I) \ (bᶠ𝕄 + A𝕄ρ bᶜρ + A𝕄𝔼 bᶜ𝔼)

Given xᶠ𝕄, we can use (1), (2), (3), and (4) to get xᶜρ, xᶜ𝔼, xᶜ𝕄, and xᶜ𝕋[i].

Note: The matrix S = A𝕄ρ Aρ𝕄 + A𝕄𝔼 A𝔼𝕄 + A𝕄𝕄 - I is the "Schur complement" of
the large -I block in A.
=#

# Function required by OrdinaryDiffEq.jl
linsolve!(::Type{Val{:init}}, f, u0; kwargs...) = _linsolve!
_linsolve!(x, A, b, update_matrix = false; kwargs...) = ldiv!(x, A, b)

# Function required by Krylov.jl (x and b can be AbstractVectors)
# See https://github.com/JuliaSmoothOptimizers/Krylov.jl/issues/605 for a
# related issue that requires the same workaround.
function LinearAlgebra.ldiv!(x, A::SchurComplementW, b)
    A.temp1 .= b
    ldiv!(A.temp2, A, A.temp1)
    x .= A.temp2
end

function LinearAlgebra.ldiv!(
    x::Fields.FieldVector,
    A::SchurComplementW,
    b::Fields.FieldVector,
)
    (; dtγ_ref, S, S_column_arrays, transform) = A
    (; ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶠ𝕄ₜ∂ᶠ𝕄, ∂ᶜ𝕋ₜ∂ᶠ𝕄_field) = A
    dtγ = dtγ_ref[]
    cond = Operators.bandwidths(eltype(∂ᶜ𝔼ₜ∂ᶠ𝕄)) != (-half, half)
    if cond
        str = "The linear solver cannot yet be run with the given ∂ᶜ𝔼ₜ/∂ᶠ𝕄 \
            block, since it has more than 2 diagonals. So, ∂ᶜ𝔼ₜ/∂ᶠ𝕄 will \
            be set to 0 for the Schur complement computation. Consider \
            changing the ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode or the energy variable."
        @warn str maxlog = 1
    end
    # @nvtx "linsolve" color = colorant"lime" begin

    # Compute Schur complement
    Fields.bycolumn(axes(x.c)) do colidx
        _ldiv_serial!(
            x.c[colidx],
            x.f[colidx],
            b.c[colidx],
            b.f[colidx],
            dtγ,
            transform,
            cond,
            ∂ᶜρₜ∂ᶠ𝕄[colidx],
            ∂ᶜ𝔼ₜ∂ᶠ𝕄[colidx],
            ∂ᶠ𝕄ₜ∂ᶜ𝔼[colidx],
            ∂ᶠ𝕄ₜ∂ᶜρ[colidx],
            ∂ᶠ𝕄ₜ∂ᶠ𝕄[colidx],
            ∂ᶜ𝕋ₜ∂ᶠ𝕄_field[colidx],
            S[colidx],
            S_column_arrays[Threads.threadid()], # can / should this be colidx?
        )
    end
    # end
end

function _ldiv_serial!(
    xc,
    xf,
    bc,
    bf,
    dtγ,
    transform,
    cond,
    ∂ᶜρₜ∂ᶠ𝕄,
    ∂ᶜ𝔼ₜ∂ᶠ𝕄,
    ∂ᶠ𝕄ₜ∂ᶜ𝔼,
    ∂ᶠ𝕄ₜ∂ᶜρ,
    ∂ᶠ𝕄ₜ∂ᶠ𝕄,
    ∂ᶜ𝕋ₜ∂ᶠ𝕄_field,
    S_column,
    S_column_array,
)
    dtγ² = dtγ^2
    # TODO: Extend LinearAlgebra.I to work with stencil fields. Allow more
    # than 2 diagonals per Jacobian block.
    FT = eltype(eltype(S_column))
    I = Ref(Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT))))
    compose = Operators.ComposeStencils()
    apply = Operators.ApplyStencil()
    if cond
        @. S_column = dtγ² * compose(∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄) + dtγ * ∂ᶠ𝕄ₜ∂ᶠ𝕄 - I
    else
        @. S_column =
            dtγ² * compose(∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄) +
            dtγ² * compose(∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶜ𝔼ₜ∂ᶠ𝕄) +
            dtγ * ∂ᶠ𝕄ₜ∂ᶠ𝕄 - I
    end

    # Compute xᶠ𝕄

    xᶜρ = xc.ρ
    bᶜρ = bc.ρ
    ᶜ𝔼_name = filter(is_energy_var, propertynames(xc))[1]
    xᶜ𝔼 = getproperty(xc, ᶜ𝔼_name)
    bᶜ𝔼 = getproperty(bc, ᶜ𝔼_name)
    ᶜ𝕄_name = filter(is_momentum_var, propertynames(xc))[1]
    xᶜ𝕄 = getproperty(xc, ᶜ𝕄_name)
    bᶜ𝕄 = getproperty(bc, ᶜ𝕄_name)
    ᶠ𝕄_name = filter(is_momentum_var, propertynames(xf))[1]
    xᶠ𝕄 = getproperty(xf, ᶠ𝕄_name).components.data.:1
    bᶠ𝕄 = getproperty(bf, ᶠ𝕄_name).components.data.:1

    @. xᶠ𝕄 = bᶠ𝕄 + dtγ * (apply(∂ᶠ𝕄ₜ∂ᶜρ, bᶜρ) + apply(∂ᶠ𝕄ₜ∂ᶜ𝔼, bᶜ𝔼))

    xᶠ𝕄_column_view = parent(xᶠ𝕄)
    @views S_column_array.dl .= parent(S_column.coefs.:1)[2:end]
    S_column_array.d .= parent(S_column.coefs.:2)
    @views S_column_array.du .= parent(S_column.coefs.:3)[1:(end - 1)]
    thomas_algorithm!(S_column_array, xᶠ𝕄_column_view)

    # Compute remaining components of x

    @. xᶜρ = -bᶜρ + dtγ * apply(∂ᶜρₜ∂ᶠ𝕄, xᶠ𝕄)
    @. xᶜ𝔼 = -bᶜ𝔼 + dtγ * apply(∂ᶜ𝔼ₜ∂ᶠ𝕄, xᶠ𝕄)
    @. xᶜ𝕄 = -bᶜ𝕄
    for ᶜ𝕋_name in filter(is_tracer_var, propertynames(xc))
        xᶜ𝕋 = getproperty(xc, ᶜ𝕋_name)
        bᶜ𝕋 = getproperty(bc, ᶜ𝕋_name)
        ∂ᶜ𝕋ₜ∂ᶠ𝕄 = getproperty(∂ᶜ𝕋ₜ∂ᶠ𝕄_field, ᶜ𝕋_name)
        @. xᶜ𝕋 = -bᶜ𝕋 + dtγ * apply(∂ᶜ𝕋ₜ∂ᶠ𝕄, xᶠ𝕄)
    end
    for var_name in filter(is_edmf_var, propertynames(xc))
        xᶜ𝕋 = getproperty(xc, var_name)
        bᶜ𝕋 = getproperty(bc, var_name)
        @. xᶜ𝕋 = -bᶜ𝕋
    end
    for var_name in filter(is_edmf_var, propertynames(xf))
        xᶠ𝕋 = getproperty(xf, var_name)
        bᶠ𝕋 = getproperty(bf, var_name)
        @. xᶠ𝕋 = -bᶠ𝕋
    end
    # Apply transform (if needed)
    if transform
        xc .*= dtγ
        xf .*= dtγ
    end
    return nothing
end

"""
    thomas_algorithm!(A, b)

Thomas algorithm for solving a linear system A x = b,
where A is a tri-diagonal matrix.
A and b are overwritten.
Solution is written to b
"""
function thomas_algorithm!(A, b)
    nrows = size(A, 1)
    # first row
    @inbounds A[1, 2] /= A[1, 1]
    @inbounds b[1] /= A[1, 1]
    # interior rows
    for row in 2:(nrows - 1)
        @inbounds fac = A[row, row] - (A[row, row - 1] * A[row - 1, row])
        @inbounds A[row, row + 1] /= fac
        @inbounds b[row] = (b[row] - A[row, row - 1] * b[row - 1]) / fac
    end
    # last row
    @inbounds fac = A[nrows, nrows] - A[nrows - 1, nrows] * A[nrows, nrows - 1]
    @inbounds b[nrows] = (b[nrows] - A[nrows, nrows - 1] * b[nrows - 1]) / fac
    # back substitution
    for row in (nrows - 1):-1:1
        @inbounds b[row] -= b[row + 1] * A[row, row + 1]
    end
    return nothing
end
