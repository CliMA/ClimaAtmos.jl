using LinearAlgebra

using ClimaCore: Spaces, Fields, Operators
using ClimaCore.Utilities: half

const compose = Operators.ComposeStencils()
const apply = Operators.ApplyStencil()

# Note: We denote energy variables with ๐ผ, momentum variables with ๐, and tracer
# variables with ๐.
is_energy_var(symbol) = symbol in (:ฯฮธ, :ฯe_tot, :ฯe_int)
is_momentum_var(symbol) = symbol in (:uโ, :ฯuโ, :w, :ฯw)
is_edmf_var(symbol) = symbol in (:turbconv,)
is_tracer_var(symbol) = !(
    symbol == :ฯ ||
    is_energy_var(symbol) ||
    is_momentum_var(symbol) ||
    is_edmf_var(symbol)
)

struct SchurComplementW{F, FT, J1, J2, J3, J4, J5, S, A}
    # whether this struct is used to compute Wfact_t or Wfact
    transform::Bool

    # flags for computing the Jacobian
    flags::F

    # reference to dtฮณ, which is specified by the ODE solver
    dtฮณ_ref::FT

    # nonzero blocks of the Jacobian
    โแถฯโโแถ ๐::J1
    โแถ๐ผโโแถ ๐::J2
    โแถ ๐โโแถ๐ผ::J3
    โแถ ๐โโแถฯ::J3
    โแถ ๐โโแถ ๐::J4
    โแถ๐โโแถ ๐_named_tuple::J5

    # cache for the Schur complement linear solve
    S::S
    S_column_arrays::A

    # whether to test the Jacobian and linear solver
    test::Bool
end

function SchurComplementW(Y, transform, flags, test = false)
    @assert length(filter(isequal(:ฯ), propertynames(Y.c))) == 1
    @assert length(filter(is_energy_var, propertynames(Y.c))) == 1
    @assert length(filter(is_momentum_var, propertynames(Y.c))) == 1
    @assert length(filter(is_momentum_var, propertynames(Y.f))) == 1

    FT = eltype(Y)
    dtฮณ_ref = Ref(zero(FT))

    bidiag_type = Operators.StencilCoefs{-half, half, NTuple{2, FT}}
    tridiag_type = Operators.StencilCoefs{-1, 1, NTuple{3, FT}}
    quaddiag_type = Operators.StencilCoefs{-(1 + half), 1 + half, NTuple{4, FT}}

    โแถ๐ผโโแถ ๐_type =
        flags.โแถ๐ผโโแถ ๐_mode == :exact && :ฯe_tot in propertynames(Y.c) ?
        quaddiag_type : bidiag_type
    โแถฯโโแถ ๐ = Fields.Field(bidiag_type, axes(Y.c))
    โแถ๐ผโโแถ ๐ = Fields.Field(โแถ๐ผโโแถ ๐_type, axes(Y.c))
    โแถ ๐โโแถ๐ผ = Fields.Field(bidiag_type, axes(Y.f))
    โแถ ๐โโแถฯ = Fields.Field(bidiag_type, axes(Y.f))
    โแถ ๐โโแถ ๐ = Fields.Field(tridiag_type, axes(Y.f))
    แถ๐_names = filter(is_tracer_var, propertynames(Y.c))
    โแถ๐โโแถ ๐_named_tuple = NamedTuple{แถ๐_names}(
        ntuple(_ -> Fields.Field(bidiag_type, axes(Y.c)), length(แถ๐_names)),
    )

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
        typeof(dtฮณ_ref),
        typeof(โแถฯโโแถ ๐),
        typeof(โแถ๐ผโโแถ ๐),
        typeof(โแถ ๐โโแถฯ),
        typeof(โแถ ๐โโแถ ๐),
        typeof(โแถ๐โโแถ ๐_named_tuple),
        typeof(S),
        typeof(S_column_arrays),
    }(
        transform,
        flags,
        dtฮณ_ref,
        โแถฯโโแถ ๐,
        โแถ๐ผโโแถ ๐,
        โแถ ๐โโแถ๐ผ,
        โแถ ๐โโแถฯ,
        โแถ ๐โโแถ ๐,
        โแถ๐โโแถ ๐_named_tuple,
        S,
        S_column_arrays,
        test,
    )
end

# We only use Wfact, but the implicit/IMEX solvers require us to pass
# jac_prototype, then call similar(jac_prototype) to obtain J and Wfact. Here
# is a temporary workaround to avoid unnecessary allocations.
Base.similar(w::SchurComplementW) = w

#=
x = [xแถฯ
     xแถ๐ผ
     xแถ๐
     โฎ
     xแถ๐[i]
     โฎ
     xแถ ๐],
b = [bแถฯ
     bแถ๐ผ
     bแถ๐
     โฎ
     bแถ๐[i]
     โฎ
     bแถ ๐], and
A = -I + dtฮณ J =
    [    -I            0       0  โฏ  0  โฏ  dtฮณ โแถฯโโแถ ๐
          0           -I       0  โฏ  0  โฏ  dtฮณ โแถ๐ผโโแถ ๐
          0            0      -I  โฏ  0  โฏ       0
          โฎ            โฎ        โฎ  โฑ  โฎ          โฎ
          0            0       0  โฏ -I  โฏ  dtฮณ โแถ๐[i]โโแถ ๐
          โฎ            โฎ        โฎ     โฎ  โฑ       โฎ
     dtฮณ โแถ ๐โโแถฯ  dtฮณ โแถ ๐โโแถ๐ผ  0  โฏ  0  โฏ  dtฮณ โแถ ๐โโแถ ๐ - I].

To simplify our notation, let us denote
A = [-I    0    0  โฏ  0  โฏ  Aฯ๐
      0   -I    0  โฏ  0  โฏ  A๐ผ๐
      0    0   -I  โฏ  0  โฏ   0
      โฎ    โฎ     โฎ  โฑ  โฎ      โฎ
      0    0    0  โฏ -I  โฏ  A๐๐[i]
      โฎ    โฎ     โฎ     โฎ  โฑ   โฎ
     A๐ฯ A๐๐ผ   0  โฏ  0  โฏ  A๐๐ - I]

If A x = b, then
    -xแถฯ + Aฯ๐ xแถ ๐ = bแถฯ ==> xแถฯ = -bแถฯ + Aฯ๐ xแถ ๐                   (1)
    -xแถ๐ผ + A๐ผ๐ xแถ ๐ = bแถ๐ผ ==> xแถ๐ผ = -bแถ๐ผ + A๐ผ๐ xแถ ๐                   (2)
    -xแถ๐ = bแถ๐ ==> xแถ๐ = -bแถ๐                                       (3)
    -xแถ๐[i] + A๐๐[i] xแถ ๐ = bแถ๐[i] ==> xแถ๐[i] = -bแถ๐[i] + A๐๐[i] xแถ ๐  (4)
    A๐ฯ xแถฯ + A๐๐ผ xแถ๐ผ + (A๐๐ - I) xแถ ๐ = bแถ ๐                        (5)

Substituting (1) and (2) into (5) gives us
    A๐ฯ (-bแถฯ + Aฯ๐ xแถ ๐) + A๐๐ผ (-bแถ๐ผ + A๐ผ๐ xแถ ๐) + (A๐๐ - I) xแถ ๐ = bแถ ๐ ==>
    (A๐ฯ Aฯ๐ + A๐๐ผ A๐ผ๐ + A๐๐ - I) xแถ ๐ = bแถ ๐ + A๐ฯ bแถฯ + A๐๐ผ bแถ๐ผ ==>
    xแถ ๐ = (A๐ฯ Aฯ๐ + A๐๐ผ A๐ผ๐ + A๐๐ - I) \ (bแถ ๐ + A๐ฯ bแถฯ + A๐๐ผ bแถ๐ผ)

Given xแถ ๐, we can use (1), (2), (3), and (4) to get xแถฯ, xแถ๐ผ, xแถ๐, and xแถ๐[i].

Note: The matrix S = A๐ฯ Aฯ๐ + A๐๐ผ A๐ผ๐ + A๐๐ - I is the "Schur complement" of
the large -I block in A.
=#

function linsolve!(::Type{Val{:init}}, f, u0; kwargs...)
    function _linsolve!(x, A, b, update_matrix = false; kwargs...)
        (; dtฮณ_ref, S, S_column_arrays) = A
        (; โแถฯโโแถ ๐, โแถ๐ผโโแถ ๐, โแถ ๐โโแถ๐ผ, โแถ ๐โโแถฯ, โแถ ๐โโแถ ๐, โแถ๐โโแถ ๐_named_tuple) = A
        dtฮณ = dtฮณ_ref[]
        dtฮณยฒ = dtฮณ^2
        @nvtx "linsolve" color = colorant"lime" begin

            # Compute Schur complement
            Fields.bycolumn(axes(x.c)) do colidx

                # TODO: Extend LinearAlgebra.I to work with stencil fields. Allow more
                # than 2 diagonals per Jacobian block.
                FT = eltype(eltype(S))
                I = Ref(
                    Operators.StencilCoefs{-1, 1}((
                        zero(FT),
                        one(FT),
                        zero(FT),
                    )),
                )
                if Operators.bandwidths(eltype(โแถ๐ผโโแถ ๐)) != (-half, half)
                    str = "The linear solver cannot yet be run with the given โแถ๐ผโ/โแถ ๐ \
                        block, since it has more than 2 diagonals. So, โแถ๐ผโ/โแถ ๐ will \
                        be set to 0 for the Schur complement computation. Consider \
                        changing the โแถ๐ผโโแถ ๐_mode or the energy variable."
                    @warn str maxlog = 1
                    @. S[colidx] =
                        dtฮณ^2 * compose(โแถ ๐โโแถฯ[colidx], โแถฯโโแถ ๐[colidx]) +
                        dtฮณ * โแถ ๐โโแถ ๐[colidx] - I
                else
                    @. S[colidx] =
                        dtฮณยฒ * compose(โแถ ๐โโแถฯ[colidx], โแถฯโโแถ ๐[colidx]) +
                        dtฮณยฒ * compose(โแถ ๐โโแถ๐ผ[colidx], โแถ๐ผโโแถ ๐[colidx]) +
                        dtฮณ * โแถ ๐โโแถ ๐[colidx] - I
                end

                # Compute xแถ ๐

                xแถฯ = x.c.ฯ
                bแถฯ = b.c.ฯ
                แถ๐ผ_name = filter(is_energy_var, propertynames(x.c))[1]
                xแถ๐ผ = getproperty(x.c, แถ๐ผ_name)
                bแถ๐ผ = getproperty(b.c, แถ๐ผ_name)
                แถ๐_name = filter(is_momentum_var, propertynames(x.c))[1]
                xแถ๐ = getproperty(x.c, แถ๐_name)
                bแถ๐ = getproperty(b.c, แถ๐_name)
                แถ ๐_name = filter(is_momentum_var, propertynames(x.f))[1]
                xแถ ๐ = getproperty(x.f, แถ ๐_name).components.data.:1
                bแถ ๐ = getproperty(b.f, แถ ๐_name).components.data.:1

                @. xแถ ๐[colidx] =
                    bแถ ๐[colidx] +
                    dtฮณ * (
                        apply(โแถ ๐โโแถฯ[colidx], bแถฯ[colidx]) +
                        apply(โแถ ๐โโแถ๐ผ[colidx], bแถ๐ผ[colidx])
                    )

                xแถ ๐_column_view = parent(xแถ ๐[colidx])
                S_column = S[colidx]
                S_column_array = S_column_arrays[Threads.threadid()]
                @views S_column_array.dl .= parent(S_column.coefs.:1)[2:end]
                S_column_array.d .= parent(S_column.coefs.:2)
                @views S_column_array.du .=
                    parent(S_column.coefs.:3)[1:(end - 1)]
                thomas_algorithm!(S_column_array, xแถ ๐_column_view)

                # Compute remaining components of x

                @. xแถฯ[colidx] =
                    -bแถฯ[colidx] + dtฮณ * apply(โแถฯโโแถ ๐[colidx], xแถ ๐[colidx])
                @. xแถ๐ผ[colidx] =
                    -bแถ๐ผ[colidx] + dtฮณ * apply(โแถ๐ผโโแถ ๐[colidx], xแถ ๐[colidx])
                @. xแถ๐[colidx] = -bแถ๐[colidx]
                for แถ๐_name in filter(is_tracer_var, propertynames(x.c))
                    xแถ๐ = getproperty(x.c, แถ๐_name)
                    bแถ๐ = getproperty(b.c, แถ๐_name)
                    โแถ๐โโแถ ๐ = getproperty(โแถ๐โโแถ ๐_named_tuple, แถ๐_name)
                    @. xแถ๐[colidx] =
                        -bแถ๐[colidx] + dtฮณ * apply(โแถ๐โโแถ ๐[colidx], xแถ ๐[colidx])
                end
                for var_name in filter(is_edmf_var, propertynames(x.c))
                    xแถ๐ = getproperty(x.c, var_name)
                    bแถ๐ = getproperty(b.c, var_name)
                    @. xแถ๐[colidx] = -bแถ๐[colidx]
                end
                for var_name in filter(is_edmf_var, propertynames(x.f))
                    xแถ๐ = getproperty(x.f, var_name)
                    bแถ๐ = getproperty(b.f, var_name)
                    @. xแถ๐[colidx] = -bแถ๐[colidx]
                end
            end
            # Verify correctness (if needed)

            if A.test && Operators.bandwidths(eltype(โแถ๐ผโโแถ ๐)) == (-half, half)
                Ni, Nj, _, Nv, Nh = size(Fields.field_values(x.c))
                Nแถf = DataLayouts.typesize(FT, eltype(x.c))
                J_col = zeros(FT, Nv * Nแถf + Nv + 1, Nv * Nแถf + Nv + 1)
                for h in 1:Nh, j in 1:Nj, i in 1:Ni
                    x_col = Fields.FieldVector(;
                        c = Spaces.column(x.c, i, j, h),
                        f = Spaces.column(x.f, i, j, h),
                    )
                    b_col = Fields.FieldVector(;
                        c = Spaces.column(b.c, i, j, h),
                        f = Spaces.column(b.f, i, j, h),
                    )
                    แถฯ_position = findfirst(isequal(:ฯ), propertynames(x.c))
                    แถฯ_offset = DataLayouts.fieldtypeoffset(
                        FT,
                        eltype(x.c),
                        แถฯ_position,
                    )
                    แถฯ_indices = (Nv * แถฯ_offset + 1):(Nv * (แถฯ_offset + 1))
                    แถ๐ผ_position = findfirst(is_energy_var, propertynames(x.c))
                    แถ๐ผ_offset = DataLayouts.fieldtypeoffset(
                        FT,
                        eltype(x.c),
                        แถ๐ผ_position,
                    )
                    แถ๐ผ_indices = (Nv * แถ๐ผ_offset + 1):(Nv * (แถ๐ผ_offset + 1))
                    แถ ๐_indices = (Nv * Nแถf + 1):(Nv * (Nแถf + 1) + 1)
                    J_col[แถฯ_indices, แถ ๐_indices] .=
                        matrix_column(โแถฯโโแถ ๐, axes(x.f), i, j, h)
                    J_col[แถ๐ผ_indices, แถ ๐_indices] .=
                        matrix_column(โแถ๐ผโโแถ ๐, axes(x.f), i, j, h)
                    J_col[แถ ๐_indices, แถฯ_indices] .=
                        matrix_column(โแถ ๐โโแถฯ, axes(x.c), i, j, h)
                    J_col[แถ ๐_indices, แถ๐ผ_indices] .=
                        matrix_column(โแถ ๐โโแถ๐ผ, axes(x.c), i, j, h)
                    J_col[แถ ๐_indices, แถ ๐_indices] .=
                        matrix_column(โแถ ๐โโแถ ๐, axes(x.f), i, j, h)
                    for แถ๐_position in
                        findall(is_tracer_var, propertynames(x.c))
                        แถ๐_offset = DataLayouts.fieldtypeoffset(
                            FT,
                            eltype(x.c),
                            แถ๐_position,
                        )
                        แถ๐_indices = (Nv * แถ๐_offset + 1):(Nv * (แถ๐_offset + 1))
                        แถ๐_name = propertynames(x.c)[แถ๐_position]
                        โแถ๐โโแถ ๐ = getproperty(โแถ๐โโแถ ๐_named_tuple, แถ๐_name)
                        J_col[แถ๐_indices, แถ ๐_indices] .=
                            matrix_column(โแถ๐โโแถ ๐, axes(x.f), i, j, h)
                    end
                    @assert (-LinearAlgebra.I + dtฮณ * J_col) * x_col โ b_col
                end
            end

            # Apply transform (if needed)

            if A.transform
                x .*= dtฮณ
            end
        end
    end
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
