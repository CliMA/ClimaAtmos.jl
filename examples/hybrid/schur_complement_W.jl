using LinearAlgebra

using ClimaCore: Spaces, Fields, Operators
using ClimaCore.Utilities: half

const compose = Operators.ComposeStencils()
const apply = Operators.ApplyStencil()

# Note: We denote energy variables with 𝔼, momentum variables with 𝕄, and tracer
# variables with 𝕋.
is_energy_var(symbol) = symbol in (:ρθ, :ρe_tot, :ρe_int)
is_momentum_var(symbol) = symbol in (:uₕ, :ρuₕ, :w, :ρw)
is_edmf_var(symbol) = symbol in (:turbconv,)
is_tracer_var(symbol) = !(
    symbol == :ρ ||
    is_energy_var(symbol) ||
    is_momentum_var(symbol) ||
    is_edmf_var(symbol)
)

struct SchurComplementW{F, FT, J1, J2, J3, J4, J5, S, A}
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
    ∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple::J5

    # cache for the Schur complement linear solve
    S::S
    S_column_arrays::A

    # whether to test the Jacobian and linear solver
    test::Bool
end

function SchurComplementW(Y, transform, flags, test = false)
    @assert length(filter(isequal(:ρ), propertynames(Y.c))) == 1
    @assert length(filter(is_energy_var, propertynames(Y.c))) == 1
    @assert length(filter(is_momentum_var, propertynames(Y.c))) == 1
    @assert length(filter(is_momentum_var, propertynames(Y.f))) == 1

    FT = eltype(Y)
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
    ∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple = NamedTuple{ᶜ𝕋_names}(
        ntuple(_ -> Fields.Field(bidiag_type, axes(Y.c)), length(ᶜ𝕋_names)),
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
        typeof(dtγ_ref),
        typeof(∂ᶜρₜ∂ᶠ𝕄),
        typeof(∂ᶜ𝔼ₜ∂ᶠ𝕄),
        typeof(∂ᶠ𝕄ₜ∂ᶜρ),
        typeof(∂ᶠ𝕄ₜ∂ᶠ𝕄),
        typeof(∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple),
        typeof(S),
        typeof(S_column_arrays),
    }(
        transform,
        flags,
        dtγ_ref,
        ∂ᶜρₜ∂ᶠ𝕄,
        ∂ᶜ𝔼ₜ∂ᶠ𝕄,
        ∂ᶠ𝕄ₜ∂ᶜ𝔼,
        ∂ᶠ𝕄ₜ∂ᶜρ,
        ∂ᶠ𝕄ₜ∂ᶠ𝕄,
        ∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple,
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
function linsolve!(::Type{Val{:init}}, f, u0; kwargs...)
    function _linsolve!(x, A, b, update_matrix = false; kwargs...)
        (; dtγ_ref, S, S_column_arrays) = A
        (; ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶠ𝕄ₜ∂ᶠ𝕄, ∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple) = A
        dtγ = dtγ_ref[]

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
                if Operators.bandwidths(eltype(∂ᶜ𝔼ₜ∂ᶠ𝕄)) != (-half, half)
                    str = "The linear solver cannot yet be run with the given ∂ᶜ𝔼ₜ/∂ᶠ𝕄 \
                        block, since it has more than 2 diagonals. So, ∂ᶜ𝔼ₜ/∂ᶠ𝕄 will \
                        be set to 0 for the Schur complement computation. Consider \
                        changing the ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode or the energy variable."
                    @warn str maxlog = 1
                    @. S[colidx] =
                        dtγ^2 * compose(∂ᶠ𝕄ₜ∂ᶜρ[colidx], ∂ᶜρₜ∂ᶠ𝕄[colidx]) +
                        dtγ * ∂ᶠ𝕄ₜ∂ᶠ𝕄[colidx] - I
                else
                    @. S[colidx] =
                        dtγ^2 * compose(∂ᶠ𝕄ₜ∂ᶜρ[colidx], ∂ᶜρₜ∂ᶠ𝕄[colidx]) +
                        dtγ^2 * compose(∂ᶠ𝕄ₜ∂ᶜ𝔼[colidx], ∂ᶜ𝔼ₜ∂ᶠ𝕄[colidx]) +
                        dtγ * ∂ᶠ𝕄ₜ∂ᶠ𝕄[colidx] - I
                end

                # Compute xᶠ𝕄

                xᶜρ = x.c.ρ
                bᶜρ = b.c.ρ
                ᶜ𝔼_name = filter(is_energy_var, propertynames(x.c))[1]
                xᶜ𝔼 = getproperty(x.c, ᶜ𝔼_name)
                bᶜ𝔼 = getproperty(b.c, ᶜ𝔼_name)
                ᶜ𝕄_name = filter(is_momentum_var, propertynames(x.c))[1]
                xᶜ𝕄 = getproperty(x.c, ᶜ𝕄_name)
                bᶜ𝕄 = getproperty(b.c, ᶜ𝕄_name)
                ᶠ𝕄_name = filter(is_momentum_var, propertynames(x.f))[1]
                xᶠ𝕄 = getproperty(x.f, ᶠ𝕄_name).components.data.:1
                bᶠ𝕄 = getproperty(b.f, ᶠ𝕄_name).components.data.:1

                @. xᶠ𝕄[colidx] =
                    bᶠ𝕄[colidx] +
                    dtγ * (
                        apply(∂ᶠ𝕄ₜ∂ᶜρ[colidx], bᶜρ[colidx]) +
                        apply(∂ᶠ𝕄ₜ∂ᶜ𝔼[colidx], bᶜ𝔼[colidx])
                    )

                xᶠ𝕄_column_view = parent(xᶠ𝕄[colidx])
                S_column = S[colidx]
                S_column_array = S_column_arrays[Threads.threadid()]
                @views S_column_array.dl .= parent(S_column.coefs.:1)[2:end]
                S_column_array.d .= parent(S_column.coefs.:2)
                @views S_column_array.du .=
                    parent(S_column.coefs.:3)[1:(end - 1)]
                ldiv!(lu!(S_column_array), xᶠ𝕄_column_view)

                # Compute remaining components of x

                @. xᶜρ[colidx] =
                    -bᶜρ[colidx] + dtγ * apply(∂ᶜρₜ∂ᶠ𝕄[colidx], xᶠ𝕄[colidx])
                @. xᶜ𝔼[colidx] =
                    -bᶜ𝔼[colidx] + dtγ * apply(∂ᶜ𝔼ₜ∂ᶠ𝕄[colidx], xᶠ𝕄[colidx])
                @. xᶜ𝕄[colidx] = -bᶜ𝕄[colidx]
                for ᶜ𝕋_name in filter(is_tracer_var, propertynames(x.c))
                    xᶜ𝕋 = getproperty(x.c, ᶜ𝕋_name)
                    bᶜ𝕋 = getproperty(b.c, ᶜ𝕋_name)
                    ∂ᶜ𝕋ₜ∂ᶠ𝕄 = getproperty(∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple, ᶜ𝕋_name)
                    @. xᶜ𝕋[colidx] =
                        -bᶜ𝕋[colidx] + dtγ * apply(∂ᶜ𝕋ₜ∂ᶠ𝕄[colidx], xᶠ𝕄[colidx])
                end
                for var_name in filter(is_edmf_var, propertynames(x.c))
                    xᶜ𝕋 = getproperty(x.c, var_name)
                    bᶜ𝕋 = getproperty(b.c, var_name)
                    @. xᶜ𝕋[colidx] = -bᶜ𝕋[colidx]
                end
                for var_name in filter(is_edmf_var, propertynames(x.f))
                    xᶜ𝕋 = getproperty(x.f, var_name)
                    bᶜ𝕋 = getproperty(b.f, var_name)
                    @. xᶜ𝕋[colidx] = -bᶜ𝕋[colidx]
                end
            end
            # Verify correctness (if needed)

            if A.test && Operators.bandwidths(eltype(∂ᶜ𝔼ₜ∂ᶠ𝕄)) == (-half, half)
                Ni, Nj, _, Nv, Nh = size(Fields.field_values(x.c))
                Nᶜf = DataLayouts.typesize(FT, eltype(x.c))
                J_col = zeros(FT, Nv * Nᶜf + Nv + 1, Nv * Nᶜf + Nv + 1)
                for h in 1:Nh, j in 1:Nj, i in 1:Ni
                    x_col = Fields.FieldVector(;
                        c = Spaces.column(x.c, i, j, h),
                        f = Spaces.column(x.f, i, j, h),
                    )
                    b_col = Fields.FieldVector(;
                        c = Spaces.column(b.c, i, j, h),
                        f = Spaces.column(b.f, i, j, h),
                    )
                    ᶜρ_position = findfirst(isequal(:ρ), propertynames(x.c))
                    ᶜρ_offset = DataLayouts.fieldtypeoffset(
                        FT,
                        eltype(x.c),
                        ᶜρ_position,
                    )
                    ᶜρ_indices = (Nv * ᶜρ_offset + 1):(Nv * (ᶜρ_offset + 1))
                    ᶜ𝔼_position = findfirst(is_energy_var, propertynames(x.c))
                    ᶜ𝔼_offset = DataLayouts.fieldtypeoffset(
                        FT,
                        eltype(x.c),
                        ᶜ𝔼_position,
                    )
                    ᶜ𝔼_indices = (Nv * ᶜ𝔼_offset + 1):(Nv * (ᶜ𝔼_offset + 1))
                    ᶠ𝕄_indices = (Nv * Nᶜf + 1):(Nv * (Nᶜf + 1) + 1)
                    J_col[ᶜρ_indices, ᶠ𝕄_indices] .=
                        matrix_column(∂ᶜρₜ∂ᶠ𝕄, axes(x.f), i, j, h)
                    J_col[ᶜ𝔼_indices, ᶠ𝕄_indices] .=
                        matrix_column(∂ᶜ𝔼ₜ∂ᶠ𝕄, axes(x.f), i, j, h)
                    J_col[ᶠ𝕄_indices, ᶜρ_indices] .=
                        matrix_column(∂ᶠ𝕄ₜ∂ᶜρ, axes(x.c), i, j, h)
                    J_col[ᶠ𝕄_indices, ᶜ𝔼_indices] .=
                        matrix_column(∂ᶠ𝕄ₜ∂ᶜ𝔼, axes(x.c), i, j, h)
                    J_col[ᶠ𝕄_indices, ᶠ𝕄_indices] .=
                        matrix_column(∂ᶠ𝕄ₜ∂ᶠ𝕄, axes(x.f), i, j, h)
                    for ᶜ𝕋_position in
                        findall(is_tracer_var, propertynames(x.c))
                        ᶜ𝕋_offset = DataLayouts.fieldtypeoffset(
                            FT,
                            eltype(x.c),
                            ᶜ𝕋_position,
                        )
                        ᶜ𝕋_indices = (Nv * ᶜ𝕋_offset + 1):(Nv * (ᶜ𝕋_offset + 1))
                        ᶜ𝕋_name = propertynames(x.c)[ᶜ𝕋_position]
                        ∂ᶜ𝕋ₜ∂ᶠ𝕄 = getproperty(∂ᶜ𝕋ₜ∂ᶠ𝕄_named_tuple, ᶜ𝕋_name)
                        J_col[ᶜ𝕋_indices, ᶠ𝕄_indices] .=
                            matrix_column(∂ᶜ𝕋ₜ∂ᶠ𝕄, axes(x.f), i, j, h)
                    end
                    @assert (-LinearAlgebra.I + dtγ * J_col) * x_col ≈ b_col
                end
            end

            # Apply transform (if needed)

            if A.transform
                x .*= dtγ
            end
        end
    end
end
