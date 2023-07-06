#####
##### Schur Complement for wfact
#####

import LinearAlgebra

import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators
using ClimaCore.Utilities: half

struct SchurComplementW{ET, F, FT, J1, J2, J3, J4, J5, J6, J7, S, T}
    # whether this struct is used to compute Wfact_t or Wfact
    transform::Bool

    # flags for computing the Jacobian
    flags::F

    # reference to dtγ, which is specified by the ODE solver
    dtγ_ref::FT

    # nonzero blocks of the "dycore Jacobian"
    ∂ᶜρₜ∂ᶠ𝕄::J1
    ∂ᶜ𝔼ₜ∂ᶠ𝕄::J2
    ∂ᶠ𝕄ₜ∂ᶜ𝔼::J3
    ∂ᶠ𝕄ₜ∂ᶜρ::J3
    ∂ᶠ𝕄ₜ∂ᶠ𝕄::J4
    ∂ᶜ𝕋ₜ∂ᶠ𝕄_field::J5

    # nonzero blocks of the "TC Jacobian"
    ∂ᶜTCₜ∂ᶜTC::J6
    ∂ᶠTCₜ∂ᶠTC::J7

    # cache for the Schur complement linear solve
    S::S

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

    if :turbconv in propertynames(Y.c)
        ᶜTC = Y.c.turbconv
        ᶠTC = Y.f.turbconv

        ∂ᶜTCₜ∂ᶜTC_type =
            DataLayouts.replace_basetype(FT, tridiag_type, eltype(ᶜTC))
        ∂ᶠTCₜ∂ᶠTC_type =
            DataLayouts.replace_basetype(FT, tridiag_type, eltype(ᶠTC))

        ∂ᶜTCₜ∂ᶜTC = similar(ᶜTC, ∂ᶜTCₜ∂ᶜTC_type)
        ∂ᶠTCₜ∂ᶠTC = similar(ᶠTC, ∂ᶠTCₜ∂ᶠTC_type)

        for var_prop_chain in Fields.property_chains(ᶜTC)
            ∂ᶜvarₜ∂ᶜvar =
                Fields.single_field(∂ᶜTCₜ∂ᶜTC, var_prop_chain, identity)
            ∂ᶜvarₜ∂ᶜvar .= tuple(tridiag_type((0, 0, 0)))
        end
        for var_prop_chain in Fields.property_chains(ᶠTC)
            ∂ᶠvarₜ∂ᶠvar =
                Fields.single_field(∂ᶠTCₜ∂ᶠTC, var_prop_chain, identity)
            ∂ᶠvarₜ∂ᶠvar .= tuple(tridiag_type((0, 0, 0)))
        end
    else
        ∂ᶜTCₜ∂ᶜTC = nothing
        ∂ᶠTCₜ∂ᶠTC = nothing
    end

    S = Fields.Field(tridiag_type, axes(Y.f))
    N = Spaces.nlevels(axes(Y.f))

    ᶜ𝕋_names = filter(is_tracer_var, propertynames(Y.c))
    ET = if isempty(ᶜ𝕋_names)
        Nothing
    else
        cid = Fields.ColumnIndex((1, 1), 1)
        typeof(getproperty(∂ᶜ𝕋ₜ∂ᶠ𝕄_field[cid], ᶜ𝕋_names[1]))
    end
    SchurComplementW{
        ET,
        typeof(flags),
        typeof(dtγ_ref),
        typeof(∂ᶜρₜ∂ᶠ𝕄),
        typeof(∂ᶜ𝔼ₜ∂ᶠ𝕄),
        typeof(∂ᶠ𝕄ₜ∂ᶜρ),
        typeof(∂ᶠ𝕄ₜ∂ᶠ𝕄),
        typeof(∂ᶜ𝕋ₜ∂ᶠ𝕄_field),
        typeof(∂ᶜTCₜ∂ᶜTC),
        typeof(∂ᶠTCₜ∂ᶠTC),
        typeof(S),
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
        ∂ᶜTCₜ∂ᶜTC,
        ∂ᶠTCₜ∂ᶠTC,
        S,
        test,
        similar(Y),
        similar(Y),
    )
end

∂ᶜ𝕋ₜ∂ᶠ𝕄_field_eltype(A::T) where {T <: SchurComplementW} =
    ∂ᶜ𝕋ₜ∂ᶠ𝕄_field_eltype(T)
∂ᶜ𝕋ₜ∂ᶠ𝕄_field_eltype(::Type{T}) where {ET, T <: SchurComplementW{ET}} = ET

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
     xᶜTC
     xᶠ𝕄
     xᶠTC],
b = [bᶜρ
     bᶜ𝔼
     bᶜ𝕄
     ⋮
     bᶜ𝕋[i]
     ⋮
     bᶜTC
     bᶠ𝕄
     bᶠTC], and
A = -I + dtγ J =
    [    -I            0       0  ⋯  0  ⋯       0            dtγ ∂ᶜρₜ∂ᶠ𝕄         0
          0           -I       0  ⋯  0  ⋯       0            dtγ ∂ᶜ𝔼ₜ∂ᶠ𝕄         0
          0            0      -I  ⋯  0  ⋯       0                 0              0
          ⋮            ⋮        ⋮  ⋱  ⋮          ⋮                 ⋮               ⋮
          0            0       0  ⋯ -I  ⋯       0            dtγ ∂ᶜ𝕋[i]ₜ∂ᶠ𝕄       0
          ⋮            ⋮        ⋮     ⋮  ⋱       ⋮                 ⋮               ⋮
          0            0       0  ⋯  0  ⋯  dtγ ∂ᶜTCₜ∂ᶜTC - I      0               0
     dtγ ∂ᶠ𝕄ₜ∂ᶜρ  dtγ ∂ᶠ𝕄ₜ∂ᶜ𝔼  0  ⋯  0  ⋯       0            dtγ ∂ᶠ𝕄ₜ∂ᶠ𝕄 - I      0
          0            0       0  ⋯  0  ⋯       0                 0          dtγ ∂ᶠTCₜ∂ᶠTC - I].

To simplify our notation, let us denote
A = [-I    0    0  ⋯  0  ⋯   0       Aρ𝕄      0
      0   -I    0  ⋯  0  ⋯   0       A𝔼𝕄      0
      0    0   -I  ⋯  0  ⋯   0        0        0
      ⋮    ⋮     ⋮  ⋱  ⋮      ⋮        ⋮        ⋮
      0    0    0  ⋯ -I  ⋯   0       A𝕋𝕄[i]    0
      ⋮    ⋮     ⋮     ⋮  ⋱   0        ⋮        ⋮
      0    0    0  ⋯  0  ⋯  AᶜTC - I  0        0
     A𝕄ρ A𝕄𝔼   0  ⋯  0  ⋯    0      A𝕄𝕄 - I  0
      0    0    0  ⋯  0  ⋯    0       0       AᶠTC - I]

If A x = b, then
    -xᶜρ + Aρ𝕄 xᶠ𝕄 = bᶜρ ==> xᶜρ = -bᶜρ + Aρ𝕄 xᶠ𝕄                   (1)
    -xᶜ𝔼 + A𝔼𝕄 xᶠ𝕄 = bᶜ𝔼 ==> xᶜ𝔼 = -bᶜ𝔼 + A𝔼𝕄 xᶠ𝕄                   (2)
    -xᶜ𝕄 = bᶜ𝕄 ==> xᶜ𝕄 = -bᶜ𝕄                                       (3)
    -xᶜ𝕋[i] + A𝕋𝕄[i] xᶠ𝕄 = bᶜ𝕋[i] ==> xᶜ𝕋[i] = -bᶜ𝕋[i] + A𝕋𝕄[i] xᶠ𝕄  (4)
    (AᶜTC - I) xᶜTC = bᶜTC                                            (5)
    A𝕄ρ xᶜρ + A𝕄𝔼 xᶜ𝔼 + (A𝕄𝕄 - I) xᶠ𝕄 = bᶠ𝕄                        (6)
    (AᶠTC - I) xᶠTC = bᶠTC                                            (7)

Substituting (1) and (2) into (6) gives us
    A𝕄ρ (-bᶜρ + Aρ𝕄 xᶠ𝕄) + A𝕄𝔼 (-bᶜ𝔼 + A𝔼𝕄 xᶠ𝕄) + (A𝕄𝕄 - I) xᶠ𝕄 = bᶠ𝕄 ==>
    (A𝕄ρ Aρ𝕄 + A𝕄𝔼 A𝔼𝕄 + A𝕄𝕄 - I) xᶠ𝕄 = bᶠ𝕄 + A𝕄ρ bᶜρ + A𝕄𝔼 bᶜ𝔼 ==>
    xᶠ𝕄 = (A𝕄ρ Aρ𝕄 + A𝕄𝔼 A𝔼𝕄 + A𝕄𝕄 - I) \ (bᶠ𝕄 + A𝕄ρ bᶜρ + A𝕄𝔼 bᶜ𝔼)

Given xᶠ𝕄, we can use (1), (2), (3), and (4) to get xᶜρ, xᶜ𝔼, xᶜ𝕄, and xᶜ𝕋[i].

Note: The matrix S = A𝕄ρ Aρ𝕄 + A𝕄𝔼 A𝔼𝕄 + A𝕄𝕄 - I is the "Schur complement" of
the large -I block in A.
=#

# Function required by OrdinaryDiffEq.jl
linsolve!(::Type{Val{:init}}, f, u0; kwargs...) = _linsolve!
_linsolve!(x, A, b, update_matrix = false; kwargs...) =
    LinearAlgebra.ldiv!(x, A, b)

# Function required by Krylov.jl (x and b can be AbstractVectors)
# See https://github.com/JuliaSmoothOptimizers/Krylov.jl/issues/605 for a
# related issue that requires the same workaround.
function LinearAlgebra.ldiv!(x, A::SchurComplementW, b)
    A.temp1 .= b
    LinearAlgebra.ldiv!(A.temp2, A, A.temp1)
    x .= A.temp2
end

function LinearAlgebra.ldiv!(
    x::Fields.FieldVector,
    A::SchurComplementW,
    b::Fields.FieldVector,
)
    (; dtγ_ref, S, transform) = A
    (; ∂ᶜρₜ∂ᶠ𝕄, ∂ᶜ𝔼ₜ∂ᶠ𝕄, ∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶠ𝕄ₜ∂ᶠ𝕄, ∂ᶜ𝕋ₜ∂ᶠ𝕄_field) = A
    (; ∂ᶜTCₜ∂ᶜTC, ∂ᶠTCₜ∂ᶠTC) = A
    dtγ = dtγ_ref[]
    cond = Operators.bandwidths(eltype(∂ᶜ𝔼ₜ∂ᶠ𝕄)) != (-half, half)
    if cond
        str = "The linear solver cannot yet be run with the given ∂ᶜ𝔼ₜ/∂ᶠ𝕄 \
            block, since it has more than 2 diagonals. So, ∂ᶜ𝔼ₜ/∂ᶠ𝕄 will \
            be set to 0 for the Schur complement computation. Consider \
            changing the ∂ᶜ𝔼ₜ∂ᶠ𝕄_mode or the energy variable."
        @warn str maxlog = 1
    end
    NVTX.@range "linsolve" color = colorant"lime" begin
        # Initialize x to -b, which correctly sets all the components of x that
        # correspond to variables without implicit tendencies.
        @. x = -b
        # TODO: Figure out why moving this into _ldiv_serial! results in a lot
        # of allocations for EDMFX.

        # Compute Schur complement
        Fields.bycolumn(axes(x.c)) do colidx
            _ldiv_serial!(
                A,
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
                isnothing(∂ᶜTCₜ∂ᶜTC) ? nothing : ∂ᶜTCₜ∂ᶜTC[colidx],
                isnothing(∂ᶠTCₜ∂ᶠTC) ? nothing : ∂ᶠTCₜ∂ᶠTC[colidx],
                S[colidx],
            )
        end
    end
end

function _ldiv_serial!(
    A::SchurComplementW,
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
    ∂ᶜTCₜ∂ᶜTC,
    ∂ᶠTCₜ∂ᶠTC,
    S,
)
    dtγ² = dtγ^2
    # TODO: Extend LinearAlgebra.I to work with stencil fields. Allow more
    # than 2 diagonals per Jacobian block.
    FT = eltype(eltype(S))
    tridiag_type = Operators.StencilCoefs{-1, 1, NTuple{3, FT}}
    I = tuple(Operators.StencilCoefs{-1, 1}((zero(FT), one(FT), zero(FT))))
    compose = Operators.ComposeStencils()
    apply = Operators.ApplyStencil()
    if cond
        @. S = dtγ² * compose(∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄) + dtγ * ∂ᶠ𝕄ₜ∂ᶠ𝕄 - I
    else
        @. S =
            dtγ² * compose(∂ᶠ𝕄ₜ∂ᶜρ, ∂ᶜρₜ∂ᶠ𝕄) +
            dtγ² * compose(∂ᶠ𝕄ₜ∂ᶜ𝔼, ∂ᶜ𝔼ₜ∂ᶠ𝕄) +
            dtγ * ∂ᶠ𝕄ₜ∂ᶠ𝕄 - I
    end

    xᶜρ = xc.ρ
    bᶜρ = bc.ρ
    ᶜ𝔼_name = filter(is_energy_var, propertynames(xc))[1]
    xᶜ𝔼 = getproperty(xc, ᶜ𝔼_name)::typeof(xc.ρ)
    bᶜ𝔼 = getproperty(bc, ᶜ𝔼_name)::typeof(xc.ρ)
    ᶠ𝕄_name = filter(is_momentum_var, propertynames(xf))[1]
    xᶠ𝕄 = getproperty(xf, ᶠ𝕄_name).components.data.:1
    bᶠ𝕄 = getproperty(bf, ᶠ𝕄_name).components.data.:1

    # Compute xᶠ𝕄.
    @. xᶠ𝕄 = bᶠ𝕄 + dtγ * (apply(∂ᶠ𝕄ₜ∂ᶜρ, bᶜρ) + apply(∂ᶠ𝕄ₜ∂ᶜ𝔼, bᶜ𝔼))
    Operators.column_thomas_solve!(S, xᶠ𝕄)

    # Compute the remaining components of x that correspond to variables with
    # implicit tendencies.
    @. xᶜρ = -bᶜρ + dtγ * apply(∂ᶜρₜ∂ᶠ𝕄, xᶠ𝕄)
    @. xᶜ𝔼 = -bᶜ𝔼 + dtγ * apply(∂ᶜ𝔼ₜ∂ᶠ𝕄, xᶠ𝕄)
    map(filter(is_tracer_var, propertynames(xc))) do ᶜ𝕋_name
        Base.@_inline_meta
        xᶜ𝕋 = getproperty(xc, ᶜ𝕋_name)::typeof(xc.ρ)
        bᶜ𝕋 = getproperty(bc, ᶜ𝕋_name)::typeof(xc.ρ)
        ∂ᶜ𝕋ₜ∂ᶠ𝕄 = getproperty(∂ᶜ𝕋ₜ∂ᶠ𝕄_field, ᶜ𝕋_name)::∂ᶜ𝕋ₜ∂ᶠ𝕄_field_eltype(A)
        @. xᶜ𝕋 = -bᶜ𝕋 + dtγ * apply(∂ᶜ𝕋ₜ∂ᶠ𝕄, xᶠ𝕄)
    end
    if :turbconv in propertynames(xc)
        xᶜTC = xc.turbconv
        xᶠTC = xf.turbconv
        bᶜTC = bc.turbconv
        bᶠTC = bf.turbconv
        for var_prop_chain in Fields.property_chains(xᶜTC)
            xᶜvar = Fields.single_field(xᶜTC, var_prop_chain, identity)
            bᶜvar = Fields.single_field(bᶜTC, var_prop_chain, identity)
            xᶜvar .= bᶜvar
            teye = tuple(tridiag_type((0, 1, 0)))
            ∂ᶜvarₜ∂ᶜvar =
                Fields.single_field(∂ᶜTCₜ∂ᶜTC, var_prop_chain, identity)
            @. ∂ᶜvarₜ∂ᶜvar = ∂ᶜvarₜ∂ᶜvar * dtγ - teye
            Operators.column_thomas_solve!(∂ᶜvarₜ∂ᶜvar, xᶜvar)
        end
        for var_prop_chain in Fields.property_chains(xᶠTC)
            xᶠvar = Fields.single_field(xᶠTC, var_prop_chain, identity)
            bᶠvar = Fields.single_field(bᶠTC, var_prop_chain, identity)
            xᶠvar .= bᶠvar
            teye = tuple(tridiag_type((0, 1, 0)))
            ∂ᶠvarₜ∂ᶠvar =
                Fields.single_field(∂ᶠTCₜ∂ᶠTC, var_prop_chain, identity)
            @. ∂ᶠvarₜ∂ᶠvar = ∂ᶠvarₜ∂ᶠvar * dtγ - teye
            Operators.column_thomas_solve!(∂ᶠvarₜ∂ᶠvar, xᶠvar)
        end
    end

    # Apply transform (if needed)
    if transform
        xc .*= dtγ
        xf .*= dtγ
    end
    return nothing
end
