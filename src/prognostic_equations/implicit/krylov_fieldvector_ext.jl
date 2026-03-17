import Krylov
import ClimaCore: Fields
import LinearAlgebra

"""
    Krylov extensions for `ClimaCore.Fields.FieldVector`

These definitions allow Krylov.jl to treat `FieldVector` as its vector type,
delegating all operations to the underlying scalar fields' storage. This works
for both CPU (`Array`) and GPU (`CuArray`) backends without specializing on a
particular array type.
"""

# Representative underlying storage for a FieldVector
_underlying_vec(b::Fields.FieldVector) = begin
    names = Fields.scalar_field_names(b)
    first_name = first(names)
    parent(Fields.get_field(b, first_name))
end

# Tell Krylov what the "k-type" of a FieldVector is (its underlying array type)
Krylov.ktypeof(b::Fields.FieldVector) = typeof(_underlying_vec(b))

# Allocate a new FieldVector with the same structure and copied data
function Krylov.kcopy(b::Fields.FieldVector)
    c = similar(b)
    c .= b
    return c
end

# Zero out a FieldVector in-place
function Krylov.kzero!(b::Fields.FieldVector)
    b .= zero(eltype(_underlying_vec(b)))
    return b
end

# y ← a * x + y
function Krylov.kaxpy!(a, x::Fields.FieldVector, y::Fields.FieldVector)
    @. y = a * x + y
    return y
end

# 2-norm of a FieldVector, aggregating over all scalar fields
function Krylov.knrm2(x::Fields.FieldVector)
    acc = zero(eltype(_underlying_vec(x)))
    for name in Fields.scalar_field_names(x)
        v = parent(Fields.get_field(x, name))
        acc += LinearAlgebra.dot(v, v)
    end
    return sqrt(acc)
end

# Dot product between two FieldVectors
function Krylov.kdot(x::Fields.FieldVector, y::Fields.FieldVector)
    acc = zero(eltype(_underlying_vec(x)))
    for name in Fields.scalar_field_names(x)
        vx = parent(Fields.get_field(x, name))
        vy = parent(Fields.get_field(y, name))
        acc += LinearAlgebra.dot(vx, vy)
    end
    return acc
end

