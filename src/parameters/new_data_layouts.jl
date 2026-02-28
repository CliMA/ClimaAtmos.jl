using UnrolledUtilities

can_use_basetype(::Type{B}, n_bytes) where {B} =
    iszero(n_bytes % Base.packedsize(B))

check_basetype(::Type{B}, n_bytes) where {B} =
    can_use_basetype(B, n_bytes) ||
    throw(ArgumentError("Cannot use $B values to represent $n_bytes bytes, as \
                         each value contains $(Base.packedsize(B)) bytes"))

function check_array_for_struct(array, ::Type{T}) where {T}
    check_basetype(eltype(array), Base.packedsize(T))
    array_val_str = Base.ispacked(eltype(array)) ? "bytes" : "non-padding bytes"
    struct_val_str = Base.ispacked(T) ? "bytes" : "non-padding bytes"
    packed_array_size = length(array) * Base.packedsize(eltype(array))
    packed_array_size == Base.packedsize(T) ||
        throw(DimensionMismatch("Cannot use array of $(packed_array_size) \
                                 $(array_val_str) to represent $T of \
                                 $(Base.packedsize(T)) $(struct_val_str)"))
end

"""
    get_struct(array, T)

Reads the data in `array` and interprets it as a value of type `T`, which can be
any `isbits` type whose size matches the amount of memory available in `array`:
```julia-repl
julia> get_struct([true, false, false, false], Int32)
1

julia> get_struct([eps(0.0), 0.0], NTuple{4, Int32})
(1, 0, 0, 0)

julia> get_struct([2, 1, 0], Tuple{Int64, Int128})
(2, 1)
```
(As with all primitive data types in Julia, the numbers in these examples are
interpreted in [little-endian order](https://en.wikipedia.org/wiki/Endianness).)

# Extended help

This behaves like `reinterpret(reshape, T, array)[1]`, but it uses a lower-level
method of `reinterpret` for `isbits` inputs. The two methods differ in how they
handle padding of nonuniform fields in data structures---the array method must
have padding in its inputs, while the lower-level method does not need padding.

The example of generating a `Tuple{Int64, Int128}` illustrates this difference.
Like `get_struct`, `reinterpret` can create such a value from a tuple of three
`Int64`s, but it needs a fourth `Int64` when given an array instead of a tuple:
```julia-repl
julia> reinterpret(Tuple{Int64, Int128}, (2, 1, 0))
(2, 1)

julia> reinterpret(reshape, Tuple{Int64, Int128}, [2, 1, 0])[1]
ERROR: ArgumentError: [...]

julia> reinterpret(reshape, Tuple{Int64, Int128}, [2, 0, 1, 0])[1]
(2, 1)
```

The C code underlying Julia adds padding to data structures so that fields of
different sizes are always aligned with each other in register memory, though it
never actually reads the data in bytes added for padding. Since `get_struct` is
implemented using a form of `reinterpret` that does not require padding, its
`array` input might be smaller than `sizeof(T)` (the smallest possible size of a
nonempty `Array{T}` or `ReinterpretArray{T}`) if `T` contains nonuniform fields.

For more information about `reinterpret` and padding, see the following:
- https://discourse.julialang.org/t/reinterpret-returns-wrong-values
- https://discourse.julialang.org/t/reinterpret-vector-into-single-struct
- https://discourse.julialang.org/t/reinterpret-vector-of-mixed-type-tuples
"""
function get_struct(array, ::Type{T}) where {T}
    @boundscheck check_array_for_struct(array, T)
    eltype(array) <: T && return array[1]
    n_indices = Base.packedsize(T) ÷ Base.packedsize(eltype(array))
    return reinterpret(T, ntuple(i -> (@inbounds array[i]), Val(n_indices)))
end

"""
    set_struct!(array, struct_val)

Fills `array` with data that represents `struct_val`, which [`get_struct`](@ref)
can read and use to recreate `struct_val` based on its type.

Like `get_struct`, this ignores Julia's internal padding of nonuniform fields.
"""
function set_struct!(array, struct_val::T) where {T}
    @boundscheck check_array_for_struct(array, T)
    eltype(array) <: T && return (array[1] = struct_val)
    n_indices = Base.packedsize(T) ÷ Base.packedsize(eltype(array))
    array_vals = reinterpret(NTuple{n_indices, eltype(array)}, struct_val)
    unrolled_foreach(enumerate(array_vals)) do (i, array_val)
        @inbounds array[i] = array_val
    end
    return struct_val
end

"""
    struct_field_view(array, T, Val(name))

Given an `array` that [`set_struct!`](@ref) has filled with data representing a
value of type `T`, creates a view of the data for a particular field of `T`. The
field is specified by a `Val` wrapping its symbolic name (or its integer index).

Like `set_struct!` and its counterpart [`get_struct`](@ref), this ignores
Julia's internal padding of nonuniform fields.
"""
function struct_field_view(array, ::Type{T}, ::Val{name}) where {T, name}
    @boundscheck check_array_for_struct(array, T)
    f = name isa Integer ? name : unrolled_findfirst(==(name), fieldnames(T))
    f isa Integer || throw(ArgumentError("Type $T has no field $name"))
    sizes = unrolled_map(Base.packedsize, unrolled_take(fieldtypes(T), Val(f)))
    check_basetype(eltype(array), last(sizes))
    check_basetype(eltype(array), unrolled_sum(sizes))
    n_indices = last(sizes) ÷ Base.packedsize(eltype(array))
    last_index = unrolled_sum(sizes) ÷ Base.packedsize(eltype(array))
    return @inbounds view(array, (last_index - n_indices + 1):last_index)
end

"""
    default_basetype(T)

Determines an array element type that [`set_struct!`](@ref) can use to represent
values of type `T`, or the `fieldtypes` of `T`, or the `fieldtypes` of those
types, and so on through all the types nested inside `T`.

If no suitable type can be found by recursively searching through `fieldtypes`,
the default result is `UInt8`, since a sequence of bytes can represent anything.
"""
function default_basetype(::Type{T}) where {T}
    (isprimitivetype(T) || Base.issingletontype(T)) && return T
    non_singleton_types = unrolled_filter(!Base.issingletontype, fieldtypes(T))
    possible_basetypes = unrolled_map(default_basetype, non_singleton_types)
    B = unrolled_argmin(Base.packedsize, possible_basetypes)
    supports_B_as_basetype = Base.Fix1(can_use_basetype, B) ∘ Base.packedsize
    return unrolled_all(supports_B_as_basetype, possible_basetypes) ? B : UInt8
end

"""
    replace_basetype(B, B′, T)

Recursively modifies the parameters of `T`, replacing each type `B` with `B′`.
This is similar to constructing a new `T` with all subfields of type `B`
converted to type `B′`, though no constructors are actually called or compiled.
"""
replace_basetype(B, B′, T) = replace_basetype(Val(Tuple{B, B′}), T)
replace_basetype(::Val{Tuple{B, B′}}, not_a_type) where {B, B′} = not_a_type
function replace_basetype(::Val{Tuple{B, B′}}, ::Type{T}) where {B, B′, T}
    T <: B && return B′
    isempty(T.parameters) && return T
    swap_B_with_B′ = Base.Fix1(replace_basetype, Val(Tuple{B, B′}))
    return T.name.wrapper{unrolled_map(swap_B_with_B′, Tuple(T.parameters))...}
end
