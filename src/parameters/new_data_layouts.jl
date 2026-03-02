using UnrolledUtilities

n_elems_for_struct(array, packed_size) =
    packed_size % sizeof(eltype(array)) == 0 ?
    packed_size ÷ sizeof(eltype(array)) :
    throw(ArgumentError("Cannot utilize $packed_size bytes of array with \
                         element type $(eltype(array))"))

"""
    get_struct(array, T)

Retrieve a value of type `T` stored in `array`. The elements of `array` must be
small enough to evenly divide the size of a value of type `T`, and the number of
elements must be large enough to fully contain a value of type `T`.
"""
function get_struct(array, ::Type{T}) where {T}
    n = n_elems_for_struct(array, Base.packedsize(T))
    @boundscheck checkbounds(array, Base.OneTo(n))
    elems = unrolled_map(Base.Fix1(Base.unsafe_getindex, array), StaticOneTo(n))
    return reinterpret(T, elems)
end

"""
    set_struct!(array, struct_val)

Store a value `struct_val` in `array`. The elements of `array` must be small
enough to evenly divide the size of `struct_val`, and the number of elements
must be large enough to fully contain `struct_val`.
"""
function set_struct!(array, struct_val::T) where {T}
    n = n_elems_for_struct(array, Base.packedsize(T))
    @boundscheck checkbounds(array, Base.OneTo(n))
    elems = reinterpret(NTuple{n, eltype(array)}, struct_val)
    unrolled_foreach(enumerate(elems)) do (index, elem)
        Base.unsafe_setindex!(array, elem, index)
    end
end

"""
    struct_field_view(array, T, Val(name))

View of a field of type `T`, which is stored in `array` through `set_struct!`.
The field's `name` can be specified either as a symbol or as an integer index.

Like `Base.reinterpret`, this assumes that the data in the array is "packed", in
order to avoid padding of nonuniform composite types (for example, see
https://discourse.julialang.org/t/reinterpret-vector-into-single-struct/107709).
"""
Base.@propagate_inbounds function struct_field_view(
    array,
    ::Type{T},
    ::Val{name},
) where {T, name}
    i = name isa Integer ? name : unrolled_findfirst(==(name), fieldnames(T))
    i isa Integer || throw(ArgumentError("Type $T has no field $name"))
    n = n_elems_for_struct(array, Base.packedsize(fieldtype(T, i)))
    packed_size_up_to_field_i =
        unrolled_sum(Base.packedsize ∘ Base.Fix1(fieldtype, T), StaticOneTo(i))
    last_index_of_view = n_elems_for_struct(array, packed_size_up_to_field_i)
    return view(array, (last_index_of_view - n + 1):last_index_of_view)
end
