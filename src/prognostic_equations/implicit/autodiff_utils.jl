# Tuple that contains the components of a generic object.
components(x) = ntuple(i -> getfield(x, i), Val(fieldcount(typeof(x))))

# Analogue of map that applies a function to each component of a generic object.
# Aside from some special cases, this assumes that the object's type has a
# default constructor (one that does not need any type parameters).
map_components(f::F, x::Union{Tuple, NamedTuple}) where {F} = unrolled_map(f, x)
map_components(f::F, x) where {F} =
    typeof(x).name.wrapper(unrolled_map(f, components(x))...)

# Determines whether there is at least one DataLayout within an object.
contains_data_layout(::DataLayouts.AbstractData) = true
contains_data_layout(x) = unrolled_any(contains_data_layout, components(x))

"""
    parent_memory(x)

Return the total memory required to allocate all `DataLayout`s within `x`
[bytes], or `0` if it contains none. Used to size the automatic differentiation
partitions in `jacobian_cache`.
"""
parent_memory(x::DataLayouts.AbstractData) = sizeof(parent(x))
parent_memory(x) =
    contains_data_layout(x) ? unrolled_sum(parent_memory, components(x)) : 0

"""
    first_column_view(x)

Return a view of the first column of every `DataLayout` within `x`, leaving
non-field components unchanged.

`DataLayout`s in `DSSBuffer`s are replaced by `nothing`, since they are not
needed in a single column. Used to build single-column states and caches for
the Jacobian debugging utilities.
"""
first_column_view(x::DataLayouts.AbstractData) = Fields.column(x, 1, 1, 1)
first_column_view(x::Fields.Field) = Fields.column(x, 1, 1, 1)
first_column_view(x::Fields.FieldVector) = Fields.column(x, 1, 1, 1)
first_column_view(x::Topologies.DSSBuffer) = nothing
first_column_view(x) =
    contains_data_layout(x) ? map_components(first_column_view, x) : x

"""
    replace_parent_eltype(x, ::Type{T})

Return a copy of `x` in which `T` is the parent array element type of every
`DataLayout`, leaving non-field components unchanged. Used to create the
dual-number copies of `Y`, `p.precomputed`, and `p.scratch`.
"""
replace_parent_eltype(x::DataLayouts.AbstractData, ::Type{T}) where {T} =
    DataLayouts.replace_basetype(x, T)
replace_parent_eltype(x::Fields.Field, ::Type{T}) where {T} =
    Fields.Field(replace_parent_eltype(Fields.field_values(x), T), axes(x))
replace_parent_eltype(x::Fields.FieldVector, ::Type{T}) where {T} =
    similar(x, T)
replace_parent_eltype(x, ::Type{T}) where {T} =
    contains_data_layout(x) ?
    map_components(Base.Fix2(replace_parent_eltype, T), x) : x

"""
    append_to_atmos_cache(atmos_cache, precomputed, scratch)

Return a copy of `atmos_cache` whose `precomputed` and `scratch` components
have the given values appended to them. Used to give `implicit_tendency!` a
cache that also holds the dual-number fields.
"""
append_to_atmos_cache(atmos_cache, precomputed, scratch) = AtmosCache(
    unrolled_map(fieldnames(typeof(atmos_cache))) do cache_component_name
        if cache_component_name == :precomputed
            (; atmos_cache.precomputed..., precomputed...)
        elseif cache_component_name == :scratch
            (; atmos_cache.scratch..., scratch...)
        else
            getfield(atmos_cache, cache_component_name)
        end
    end...,
)

# The horizontal SpectralElementSpace of the fields in a FieldVector.
function horizontal_space(field_vector)
    all_values = Fields._values(field_vector)
    space = axes(unrolled_argfirst(value -> value isa Fields.Field, all_values))
    return space isa Spaces.FiniteDifferenceSpace ? nothing :
           Spaces.horizontal_space(space)
end

"""
    column_index_iterator(field_vector)

Return an iterator over the column indices `(i, j, h)` of all fields in
`field_vector`, or `((1, 1, 1),)` when there is no horizontal space. For a
`SpectralElementSpace1D` the indices are pairs `(i, h)`.
"""
function column_index_iterator(field_vector)
    horz_space = horizontal_space(field_vector)
    isnothing(horz_space) && return ((1, 1, 1),)
    qs = 1:Quadratures.degrees_of_freedom(Spaces.quadrature_style(horz_space))
    hs = Spaces.eachslabindex(horz_space)
    return horz_space isa Spaces.SpectralElementSpace1D ?
           Iterators.product(qs, hs) : Iterators.product(qs, qs, hs)
end

"""
    scalar_field_names(field_vector)

Return an iterator over the `FieldName`s of all scalar fields in
`field_vector`, i.e. of the fields whose element type is the floating-point
type of the `FieldVector`. Vector-valued fields are split into their scalar
components.
"""
scalar_field_names(field_vector) =
    MatrixFields.filtered_names(field_vector) do x
        x isa Fields.Field && eltype(x) == eltype(field_vector)
    end

"""
    field_vector_index_iterator(field_vector)

Return an iterator over tuples `(scalar_index, level_index)`, where
`scalar_index` indexes into `scalar_field_names(field_vector)` and
`level_index` is a vertical index into the scalar field with that name.

The iteration order defines the row and column order of the dense per-column
Jacobian matrices. Single-level fields (`PointField`s and
`SpectralElementField`s) contribute exactly one index.
"""
function field_vector_index_iterator(field_vector)
    scalar_names = scalar_field_names(field_vector)
    scalar_index_and_level_pairs =
        unrolled_map(enumerate(scalar_names)) do (scalar_index, name)
            field = MatrixFields.get_field(field_vector, name)
            is_one_level =
                field isa Union{Fields.PointField, Fields.SpectralElementField}
            Iterators.map(
                level_index -> (scalar_index, level_index),
                Base.OneTo(is_one_level ? 1 : Spaces.nlevels(axes(field))),
            )
        end
    return Iterators.flatten(scalar_index_and_level_pairs)
end

"""
    point(value, level_index, column_index...)

Return a view of one point in a `Field` or `DataLayout`, which can be indexed
and assigned to like a `Ref`. `level_index` counts from the first interior
level, and is shifted internally to account for the space's left index.
"""
Base.@propagate_inbounds function point(value, level_index, column_index...)
    column_value = Fields.column(value, column_index...)
    column_value isa Union{Fields.PointField, DataLayouts.DataF} &&
        return column_value
    level_index_offset =
        column_value isa Fields.Field ?
        Operators.left_idx(axes(column_value)) - 1 : 0
    return Fields.level(column_value, level_index + level_index_offset)
end

# TODO: This needs to be moved into ClimaCore to avoid breaking precompilation.
# function ClimaComms.context(x::Fields.FieldVector)
#     values = Fields._values(x)
#     isempty(values) && error("Empty FieldVector has no device or context")
#     index = unrolled_findfirst(Base.Fix2(!isa, Fields.PointField), values)
#     return ClimaComms.context(values[isnothing(index) ? 1 : index])
# end

##################################################
## Nonstandard rules for automatic differentiation
##################################################

# Every nonstandard rule should be defined using a method that specializes on
# the tag Jacobian. Not specializing on any tag overwrites the generic method
# for Dual in ForwardDiff and breaks precompilation, while specializing on the
# default tag Nothing causes the type piracy test to fail.

# Set the derivative of sqrt(x) to iszero(x) ? zero(x) : inv(2 * sqrt(x)) in
# order to properly handle derivatives of x * sqrt(x). Without this change, the
# derivative of the turbulent kinetic energy dissipation tendency, which is a
# linear function of ᶜtke * sqrt(ᶜtke), evaluates to NaN at every point where
# tke = 0. In general, this change is valid if all functions of sqrt(x) can be
# expanded around x = 0 with a leading term of the form c * x^p * sqrt(x), where
# c and p are constants and p ≥ 1/2. For example, this will be the case for any
# linear function of f(x) * sqrt(x), where f(x) has a non-constant Taylor
# expansion around x = 0. On the other hand, this change will not be valid for
# functions like sqrt(x) or sqrt(x) / x, whose derivatives at x = 0 should be
# ±Inf, but will end up being 0 or NaN due to this change. If a function whose
# leading term has p < 1/2 is called by set_implicit_precomputed_quantities! or
# implicit_tendency!, this strategy for avoiding NaNs in the dissipation
# tendency derivative will need to be modified.
@inline function Base.sqrt(d::ForwardDiff.Dual{Jacobian})
    tag = Val(Jacobian)
    x = ForwardDiff.value(d)
    partials = ForwardDiff.partials(d)
    val = sqrt(x)
    deriv = iszero(x) ? zero(x) : inv(2 * val)
    return ForwardDiff.dual_definition_retval(tag, val, deriv, partials)
end

# Ignore all derivative information when comparing Duals to other Numbers. This
# ensures that conditional statements are always equivalent for Duals and Reals.
for func in (:iszero,)
    @eval @inline Base.$func(arg::ForwardDiff.Dual{Jacobian}) =
        $func(ForwardDiff.value(arg))
end
for func in (:isequal, :isless, :<, :>, :(==), :!=, :<=, :>=)
    # These methods handle combinations of Duals without ambiguities.
    @eval @inline Base.$func(
        arg1::ForwardDiff.Dual{Jacobian},
        arg2::ForwardDiff.Dual{Jacobian},
    ) = $func(ForwardDiff.value(arg1), ForwardDiff.value(arg2))
    @eval @inline Base.$func(
        arg1::ForwardDiff.Dual{Tx},
        arg2::ForwardDiff.Dual{Jacobian},
    ) where {Tx} = $func(ForwardDiff.value(arg1), ForwardDiff.value(arg2))
    @eval @inline Base.$func(
        arg1::ForwardDiff.Dual{Jacobian},
        arg2::ForwardDiff.Dual{Ty},
    ) where {Ty} = $func(ForwardDiff.value(arg1), ForwardDiff.value(arg2))

    # These methods handle combinations of Reals and Duals without ambiguities.
    for R in (ForwardDiff.AMBIGUOUS_TYPES..., AbstractIrrational)
        R == Real && continue # defining these methods for Real causes ambiguities from StatsBase
        @eval @inline Base.$func(arg1::ForwardDiff.Dual{Jacobian}, arg2::$R) =
            $func(ForwardDiff.value(arg1), arg2)
        @eval @inline Base.$func(arg1::$R, arg2::ForwardDiff.Dual{Jacobian}) =
            $func(arg1, ForwardDiff.value(arg2))
    end
end
