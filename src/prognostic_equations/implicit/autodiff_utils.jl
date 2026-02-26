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

# Adds up the memory required to allocate all DataLayouts within an object.
parent_memory(x::DataLayouts.AbstractData) = sizeof(parent(x))
parent_memory(x) =
    contains_data_layout(x) ? unrolled_sum(parent_memory, components(x)) : 0

# Extracts a view of the first column from every DataLayout within an object,
# except for DataLayouts in DSSBuffers, which are not needed in a single column.
first_column_view(x::DataLayouts.AbstractData) = Fields.column(x, 1, 1, 1)
first_column_view(x::Fields.Field) = Fields.column(x, 1, 1, 1)
first_column_view(x::Fields.FieldVector) = Fields.column(x, 1, 1, 1)
first_column_view(x::Topologies.DSSBuffer) = nothing
first_column_view(x) =
    contains_data_layout(x) ? map_components(first_column_view, x) : x

# Makes T the parent array element type of every DataLayout within an object.
replace_parent_eltype(x::DataLayouts.AbstractData, ::Type{T}) where {T} =
    DataLayouts.replace_basetype(x, T)
replace_parent_eltype(x::Fields.Field, ::Type{T}) where {T} =
    Fields.Field(replace_parent_eltype(Fields.field_values(x), T), axes(x))
replace_parent_eltype(x::Fields.FieldVector, ::Type{T}) where {T} =
    similar(x, T)
replace_parent_eltype(x, ::Type{T}) where {T} =
    contains_data_layout(x) ?
    map_components(Base.Fix2(replace_parent_eltype, T), x) : x

# Appends values to the precomputed and scratch components of an AtmosCache.
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

# Full-tendency Jacobian: also replace hyperdiff (written in prep_hyperdiffusion_tendency!)
# and core (read in implicit_tendency!; mixing Float32 core with Dual causes method errors).
append_to_atmos_cache(atmos_cache, precomputed, scratch, hyperdiff, core) = AtmosCache(
    unrolled_map(fieldnames(typeof(atmos_cache))) do cache_component_name
        if cache_component_name == :precomputed
            (; atmos_cache.precomputed..., precomputed...)
        elseif cache_component_name == :scratch
            (; atmos_cache.scratch..., scratch...)
        elseif cache_component_name == :hyperdiff
            hyperdiff
        elseif cache_component_name == :core
            core
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

# An iterator over the column indices of all fields in a FieldVector.
function column_index_iterator(field_vector)
    horz_space = horizontal_space(field_vector)
    isnothing(horz_space) && return ((1, 1, 1),)
    qs = 1:Quadratures.degrees_of_freedom(Spaces.quadrature_style(horz_space))
    hs = Spaces.eachslabindex(horz_space)
    return horz_space isa Spaces.SpectralElementSpace1D ?
           Iterators.product(qs, hs) : Iterators.product(qs, qs, hs)
end

# An iterator over the FieldNames of all scalar fields in a FieldVector.
scalar_field_names(field_vector) =
    MatrixFields.filtered_names(field_vector) do x
        x isa Fields.Field && eltype(x) == eltype(field_vector)
    end

# An iterator with tuples of the form (scalar_index, level_index)), where
# scalar_index is an index into scalar_field_names(field_vector) and level_index
# is a vertical index into the scalar field with this name.
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

# A view of one point in a Field or DataLayout, which can be used like a Ref.
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
