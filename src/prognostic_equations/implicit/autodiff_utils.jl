# Replaces all numbers in Fields and DataLayouts with values of type T.
replace_parent_type(x::Fields.FieldVector, ::Type{T}) where {T} = similar(x, T)
replace_parent_type(x::Fields.Field, ::Type{T}) where {T} =
    Fields.Field(replace_parent_type(Fields.field_values(x), T), axes(x))
replace_parent_type(x::DataLayouts.AbstractData, ::Type{T}) where {T} =
    DataLayouts.replace_basetype(x, T)
replace_parent_type(x::Union{Tuple, NamedTuple}, ::Type{T}) where {T} =
    unrolled_map(Base.Fix2(replace_parent_type, T), x)

# An iterator over the column indices of all fields in a FieldVector.
function column_index_iterator(field_vector)
    all_values = Fields._values(field_vector)
    first_field = unrolled_argfirst(value -> value isa Fields.Field, all_values)
    horz_space = Spaces.horizontal_space(axes(first_field))
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

# An iterator with pairs of the form
# (scalar_level_index, (scalar_index, level_index)), where scalar_index is an
# index into scalar_field_names(field_vector), level_index is a vertical index
# into the scalar field with this name, and scalar_level_index is a unique
# linear index for each combination of scalar field name and vertical index.
function scalar_level_index_pairs(field_vector)
    scalar_names = scalar_field_names(field_vector)
    scalar_index_and_level_pairs =
        unrolled_map(enumerate(scalar_names)) do (scalar_index, name)
            field = MatrixFields.get_field(field_vector, name)
            is_one_level = field isa Fields.SpectralElementField
            Iterators.map(
                level_index -> (scalar_index, level_index),
                Base.OneTo(is_one_level ? 1 : Spaces.nlevels(axes(field))),
            )
        end
    return enumerate(Iterators.flatten(scalar_index_and_level_pairs))
end

# A view of the data for one point in a field, which can be used like a Ref.
Base.@propagate_inbounds function point(field, level_index, column_index...)
    column_field = Fields.column(field, column_index...)
    column_field isa Fields.PointField && return column_field
    level_index_offset = Operators.left_idx(axes(column_field)) - 1
    return Fields.level(column_field, level_index + level_index_offset)
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
# linear function of ᶜtke⁰ * sqrt(ᶜtke⁰), evaluates to NaN at every point where
# tke⁰ = 0. In general, this change is valid if all functions of sqrt(x) can be
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
