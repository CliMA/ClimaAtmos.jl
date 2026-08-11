import ClimaCore.Fields as Fields

#####
##### Recursive function for filling auxiliary state with NaNs
#####

"""
    fill_with_nans_generic!(var)
    fill_with_nans_generic!(state, var, colidx)

Recurse through `var` and fill every `Field` it contains with `NaN`s, mutating
them in place.

`FieldVector`s and `NamedTuple`s are descended into; everything that is not a
`Field` is left alone.
"""
function fill_with_nans_generic!(var::Union{Fields.FieldVector, NamedTuple})
    for pn in propertynames(var)
        fill_with_nans_generic!(getproperty(var, pn))
    end
end

function fill_with_nans_generic!(
    state,
    var::Union{Fields.FieldVector, NamedTuple},
    colidx,
)
    for pn in propertynames(var)
        fill_with_nans_generic!(getproperty(var, pn), colidx)
    end
end

fill_with_nans_generic!(var::Number) = nothing
fill_with_nans_generic!(var::AbstractString) = nothing
fill_with_nans_generic!(var::Bool) = nothing
fill_with_nans_generic!(var::Nothing) = nothing
fill_with_nans_generic!(var::Any) = nothing # TODO: should we try to catch more types?

fill_with_nans_generic!(var::Number, colidx) = nothing
fill_with_nans_generic!(var::AbstractString, colidx) = nothing
fill_with_nans_generic!(var::Bool, colidx) = nothing
fill_with_nans_generic!(var::Nothing, colidx) = nothing
fill_with_nans_generic!(var::Any, colidx) = nothing # TODO: should we try to catch more types?

fill_with_nans_generic!(var::Fields.Field) = fill_with_nans_field!(var)

"""
    fill_with_nans_field!(field)

Fill `field` with `NaN`s in place; do nothing for `nothing`.
"""
fill_with_nans_field!(::Nothing) = nothing
fill_with_nans_field!(::Nothing, colidx) = nothing
function fill_with_nans_field!(prog::Fields.Field)
    parent(prog) .= NaN
end

"""
    fill_with_nans!(p)

Fill every `Field` in the cache `p` with `NaN`s, but only when the
`test_dycore_consistency` numerics option is set.

Poisoning the cache before a tendency evaluation exposes any quantity that the
tendency reads without first recomputing it. Returns `nothing`.
"""
fill_with_nans!(p) =
    fill_with_nans!(p, p.atmos.numerics.test_dycore_consistency)
fill_with_nans!(p, ::Nothing) = nothing
fill_with_nans!(p, ::TestDycoreConsistency) = fill_with_nans_generic!(p)
