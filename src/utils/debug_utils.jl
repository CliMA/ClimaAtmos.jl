#####
##### State debugging tools
#####

import ClimaCore.Fields as Fields

function debug_state_generic!(
    state,
    var::Union{Fields.FieldVector, NamedTuple};
    name = "",
)
    for pn in propertynames(var)
        pfx = isempty(name) ? "" : "$name."
        debug_state_generic!(state, getproperty(var, pn); name = "$pfx$pn")
    end
end

function debug_state_generic!(
    state,
    var::Union{Fields.FieldVector, NamedTuple},
    colidx,
    name = "",
)
    for pn in propertynames(var)
        pfx = isempty(name) ? "" : "$name."
        debug_state_generic!(
            state,
            getproperty(var, pn),
            colidx;
            name = "$pfx$pn",
        )
    end
end

debug_state_generic!(state, var::Number; name = "") = nothing
debug_state_generic!(state, var::AbstractString; name = "") = nothing
debug_state_generic!(state, var::Bool; name = "") = nothing
debug_state_generic!(state, var::Nothing; name = "") = nothing
debug_state_generic!(state, var::Any; name = "") = nothing # TODO: should we try to catch more types?

debug_state_generic!(state, var::Number, colidx; name = "") = nothing
debug_state_generic!(state, var::AbstractString, colidx; name = "") = nothing
debug_state_generic!(state, var::Bool, colidx; name = "") = nothing
debug_state_generic!(state, var::Nothing, colidx; name = "") = nothing
debug_state_generic!(state, var::Any, colidx; name = "") = nothing # TODO: should we try to catch more types?

debug_state_generic!(state, var::Fields.Field, colidx; name = "") =
    debug_state_column_field!(state, var[colidx]; name)
debug_state_generic!(state, var::Fields.Field; name = "") =
    debug_state_field!(state, var; name)

debug_state_field!(state, ::Nothing; name = "") = nothing
debug_state_field!(state, ::Nothing, colidx; name = "") = nothing

debug_state_field!(state, prog::Fields.Field, colidx; name = "") =
    debug_state_column_field!(state, prog[colidx]; name)
debug_state_field!(state, prog::Fields.Field; name = "") =
    debug_state_full_field!(state, prog; name)

function debug_state_full_field!(state, prog::Fields.Field; name = "")
    isbad(x) = isnan(x) || isinf(x)
    (; msg) = state
    for prop_chain in Fields.property_chains(prog)
        var = Fields.single_field(prog, prop_chain)
        nan = any(isnan.(parent(var)))
        inf = any(isinf.(parent(var)))
        any(isbad.(parent(var))) || continue
        pfx = isempty(name) ? "" : "$name."
        push!(
            msg,
            "-------------------- Bad data (nan=$nan, inf=$inf) in $(state.name).$pfx$prop_chain",
        )
        push!(msg, sprint(show, var))
    end
end

function debug_state_column_field!(state, prog::Fields.Field; name = "") # can we change this to prof::Fields.ColumnField ?
    isbad(x) = isnan(x) || isinf(x)
    (; msg) = state
    for prop_chain in Fields.property_chains(prog)
        var = Fields.single_field(prog, prop_chain)
        nan = any(isnan.(parent(var)))
        inf = any(isinf.(parent(var)))
        pfx = isempty(name) ? "" : "$name."
        any(isbad.(parent(var))) || continue
        push!(
            msg,
            "-------------------- Bad data (nan=$nan, inf=$inf) in $(state.name).$pfx$prop_chain",
        )
        push!(msg, sprint(show, var))
    end
end

"""
    debug_state(t
        [, colidx];                                            # colidx is optional
        Yₜ::Union{Fields.Field, Fields.FieldVector} = nothing, # Yₜ is optional
        Y::Union{Fields.Field, Fields.FieldVector} = nothing,  # Y is optional
        p::Any = nothing,                                      # p is optional
    )

Helper function for debugging `NaN`s and `Inf`s.

To avoid jumbled printed messages, it's recommended to use this
feature with threading disabled.

## Example
```julia
function precomputed_quantities!(Y, p, t, colidx)
    ᶜuₕ = Y.c.uₕ
    ᶠu₃ = Y.f.u₃

    @. ᶜu_bar[colidx] = C123(ᶜuₕ[colidx]) + C123(ᶜinterp(ᶠu₃[colidx]))
    @. ᶜK[colidx] = norm_sqr(ᶜu_bar[colidx]) / 2
    thermo_params = CAP.thermodynamics_params(params)

    CA.debug_state(t, colidx; Y, p) # Debug Y and p state here!

    CA.thermo_state!(Y, p, ᶜinterp, colidx)
    @. ᶜp[colidx] = TD.air_pressure(thermo_params, ᶜts[colidx])
    return nothing
end
```
"""
debug_state(t; kwargs...) = debug_state(t, (); kwargs...)
debug_state(t, colidx::Fields.ColumnIndex; kwargs...) =
    debug_state(t, (colidx,); kwargs...)

function debug_state(t, colidx; Yₜ = nothing, Y = nothing, p = nothing)
    states = Dict()
    states["Yₜ"] = (; msg = String[], name = "Yₜ")
    states["Y"] = (; msg = String[], name = "Y")
    states["p"] = (; msg = String[], name = "p")
    debug_state_generic!(states["Yₜ"], Yₜ, colidx...)
    debug_state_generic!(states["Y"], Y, colidx...)
    debug_state_generic!(states["p"], p, colidx...)
    if !all(map(x -> isempty(states[x].msg), collect(keys(states))))
        for key in keys(states)
            for msg in states[key].msg
                println(msg)
            end
        end
        if colidx == ()
            error("Bad state at time $t")
        else
            error("Bad state at time $t in column $colidx")
        end
    end
    return nothing
end

#####
##### Recursive function for filling auxiliary state with NaNs
#####

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

fill_with_nans_field!(::Nothing) = nothing
fill_with_nans_field!(::Nothing, colidx) = nothing
function fill_with_nans_field!(prog::Fields.Field)
    parent(prog) .= NaN
end

"""
    fill_with_nans!(p)

Fill a data structure's `Field`s / `FieldVector`s with NaNs.
"""
fill_with_nans!(p) =
    fill_with_nans!(p, p.atmos.numerics.test_dycore_consistency)
fill_with_nans!(p, ::Nothing) = nothing
fill_with_nans!(p, ::TestDycoreConsistency) = fill_with_nans_generic!(p)


import ClimaCore.Fields as Fields

function find_column(f::Fields.Field, cond)::Fields.ColumnIndex
    _colidx = Fields.ColumnIndex((-1,), -1))
    Fields.bycolumn(axes(f)) do colidx
        if cond(f[colidx])
            _colidx = colidx
        end
    end
    found = _colidx.h ≠ -1
    return (_colidx, found)
end
find_level(f::Fields.Field, cond, ::Spaces.CenterExtrudedFiniteDifferenceSpace) =
    find_level(f, cond, 1:Fields.nlevels(axes(field)))
find_level(f::Fields.Field, cond, ::Spaces.FaceExtrudedFiniteDifferenceSpace) =
    find_level(f, cond, PlusHalf(0):Fields.nlevels(axes(field)))
function find_level(f::Fields.Field, cond, R)
    L = -1
    for lev in R
        flev = Spaces.level(f, lev)
        if cond(flev)
            L = lev
        end
    end
    found = L ≠ -1
    return (L, found)
end

function debug_plot(Y, p, t, Yₜ = nothing)
    @info "Debug plotting at time $t"
    for prop_chain in Fields.property_chains(Y)
        var = Fields.single_field(Y, prop_chain)
        (col, found) = find_column(var, x->any(parent(x) .== maximum(var)))
        if found
            @info "Found maximum for variable $prop_chain"
            Plots.plot(var, level=7, size=(600,450), clim=(-0.01, 0.01))
        end
    end
end

struct DebugCrash{T}
    tends::T
    dict::Dict{String,Fields.Field}
end
DebugCrash(Y) = DebugCrash(Y, Dict())
Base.getindex(x::DebugCrash, i) = Base.getindex(x.dict, i)

dbt = DebugCrash(similar(Y))

dbt["u₃","-ᶠω¹²×ᶜu+ᶜK"] = @. - ᶠω¹² × ᶠinterp(CT12(ᶜu[colidx])) + ᶠgradᵥ(ᶜK[colidx])
