#####
##### Utility functions
#####

is_energy_var(symbol) = symbol in (:ρθ, :ρe_tot, :ρe_int)
is_momentum_var(symbol) = symbol in (:uₕ, :ρuₕ, :w, :ρw)
is_turbconv_var(symbol) = symbol == :turbconv
is_tracer_var(symbol) = !(
    symbol == :ρ ||
    is_energy_var(symbol) ||
    is_momentum_var(symbol) ||
    is_turbconv_var(symbol)
)

# we may be hitting a slow path:
# https://stackoverflow.com/questions/14687665/very-slow-stdpow-for-bases-very-close-to-1
fast_pow(x::FT, y::FT) where {FT <: AbstractFloat} = exp(y * log(x))

"""
    time_from_filename(file)

Returns the time (Float64) from a given filename

## Example
```
time_from_filename("day4.46906.hdf5")
392506.0
```
"""
function time_from_filename(file)
    arr = split(basename(file), ".")
    day = parse(Float64, replace(arr[1], "day" => ""))
    sec = parse(Float64, arr[2])
    return day * (60 * 60 * 24) + sec
end

"""
    sort_files_by_time(files)

Sorts an array of files by the time,
determined by the filename.
"""
sort_files_by_time(files) =
    permute!(files, sortperm(time_from_filename.(files)))

#####
##### State debugging tools
#####

import ClimaCore.Fields as Fields

function debug_state_generic!(state, var::Union{Fields.FieldVector, NamedTuple})
    for pn in propertynames(var)
        debug_state_generic!(state, getproperty(var, pn))
    end
end

function debug_state_generic!(
    state,
    var::Union{Fields.FieldVector, NamedTuple},
    colidx,
)
    for pn in propertynames(var)
        debug_state_generic!(state, getproperty(var, pn), colidx)
    end
end

debug_state_generic!(state, var::Number) = nothing
debug_state_generic!(state, var::AbstractString) = nothing
debug_state_generic!(state, var::Bool) = nothing
debug_state_generic!(state, var::Nothing) = nothing
debug_state_generic!(state, var::Any) = nothing # TODO: should we try to catch more types?

debug_state_generic!(state, var::Number, colidx) = nothing
debug_state_generic!(state, var::AbstractString, colidx) = nothing
debug_state_generic!(state, var::Bool, colidx) = nothing
debug_state_generic!(state, var::Nothing, colidx) = nothing
debug_state_generic!(state, var::Any, colidx) = nothing # TODO: should we try to catch more types?

debug_state_generic!(state, var::Fields.Field, colidx) =
    debug_state_column_field!(state, var[colidx])
debug_state_generic!(state, var::Fields.Field) = debug_state_field!(state, var)

debug_state_field!(state, ::Nothing) = nothing
debug_state_field!(state, ::Nothing, colidx) = nothing

debug_state_field!(state, prog::Fields.Field, colidx) =
    debug_state_column_field!(state, prog[colidx])
debug_state_field!(state, prog::Fields.Field) =
    debug_state_full_field!(state, prog)

function debug_state_full_field!(state, prog::Fields.Field)
    isbad(x) = isnan(x) || isinf(x)
    (; msg, name) = state
    for prop_chain in Fields.property_chains(prog)
        var = Fields.single_field(prog, prop_chain)
        nan = any(isnan.(parent(var)))
        inf = any(isinf.(parent(var)))
        any(isbad.(parent(var))) || continue
        push!(
            msg,
            "-------------------- Bad data (nan=$nan, inf=$inf) in $name.$prop_chain",
        )
        push!(msg, sprint(show, var))
    end
end

function debug_state_column_field!(state, prog::Fields.Field) # can we change this to prof::Fields.ColumnField ?
    isbad(x) = isnan(x) || isinf(x)
    (; msg, name) = state
    for prop_chain in Fields.property_chains(prog)
        var = Fields.single_field(prog, prop_chain)
        nan = any(isnan.(parent(var)))
        inf = any(isinf.(parent(var)))
        any(isbad.(parent(var))) || continue
        push!(
            msg,
            "-------------------- Bad data (nan=$nan, inf=$inf) in $name.$prop_chain",
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
feature with `ClimaCore.enable_threading() = false`.

## Example
```julia
function precomputed_quantities!(Y, p, t, colidx)
    ᶜuₕ = Y.c.uₕ
    ᶠw = Y.f.w
    (; ᶜuvw, ᶜK, ᶜts, ᶜp, params, thermo_dispatcher) = p

    @. ᶜuvw[colidx] = C123(ᶜuₕ[colidx]) + C123(ᶜinterp(ᶠw[colidx]))
    @. ᶜK[colidx] = norm_sqr(ᶜuvw[colidx]) / 2
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
        error("Bad state at time $t")
    end
    return nothing
end

#####
##### Recursive function for filling auxiliary state with NaNs
#####

import ClimaCore.Fields as Fields

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
fill_with_nans!(p) = fill_with_nans_generic!(p)
