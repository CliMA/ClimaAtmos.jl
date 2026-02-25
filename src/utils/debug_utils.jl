#####
##### State debugging tools
#####

import ClimaCore.Fields as Fields
import ForwardDiff

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

"""
    check_tendency_nans(Yₜ, context::String; Y = nothing)

If the tendency vector `Yₜ` contains any NaN, throw an error naming `context`.
Call this after each tendency in computation order so the first failure
identifies which tendency first wrote a NaN.

When `Y` is provided we are in the implicit tendency path. The check is skipped
only when `Y` has `ForwardDiff.Dual` eltype (Jacobian build). For residual
evaluation (Float state), we error so the first NaN in the tendency identifies
the exact routine and the Newton trial state `Y` that caused it (e.g. inspect
min(ρ), min(p) for that state; try smaller dt).
"""
function check_tendency_nans(Yₜ, context::String; Y = nothing)
    if Y !== nothing && hasproperty(Y, :c)
        first_name = first(propertynames(Y.c))
        T = eltype(parent(getproperty(Y.c, first_name)))
        if T <: ForwardDiff.Dual
            return nothing  # Jacobian build: skip so AD does not error
        end
    end
    data = parent(Yₜ)
    any(isnan, data) &&
        error("NaN in Yₜ after $context (first NaN in tendency computation order; " *
              "if implicit: Newton trial state Y was unphysical — inspect min(ρ), min(p); try smaller dt)")
    any(isinf, data) &&
        error("Inf in Yₜ after $context (Float overflow in tendency; " *
              "check for super-CFL vertical u₃, large kinetic energy, or division by near-zero density)")
    return nothing
end

"""
    check_state_nans(Y, context::String)

If the state vector `Y` contains any NaN, throw an error naming `context`.
Call at the start of each tendency so the first failure identifies which
tendency first received a bad state (i.e. the state was corrupted by the
previous tendency or by the previous IMEX stage update).

When `Y` has `ForwardDiff.Dual` eltype (Jacobian build), the check is skipped.
"""
function check_state_nans(Y, context::String)
    if hasproperty(Y, :c)
        first_name = first(propertynames(Y.c))
        T = eltype(parent(getproperty(Y.c, first_name)))
        if T <: ForwardDiff.Dual
            return nothing
        end
    end
    data = parent(Y)
    any(isnan, data) &&
        error("State Y contains NaN at entry to $context (state was already bad; NaN came from previous tendency or stage update)")
    any(isinf, data) &&
        error("State Y contains Inf at entry to $context (Float overflow in stage combination or tendency; " *
              "check for super-CFL vertical u₃ or kinetic energy overflow)")
    return nothing
end
