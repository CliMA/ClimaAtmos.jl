"""
This file contains helper methods for adopting show methods for model structs.

The main helper methods follow:
- `verbose_show_type_and_fields(io::IO, ::MIME"text/plain", x)`
- `parseable_show_with_fields_no_type_header(io::IO, x; with_kwargs = true)`

Below these, there are implementations of `Base.show` and `Base.summary` for model structs.

More reading: https://discourse.julialang.org/t/print-vs-two-argument-show-repr-vs-three-argument-show-repr-with-mime-text-plain/117790
"""


### ----------------------- ###
### Helper methods for show ###
### ----------------------- ###

_type_name(x; with_module_prefix = false) =
    if with_module_prefix
        # name with module prefix, e.g. CloudMicrophysics.Parameters.SB2006
        typeof(x).name.wrapper
    else
        # name without module prefix, e.g. SB2006
        string(typeof(x).name.name)
    end

"""
    field_units(x)

Return a `NamedTuple` mapping field names to unit strings for type `x`,
or `nothing` if no units are defined.

Types that want unit annotations in their verbose `show` output should
define a method, e.g.:

```julia
ShowMethods.field_units(::MyType) = (; mass = "kg", velocity = "m/s")
```

Fields present in the `NamedTuple` get `[unit]` printed after their value.
Fields *not* in the `NamedTuple` (but whose type does define `field_units`)
get `[-]` (dimensionless). Types with no `field_units` method get no
unit annotations at all.
"""
field_units(_) = nothing

"Print unit annotation for field `k` based on the units spec."
_print_unit(::IO, _, _) = nothing
function _print_unit(io::IO, units::NamedTuple, k::Symbol)
    unit = haskey(units, k) ? units[k] : "-"
    print(io, " [", unit, "]")
end

"""
    _is_nested_struct(v)

Return `true` if `v` is a composite struct that should be recursively
pretty-printed (i.e. a non-trivial struct with fields, but not a
`Number`, `AbstractString`, `Symbol`, or `Nothing`).
"""
_is_nested_struct(v) =
    isstructtype(typeof(v)) &&
    fieldcount(typeof(v)) > 0 &&
    !(v isa Union{Number, AbstractString, Symbol, Nothing})


"""
    verbose_show_type_and_fields(io::IO, ::MIME"text/plain", x)

Print a verbose representation of the type and fields of `x` to `io`.
"""
function verbose_show_type_and_fields(io::IO, mime::MIME"text/plain", x; with_module_prefix = false)
    compact = get(io, :compact, false)::Bool
    indent = get(io, :indent, "")::String
    typename = _type_name(x; with_module_prefix)
    keys = fieldnames(typeof(x))
    vals = [getfield(x, k) for k in keys]
    units = field_units(x)
    print(io, "$(typename)")
    if compact
        print(io, "(")
        for (k, v) in zip(keys, vals)
            print(io, "$k = ", v, ",")
            k == keys[end] || print(io, " ")
        end
        print(io, ")")
    else
        for (i, (k, v)) in enumerate(zip(keys, vals))
            is_last = i == length(keys)
            connector = is_last ? "└─ " : "├─ "
            continuation = is_last ? "   " : "│  "
            if _is_nested_struct(v)
                print(io, "\n$(indent)$(connector)$(k): ")
                nested_io = IOContext(io, :indent => indent * continuation)
                show(nested_io, mime, v)
            else
                print(io, "\n$(indent)$(connector)$(k) = ", v)
                _print_unit(io, units, k)
            end
        end
    end
end

"""
    compact_show_type_and_fields(io::IO, ::MIME"text/plain", x; kw...)

Print a compact one-line representation of `x`: `TypeName(field = value, ...)`.

# Keyword arguments
- `with_units = true`: Append unit annotations from [`field_units`](@ref).
- `with_kwargs = true`: Print fields as keyword arguments (`field = value`).
- `with_module_prefix = false`: Use module-qualified type name.
- `skip_fields_by_value = ()`: Skip fields whose values are in this tuple.

# Example output
```julia-repl
julia> import CloudMicrophysics.Parameters as CMP

julia> ap = CMP.AirProperties(Float32)
AirProperties(K_therm = 0.024 [W/m/K], D_vapor = 2.26e-5 [m²/s], ν_air = 1.6e-5 [m²/s])
```
"""
function compact_show_type_and_fields(io::IO, ::MIME"text/plain", x;
    with_units = true, with_kwargs = true,
    with_module_prefix = false, skip_fields_by_value = (),
)
    typename = _type_name(x; with_module_prefix)
    keys = fieldnames(typeof(x))
    vals = [getfield(x, k) for k in keys]
    units = with_units ? field_units(x) : nothing
    print(io, typename, "(")
    printed_first = false
    for (k, v) in zip(keys, vals)
        v ∈ skip_fields_by_value && continue
        printed_first && print(io, ", ")
        with_kwargs && print(io, k, " = ")
        show(io, v)
        _print_unit(io, units, k)
        printed_first = true
    end
    print(io, ")")
end

"""
	parseable_show_with_fields_no_type_header(io::IO, x; kw...)

Print a parseable (copy-pasteable) one-line representation of `x`.
Thin wrapper around [`compact_show_type_and_fields`](@ref) with
`with_units = false` and `with_module_prefix = true` by default.

Note: This assumes that the type 

# Examples
```julia-repl
julia> show(stdout, CM.Parameters.NumberAdjustmentHorn2012(τ = 3.0f0))
CloudMicrophysics.Parameters.NumberAdjustmentHorn2012(τ = 3.0f0)
```
"""
parseable_show_with_fields_no_type_header(io::IO, x; with_module_prefix = true, kw...) =
    compact_show_type_and_fields(io, MIME("text/plain"), x;
        with_units = false, with_module_prefix, kw...,
    )


### ------------------------------ ###
### Show methods for model structs ###
### ------------------------------ ###

## Implementations of `Base.show(io::IO, x::MyModel)` should enable copy-pasting
## of the output to recreate the model.
##
## Implementations of `Base.show(io::IO, mime::MIME"text/plain", x::MyModel)` should
## provide a human-readable representation of the type and fields of `x`.
## Optionally, provide a compact version of the human-readable representation
## considering the value of `get(io, :compact, false)`.


### src/solver/types.jl
Base.show(io::IO, x::ConstantAlbedo) =
    parseable_show_with_fields_no_type_header(io, x)
Base.show(io::IO, x::AbstractCloudModel) =
    parseable_show_with_fields_no_type_header(io, x; with_kwargs = false)

function Base.show(io::IO, x::SGSQuadrature)
    name = typeof(x).name.wrapper
    FT = eltype(x.a)
    N = quadrature_order(x)
    print(io, "$(name)($(FT); quadrature_order = $(N))")
end

Base.show(io::IO, x::Hyperdiffusion) = parseable_show_with_fields_no_type_header(io, x)

Base.show(io::IO, mime::MIME"text/plain", x::AtmosWater) =
    verbose_show_type_and_fields(io, mime, x)

Base.show(io::IO, x::AtmosNumerics) =
    parseable_show_with_fields_no_type_header(io, x; skip_fields_by_value = (nothing,))
Base.show(io::IO, x::EDMFXModel) =
    parseable_show_with_fields_no_type_header(io, x; skip_fields_by_value = (nothing,))
Base.show(io::IO, x::SCMSetup) =
    parseable_show_with_fields_no_type_header(io, x; skip_fields_by_value = (nothing,))
Base.show(io::IO, x::AtmosWater) =
    parseable_show_with_fields_no_type_header(io, x; skip_fields_by_value = (nothing,))
Base.show(io::IO, x::AtmosRadiation) =
    parseable_show_with_fields_no_type_header(io, x; skip_fields_by_value = (nothing,))
Base.show(io::IO, x::AtmosTurbconv) = parseable_show_with_fields_no_type_header(
    io, x; skip_fields_by_value = (nothing, Explicit()),
)
Base.show(io::IO, x::AtmosGravityWave) =
    parseable_show_with_fields_no_type_header(io, x; skip_fields_by_value = (nothing,))
Base.show(io::IO, x::AtmosSponge) =
    parseable_show_with_fields_no_type_header(io, x; skip_fields_by_value = (nothing,))
Base.show(io::IO, x::AtmosSurface) = parseable_show_with_fields_no_type_header(io, x)
Base.show(io::IO, x::AtmosModel) = parseable_show_with_fields_no_type_header(io, x)

Base.show(io::IO, mime::MIME"text/plain", model::AtmosModel) =
    verbose_show_type_and_fields(io, mime, model)

Base.show(io::IO, x::SpongeModel) = parseable_show_with_fields_no_type_header(io, x)

# src/parameterized_tendencies/radiation/RRTMGPInterface.jl
Base.show(io::IO, mime::MIME"text/plain", mode::RRTMGPI.AbstractRRTMGPMode) =
    verbose_show_type_and_fields(io, mime, mode)
Base.show(io::IO, x::RRTMGPI.AbstractRRTMGPMode) =
    parseable_show_with_fields_no_type_header(io, x)

# src/initial_conditions/initial_conditions.jl
Base.show(io::IO, x::Setups.RCEMIPIIProfile) =
    parseable_show_with_fields_no_type_header(io, x)

import ClimaCore: Fields, Spaces
function Base.show(
    io::IO, ::MIME"text/plain", x::Setups.ColumnInterpolatableField,
)
    # Extract z grid from the wrapped column field
    z = Fields.coordinate_field(x.f).z
    nz = Spaces.nlevels(z)
    zmin, zmax = extrema(z)
    val_eltype = eltype(x.f)
    # These are fixed by the constructor
    interp_str = "Linear"
    extrap_str = "Flat"
    print(io,
        "ColumnInterpolatableField(Nz=$nz, z∈[$zmin, $zmax], value_eltype=$val_eltype, ",
        "interpolation=$interp_str, extrapolation=$extrap_str)",
    )
end

# src/simulation/AtmosSimulations.jl
import ClimaComms
function Base.show(io::IO, ::MIME"text/plain", sim::AtmosSimulation)
    device_type = nameof(typeof(ClimaComms.device(sim)))
    return print(
        io,
        "Simulation $(sim.job_id)\n",
        "├── Running on: $(device_type)\n",
        "├── Output folder: $(sim.output_dir)\n",
        "├── Start date: $(sim.start_date)\n",
        "├── Current time: $(sim.integrator.t) seconds\n",
        "└── Stop time: $(sim.t_end) seconds",
    )
end

# src/solver/solve.jl
function Base.show(io::IO, ::MIME"text/plain", sim::AtmosSolveResults)
    return print(
        io,
        "Simulation completed\n",
        "├── Return code: $(sim.ret_code)\n",
        "└── Walltime: $(sim.walltime) seconds",
    )
end

# src/parameters/Parameters.jl
Base.show(io::IO, mime::MIME"text/plain", params::Parameters.ClimaAtmosParameters) =
    verbose_show_type_and_fields(io, mime, params)

Base.show(io::IO, params::Parameters.ClimaAtmosParameters) =
    parseable_show_with_fields_no_type_header(io, params)

### ---------------------- ###
### Custom summary methods ###
### ---------------------- ###

Base.summary(::AtmosSimulation) = "AtmosSimulation"

function Base.summary(io::IO, numerics::AtmosNumerics)
    pns = string.(propertynames(numerics))
    buf = maximum(length.(pns))
    keys = propertynames(numerics)
    vals = repeat.(" ", map(s -> buf - length(s) + 2, pns))
    bufs = (; zip(keys, vals)...)
    print(io, '\n')
    for pn in propertynames(numerics)
        prop = getproperty(numerics, pn)
        s = string(
            "  ", # needed for some reason
            getproperty(bufs, pn),
            '`',
            string(pn),
            '`',
            "::",
            '`',
            typeof(prop),
            '`',
            '\n',
        )
        print(io, s)
    end
end

function Base.summary(io::IO, atmos::AtmosModel)
    pns = string.(propertynames(atmos))
    buf = maximum(length.(pns))
    keys = propertynames(atmos)
    vals = repeat.(" ", map(s -> buf - length(s) + 2, pns))
    bufs = (; zip(keys, vals)...)
    print(io, '\n')
    for pn in propertynames(atmos)
        prop = getproperty(atmos, pn)
        # Skip some data:
        prop isa Bool && continue
        prop isa NTuple && continue
        prop isa Int && continue
        prop isa Float64 && continue
        prop isa Float32 && continue
        s = string(
            "  ", # needed for some reason
            getproperty(bufs, pn),
            '`',
            string(pn),
            '`',
            "::",
            '`',
            typeof(prop),
            '`',
            '\n',
        )
        print(io, s)
    end
end
