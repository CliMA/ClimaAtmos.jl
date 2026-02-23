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

"""
    verbose_show_type_and_fields(io::IO, ::MIME"text/plain", x)

Print a verbose representation of the type and fields of `x` to `io`.

# Examples
```julia-repl
julia> import ClimaAtmos as CA

julia> CA.RRTMGPInterface.AllSkyRadiation()
AllSkyRadiation
  ├─ idealized_h2o = false
  ├─ idealized_clouds = false
  ├─ cloud = ClimaAtmos.InteractiveCloudInRadiation()
  ├─ add_isothermal_boundary_layer = true
  ├─ aerosol_radiation = false
  ├─ reset_rng_seed = false
  └─ deep_atmosphere = true

```
"""
function verbose_show_type_and_fields(io::IO, ::MIME"text/plain", x)
    compact = get(io, :compact, false)::Bool
    typename = typeof(x).name.name  # name without module prefix
    keys = fieldnames(typeof(x))
    vals = [getfield(x, k) for k in keys]
    print(io, "$(typename)")
    if compact
        print(io, "(")
        for (k, v) in zip(keys, vals)
            print(io, "$k = ", v, ",")
            k == keys[end] || print(io, " ")
        end
        print(io, ")")
    else
        for (k, v) in zip(keys, vals)
            prefix = k == keys[end] ? "└─ " : "├─ "
            print(io, "\n  " * prefix * "$k = ", v)
        end
    end
end

"""
	parseable_show_with_fields_no_type_header(io::IO, x; with_kwargs = ())

Print a parseable representation of the type that allows reconstruction of `x`.

# Arguments
- `with_kwargs`: If true, print the fields as keyword arguments.
	- This is relevant for structs definitions prefixed with `@kwdef`
- `skip_fields_by_value`: A tuple of values that are not printed for any fields with those values.
	- Note: Because the printed representation should reconstruct an equivalent instance,
		the values provided here should be the default values for the fields.
	- By default, no fields are skipped.

# Examples
```julia-repl
julia> show(stdout, CA.AtmosRadiation())
ClimaAtmos.AtmosRadiation(insolation = ClimaAtmos.IdealizedInsolation())
```
"""
function parseable_show_with_fields_no_type_header(io::IO, x;
    with_kwargs = true, skip_fields_by_value = (),
)
    typename = typeof(x).name.wrapper  # name with module prefix, e.g. ClimaAtmos.QuadratureCloud
    keys = fieldnames(typeof(x))
    vals = [getfield(x, k) for k in keys]
    show(io, typename)
    print(io, "(")
    printed_first_arg = false
    for (k, v) in zip(keys, vals)
        v ∈ skip_fields_by_value && continue
        printed_first_arg && print(io, ", ")
        with_kwargs && print(io, "$k = ")
        show(io, v)
        printed_first_arg = true
    end
    print(io, ")")
end

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
Base.show(io::IO, x::InitialConditions.RCEMIPIIProfile) =
    parseable_show_with_fields_no_type_header(io, x)

import ClimaCore: Fields, Spaces
function Base.show(
    io::IO, ::MIME"text/plain", x::InitialConditions.ColumnInterpolatableField,
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
