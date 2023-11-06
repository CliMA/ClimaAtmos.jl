#####
##### Utility functions
#####
import ClimaComms
import ClimaCore: Spaces, Topologies, Fields, Geometry
import LinearAlgebra: norm_sqr

is_energy_var(symbol) = symbol in (:ρθ, :ρe_tot, :ρaθ, :ρae_tot)
is_momentum_var(symbol) = symbol in (:uₕ, :ρuₕ, :u₃, :ρw)
is_turbconv_var(symbol) = symbol in (:turbconv, :sgsʲs, :sgs⁰)
is_tracer_var(symbol) = !(
    symbol == :ρ ||
    symbol == :ρa ||
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

"""
    compute_kinetic!(κ::Field, uₕ::Field, uᵥ::Field)

Compute the specific kinetic energy at cell centers, storing in `κ` from
individual velocity components:
κ = 1/2 (uₕ⋅uʰ + 2uʰ⋅ᶜI(uᵥ) + ᶜI(uᵥ⋅uᵛ))
- `uₕ` should be a `Covariant1Vector` or `Covariant12Vector`-valued field at
    cell centers, and 
- `uᵥ` should be a `Covariant3Vector`-valued field at cell faces.
"""
function compute_kinetic!(κ::Fields.Field, uₕ::Fields.Field, uᵥ::Fields.Field)
    @assert eltype(uₕ) <: Union{C1, C2, C12}
    @assert eltype(uᵥ) <: C3
    @. κ =
        1 / 2 * (
            dot(C123(uₕ), CT123(uₕ)) +
            ᶜinterp(dot(C123(uᵥ), CT123(uᵥ))) +
            2 * dot(CT123(uₕ), ᶜinterp(C123(uᵥ)))
        )
end

"""
    compute_kinetic!(κ::Field, Y::FieldVector)

Compute the specific kinetic energy at cell centers, storing in `κ`, where `Y`
is the model state.
"""
compute_kinetic!(κ::Fields.Field, Y::Fields.FieldVector) =
    compute_kinetic!(κ, Y.c.uₕ, Y.f.u₃)

"""
    compute_strain_rate_center!(ϵ::Field, u::Field)

Compute the strain_rate at cell centers, storing in `ϵ` from
velocity at cell faces.
"""
function compute_strain_rate_center!(ϵ::Fields.Field, u::Fields.Field)
    @assert eltype(u) <: C123
    axis_uvw = Geometry.UVWAxis()
    @. ϵ =
        (
            Geometry.project((axis_uvw,), ᶜgradᵥ(UVW(u))) +
            adjoint(Geometry.project((axis_uvw,), ᶜgradᵥ(UVW(u))))
        ) / 2
end

"""
    compute_strain_rate_face!(ϵ::Field, u::Field)

Compute the strain_rate at cell faces, storing in `ϵ` from
velocity at cell centers.
"""
function compute_strain_rate_face!(ϵ::Fields.Field, u::Fields.Field)
    @assert eltype(u) <: C123
    ∇ᵥuvw_boundary =
        Geometry.outer(Geometry.WVector(0), Geometry.UVWVector(0, 0, 0))
    ᶠgradᵥ = Operators.GradientC2F(
        bottom = Operators.SetGradient(∇ᵥuvw_boundary),
        top = Operators.SetGradient(∇ᵥuvw_boundary),
    )
    axis_uvw = Geometry.UVWAxis()
    @. ϵ =
        (
            Geometry.project((axis_uvw,), ᶠgradᵥ(UVW(u))) +
            adjoint(Geometry.project((axis_uvw,), ᶠgradᵥ(UVW(u))))
        ) / 2
end

"""
    g³³_field(field)

Extracts the value of `g³³` from `Fields.local_geometry_field(field)`.
"""
function g³³_field(field)
    g_field = Fields.local_geometry_field(field).gⁱʲ.components.data
    end_index = fieldcount(eltype(g_field)) # This will be 4 in 2D and 9 in 3D.
    return g_field.:($end_index) # For both 2D and 3D spaces, g³³ = g[end].
end

"""
    unit_basis_vector_data(type, local_geometry)

The component of the vector of the specified type with length 1 in physical units.
The type should correspond to a vector with only one component, i.e., a basis vector.
"""
function unit_basis_vector_data(::Type{V}, local_geometry) where {V}
    FT = Geometry.undertype(typeof(local_geometry))
    return FT(1) / Geometry._norm(V(FT(1)), local_geometry)
end

"""
    projected_vector_data(::Type{V}, vector, local_geometry)

Projects the given vector onto the axis of V, then extracts the component data and rescales it to physical units.
The type should correspond to a vector with only one component, i.e., a basis vector.
"""
projected_vector_data(::Type{V}, vector, local_geometry) where {V} =
    V(vector, local_geometry)[1] / unit_basis_vector_data(V, local_geometry)

"""
    get_physical_w(u, local_geometry)
    
Return physical vertical velocity - a projection of full velocity vector
onto the vertical axis.
"""
get_physical_w(u, local_geometry) = Geometry.WVector(u, local_geometry)[1]

time_to_seconds(t::Number) =
    t == Inf ? t : error("Uncaught case in computing time from given string.")

function time_to_seconds(s::String)
    factor = Dict(
        "secs" => 1,
        "mins" => 60,
        "hours" => 60 * 60,
        "days" => 60 * 60 * 24,
    )
    s == "Inf" && return Inf
    if count(occursin.(keys(factor), Ref(s))) != 1
        error(
            "Bad format for flag $s. Examples: [`10secs`, `20mins`, `30hours`, `40days`]",
        )
    end
    for match in keys(factor)
        occursin(match, s) || continue
        return parse(Float64, first(split(s, match))) * factor[match]
    end
    error("Uncaught case in computing time from given string.")
end

function verify_callbacks(t)
    if length(t) ≠ length(unique(t))
        @show length(t)
        @show length(unique(t))
        error(
            string(
                "Saving duplicate solutions at the same time.",
                "Please change the callbacks to not save ",
                "duplicate solutions at the same timestep.",
            ),
        )
    end
end

using ClimaComms
is_distributed(::ClimaComms.SingletonCommsContext) = false
is_distributed(::ClimaComms.MPICommsContext) = true

using Printf
# From BenchmarkTools
function prettytime(t)
    if t < 1e3
        value, units = t, "ns"
    elseif t < 1e6
        value, units = t / 1e3, "μs"
    elseif t < 1e9
        value, units = t / 1e6, "ms"
    else
        value, units = t / 1e9, "s"
    end
    return @sprintf("%.3f %s", value, units)
end

function prettymemory(b)
    if b < 1024
        return string(b, " bytes")
    elseif b < 1024^2
        value, units = b / 1024, "KiB"
    elseif b < 1024^3
        value, units = b / 1024^2, "MiB"
    else
        value, units = b / 1024^3, "GiB"
    end
    return @sprintf("%.2f %s", value, units)
end

"""
    @timed_str expr
    @timed_str "description" expr

Returns a string containing `@timed` information.
"""
macro timed_str(ex)
    quote
        local stats = @timed $(esc(ex))
        "$(prettytime(stats.time*1e9)) ($(Base.gc_alloc_count(stats.gcstats)) allocations: $(prettymemory(stats.gcstats.allocd)))"
    end
end

struct AllNothing end
const all_nothing = AllNothing()
Base.getproperty(::AllNothing, ::Symbol) = nothing

"""
    run_model_from_file(config_file)

Convenience function to run the driver with a given config file.
Note that this does not set the global Random seed, 
so results may not be fully reproducible. 
"""
function run_model_from_file(config_file)
    config = AtmosConfig(; parsed_args = Dict("config_file" => config_file))
    integrator = get_integrator(config)
    return solve_atmos!(integrator)
end
