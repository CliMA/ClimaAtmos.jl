#####
##### Utility functions
#####
import ClimaComms
import ClimaCore: Spaces, Topologies, Fields
import LinearAlgebra: norm_sqr
import DiffEqBase
import JLD2

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
is the model state.s
"""
compute_kinetic!(κ::Fields.Field, Y::Fields.FieldVector) =
    compute_kinetic!(κ, Y.c.uₕ, Y.f.u₃)

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

function export_scaling_file(sol, output_dir, walltime, comms_ctx, nprocs)
    # replace sol.u on the root processor with the global sol.u
    if ClimaComms.iamroot(comms_ctx)
        Y = sol.u[1]
        center_space = axes(Y.c)
        horz_space = Spaces.horizontal_space(center_space)
        horz_topology = horz_space.topology
        Nq = Spaces.Quadratures.degrees_of_freedom(horz_space.quadrature_style)
        nlocalelems = Topologies.nlocalelems(horz_topology)
        ncols_per_process = nlocalelems * Nq * Nq
        scaling_file =
            joinpath(output_dir, "scaling_data_$(nprocs)_processes.jld2")
        @info(
            "Writing scaling data",
            "walltime (seconds)" = walltime,
            scaling_file
        )
        JLD2.jldsave(scaling_file; nprocs, ncols_per_process, walltime)
    end
    return nothing
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
is_distributed(::ClimaComms.AbstractCommsContext) = true # deprecate after upgrading
# is_distributed(::ClimaComms.MPICommsContext) = true # use this after upgrading


# TODO: remove after upgrading to latest ClimaCore:
Spaces_comm_context(field::Fields.Field) = Spaces_comm_context(axes(field))

Spaces_comm_context(space::Spaces.ExtrudedFiniteDifferenceSpace) =
    Spaces_comm_context(space.horizontal_space)
Spaces_comm_context(space::Spaces.SpectralElementSpace2D) =
    Spaces_comm_context(space.topology)
Spaces_comm_context(space::S) where {S <: Spaces.AbstractSpace} =
    ClimaComms.SingletonCommsContext()
import ClimaCore.Topologies as Topologies
Spaces_comm_context(topology::Topologies.Topology2D) = topology.context
Spaces_comm_context(topology::T) where {T <: Topologies.AbstractTopology} =
    ClimaComms.SingletonCommsContext()
