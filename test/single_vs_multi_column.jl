# Run the ERA5-driven column configuration once per site and once on a
# multi-column grid with one site per column, and check that column `h` of the
# multi-column run is identical to the single-column run at site `h` in the
# prognostic state and the cache
!isnothing(Base.find_package("PrecompileCI")) && (using PrecompileCI)

import ClimaAtmos as CA
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore: Fields
using Test

const CONFIG_DIR = joinpath(@__DIR__, "..", "config")
const JOB_ID = "prognostic_edmfx_tv_era5driven_column"
const CONFIG_FILES = [
    joinpath(CONFIG_DIR, "common_configs", "diagnostics_column_progedmf_0M.yml"),
    joinpath(CONFIG_DIR, "model_configs", "$JOB_ID.yml"),
]
# Use clear-sky radiation since all-sky radiation seed each column with a
# different random number
const OVERRIDES = Dict{String, Any}("rad" => "clearsky", "t_end" => "1hours")
# The first and second columns share a site and the same forcing file
const SITE_LONGITUDES = [-149.0, -149.0, -150.0]

const IGNORED = Set([
    :atmos,
    :scratch,
    :output_dir,
    :ghost_buffer,
    :hyperdiffusion_ghost_buffer,
    :data_handler,
    :rc,
    :ᶜmp_tendency,
    :rrtmgp_solver,
    # Unwritten when every SGS tracer is a microphysics species
    :ᶜ∇²sgs_tracerʲs,
])

# Note that isequal(-0.0, 0.0) is false
same(x::Number, y::Number) = x == y || (isnan(x) && isnan(y))
same(x, y) = isequal(x, y)
same(x::AbstractArray, y::AbstractArray) =
    size(x) == size(y) && all(splat(same), zip(x, y))

array(field) = vec(Array(parent(field)))

# Don't specialize compare_column! since many different types will be passed
# from the single and multi-col simulations
@nospecialize

"""
    compare_column!(report, single, multi, h, name)

Compare `single`, from a single-column run, with column `h` of `multi`, from a
multi-column run, recursing into named tuples, tuples, `FieldVector`s, the
cache, and `Field`s of named tuples. Record each path in `report.compared`,
`report.differing`, or `report.skipped`.
"""
function compare_column!(report, single, multi, h, name)
    # Field of plain values
    if single isa Fields.Field && !(eltype(single) <: NamedTuple)
        push!(report.compared, name)
        same(array(single), array(Fields.column(multi, 1, 1, h))) ||
            push!(report.differing, name)
    elseif single isa Number
        push!(report.compared, name)
        same(single, multi) || push!(report.differing, name)
    elseif single isa Union{
        NamedTuple,
        Tuple,
        Fields.FieldVector,
        CA.AtmosCache,
        Fields.Field,
    }
        # Recurse into each property
        for p in propertynames(single)
            p in IGNORED && continue
            compare_column!(
                report,
                getproperty(single, p),
                getproperty(multi, p),
                h,
                "$name.$p",
            )
        end
    else
        push!(report.skipped, name)
    end
    return report
end

@specialize

"""
    run_simulation(overrides, job_id)

Build the simulation of `CONFIG_FILES` with `OVERRIDES` and `overrides`, and run it
to the end.
"""
function run_simulation(overrides, job_id)
    base = CA.AtmosConfig(CONFIG_FILES; job_id)
    output_dir = mktempdir()
    config = CA.AtmosConfig(
        (
            base.parsed_args,
            merge(OVERRIDES, overrides, Dict("output_dir" => output_dir)),
        );
        job_id,
    )
    simulation = CA.get_simulation(config)
    @test CA.solve_atmos!(simulation).ret_code == :success
    return simulation
end

@testset "Multi-column runs against single columns" begin
    singles = Dict(
        lon => run_simulation(
            Dict("site_longitude" => lon),
            "$(JOB_ID)_single_$lon",
        ) for lon in unique(SITE_LONGITUDES)
    )
    multi = run_simulation(
        Dict("n_columns" => length(SITE_LONGITUDES), "site_longitude" => SITE_LONGITUDES),
        "$(JOB_ID)_multi",
    )
    @testset "column $h, longitude $lon" for (h, lon) in enumerate(SITE_LONGITUDES)
        (; u, p) = singles[lon].integrator
        report = (; compared = String[], differing = String[], skipped = String[])
        compare_column!(report, u, multi.integrator.u, h, "Y")
        compare_column!(report, p, multi.integrator.p, h, "p")
        h == 1 && @info "Not compared: $(join(report.skipped, ", "))"
        @test !isempty(report.compared)
        @test report.differing == String[]
    end
    # Columns at different sites should differ
    @test array(singles[-149.0].integrator.u.c.ρe_tot) !=
          array(singles[-150.0].integrator.u.c.ρe_tot)
end
