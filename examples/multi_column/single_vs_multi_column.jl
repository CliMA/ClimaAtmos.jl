# Run a column configuration on a single column and on a `MultiColumnGrid` of duplicate
# columns at (lat, long) = (0, 0), then check that
#  1. the duplicate columns are bitwise identical to each other, and
#  2. every column matches the single-column run within a tolerance,
# for the prognostic state, the cache, and the NetCDF diagnostics. Modeled on ClimaLand's
# experiments/integrated/era5/column_ensemble_comparison.jl.
#
# Equivalence holds up to floating-point rounding only when nothing reads latitude or
# longitude, so the columns sit at the equator. `deep_atmosphere` is left at the config
# value: neither column kind propagates it to the grid, so it acts identically on both.
#
# A field passes when its worst relative error is within `rtol` or its worst absolute
# error is within `atol` (near-zero fields); duplicate columns must be bitwise identical.
#
# Usage: julia --project=.buildkite examples/multi_column/single_vs_multi_column.jl \
#            --config_file <config.yml> [--config_file <overlay.yml> ...] --job_id <id>
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Spaces
import YAML
using Test

include(joinpath(@__DIR__, "comparison_utils.jl"))

const N_COLUMNS = 3

function run_simulation(config_files, job_id, overrides)
    overlay = joinpath(mktempdir(), "$(job_id).yml")
    output_dir = joinpath("output", job_id)
    YAML.write_file(overlay, merge(overrides, Dict("output_dir" => output_dir)))
    config = CA.AtmosConfig([config_files..., overlay]; job_id)
    simulation = CA.get_simulation(config)
    CA.solve_atmos!(simulation)
    return simulation
end

(; config_file, job_id) = CA.commandline_kwargs()
config_files = config_file isa AbstractString ? [config_file] : config_file
common = Dict("output_dir_style" => "removepreexisting")

single = run_simulation(config_files, "$(job_id)_single", common)
multi = run_simulation(
    config_files,
    "$(job_id)_multi",
    merge(
        common,
        Dict(
            "config" => "multicolumn",
            "column_latitudes" => zeros(N_COLUMNS),
            "column_longitudes" => zeros(N_COLUMNS),
        ),
    ),
)

Y1, p1 = single.integrator.u, single.integrator.p
Yn, pn = multi.integrator.u, multi.integrator.p
FT = eltype(Y1)
# rtol = FT == Float64 ? 1e-9 : 1e-4
# atol = FT == Float64 ? 1e-10 : 1e-6
rtol = zero(FT)
atol = zero(FT)

@testset "Single column vs $(N_COLUMNS) duplicate columns: $job_id" begin
    @test Spaces.ncolumns(axes(Yn.c)) == N_COLUMNS
    @testset "Duplicate columns are identical" begin
        for k in 2:N_COLUMNS
            dY = field_diffs(Yn, Yn; col1 = 1, col2 = k, name = "Y")
            dp = field_diffs(pn, pn; col1 = 1, col2 = k, name = "p", ignore = CACHE_IGNORE)
            @test report_diffs(dY; label = "Y column $k vs column 1") == 0
            @test report_diffs(dp; label = "p column $k vs column 1") == 0
        end
    end
    @testset "Columns match the single column" begin
        for k in 1:N_COLUMNS
            dY = field_diffs(Y1, Yn; col2 = k, name = "Y")
            dp = field_diffs(p1, pn; col2 = k, name = "p", ignore = CACHE_IGNORE)
            @test report_diffs(dY; label = "Y single vs column $k", rtol, atol) == 0
            @test report_diffs(dp; label = "p single vs column $k", rtol, atol) == 0
        end
        dd = diagnostic_diffs(single.output_dir, multi.output_dir; col = 1)
        label = "NetCDF diagnostics single vs column 1"
        @test report_diffs(dd; label, rtol, atol) == 0
    end
end
