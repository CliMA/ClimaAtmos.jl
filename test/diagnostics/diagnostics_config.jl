using Test
import ClimaAtmos as CA
import ClimaAtmos.Diagnostics as CAD
using Dates

@testset "DiagnosticsConfig defaults" begin
    cfg = CA.DiagnosticsConfig()
    @test cfg.default == true
    @test cfg.additional == ()
    @test cfg.interpolation_num_points === nothing
    @test cfg.output_at_levels == true
end

@testset "DiagnosticsConfig overrides" begin
    spec = ["ts" => (; period = "1hours", reduction = "average")]
    cfg = CA.DiagnosticsConfig(;
        default = false,
        additional = spec,
        interpolation_num_points = (180, 90, 10),
        output_at_levels = false,
    )
    @test cfg.default == false
    @test cfg.additional == spec
    @test cfg.interpolation_num_points == (180, 90, 10)
    @test cfg.output_at_levels == false
end

@testset "normalize_diag_entry: bare Pair{String,String} is rejected" begin
    # The bare short_name => period shorthand is ambiguous (no reduction);
    # callers must specify the reduction via NamedTuple options.
    @test_throws ErrorException CAD.normalize_diag_entry("ts" => "1hours")
end

@testset "normalize_diag_entry: Pair{String,NamedTuple}" begin
    d = CAD.normalize_diag_entry("ua" => (; period = "30mins", reduction = "average"))
    @test d isa Dict{String, Any}
    @test d["short_name"] == "ua"
    @test d["period"] == "30mins"
    # `reduction` aliases to canonical YAML key `reduction_time`
    @test d["reduction_time"] == "average"
    @test !haskey(d, "reduction")
end

@testset "normalize_diag_entry: NamedTuple" begin
    d = CAD.normalize_diag_entry((;
        short_name = "va",
        period = "1hours",
        writer = "netcdf",
    ))
    @test d isa Dict{String, Any}
    @test d["short_name"] == "va"
    @test d["period"] == "1hours"
    @test d["writer"] == "netcdf"
end

@testset "normalize_diag_entry: NamedTuple with reduction alias" begin
    d = CAD.normalize_diag_entry((;
        short_name = "ts",
        period = "1hours",
        reduction = "max",
    ))
    @test d["reduction_time"] == "max"
    @test !haskey(d, "reduction")
end

@testset "normalize_diag_entry: AbstractDict passes through with stringified keys" begin
    d = CAD.normalize_diag_entry(Dict("short_name" => "ts", "period" => "1hours"))
    @test d isa Dict{String, Any}
    @test d["short_name"] == "ts"
    @test d["period"] == "1hours"
end

@testset "normalize_diag_entry: rejects unsupported types" begin
    @test_throws ErrorException CAD.normalize_diag_entry(42)
    @test_throws ErrorException CAD.normalize_diag_entry("just a string")
    @test_throws ErrorException CAD.normalize_diag_entry(:symbol_only)
end

@testset "Default diagnostics cadence at exact threshold boundaries" begin
    # Every bucket requires three of its own averaging windows, so a cadence is only
    # selected once the run can produce more than one sample at it.
    cadence(duration) = CAD.extract_diagnostic_periods(
        CAD.frequency_averages(duration)(
            "ta";
            output_writer = CAD.DictWriter(),
            start_date = DateTime(2010, 1, 1),
            t_start = 0.0,
        ),
    )

    @test cadence(90 * 86400.0) == Set([Month(1)])
    @test cadence(30 * 86400.0) == Set([Day(10)])
    @test cadence(3 * 86400.0) == Set([Day(1)])
    @test cadence(3 * 3600.0) == Set([Hour(1)])

    # One timestep short of a boundary selects the next cadence down.
    @test cadence(30 * 86400.0 - 30) == Set([Day(1)])
    @test cadence(3 * 86400.0 - 30) == Set([Hour(1)])
    @test isempty(cadence(3 * 3600.0 - 30))

    # A run lasting exactly one averaging window must not select it; 86400 and 3600
    # are the lengths used by the CI column cases.
    @test cadence(86400.0) == Set([Hour(1)])
    @test isempty(cadence(3600.0))
end
