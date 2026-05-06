using Test
import ClimaAtmos as CA
import ClimaAtmos.Diagnostics as CAD

@testset "DiagnosticsConfig defaults" begin
    cfg = CA.DiagnosticsConfig()
    @test cfg.default == true
    @test cfg.additional == ()
    @test cfg.interpolation_num_points === nothing
    @test cfg.output_at_levels == true
end

@testset "DiagnosticsConfig overrides" begin
    cfg = CA.DiagnosticsConfig(;
        default = false,
        additional = ["ts" => "1hours"],
        interpolation_num_points = (180, 90, 10),
        output_at_levels = false,
    )
    @test cfg.default == false
    @test cfg.additional == ["ts" => "1hours"]
    @test cfg.interpolation_num_points == (180, 90, 10)
    @test cfg.output_at_levels == false
end

@testset "normalize_diag_entry: Pair{String,String}" begin
    d = CAD.normalize_diag_entry("ts" => "1hours")
    @test d isa Dict{String, Any}
    @test d["short_name"] == "ts"
    @test d["period"] == "1hours"
    @test length(d) == 2
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
