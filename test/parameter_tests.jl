using Test
import CLIMAParameters as CP
import ClimaAtmos as CA

FT = Float64

default_args = CA.cli_defaults(CA.argparse_settings())

@testset "Test types" begin

    dict = Dict(
        "krylov_rtol" => Float64(1),
        "newton_rtol" => Float32(1),
        "max_newton_iters" => Int64(1),
        "nh_poly" => Int32(1),
        "dt_save_to_disk" => "str",
        "bubble" => true,
    )
    toml_dict = CP.create_toml_dict(FT)
    toml_dict, _ = CA.merge_parsed_args_with_toml(toml_dict, dict, default_args)
    values = (; CP.get_parameter_values!(toml_dict, collect(keys(dict)))...)

    @test values.krylov_rtol isa FT
    @test toml_dict.data["krylov_rtol"]["type"] == "float"

    @test values.newton_rtol isa FT
    @test toml_dict.data["newton_rtol"]["type"] == "float"

    @test values.max_newton_iters isa Int
    @test toml_dict.data["max_newton_iters"]["type"] == "integer"

    @test values.nh_poly isa Int
    @test toml_dict.data["nh_poly"]["type"] == "integer"

    @test values.dt_save_to_disk isa String
    @test toml_dict.data["dt_save_to_disk"]["type"] == "string"

    @test values.bubble isa Bool
    @test toml_dict.data["bubble"]["type"] == "bool"
end

@testset "Test override and alias" begin
    toml_dict = CP.create_toml_dict(
        FT;
        override_file = joinpath(@__DIR__, "parameter_tests.toml"),
    )
    @test CP.get_parameter_values!(toml_dict, "y_elem").second == 0

    dict = Dict("y_elem" => 1)
    toml_dict, _ = CA.merge_parsed_args_with_toml(toml_dict, dict, default_args)
    println(CP.get_parameter_values!(toml_dict, "y_elem"))
    println(typeof(CP.get_parameter_values!(toml_dict, "y_elem")))
    @test CP.get_parameter_values!(toml_dict, "y_elem").second == 1

    # Ensure that conflicting aliases use the final alias in order
    @test CP.get_parameter_values!(toml_dict, "same_alias").second == 0
end

@testset "Test priorities" begin
    toml_dict = CP.create_toml_dict(
        FT;
        override_file = joinpath(@__DIR__, "parameter_tests.toml"),
    )
    parsed_args = Dict(
        "toml" => "test/parameter_tests.toml",
        "dt" => "35secs",
        "t_end" => "3hours",
    )
    toml_dict, _ =
        CA.merge_parsed_args_with_toml(toml_dict, parsed_args, default_args)
    param_names = ["dt", "dt_save_to_disk", "z_elem"]
    params = CP.get_parameter_values!(toml_dict, param_names)
    params = (; params...)
    # Check that CLI takes priority over TOML
    @test params.dt == "35secs"
    # Check that TOML takes priority over CLI defaults
    @test params.dt_save_to_disk == "5mins"
    @test params.z_elem == 60
end
