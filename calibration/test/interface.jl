# Tests for ensuring that the calibration/model interface sets the AtmosConfig correctly.
import ClimaAtmos as CA
import ClimaCalibrate
using Test

include(joinpath(pkgdir(CA), "calibration", "model_interface.jl"))

member = 1
iter = 0

mktempdir() do output_dir
    member_path =
        ClimaCalibrate.path_to_ensemble_member(output_dir, iter, member)
    mkpath(member_path)
    # We need to check that a "base" TOML is support alongside the TOML sampled from the prior
    base_toml_file = touch(joinpath(output_dir, "default_parameters.toml"))
    sampled_toml_file = touch(joinpath(member_path, "parameters.toml"))
    config_dict = Dict(
        "output_default_diagnostics" => true,
        "moist" => "equil",
        "toml" => [base_toml_file],
        "output_dir" => output_dir,
    )
    (; parsed_args) =
        ClimaCalibrate.set_up_forward_model(member, iter, config_dict)

    @testset "Atmos Configuration" begin
        @test parsed_args["moist"] == "equil"
        @test parsed_args["output_dir"] == member_path
        @test parsed_args["output_default_diagnostics"] == false
        @test parsed_args["toml"] == [base_toml_file, sampled_toml_file]
    end
end
