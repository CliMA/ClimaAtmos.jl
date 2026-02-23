import ClimaAtmos as CA
import YAML
using Test

@testset "Output YAML Path Test" begin
    mktempdir() do output_dir
        job_id = "test_yaml_path_fix"

        input_toml_path = joinpath(@__DIR__, "..", "toml", "bomex_box_rhoe.toml")

        config_dict = Dict(
            "job_id" => job_id,
            "output_dir" => output_dir,
            "config" => "box",
            "toml" => [input_toml_path],
            "x_elem" => 2,
            "y_elem" => 2,
            "z_elem" => 4,
            "dt" => "1s",
            "t_end" => "2s",
            "dt_save_state_to_disk" => "2s",
        )

        config = CA.AtmosConfig(config_dict; job_id)

        simulation = CA.get_simulation(config)
        real_output_dir = simulation.output_dir

        output_yaml_path = joinpath(real_output_dir, "$(job_id).yml")
        output_toml_path = joinpath(real_output_dir, "$(job_id)_parameters.toml")

        @test isfile(output_yaml_path)
        @test isfile(output_toml_path)

        yaml_data = YAML.load_file(output_yaml_path)
        @test yaml_data["toml"] == [abspath(output_toml_path)]
        @test yaml_data["toml"] != [input_toml_path]
    end
end
