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
    # Write our own parameter file
    open(joinpath(member_path, "parameters.toml"), "w") do file
        toml_contents = """
        [gravitational_acceleration]
        value = 10.0
        """
        write(file, toml_contents)
    end
    config_dict = Dict(
        "output_default_diagnostics" => true,
        "microphysics_model" => "0M",
        "toml" => [base_toml_file],
        "output_dir" => output_dir,
        "t_end" => "600secs",
    )
    simulation = ClimaCalibrate.forward_model(iter, member, config_dict)

    @testset "Atmos Configuration" begin
        @test float(simulation.t_end) == 600
        @test simulation.output_dir == joinpath(member_path, "output_0000")
        @test simulation.integrator.p.atmos.microphysics_model ==
              CA.EquilibriumMicrophysics0M()
        @test simulation.integrator.p.params.rrtmgp_params.grav == 10.0
    end
end
