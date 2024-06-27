import ClimaAtmos as CA
import SurfaceFluxes as SF
import ClimaAtmos.Parameters as CAP
import ClimaCore as CC
import Thermodynamics as TD
import CloudMicrophysics as CM
import SciMLBase
import ClimaCore.Spaces
import ClimaComms
using Test

@testset begin
    "Test restarts across configuration combinations"
    ### Test Description
    # Generate a simulation with some complexity of 
    # config arguments. Some config combinations are
    # incompatible so we do not sweep over all possible
    # iterations.

    # Modify the timestep to 1-second increments. 
    # Save simulation state at each timestep, 
    # and generate a restart file at 0secs, 2secs simulation time.
    # Verify objects read in using ClimaCore.InputOutput functions 
    # are identical (i.e. restarts result
    # in the same simulation states as if one were to advance
    # the timestepper uninterrupted.)

    # TODO: Restart and diagnostic behaviour needs to be 
    # clearly defined when config files have different 
    # settings (or when tendency computations conflict with
    # dt or t_end parsed args)

    for configuration in ["sphere", "column"]
        for moisture in ["equil"]
            for turb_conv in ["diagnostic_edmfx", "prognostic_edmfx"]
                for precip in ["0M", "1M"]

                    mktempdir() do output_loc
                        test_dict = Dict(
                            "check_nan_every" => 3,
                            "log_progress" => false,
                            "moist" => moisture,
                            "precip_model" => precip,
                            "config" => configuration,
                            "turbconv" => turb_conv,
                            "perturb_initstate" => false,
                            "dt" => "1secs",
                            "t_end" => "3secs",
                            "dt_save_state_to_disk" => "1secs",
                            "enable_diagnostics" => false,
                        )

                        ### Boilerplate default integrator objects
                        config = CA.AtmosConfig(
                            merge(
                                test_dict,
                                Dict(
                                    "output_dir" => joinpath(
                                        output_loc,
                                        "gen_test_output",
                                    ),
                                ),
                            ),
                        )

                        @info config.parsed_args["output_dir"]
                        simulation = CA.get_simulation(config)
                        Y₀ = simulation.integrator.u
                        CA.solve_atmos!(simulation)

                        restart_dir = simulation.output_dir
                        @test isfile(joinpath(restart_dir), "day0.2.hdf5")
                        restart₁ = joinpath(restart_dir, "day0.2.hdf5")

                        @info "Restart #1 from: " restart₁
                        config₁ = CA.AtmosConfig(
                            merge(
                                test_dict,
                                Dict(
                                    "restart_file" => restart₁,
                                    "output_dir" => joinpath(
                                        output_loc,
                                        "restart_test_output",
                                    ),
                                ),
                            ),
                        )
                        simulation_test₁ = CA.get_simulation(config₁)
                        @info "Advancing restarted simulation"
                        CA.solve_atmos!(simulation_test₁)
                        @info "Restarted simulation complete"

                        # TODO: Extend for MPI and GPU checks as part of standard runtests.jl
                        file0 = joinpath(
                            output_loc,
                            "gen_test_output/output_active/day0.0.hdf5",
                        )
                        f0 = CC.InputOutput.HDF5Reader(
                            file0,
                            ClimaComms.SingletonCommsContext(),
                        )
                        f0Y = CC.InputOutput.read_field(f0, "Y")

                        file1 = joinpath(
                            output_loc,
                            "gen_test_output/output_active/day0.3.hdf5",
                        )
                        file2 = joinpath(
                            output_loc,
                            "restart_test_output/output_active/day0.3.hdf5",
                        )
                        f1 = CC.InputOutput.HDF5Reader(
                            file1,
                            ClimaComms.SingletonCommsContext(),
                        )
                        f2 = CC.InputOutput.HDF5Reader(
                            file2,
                            ClimaComms.SingletonCommsContext(),
                        )
                        f1Y = CC.InputOutput.read_field(f1, "Y")
                        f2Y = CC.InputOutput.read_field(f2, "Y")
                        close(f1)
                        close(f2)
                        FT = eltype(f1Y)
                        @info "Check file-read from checkpoint data"
                        @test maximum(abs.(parent(f1Y.c) .- parent(f2Y.c))) <=
                              sqrt(eps(FT))
                        @test maximum(abs.(parent(f1Y.f) .- parent(f2Y.f))) <=
                              sqrt(eps(FT))
                        # TODO: Check cache variables upon restart
                        # TODO: Check behaviour of callbacks upon restart
                    end
                end
            end
        end
    end
end
