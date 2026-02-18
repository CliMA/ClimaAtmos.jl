import ClimaAtmos as CA
import ClimaCore
import ClimaCore: DataLayouts, Fields, Geometry
import ClimaCore.Fields: Field, FieldVector, field_values
import ClimaCore.DataLayouts: AbstractData
import ClimaCore.Geometry: AxisTensor
import ClimaCore.Spaces: AbstractSpace
import ClimaComms
import ClimaUtilities.OutputPathGenerator: maybe_wait_filesystem
pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
import Logging
import NCDatasets
import YAML
using Test

import Random
Random.seed!(1234)

const device = ClimaComms.device()
const comms_ctx = ClimaComms.context(device)
ClimaComms.init(comms_ctx)

include("restart_utils.jl")

# This test checks that:

# 1. A simulation, saved to a checkpoint, is read back identically (up to some
#   tolerance and excluding those fields that are computed during the
#   calculation of the tendencies)
# 2. A simulation, saved to a previous checkpoint, and read back and evolved to
#   the same time is identical (up to some tolerance)
# 3. ClimaAtmos can automatically detect restarts
#
# This test checks a variety of configurations and spends a long time compiling
# functions. For this reason, the test takes an optional command-line argument
# to only many tests instead of comprehensive test.


"""
    test_restart(test_dict; job_id, comms_ctx, more_ignore = Symbol[])

Test if the restarts are consistent for a simulation defined by the `test_dict` config.

`more_ignore` is a Vector of Symbols that identifies config-specific keys that
have to be ignored when reading a simulation.
"""
function test_restart(test_dict; job_id, comms_ctx, more_ignore = Symbol[])
    ClimaComms.iamroot(comms_ctx) && println("job_id = $(job_id)")

    local_success = true

    simulation = CA.AtmosSimulation(CA.AtmosConfig(test_dict; job_id, comms_ctx))
    CA.solve_atmos!(simulation)

    # Check re-importing the same state
    restart_dir = simulation.output_dir
    @test isfile(joinpath(restart_dir), "day0.3.hdf5")

    # Reset random seed for RRTMGP
    Random.seed!(1234)

    ClimaComms.iamroot(comms_ctx) && println("    just reading data")
    config_should_be_same = CA.AtmosConfig(
        merge(test_dict, Dict("detect_restart_file" => true));
        job_id,
        comms_ctx,
    )

    simulation_restarted = CA.AtmosSimulation(config_should_be_same)

    if pkgversion(CA.RRTMGP) < v"0.22"
        # Versions of RRTMGP older than 0.22 have a bug and do not set the
        # flux_dn_dir, so that face_clear_sw_direct_flux_dn and
        # face_sw_direct_flux_dn is uninitialized and not deterministic
        rrtmgp_clear_fix =
            (:face_clear_sw_direct_flux_dn, :face_sw_direct_flux_dn)
    else
        rrtmgp_clear_fix = ()
    end

    local_success &= compare(
        simulation.integrator.u,
        simulation_restarted.integrator.u;
        name = "integrator.u",
    )
    local_success &= compare(
        axes(simulation.integrator.u.c),
        axes(simulation_restarted.integrator.u.c);
        name = "space",
    )
    local_success &= compare(
        simulation.integrator.p,
        simulation_restarted.integrator.p;
        name = "integrator.p",
        ignore = Set([
            :ghost_buffer,
            :hyperdiffusion_ghost_buffer,
            :scratch,
            :output_dir,
            :ghost_buffer,
            # Computed in tendencies (which are not computed in this case)
            :hyperdiff,
            # rc is some CUDA/CuArray internal object that we don't care about
            :rc,
            # DataHandlers contains caches, so they are stateful
            :data_handler,
            # Scratch field for microphysics NamedTuple results
            :ᶜmp_tendency,
            # Covariance fields depend on scratch state
            :ᶜT′T′,
            :ᶜq′q′,
            # RRTMGP internal arrays may differ due to RNG state
            rrtmgp_clear_fix...,
            # Config-specific
            more_ignore...,
        ]),
    )

    # Check re-importing from previous state and advancing one step
    ClimaComms.iamroot(comms_ctx) && println("    reading and simulating")
    # Reset random seed for RRTMGP
    Random.seed!(1234)

    restart_file = joinpath(simulation.output_dir, "day0.2.hdf5")
    @test isfile(joinpath(restart_dir), "day0.2.hdf5")
    # Restart from specific file
    config2 = CA.AtmosConfig(
        merge(test_dict, Dict("restart_file" => restart_file));
        job_id,
        comms_ctx,
    )

    simulation_restarted2 = CA.AtmosSimulation(config2)
    CA.fill_with_nans!(simulation_restarted2.integrator.p)

    CA.solve_atmos!(simulation_restarted2)
    local_success &= compare(
        simulation.integrator.u,
        simulation_restarted2.integrator.u;
        name = "integrator.u",
    )
    local_success &= compare(
        simulation.integrator.p,
        simulation_restarted2.integrator.p;
        name = "integrator.p",
        ignore = Set([
            :scratch,
            :output_dir,
            :ghost_buffer,
            :hyperdiffusion_ghost_buffer,
            :data_handler,
            :rc,
            # Scratch field for microphysics NamedTuple results
            :ᶜmp_tendency,
            # Covariance fields depend on scratch state
            :ᶜT′T′,
            :ᶜq′q′,
            # RRTMGP internal arrays are not deterministic through fill_with_nans!
            rrtmgp_clear_fix...,
        ]),
    )

    return (
        local_success,
        simulation,
        simulation_restarted,
        simulation_restarted2,
    )
end

# Begin tests

# Let's prepare the test_dicts. TESTING is a Vector of NamedTuples, each element
# has a test_dict, a job_id, and a more_ignore

TESTING = Any[]

# Add a configuration with all the bells and whistles
if MANYTESTS
    if comms_ctx isa ClimaComms.SingletonCommsContext
        configurations = ["sphere", "box", "column"]
    else
        configurations = ["sphere", "box"]
    end

    for configuration in configurations
        if configuration == "sphere"
            moistures = ["nonequil"]
            precips = ["1M"]
            topography = "Earth"
            turbconv_models = [nothing, "diagnostic_edmfx"]
            # turbconv_models = ["prognostic_edmfx"]
            radiations = [nothing, "allsky"]
        else
            moistures = ["equil"]
            precips = ["0M"]
            topography = "NoWarp"
            turbconv_models = ["diagnostic_edmfx"]
            radiations = ["gray", "allskywithclear"]
        end

        for turbconv_mode in turbconv_models
            for radiation in radiations
                for moisture in moistures
                    for precip in precips
                        if !isnothing(turbconv_mode)
                            # EDMF only supports equilibrium moisture
                            if occursin("edmfx", turbconv_mode)
                                moisture == "equil" || continue
                            end
                        end

                        # The `enable_bubble` case is broken for ClimaCore < 0.14.6, so we
                        # hard-code this to be always false for those versions
                        bubble = pkgversion(ClimaCore) > v"0.14.5"

                        # Make sure that all MPI processes agree on the output_loc
                        output_loc =
                            ClimaComms.iamroot(comms_ctx) ? mktempdir(pwd()) :
                            ""
                        output_loc = ClimaComms.bcast(comms_ctx, output_loc)
                        # Sometimes the shared filesystem doesn't work properly
                        # and the folder is not synced across MPI processes.
                        # Let's add an additional check here.
                        maybe_wait_filesystem(comms_ctx, output_loc)

                        job_id = "$(configuration)_$(moisture)_$(precip)_$(topography)_$(radiation)_$(turbconv_mode)"
                        test_dict = Dict(
                            "test_dycore_consistency" => true, # We will add NaNs to the cache, just to make sure
                            "reproducible_restart" => true,
                            "check_nan_every" => 3,
                            "log_progress" => false,
                            "moist" => moisture,
                            "precip_model" => precip,
                            "config" => configuration,
                            "topography" => topography,
                            "turbconv" => turbconv_mode,
                            "dt" => "1secs",
                            "bubble" => bubble,
                            "viscous_sponge" => true,
                            "rayleigh_sponge" => true,
                            "insolation" => "timevarying",
                            "rad" => radiation,
                            "dt_rad" => "1secs",
                            "surface_setup" => "DefaultMoninObukhov",
                            "radiation_reset_rng_seed" => true,
                            "t_end" => "3secs",
                            "dt_save_state_to_disk" => "1secs",
                            "enable_diagnostics" => false,
                            "output_dir" => joinpath(output_loc, job_id),
                        )
                        push!(
                            TESTING,
                            (; test_dict, job_id, more_ignore = Symbol[]),
                        )
                    end
                end
            end
        end
    end
else
    amip_output_loc = ClimaComms.iamroot(comms_ctx) ? mktempdir(pwd()) : ""
    amip_output_loc = ClimaComms.bcast(comms_ctx, amip_output_loc)
    # Sometimes the shared filesystem doesn't work properly and the folder is
    # not synced across MPI processes. Let's add an additional check here.
    maybe_wait_filesystem(comms_ctx, amip_output_loc)

    amip_job_id = "amip_target_diagedmf"

    amip_test_dict = merge(
        YAML.load_file(
            joinpath(
                @__DIR__,
                "../config/common_configs/numerics_sphere_he16ze63.yml",
            ),
        ),
        YAML.load_file(
            joinpath(
                @__DIR__,
                "../config/longrun_configs/amip_target_diagedmf.yml",
            ),
        ),
        Dict(
            "h_elem" => 4,
            "z_elem" => 15,
            "test_dycore_consistency" => true, # We will add NaNs to the cache, just to make sure
            "reproducible_restart" => true,
            "check_nan_every" => 3,
            "log_progress" => false,
            "dt" => "1secs",
            "dt_rad" => "1secs", "t_end" => "3secs",
            "dt_save_state_to_disk" => "1secs",
            "output_dir" => joinpath(amip_output_loc, amip_job_id),
            "dt_cloud_fraction" => "1secs",
            "rad" => "allskywithclear",
            "radiation_reset_rng_seed" => true,
            "toml" => [
                joinpath(@__DIR__, "../toml/longrun_aquaplanet_diagedmf.toml"),
            ],
        ),
    )

    push!(
        TESTING,
        (;
            test_dict = amip_test_dict,
            job_id = amip_job_id,
            more_ignore = Symbol[],
        ),
    )
end

# We know that this test is broken for old versions of ClimaCore
@test all(
    @time test_restart(t.test_dict; comms_ctx, t.job_id, t.more_ignore)[1] for
    t in TESTING
) skip = pkgversion(ClimaCore) < v"0.14.18"
