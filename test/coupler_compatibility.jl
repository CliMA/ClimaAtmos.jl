using Test
import ClimaComms
ClimaComms.@import_required_backends
import Thermodynamics as TD
import SurfaceFluxes as SF
import StaticArrays as SA
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaAtmos.SurfaceConditions: projected_vector_data, CT1, CT2
import ClimaCore: Spaces, Fields
import ClimaCore.Utilities: half

# TODO: Add surface flux exchange tests after new SurfaceFluxes.jl is integrated

@testset "Coupler Initialization" begin
    # Verify PrescribedSurface works with RRTMGP and diagnostic EDMF.
    # Also verify non-zero t_start works.
    config = CA.AtmosConfig(
        Dict(
            "surface_setup" => "PrescribedSurface",
            "moist" => "equil",
            "rad" => "clearsky",
            "co2_model" => "fixed",
            "turbconv" => "diagnostic_edmfx",
            "output_default_diagnostics" => false,
            "t_start" => "1secs",
        );
        job_id = "coupler_compatibility3",
    )
    simulation = CA.AtmosSimulation(config)
end
