using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaAtmos.Setups as Setups
import AtmosphericProfilesLibrary as APL

@testset "Larcform1 setup" begin
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "Larcform1",
            "microphysics_model" => "0M",
            "config" => "column",
            "output_default_diagnostics" => false,
        );
        job_id = "larcform1_setup_test",
    )
    simulation = CA.get_simulation(config)
    thermo_params = CAP.thermodynamics_params(simulation.integrator.p.params)
    FT = Float64
    LC = APL.Larcform1_constants

    setup = Setups.Larcform1(; prognostic_tke = true, thermo_params)
    profs = setup.profiles

    # Temperature: surface and tropopause match Pithan 2016 Table 1
    @test profs.T(0) ≈ FT(LC.T_0) atol = 0.1
    @test profs.T(FT(LC.z_tropopause)) ≈ FT(LC.T_tropopause) atol = 0.5

    # Pressure: surface and tropopause match Pithan 2016 Table 1
    @test profs.p(0) ≈ FT(LC.P_0) atol = 10
    @test profs.p(FT(LC.z_tropopause)) ≈ FT(LC.P_tropopause) rtol = 0.01

    # Humidity: moist at surface, fixed q_top above tropopause
    @test profs.q_tot(0) > FT(1e-4)
    @test profs.q_tot(FT(LC.z_tropopause) + 1000) ≈ FT(LC.q_top) rtol = 1e-6

    # Winds: 5 m/s below tropopause, 0 above
    @test profs.u(0) ≈ FT(5)
    @test profs.u(FT(LC.z_tropopause) + 500) ≈ FT(0)
    @test profs.v(0) ≈ FT(0)

    # Coriolis: 2Ω sin(80°N)
    cf = Setups.coriolis_forcing(setup, FT)
    @test cf.coriolis_param ≈ FT(1.432e-4) rtol = 1e-3
    @test cf.prof_ug(0) ≈ FT(5)
end
