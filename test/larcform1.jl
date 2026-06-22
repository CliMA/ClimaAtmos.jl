using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaAtmos.Setups as Setups
import AtmosphericProfilesLibrary as APL
import Thermodynamics as TD

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

    # Surface condition: temperature initialized to 250 K
    surf_cond = Setups.surface_condition(setup, simulation.integrator.p.params)
    T_surface = surf_cond.temperature.f()  # Returns(250) function
    @test T_surface ≈ FT(250) atol = 1e-3

    # Temperature: dry adiabatic lapse rate γ = 8e-3 K/m in troposphere (Pithan 2016 Table 1)
    z_mid = FT(LC.z_tropopause / 2)
    @test profs.T(z_mid) ≈ FT(LC.T_0 - LC.γ * z_mid) rtol = 1e-4

    # Temperature: isothermal above tropopause (two heights)
    @test profs.T(FT(LC.z_tropopause) + 1000) ≈ FT(LC.T_tropopause) atol = 0.1
    @test profs.T(FT(LC.z_tropopause) + 3000) ≈ FT(LC.T_tropopause) atol = 0.1

    # RH profile: 80% at surface, 20% at 600 hPa (Pithan 2016 Table 1, wrt liquid)
    # Pithan 2016 Table 1 specifies RH wrt liquid throughout, so we reconstruct wrt liquid
    # explicitly: invert q_tot = ε*e/(p - e + ε*e) to get e, then RH = e/e_sat_liquid(T)
    function rh_liquid_from_profs(z)
        T = profs.T(z)
        p = profs.p(z)
        q = profs.q_tot(z)
        Rv_over_Rd = TD.Parameters.Rv_over_Rd(thermo_params)
        ε = 1 / Rv_over_Rd
        e = q * p / (ε * (1 - q) + q)
        e_sat = TD.saturation_vapor_pressure(thermo_params, T, TD.Liquid())
        e / e_sat
    end
    @test rh_liquid_from_profs(FT(0)) ≈ FT(0.8) rtol = 0.01
    # Height of 600 hPa level via analytic pressure profile (Table 1 knee point)
    z_600hPa = FT((LC.T_0 / LC.γ) * (1 - (60000 / LC.P_0)^LC.α))
    @test rh_liquid_from_profs(z_600hPa) ≈ FT(0.2) rtol = 0.02

    # Meridional wind: zero everywhere (Pithan 2016 Table 1, v_geo = 0)
    @test profs.v(FT(LC.z_tropopause) + 500) ≈ FT(0)

    # Pressure: monotonically decreasing with height
    zs = range(FT(0), FT(LC.z_tropopause) + 2000, length = 20)
    ps = [profs.p(z) for z in zs]
    @test issorted(ps; rev = true)

    # Surface flux scheme: roughness z0m = 1e-3 m for sea ice (Pithan 2016 Table 3)
    @test surf_cond.flux_scheme.z0m ≈ FT(1e-3) rtol = 1e-6

    # Surface q_vap: liquid saturation at T_surface = 250 K, p_surface = P_0
    p_v_sat = TD.saturation_vapor_pressure(thermo_params, FT(250), TD.Liquid())
    ϵ_v = TD.Parameters.R_d(thermo_params) / TD.Parameters.R_v(thermo_params)
    p_surface = FT(LC.P_0)
    q_vap_expected = ϵ_v * p_v_sat / (p_surface - p_v_sat * (1 - ϵ_v))
    @test surf_cond.overrides.q_vap ≈ q_vap_expected rtol = 1e-4  # Float32 precision
end
