using Test
using ClimaAtmos
import ClimaTimeSteppers as CTS
import ClimaComms
using ClimaAtmos.InitialConditions
using ClimaAtmos.Parameters

@testset "Composability" begin
    @testset "Column Simulation" begin
        FT = Float64
        params = ClimaAtmosParameters(FT)
        comms_ctx = ClimaComms.SingletonCommsContext()

        model = AtmosModel(FT, moisture_model = ClimaAtmos.DryModel())

        domain = ColumnDomain(
            FT,
            z_elem = 10,
            z_max = 1000.0,
            z_stretch = false,
            dz_bottom = 100.0,
        )

        initial_condition = IsothermalProfile()

        simulation = AtmosSimulation(
            model,
            domain,
            initial_condition,
            params,
            comms_ctx;
            dt = 10.0,
            t_span = (0.0, 10.0),
            ode_algo_type = CTS.IMEXEuler,
        )

        @test simulation isa AtmosSimulation
        @test simulation.integrator.t == 0.0
        @test simulation.integrator.sol.prob.tspan == (0.0, 10.0)
    end

    @testset "Sphere Simulation" begin
        FT = Float64
        params = ClimaAtmosParameters(FT)
        comms_ctx = ClimaComms.context(ClimaComms.CPUSingleThreaded())
        ClimaComms.init(comms_ctx)

        model = AtmosModel(
            FT,
            moisture_model = ClimaAtmos.EquilMoistModel(),
            radiation_mode = ClimaAtmos.HeldSuarezForcing(),
        )

        domain = SphereDomain(
            FT,
            radius = 6.371229e6,
            h_elem = 4,
            nh_poly = 4,
            z_elem = 10,
            z_max = 40000.0,
            z_stretch = true,
            dz_bottom = 1000.0,
            bubble = false,
            deep_atmosphere = false,
        )

        initial_condition = DryBaroclinicWave()

        simulation = AtmosSimulation(
            model,
            domain,
            initial_condition,
            params,
            comms_ctx;
            dt = 300.0,
            t_span = (0.0, 300.0),
            ode_algo_type = CTS.IMEXEuler,
        )

        @test simulation isa AtmosSimulation
        @test simulation.integrator.t == 0.0
        @test simulation.integrator.sol.prob.tspan == (0.0, 300.0)
    end
end
