import ClimaAtmos as CA

simulation = CA.AtmosSimulation(
    CA.AtmosConfig(
        Dict(
            "initial_condition" => "artifact\"DYAMOND_SUMMER_ICS_p98deg\"/DYAMOND_SUMMER_ICS_p98deg.nc",
            "microphysics_model" => "0M",
        ),
        job_id = "test_init_with_file_dyamond",
    ),
)

# Just a small test to see that we got here
@test maximum(simulation.integrator.u.c.ρ) > 0
