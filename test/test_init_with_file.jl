import ClimaAtmos as CA

simulation = CA.get_simulation(
    CA.AtmosConfig(
        Dict(
            "initial_condition" => "artifact\"DYAMOND_SUMMER_ICS_p98deg\"/DYAMOND_SUMMER_ICS_p98deg.nc",
            "moist" => "equil",
        ),
        job_id = "test_init_with_file_dyamond",
    ),
)

# Just a small test to see that we got here
@test maximum(simulation.integrator.u.c.ρ) > 0
