import ClimaAtmos as CA

# ("Explicit is better than implicit" -- the Zen of Python)

# Define some units
m = 1.0; km = 1000m; sec = 1.; min = 60sec; hour = 60min; hours = hour; day=24hour; days = day;

# Set up our the computational domain, a column
domain = CA.ExponentiallyStretchedColumn(z_elem = 45,
                                         dz_bottom = 30m,
                                         dz_top = 5km,
                                         z_max = 30km)

initial_data = CA.InitialConditions.IsothermalProfile()


# Set up the model
model = CA.AtmosModel(domain, initial_data;
                      params = CA.default_parameter_set(), # This is the default choice
                      moisture_model = CA.DryModel(),
                      energy_form = CA.TotalEnergy(),
                      sfc_temperature_form = CA.ZonallySymmetricSST(),
                      surface_model = CA.PrescribedSurfaceTemperature(),
                      surface_setup = CA.SurfaceConditions.DefaultExchangeCoefficients()
)


# Interacting with the simulation
dt_save_to_disk = 5day
callbacks = [CA.save_to_disk(dt_save_to_disk)]

Δt = 3hours

sim = CA.AtmosSimulation(model, Δt,
                         stop_time = 10days,
                         callbacks = callbacks,
                         output_dir = "output_script_based")

CA.solve!(sim)
