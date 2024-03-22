redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import Random
Random.seed!(1234)
import ClimaAtmos as CA
import Profile, ProfileCanvas
import SciMLBase

length(ARGS) != 1 && error("Usage: longer_flame.jl <config_file>")
config_file = ARGS[1]
config = CA.AtmosConfig(config_file)
FT = eltype(config)
# 1 day is a good compromise to go through most of the functions and diagnostics with almost
# realistic cadence. For diagnostics, production runs will probably have a monthly output,
# so this is a worst case to optimize for.
config.parsed_args["t_end"] = "1days"
simulation = CA.get_simulation(config)

# First, we run the simulation for a few timesteps to compile most of the functions
# (including top level function such as timed_solve!)

simulation_dt = FT(CA.time_to_seconds(config.parsed_args["dt"]))
# Change the final time to 5dt
SciMLBase.reinit!(simulation.integrator, tf = 5simulation_dt)

# We force the compilation of all callbacks to also include radiation
CA.call_all_callbacks!(simulation.integrator)
# Solve for a little bit
CA.timed_solve!(simulation.integrator)

# Change the final time to 1day
SciMLBase.add_tstop!(simulation.integrator, FT(86400))

@info "$simulation"

@info "Collecting profile"
Profile.init(n = 10^7, delay = 0.1)
prof = Profile.@profile CA.timed_solve!(simulation.integrator)
results = Profile.fetch()

flame_path = joinpath(simulation.output_dir, "flame.html")

ProfileCanvas.html_file(flame_path, results)
@info "Flame saved in $flame_path"
