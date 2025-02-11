import ClimaAtmos as CA
import ClimaUtilities
import ClimaUtilities: TimeVaryingInputs
using Dates
import ClimaCoreTempestRemap
import ClimaCore.Fields
using NCDatasets

cd("../../")
config = CA.AtmosConfig("config/model_configs/prognostic_edmfx_gcmdriven_column.yml")
simulation = CA.get_simulation(config)


ds = NCDataset("/home/jschmitt/ClimaAtmos.jl/examples/tvforcing/sim_forcing_site23_ng_xyz.nc")


start_time = DateTime(2007, 7, 1)
tv = TimeVaryingInputs.TimeVaryingInput(["/home/jschmitt/ClimaAtmos.jl/examples/tvforcing/sim_forcing_site23_ng_xyz.nc"],
                                ["ta"],
                                axes(similar(simulation.integrator.u.c.ρ));
                                reference_date = start_time,
                                regridder_type = :InterpolationsRegridder,
);

