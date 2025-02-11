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


# ds = NCDataset("/home/jschmitt/ClimaAtmos.jl/examples/tvforcing/sim_forcing_site23_ng_xyz.nc")


start_time = DateTime(2007, 7, 1)
tv = TimeVaryingInputs.TimeVaryingInput(["examples/tvforcing/sim_forcing_site23_ng_xyz.nc"],
                                ["ta"],
                                axes(similar(simulation.integrator.u.c.ρ));
                                reference_date = start_time,
                                regridder_type = :InterpolationsRegridder,
);

import ClimaCore: Fields
FT = Float32
tv = TimeVaryingInputs.TimeVaryingInput(["examples/tvforcing/sim_forcing_site23_ng_xyz.nc"],
                                ["coszen"],
                                axes(similar(Fields.level(simulation.integrator.u.c.ρ, 1), FT));
                                reference_date = start_time,
                                regridder_type = :InterpolationsRegridder,
);
F1 = zero(Fields.level(simulation.integrator.u.c.ρ, 1))
next_t = start_time + Dates.Second(1000)
next_t
TimeVaryingInputs.evaluate!(F1, tv, next_t)
