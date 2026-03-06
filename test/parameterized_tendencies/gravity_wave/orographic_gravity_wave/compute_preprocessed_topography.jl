import ClimaAtmos as CA

include(
    joinpath(
        pkgdir(CA),
        "src/parameterized_tendencies/gravity_wave_drag",
        "preprocess_topography.jl",
    ),
)

import CUDA
import ClimaComms
import ClimaComms.@import_required_backends

comms_ctx = ClimaComms.SingletonCommsContext()
@show ClimaComms.device(comms_ctx)

if !(@isdefined config)
    (; config_file, job_id) = CA.commandline_kwargs()
    config = CA.AtmosConfig(config_file; job_id)
end

config.parsed_args["h_elem"] = 8;
config.parsed_args["topography"] = "Earth";
config.parsed_args["topo_smoothing"] = false;
config.parsed_args["topography_damping_factor"] = 1;
config.parsed_args["orographic_gravity_wave"] = "raw_topo";

params = CA.ClimaAtmosParameters(config)
atmos = CA.get_atmos(config, params)
grid = CA.get_grid(config.parsed_args, params, config.comms_ctx)
spaces = CA.get_spaces(grid)
initial_condition = CA.get_initial_condition(config.parsed_args, atmos)
surface_setup = CA.get_surface_setup(config.parsed_args)
hspace = Spaces.horizontal_space(spaces.center_space)

Y = CA.ICs.atmos_state(
    initial_condition(params),
    atmos,
    spaces.center_space,
    spaces.face_space,
)

parsed_args = config.parsed_args

if parsed_args["topography"] != "Earth"
    error("Topography must be 'Earth', got: $(parsed_args["topography"])")
end

earth_radius = Spaces.topology(hspace).mesh.domain.radius

(; γ, h_frac) = params.orographic_gravity_wave_params

elevation_data =
    CA.AA.earth_orography_file_path(; context = ClimaComms.context(Y.c))

α_smoothing = 0.15  # smoothing scale as fraction of grid resolution

load_preprocessed_topography = false

if load_preprocessed_topography
    (; output_filename, topography, topo_smoothing, topo_damping_factor, h_elem) =
        CA.gen_fn(parsed_args)
    topo_cg = CA.load_preprocessed_topography(
        parsed_args,
    )
else
    topo_cg = CA.compute_OGW_info(
        Y, elevation_data, earth_radius, γ, h_frac;
        α_smoothing,
    )
end

output_filename = write_computed_drag!(topo_cg, parsed_args, config)

datafile_cg, weightfile = save_nc_data(output_filename, topo_cg, spaces)

datafile_rll = remap_nc_data(output_filename)
(; lon, lat, hmax, hmin, t11, t12, t21, t22) = diagnostics(datafile_rll)
plot_diagnostics(lon, lat, hmax, hmin, t11, t12, t21, t22);
