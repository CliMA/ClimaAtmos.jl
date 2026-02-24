import ClimaAtmos as CA
import ClimaCore.InputOutput
include(
    joinpath(
        pkgdir(CA),
        "src/parameterized_tendencies/gravity_wave_drag",
        "preprocess_topography.jl",
    ),
)
include("../gw_remap_plot_utils.jl")
include("ogw_test_utils.jl")

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

params = CA.ClimaAtmosParameters(config)
atmos = CA.get_atmos(config, params)
grid = CA.get_grid(config.parsed_args, params, config.comms_ctx)
spaces = CA.get_spaces(grid)
initial_condition = CA.get_initial_condition(config.parsed_args, atmos)

parsed_args = config.parsed_args

(; output_filename) = CA.gen_fn(parsed_args)


computed_drag = load_computed_drag(parsed_args, comms_ctx)

#########################################

Y = CA.ICs.atmos_state(
    initial_condition(params),
    atmos,
    spaces.center_space,
    spaces.face_space,
)

# Initialize cache vars for orographic gravity wave
γ = 0.4
ϵ = 0.0
β = 0.5
h_frac = 0.1
ρscale = 1.2
L0 = 80000.0
a0 = 0.9
a1 = 3.0
Fr_crit = 0.7
topo_info = Val(:gfdl_restart)
topography = Val(:Earth)
FT = eltype(Y.c.ρ)
ogw = CA.FullOrographicGravityWave{FT, typeof(topo_info), typeof(topography)}(;
    γ,
    ϵ,
    β,
    h_frac,
    ρscale,
    L0,
    a0,
    a1,
    Fr_crit,
    topo_info,
    topography,
)

topo_info = CA.get_topo_info(Y, ogw)

###########################################

ENV["GKSwstype"] = "nul"
output_dir = "orographic_gravity_wave_test_computed_drag"

field_data = Dict(
    "gfdl_hmax" => topo_info.hmax,
    "gfdl_hmin" => topo_info.hmin,
    "gfdl_t11" => topo_info.t11,
    "gfdl_t12" => topo_info.t12,
    "gfdl_t21" => topo_info.t21,
    "gfdl_t22" => topo_info.t22,
    "computed_drag_hmax" => computed_drag.hmax,
    "computed_drag_hmin" => computed_drag.hmin,
    "computed_drag_t11" => computed_drag.t11,
    "computed_drag_t12" => computed_drag.t12,
    "computed_drag_t21" => computed_drag.t21,
    "computed_drag_t22" => computed_drag.t22,
)

"""
    compute_shared_colorrange(field_data, var_names...)

Compute a shared color range across multiple fields for consistent colormaps.

# Arguments
- `field_data`: Dictionary mapping variable names to field objects
- `var_names...`: Variable arguments of field names to compute range across

# Returns
- Tuple (min_val, max_val) representing the shared color range
"""
function compute_shared_colorrange(field_data, var_names...)
    all_mins = [
        minimum(parent(field_data[name])) for name in var_names if haskey(field_data, name)
    ]
    all_maxs = [
        maximum(parent(field_data[name])) for name in var_names if haskey(field_data, name)
    ]
    return (minimum(all_mins), maximum(all_maxs))
end

# Compute shared color ranges for each row
hmax_range = compute_shared_colorrange(field_data, "gfdl_hmax", "computed_drag_hmax")
hmin_range = compute_shared_colorrange(field_data, "gfdl_hmin", "computed_drag_hmin")
t11_range = compute_shared_colorrange(field_data, "gfdl_t11", "computed_drag_t11")
t12_range = compute_shared_colorrange(field_data, "gfdl_t12", "computed_drag_t12")
t21_range = compute_shared_colorrange(field_data, "gfdl_t21", "computed_drag_t21")
t22_range = compute_shared_colorrange(field_data, "gfdl_t22", "computed_drag_t22")

panels = [
    PlotPanel("gfdl_hmax", "GFDL hmax", (1, 1); colorrange = hmax_range),
    PlotPanel("computed_drag_hmax", "Computed hmax", (1, 2); colorrange = hmax_range),
    PlotPanel("gfdl_hmin", "GFDL hmin", (2, 1); colorrange = hmin_range),
    PlotPanel("computed_drag_hmin", "Computed hmin", (2, 2); colorrange = hmin_range),
    PlotPanel("gfdl_t11", "GFDL t11", (3, 1); colorrange = t11_range),
    PlotPanel("computed_drag_t11", "Computed t11", (3, 2); colorrange = t11_range),
    PlotPanel("gfdl_t12", "GFDL t12", (4, 1); colorrange = t12_range),
    PlotPanel("computed_drag_t12", "Computed t12", (4, 2); colorrange = t12_range),
    PlotPanel("gfdl_t21", "GFDL t21", (5, 1); colorrange = t21_range),
    PlotPanel("computed_drag_t21", "Computed t21", (5, 2); colorrange = t21_range),
    PlotPanel("gfdl_t22", "GFDL t22", (6, 1); colorrange = t22_range),
    PlotPanel("computed_drag_t22", "Computed t22", (6, 2); colorrange = t22_range),
]

plot_config = PlotConfig(
    plot_mode = :horizontal_slice,
    # contour_levels = range(-10, 10; length = 20),
    contour_levels = 20,
    nlat = 90,
    nlon = 180,
    yreversed = false,
    output_format = "pdf",
)

figure_specs = Dict(
    "drag_comparison" => panels,
)

create_figure_set(
    output_dir,
    collect(keys(field_data)),
    field_data,
    Y,
    spaces.center_space,
    figure_specs,
    plot_config;
    remap_dir = joinpath(@__DIR__, "ogwd_3d", "remap_data/"),
    FT = FT,
)
