import CUDA
ENV["CLIMACOMMS_DEVICE"] = "CUDA"

import ClimaComms
import ClimaComms.@import_required_backends

using ClimaCore
using ClimaCore.CommonSpaces
import ClimaAtmos as CA
import Thermodynamics as TD
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, Geometry, Operators, Spaces, Grids, Utilities, to_cpu

include("../gw_remap_plot_utils.jl")

const FT = Float64
ᶜgradᵥ = Operators.GradientF2C()
ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)

comms_ctx = ClimaComms.SingletonCommsContext()
@show CUDA.functional()
@show ClimaComms.device(comms_ctx)
# driver
# config_file = "./config/model_configs/plane_schar_mountain_float64_test.yml"

(; config_file, job_id) = CA.commandline_kwargs()
config = CA.AtmosConfig(config_file; job_id, comms_ctx)
config.parsed_args["orographic_gravity_wave"] = "raw_topo"
config.parsed_args["topography"] = "Earth";
(; parsed_args) = config

simulation = CA.get_simulation(config)
p = simulation.integrator.p
Y = simulation.integrator.u

# prepare physical uv input variables for gravity_wave_forcing()
u_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:1
v_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:2

ᶜz = Fields.coordinate_field(Y.c).z

# Create realistic wind profile for OGWD testing (GPU-compatible)
# Strong low-level winds with jet at ~10km and stratospheric decay
function realistic_wind_profile(u_phy, v_phy, z)
    FT = eltype(z)

    # Surface wind
    u_surface = FT(15.0)  # Strong surface wind for wave generation (m/s)

    # Jet parameters
    u_jet = FT(45.0)      # Jet maximum (m/s)
    z_jet = FT(10e3)      # Jet altitude (m)
    z_boundary = FT(2e3)  # Boundary layer top (m)

    # Low-level (0-2 km): Strong, nearly constant for wave launch
    u_low = @. u_surface * (FT(1) + FT(0.15) * z / z_boundary)

    # Mid-levels (2-10 km): Smooth transition to jet
    progress = @. (z - z_boundary) / (z_jet - z_boundary)
    u_mid = @. u_surface + (u_jet - u_surface) * progress^FT(1.5)

    # Upper levels (>10 km): Jet decay into stratosphere
    u_high = @. u_jet * exp(-((z - z_jet) / FT(12e3))^2)

    # Use nested ifelse for GPU compatibility (no branching)
    @. u_phy = ifelse(z < z_boundary, u_low, ifelse(z < z_jet, u_mid, u_high))

    # v-component: Much weaker, with slight turning with height
    @. v_phy = FT(0.08) * u_phy * sin(FT(π) * z / FT(20e3))

    # return u, v
end

# Apply realistic wind profile
realistic_wind_profile(u_phy, v_phy, ᶜz)

# Unpack cache and scratch vars
ᶜT = p.scratch.ᶜtemp_scalar
(; topo_ᶜz_pbl, topo_τ_x, topo_τ_y, topo_τ_l, topo_τ_p, topo_τ_np) =
    p.orographic_gravity_wave
(; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) =
    p.orographic_gravity_wave
(; values_at_z_pbl, topo_info) =
    p.orographic_gravity_wave
(; ᶜdTdz, ᶜbuoyancy_frequency) = p.orographic_gravity_wave
(; ᶜts) = p.precomputed
(; params) = p

# Extract parameters
ogw_params = p.orographic_gravity_wave.ogw_params

grav = CAP.grav(params)
thermo_params = CAP.thermodynamics_params(params)

# compute buoyancy frequency
@. ᶜT = TD.air_temperature(thermo_params, ᶜts)
ᶜdTdz .= Geometry.WVector.(ᶜgradᵥ.(ᶠinterp.(ᶜT))).components.data.:1
@. ᶜbuoyancy_frequency =
    (grav / ᶜT) * (ᶜdTdz + grav / TD.cp_m(thermo_params, ᶜts))
@. ᶜbuoyancy_frequency =
    ifelse(ᶜbuoyancy_frequency < eps(FT), sqrt(eps(FT)), sqrt(abs(ᶜbuoyancy_frequency))) # to avoid small numbers

# Compute base flux
CA.calc_base_flux!(
    topo_τ_x,
    topo_τ_y,
    topo_τ_l,
    topo_τ_p,
    topo_τ_np,
    #
    topo_U_sat,
    topo_FrU_sat,
    topo_FrU_clp,
    topo_FrU_max,
    topo_FrU_min,
    topo_ᶜz_pbl,
    #
    values_at_z_pbl,
    #
    ogw_params,
    topo_info,
    #
    Y.c.ρ,
    u_phy,
    v_phy,
    ᶜz,
    ᶜbuoyancy_frequency,
)

ρ_pbl = values_at_z_pbl.:1
u_pbl = values_at_z_pbl.:2
v_pbl = values_at_z_pbl.:3
N_pbl = values_at_z_pbl.:4

#######################################
# PLOTTING SECTION
#######################################

# Move GPU arrays back to CPU for plotting
ρ_pbl_cpu = to_cpu(ρ_pbl)
u_pbl_cpu = to_cpu(u_pbl)
v_pbl_cpu = to_cpu(v_pbl)
N_pbl_cpu = to_cpu(N_pbl)
ᶜz_cpu = to_cpu(ᶜz)
Y_cpu = to_cpu(Y)
ᶜspace = axes(Y_cpu.c)

ENV["GKSwstype"] = "nul"
output_dir = "orographic_gravity_wave_test_3d"

# REFACTORED: Replace lines 281-396 with this:

# Prepare field data dictionary
field_data = Dict(
    "ρ_pbl" => ρ_pbl_cpu,
    "u_pbl" => u_pbl_cpu,
    "v_pbl" => v_pbl_cpu,
    "N_pbl" => N_pbl_cpu,
    "z_3d" => ᶜz_cpu,
)

comparison_panels = [
    PlotPanel("ρ_pbl", "ρ_pbl", (1, 1)),
    PlotPanel("u_pbl", "u_pbl", (1, 2)),
    PlotPanel("v_pbl", "v_pbl", (2, 1)),
    PlotPanel("N_pbl", "N_pbl", (2, 2)),
]

# Configure plots
config = PlotConfig(
    plot_mode = :horizontal_slice,
    # contour_levels = range(-10, 10; length = 20),
    contour_levels = 20,
    nlat = 90,
    nlon = 180,
    yreversed = false,
)

figure_specs = Dict(
    "everything" => comparison_panels,
)

create_figure_set(
    output_dir,
    collect(keys(field_data)),
    field_data,
    Y_cpu,
    ᶜspace,
    figure_specs,
    config;
    remap_dir = joinpath(@__DIR__, "ogwd_3d", "remap_data/"),
    FT = FT,
)
