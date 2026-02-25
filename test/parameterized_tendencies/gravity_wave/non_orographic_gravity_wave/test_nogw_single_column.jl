import ClimaComms
ClimaComms.@import_required_backends

using NCDatasets
using Dates
using Statistics
import Interpolations

import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry

include("../gw_plotutils.jl")

const FT = Float64

# single column test Figure 6 of the Alexander and Dunkerton (1999) paper:
# https://journals.ametsoc.org/view/journals/atsc/56/24/1520-0469_1999_056_4167_aspomf_2.0.co_2.xml?tab_body=pdf
# zonal mean monthly wind, temperature 1958-1973; at 40N in latitude for Jan, April, July, Oct.

# Set up ClimaAtmos infrastructure
comms_ctx = ClimaComms.SingletonCommsContext()
config_file = joinpath(
    @__DIR__,
    "../../../../config/model_configs/single_column_nonorographic_gravity_wave.yml",
)
config = CA.AtmosConfig(config_file; job_id = "nogw_single_column_test", comms_ctx)

# Build simulation
simulation = CA.get_simulation(config)
p = simulation.integrator.p
Y = simulation.integrator.u

# Extract physical constants from parameters
(; params) = p
grav = CAP.grav(params)
R_d = CAP.R_d(params)
cp_d = CAP.cp_d(params)

# Extract NOGW cache from simulation
(;
    gw_source_height,
    gw_ncval,
    ᶜbuoyancy_frequency,
    ᶜlevel,
    u_waveforcing,
    v_waveforcing,
    uforcing,
    vforcing,
) = p.non_orographic_gravity_wave

# Get spaces and coordinate fields
center_space = axes(Y.c)
ᶜz = Fields.coordinate_field(Y.c).z

# Compute source_level and damp_level based on height
center_z = Array(Fields.field2array(ᶜz))[:, 1]
source_level = argmin(abs.(center_z .- gw_source_height))
damp_level = Spaces.nlevels(center_space)

# Load ERA5 data from artifacts
include(joinpath(@__DIR__, "../../../artifact_funcs.jl"))
era_data = joinpath(era_single_column_dataset_path(), "box-single_column_test.nc")

nt = NCDataset(era_data) do ds
    # Dimensions:  longitude × latitude × level × time
    # data at 40N, all longitude, all levels, all time
    lon = Array(ds["longitude"])
    lat = Array(ds["latitude"])
    lev = Array(ds["level"]) .* 100
    time = Array(ds["time"])
    gZ = ds["z"][:, 5, :, :]
    T = ds["t"][:, 5, :, :]
    u = ds["u"][:, 5, :, :]
    (; lon, lat, lev, time, gZ, T, u)
end
(; lon, lat, lev, time, gZ, T, u) = nt

# Compute density and buoyancy frequency from ERA5 data
Z = gZ ./ grav
ρ = ones(size(T)) .* reshape(lev, (1, length(lev), 1)) ./ T / R_d

dTdz = zeros(size(T))
@. dTdz[:, 1, :] = (T[:, 2, :] - T[:, 1, :]) / (Z[:, 2, :] - Z[:, 1, :])
@. dTdz[:, end, :] =
    (T[:, end, :] - T[:, end - 1, :]) / (Z[:, end, :] - Z[:, end - 1, :])
@. dTdz[:, 2:(end - 1), :] =
    (T[:, 3:end, :] - T[:, 1:(end - 2), :]) /
    (Z[:, 3:end, :] - Z[:, 1:(end - 2), :])
bf = @. (grav / T) * (dTdz + grav / cp_d)
bf = @. ifelse(bf < 2.5e-5, sqrt(2.5e-5), sqrt(abs(bf)))

# Interpolation to center_z grid
center_u = zeros(length(lon), length(center_z), length(time))
center_bf = zeros(length(lon), length(center_z), length(time))
center_ρ = zeros(length(lon), length(center_z), length(time))
for i in 1:length(lon)
    for it in 1:length(time)
        interp_linear = Interpolations.LinearInterpolation(
            Z[i, :, it][end:-1:1],
            u[i, :, it][end:-1:1],
            extrapolation_bc = Interpolations.Line(),
        )
        center_u[i, :, it] = interp_linear.(center_z)

        interp_linear = Interpolations.LinearInterpolation(
            Z[i, :, it][end:-1:1],
            bf[i, :, it][end:-1:1],
            extrapolation_bc = Interpolations.Line(),
        )
        center_bf[i, :, it] = interp_linear.(center_z)

        interp_linear = Interpolations.LinearInterpolation(
            Z[i, :, it][end:-1:1],
            ρ[i, :, it][end:-1:1],
            extrapolation_bc = Interpolations.Line(),
        )
        center_ρ[i, :, it] = interp_linear.(center_z)
    end
end

# Zonal mean
center_u_mean = mean(center_u, dims = 1)[1, :, :]
center_bf_mean = mean(center_bf, dims = 1)[1, :, :]
center_ρ_mean = mean(center_ρ, dims = 1)[1, :, :]

# Monthly averaging for Jan, April, July, Oct
month = Dates.month.(time)

ENV["GKSwstype"] = "nul"
output_dir = "nonorographic_gravity_wave_test_single_column"
mkpath(output_dir)

# Get state fields from simulation
ᶜρ = Y.c.ρ
ᶜu = similar(ᶜρ, FT)
ᶜv = similar(ᶜρ, FT)

# Jan
Jan_u = mean(center_u_mean[:, month .== 1], dims = 2)[:, 1]
Jan_bf = mean(center_bf_mean[:, month .== 1], dims = 2)[:, 1]
Jan_ρ = mean(center_ρ_mean[:, month .== 1], dims = 2)[:, 1]
parent(ᶜρ) .= Jan_ρ
parent(ᶜu) .= Jan_u
parent(ᶜv) .= 0
parent(ᶜbuoyancy_frequency) .= Jan_bf
ᶜρ_source = Fields.level(ᶜρ, source_level)
ᶜu_source = Fields.level(ᶜu, source_level)
ᶜv_source = Fields.level(ᶜv, source_level)
uforcing .= 0
vforcing .= 0

CA.non_orographic_gravity_wave_forcing(
    ᶜu,
    ᶜv,
    ᶜbuoyancy_frequency,
    ᶜρ,
    ᶜz,
    ᶜlevel,
    source_level,
    damp_level,
    ᶜρ_source,
    ᶜu_source,
    ᶜv_source,
    uforcing,
    vforcing,
    gw_ncval,
    u_waveforcing,
    v_waveforcing,
    p,
)
Jan_uforcing = Array(Fields.field2array(uforcing))[:, 1]
fig = generate_empty_figure()
create_plot!(
    fig;
    X = Jan_uforcing[source_level:(end - 1)] * 86400,
    Y = center_z[source_level:(end - 1)],
    label = "Jan",
)
CairoMakie.save(joinpath(output_dir, "fig6jan.png"), fig)

# April
April_u = mean(center_u_mean[:, month .== 4], dims = 2)[:, 1]
April_bf = mean(center_bf_mean[:, month .== 4], dims = 2)[:, 1]
April_ρ = mean(center_ρ_mean[:, month .== 4], dims = 2)[:, 1]
parent(ᶜρ) .= April_ρ
parent(ᶜu) .= April_u
parent(ᶜv) .= 0
parent(ᶜbuoyancy_frequency) .= April_bf
uforcing .= 0
vforcing .= 0

CA.non_orographic_gravity_wave_forcing(
    ᶜu,
    ᶜv,
    ᶜbuoyancy_frequency,
    ᶜρ,
    ᶜz,
    ᶜlevel,
    source_level,
    damp_level,
    ᶜρ_source,
    ᶜu_source,
    ᶜv_source,
    uforcing,
    vforcing,
    gw_ncval,
    u_waveforcing,
    v_waveforcing,
    p,
)
April_uforcing = Array(Fields.field2array(uforcing))[:, 1]
fig = generate_empty_figure()
create_plot!(
    fig;
    X = April_uforcing[source_level:(end - 1)] * 86400,
    Y = center_z[source_level:(end - 1)],
    label = "Apr",
)
CairoMakie.save(joinpath(output_dir, "fig6apr.png"), fig)

# July
July_u = mean(center_u_mean[:, month .== 7], dims = 2)[:, 1]
July_bf = mean(center_bf_mean[:, month .== 7], dims = 2)[:, 1]
July_ρ = mean(center_ρ_mean[:, month .== 7], dims = 2)[:, 1]
parent(ᶜρ) .= July_ρ
parent(ᶜu) .= July_u
parent(ᶜv) .= 0
parent(ᶜbuoyancy_frequency) .= July_bf
uforcing .= 0
vforcing .= 0

CA.non_orographic_gravity_wave_forcing(
    ᶜu,
    ᶜv,
    ᶜbuoyancy_frequency,
    ᶜρ,
    ᶜz,
    ᶜlevel,
    source_level,
    damp_level,
    ᶜρ_source,
    ᶜu_source,
    ᶜv_source,
    uforcing,
    vforcing,
    gw_ncval,
    u_waveforcing,
    v_waveforcing,
    p,
)
July_uforcing = Array(Fields.field2array(uforcing))[:, 1]
fig = generate_empty_figure()
create_plot!(
    fig;
    X = July_uforcing[source_level:(end - 1)] * 86400,
    Y = center_z[source_level:(end - 1)],
    label = "Jul",
)
CairoMakie.save(joinpath(output_dir, "fig6jul.png"), fig)

# Oct
Oct_u = mean(center_u_mean[:, month .== 10], dims = 2)[:, 1]
Oct_bf = mean(center_bf_mean[:, month .== 10], dims = 2)[:, 1]
Oct_ρ = mean(center_ρ_mean[:, month .== 10], dims = 2)[:, 1]
parent(ᶜρ) .= Oct_ρ
parent(ᶜu) .= Oct_u
parent(ᶜv) .= 0
parent(ᶜbuoyancy_frequency) .= Oct_bf
uforcing .= 0
vforcing .= 0

CA.non_orographic_gravity_wave_forcing(
    ᶜu,
    ᶜv,
    ᶜbuoyancy_frequency,
    ᶜρ,
    ᶜz,
    ᶜlevel,
    source_level,
    damp_level,
    ᶜρ_source,
    ᶜu_source,
    ᶜv_source,
    uforcing,
    vforcing,
    gw_ncval,
    u_waveforcing,
    v_waveforcing,
    p,
)
Oct_uforcing = Array(Fields.field2array(uforcing))[:, 1]
fig = generate_empty_figure()
create_plot!(
    fig;
    X = Oct_uforcing[source_level:(end - 1)] * 86400,
    Y = center_z[source_level:(end - 1)],
    label = "Oct",
)
CairoMakie.save(joinpath(output_dir, "fig6oct.png"), fig)
