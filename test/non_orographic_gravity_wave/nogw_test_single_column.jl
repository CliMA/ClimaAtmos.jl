using NCDatasets
using Dates
using Interpolations
using Statistics
using Plots
import ClimaAtmos
import ClimaAtmos as CA

const FT = Float64
# single column test Figure 6 of the Alexander and Dunkerton (1999) paper:
# https://journals.ametsoc.org/view/journals/atsc/56/24/1520-0469_1999_056_4167_aspomf_2.0.co_2.xml?tab_body=pdf
# zonal mean monthly wind, temperature 1958-1973; at 40N in latitude for Jan, April, July, Oct.

face_z = FT.(0:1e3:0.5e5)
center_z = FT(0.5) .* (face_z[1:(end - 1)] .+ face_z[2:end])
model_config = CA.SingleColumnModel()

# compute the source parameters
function non_orographic_gravity_wave_cache(
    ::Type{FT};
    source_height = FT(15000),
    Bm = FT(1.2),
    F_S0 = FT(4e-3),
    dc = FT(0.6),
    cmax = FT(99.6),
    c0 = FT(0),
    kwv = FT(2π / 100e5),
    cw = FT(40.0),
) where {FT}

    nc = Int(floor(FT(2 * cmax / dc + 1)))
    c = [FT((n - 1) * dc - cmax) for n in 1:nc]

    return (;
        gw_source_height = source_height,
        gw_F_S0 = F_S0,
        gw_Bm = Bm,
        gw_c = c,
        gw_cw = cw,
        gw_c0 = c0,
        gw_nk = length(kwv),
        gw_k = kwv,
        gw_k2 = kwv .^ 2,
    )
end

params = non_orographic_gravity_wave_cache(
    FT;
    Bm = 0.4,
    cmax = 150,
    kwv = 2π / 100e3,
)
source_level = argmin(abs.(center_z .- params.gw_source_height))

include(joinpath(pkgdir(ClimaAtmos), "artifacts", "artifact_funcs.jl"))

era_data =
    joinpath(era_single_column_dataset_path(), "box-single_column_test.nc")

nt = NCDataset(era_data) do ds
    # Dimensions:  longitude × latitude × level × time
    # data at 40N, all longitude, all levels, all time
    lon = ds["longitude"][:]
    lat = ds["latitude"][:]
    lev = ds["level"][:] .* 100
    time = ds["time"][:]
    gZ = ds["z"][:, 5, :, :]
    T = ds["t"][:, 5, :, :]
    u = ds["u"][:, 5, :, :]
    (; lon, lat, lev, time, gZ, T, u)
end
(; lon, lat, lev, time, gZ, T, u) = nt

# compute density and buoyancy frequency
R_d = 287.0
grav = 9.8
cp_d = 1004.0

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

# interpolation to center_z grid
center_u = zeros(length(lon), length(center_z), length(time))
center_bf = zeros(length(lon), length(center_z), length(time))
center_ρ = zeros(length(lon), length(center_z), length(time))
for i in 1:length(lon)
    for it in 1:length(time)
        interp_linear = LinearInterpolation(
            Z[i, :, it][end:-1:1],
            u[i, :, it][end:-1:1],
            extrapolation_bc = Line(),
        )
        center_u[i, :, it] = interp_linear.(center_z)

        interp_linear = LinearInterpolation(
            Z[i, :, it][end:-1:1],
            bf[i, :, it][end:-1:1],
            extrapolation_bc = Line(),
        )
        center_bf[i, :, it] = interp_linear.(center_z)

        interp_linear = LinearInterpolation(
            Z[i, :, it][end:-1:1],
            ρ[i, :, it][end:-1:1],
            extrapolation_bc = Line(),
        )
        center_ρ[i, :, it] = interp_linear.(center_z)
    end
end

# zonal mean
center_u_mean = mean(center_u, dims = 1)[1, :, :]
center_bf_mean = mean(center_bf, dims = 1)[1, :, :]
center_ρ_mean = mean(center_ρ, dims = 1)[1, :, :]

# monthly ave Jan, April, July, Oct
month = Dates.month.(time)

ENV["GKSwstype"] = "nul"
output_dir = "nonorographic_gravity_wave_test_single_column"
mkpath(output_dir)

# Jan
Jan_u = mean(center_u_mean[:, month .== 1], dims = 2)[:, 1]
Jan_bf = mean(center_bf_mean[:, month .== 1], dims = 2)[:, 1]
Jan_ρ = mean(center_ρ_mean[:, month .== 1], dims = 2)[:, 1]
Jan_uforcing = CA.non_orographic_gravity_wave_forcing(
    model_config,
    Jan_u,
    source_level,
    params.gw_F_S0,
    params.gw_Bm,
    params.gw_c,
    params.gw_cw,
    params.gw_c0,
    params.gw_nk,
    params.gw_k,
    params.gw_k2,
    Jan_bf,
    Jan_ρ,
    face_z,
)
png(
    plot(Jan_uforcing[source_level:end] * 86400, center_z[source_level:end]),
    joinpath(output_dir, "fig6jan.png"),
)

# April
April_u = mean(center_u_mean[:, month .== 4], dims = 2)[:, 1]
April_bf = mean(center_bf_mean[:, month .== 4], dims = 2)[:, 1]
April_ρ = mean(center_ρ_mean[:, month .== 4], dims = 2)[:, 1]
April_uforcing = CA.non_orographic_gravity_wave_forcing(
    model_config,
    April_u,
    source_level,
    params.gw_F_S0,
    params.gw_Bm,
    params.gw_c,
    params.gw_cw,
    params.gw_c0,
    params.gw_nk,
    params.gw_k,
    params.gw_k2,
    April_bf,
    April_ρ,
    face_z,
)
png(
    plot(April_uforcing[source_level:end] * 86400, center_z[source_level:end]),
    joinpath(output_dir, "fig6apr.png"),
)

# July
July_u = mean(center_u_mean[:, month .== 7], dims = 2)[:, 1]
July_bf = mean(center_bf_mean[:, month .== 7], dims = 2)[:, 1]
July_ρ = mean(center_ρ_mean[:, month .== 7], dims = 2)[:, 1]
July_uforcing = CA.non_orographic_gravity_wave_forcing(
    model_config,
    July_u,
    source_level,
    params.gw_F_S0,
    params.gw_Bm,
    params.gw_c,
    params.gw_cw,
    params.gw_c0,
    params.gw_nk,
    params.gw_k,
    params.gw_k2,
    July_bf,
    July_ρ,
    face_z,
)
png(
    plot(July_uforcing[source_level:end] * 86400, center_z[source_level:end]),
    joinpath(output_dir, "fig6jul.png"),
)

# Oct
Oct_u = mean(center_u_mean[:, month .== 10], dims = 2)[:, 1]
Oct_bf = mean(center_bf_mean[:, month .== 10], dims = 2)[:, 1]
Oct_ρ = mean(center_ρ_mean[:, month .== 10], dims = 2)[:, 1]
Oct_uforcing = CA.non_orographic_gravity_wave_forcing(
    model_config,
    Oct_u,
    source_level,
    params.gw_F_S0,
    params.gw_Bm,
    params.gw_c,
    params.gw_cw,
    params.gw_c0,
    params.gw_nk,
    params.gw_k,
    params.gw_k2,
    Oct_bf,
    Oct_ρ,
    face_z,
)
png(
    plot(Oct_uforcing[source_level:end] * 86400, center_z[source_level:end]),
    joinpath(output_dir, "fig6oct.png"),
)
