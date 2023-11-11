using NCDatasets
using Dates
using Interpolations
using Statistics
import ClimaAtmos
import ClimaAtmos as CA
using Plots
const FT = Float64

# test Figure 8 of the Alexander and Dunkerton (1999) paper:
# https://journals.ametsoc.org/view/journals/atsc/56/24/1520-0469_1999_056_4167_aspomf_2.0.co_2.xml?tab_body=pdf

face_z = FT.(0:1e3:0.47e5)
center_z = FT(0.5) .* (face_z[1:(end - 1)] .+ face_z[2:end])

# compute the source parameters
function non_orographic_gravity_wave(
    ::Type{FT};
    source_height = FT(15000),
    Bw = FT(1.2),
    Bt_0 = FT(4e-3),
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
        gw_source_ampl = Bt_0,
        gw_Bw = Bw,
        gw_Bn = FT(0),
        gw_c = c,
        gw_cw = cw,
        gw_cn = FT(1),
        gw_flag = FT(1),
        gw_c0 = c0,
        gw_nk = length(kwv),
    )
end

params = non_orographic_gravity_wave(FT; Bw = 0.4, cmax = 150, kwv = 2π / 100e3)
source_level = argmin(abs.(center_z .- params.gw_source_height))
damp_level = length(center_z)

include(joinpath(pkgdir(ClimaAtmos), "artifacts", "artifact_funcs.jl"))

era_data = joinpath(era_global_dataset_path(), "box-era5-monthly.nc")

nt = NCDataset(era_data) do ds
    lon = Array(ds["longitude"])
    lat = Array(ds["latitude"])
    lev = Array(ds["level"]) .* 100
    time = Array(ds["time"])
    gZ = Array(ds["z"])
    T = Array(ds["t"])
    u = Array(ds["u"])
    (; lon, lat, lev, time, gZ, T, u)
end
(; lon, lat, lev, time, gZ, T, u) = nt

# compute density and buoyancy frequency
R_d = 287.0
grav = 9.8
cp_d = 1004.0

Z = gZ ./ grav
ρ = ones(size(T)) .* reshape(lev, (1, 1, length(lev), 1)) ./ T / R_d

dTdz = zeros(size(T))
@. dTdz[:, :, 1, :] =
    (T[:, :, 2, :] - T[:, :, 1, :]) / (Z[:, :, 2, :] - Z[:, :, 1, :])
@. dTdz[:, :, end, :] =
    (T[:, :, end, :] - T[:, :, end - 1, :]) /
    (Z[:, :, end, :] - Z[:, :, end - 1, :])
@. dTdz[:, :, 2:(end - 1), :] =
    (T[:, :, 3:end, :] - T[:, :, 1:(end - 2), :]) /
    (Z[:, :, 3:end, :] - Z[:, :, 1:(end - 2), :])
bf = @. (grav / T) * (dTdz + grav / cp_d)
bf = @. ifelse(bf < 2.5e-5, sqrt(2.5e-5), sqrt(abs(bf)))

# interpolation to center_z grid
center_u = zeros(length(lon), length(lat), length(center_z), length(time))
center_bf = zeros(length(lon), length(lat), length(center_z), length(time))
center_ρ = zeros(length(lon), length(lat), length(center_z), length(time))
for i in 1:length(lon)
    for j in 1:length(lat)
        for it in 1:length(time)
            interp_linear = LinearInterpolation(
                Z[i, j, :, it][end:-1:1],
                u[i, j, :, it][end:-1:1],
                extrapolation_bc = Line(),
            )
            center_u[i, j, :, it] = interp_linear.(center_z)

            interp_linear = LinearInterpolation(
                Z[i, j, :, it][end:-1:1],
                bf[i, j, :, it][end:-1:1],
                extrapolation_bc = Line(),
            )
            center_bf[i, j, :, it] = interp_linear.(center_z)

            interp_linear = LinearInterpolation(
                Z[i, j, :, it][end:-1:1],
                ρ[i, j, :, it][end:-1:1],
                extrapolation_bc = Line(),
            )
            center_ρ[i, j, :, it] = interp_linear.(center_z)
        end
    end
end

# compute zonal mean profile first and apply parameterization
center_u_zonalave = mean(center_u, dims = 1)[1, :, :, :]
center_bf_zonalave = mean(center_bf, dims = 1)[1, :, :, :]
center_ρ_zonalave = mean(center_ρ, dims = 1)[1, :, :, :]

# Jan
month = Dates.month.(time)

Jan_u = mean(center_u_zonalave[:, :, month .== 1], dims = 3)[:, :, 1]
Jan_bf = mean(center_bf_zonalave[:, :, month .== 1], dims = 3)[:, :, 1]
Jan_ρ = mean(center_ρ_zonalave[:, :, month .== 1], dims = 3)[:, :, 1]
Jan_uforcing = zeros(length(lat), length(center_z))
for j in 1:length(lat)
    Jan_uforcing[j, :] = CA.non_orographic_gravity_wave_forcing(
        Jan_u[j, :],
        Jan_bf[j, :],
        Jan_ρ[j, :],
        copy(center_z),
        source_level,
        damp_level,
        params.gw_source_ampl,
        params.gw_Bw,
        params.gw_Bn,
        params.gw_cw,
        params.gw_cn,
        params.gw_flag,
        params.gw_c,
        params.gw_c0,
        params.gw_nk,
    )
end

ENV["GKSwstype"] = "nul"
output_dir = "nonorographic_gravity_wave_test_3d"
mkpath(output_dir)
png(
    contourf(
        lat[end:-1:1],
        center_z[source_level:end],
        86400 * Jan_uforcing[end:-1:1, source_level:end]',
        color = :balance,
        clim = (-1, 1),
    ),
    joinpath(output_dir, "test-fig8.png"),
)
