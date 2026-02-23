#=
Non-orographic gravity wave visualization script (3D)

This script generates Figure 8 from Alexander & Dunkerton (1999) for visual
comparison. It is NOT included in the automated test suite because it has no
@test assertions - it only generates plots for manual verification.

To run manually:
    julia --project test/parameterized_tendencies/gravity_wave/non_orographic_gravity_wave/nogw_test_3d.jl

Reference: https://journals.ametsoc.org/view/journals/atsc/56/24/1520-0469_1999_056_4167_aspomf_2.0.co_2.xml
=#

using ClimaComms
ClimaComms.@import_required_backends
using NCDatasets
using Dates
using Statistics
import Interpolations
import ClimaAtmos
import ClimaAtmos as CA
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Operators as Operators
const FT = Float64

include("../gw_plotutils.jl")

# test Figure 8 of the Alexander and Dunkerton (1999) paper:
# https://journals.ametsoc.org/view/journals/atsc/56/24/1520-0469_1999_056_4167_aspomf_2.0.co_2.xml?tab_body=pdf

face_z = FT.(0:1e3:0.47e5)
center_z = FT(0.5) .* (face_z[1:(end - 1)] .+ face_z[2:end])

# compute the source parameters
function non_orographic_gravity_wave_param(
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

non_orographic_gravity_wave = non_orographic_gravity_wave_param(
    FT;
    Bw = 0.4,
    cmax = 150,
    kwv = 2π / 100e3,
)
source_level =
    argmin(abs.(center_z .- non_orographic_gravity_wave.gw_source_height))
damp_level = length(center_z)

include(joinpath(@__DIR__, "../../../artifact_funcs.jl"))

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
            interp_linear = Interpolations.LinearInterpolation(
                Z[i, j, :, it][end:-1:1],
                u[i, j, :, it][end:-1:1],
                extrapolation_bc = Interpolations.Line(),
            )
            center_u[i, j, :, it] = interp_linear.(center_z)

            interp_linear = Interpolations.LinearInterpolation(
                Z[i, j, :, it][end:-1:1],
                bf[i, j, :, it][end:-1:1],
                extrapolation_bc = Interpolations.Line(),
            )
            center_bf[i, j, :, it] = interp_linear.(center_z)

            interp_linear = Interpolations.LinearInterpolation(
                Z[i, j, :, it][end:-1:1],
                ρ[i, j, :, it][end:-1:1],
                extrapolation_bc = Interpolations.Line(),
            )
            center_ρ[i, j, :, it] = interp_linear.(center_z)
        end
    end
end

# compute zonal mean profile first and apply parameterization
center_u_zonalave = mean(center_u, dims = 1)[1, :, :, :]
center_bf_zonalave = mean(center_bf, dims = 1)[1, :, :, :]
center_ρ_zonalave = mean(center_ρ, dims = 1)[1, :, :, :]

#generate domain, space and field
column_domain = ClimaCore.Domains.IntervalDomain(
    ClimaCore.Geometry.ZPoint(0.0) .. ClimaCore.Geometry.ZPoint(47000),
    boundary_names = (:bottom, :top),
)

column_mesh = ClimaCore.Meshes.IntervalMesh(column_domain, nelems = 47)

column_center_space = ClimaCore.Spaces.CenterFiniteDifferenceSpace(column_mesh)

column_face_space =
    ClimaCore.Spaces.FaceFiniteDifferenceSpace(column_center_space)

coordinate = Fields.coordinate_field(column_center_space)
gw_ncval = Val(500)
ᶜz = coordinate.z
ᶜρ = copy(ᶜz)
ᶜu = copy(ᶜz)
ᶜv = copy(ᶜz)
ᶜbf = copy(ᶜz)
ᶜlevel = similar(ᶜρ, FT)
u_waveforcing = similar(ᶜv)
v_waveforcing = similar(ᶜv)
for i in 1:Spaces.nlevels(axes(ᶜρ))
    fill!(Fields.level(ᶜlevel, i), i)
end
uforcing = similar(ᶜρ, FT)
vforcing = similar(ᶜρ, FT)

scratch = (;
    ᶜtemp_scalar = similar(ᶜz, FT),
    ᶜtemp_scalar_2 = similar(ᶜz, FT),
    ᶜtemp_scalar_3 = similar(ᶜz, FT),
    ᶜtemp_scalar_4 = similar(ᶜz, FT),
    ᶜtemp_scalar_5 = similar(ᶜz, FT),
    temp_field_level = similar(Fields.level(ᶜz, 1), FT),
)

# create input parameter
params = (; non_orographic_gravity_wave, scratch)

# Jan
month = Dates.month.(time)

Jan_u = mean(center_u_zonalave[:, :, month .== 1], dims = 3)[:, :, 1]
Jan_bf = mean(center_bf_zonalave[:, :, month .== 1], dims = 3)[:, :, 1]
Jan_ρ = mean(center_ρ_zonalave[:, :, month .== 1], dims = 3)[:, :, 1]
Jan_uforcing = similar(Jan_u)

for j in 1:length(lat)
    Base.parent(ᶜρ) .= Jan_ρ[j, :]
    Base.parent(ᶜu) .= Jan_u[j, :]
    Base.parent(ᶜbf) .= Jan_bf[j, :]
    ᶜρ_source = Fields.level(ᶜρ, source_level)
    ᶜu_source = Fields.level(ᶜu, source_level)
    ᶜv_source = Fields.level(ᶜv, source_level)
    uforcing .= 0
    vforcing .= 0

    CA.non_orographic_gravity_wave_forcing(
        ᶜu,
        ᶜv,
        ᶜbf,
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
        params,
    )
    Jan_uforcing[j, :] = parent(uforcing)
end

output_dir = "nonorographic_gravity_wave_test_3d"
mkpath(output_dir)
fig = generate_empty_figure();
create_plot!(
    fig;
    X = lat[end:-1:1],
    Y = center_z[source_level:end],
    Z = 86400 * Jan_uforcing[end:-1:1, source_level:end],
    levels = range(-1, 1; length = 20),
    yreversed = false,
    yscale = identity,
)
CairoMakie.save(joinpath(output_dir, "test-fig8.png"), fig)
