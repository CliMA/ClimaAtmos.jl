using ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
using NCDatasets
using Dates
using Interpolations
using Statistics
import ClimaAtmos
import ClimaAtmos as CA
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore.Operators as Operators

include("../gw_plotutils.jl")

const FT = Float64
# single column test Figure 6 of the Alexander and Dunkerton (1999) paper:
# https://journals.ametsoc.org/view/journals/atsc/56/24/1520-0469_1999_056_4167_aspomf_2.0.co_2.xml?tab_body=pdf
# zonal mean monthly wind, temperature 1958-1973; at 40N in latitude for Jan, April, July, Oct.

face_z = FT.(0:1e3:0.5e5)
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

include(joinpath(pkgdir(ClimaAtmos), "artifacts", "artifact_funcs.jl"))

era_data =
    joinpath(era_single_column_dataset_path(), "box-single_column_test.nc")

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

column_domain = ClimaCore.Domains.IntervalDomain(
    ClimaCore.Geometry.ZPoint(0.0) .. ClimaCore.Geometry.ZPoint(50000.0),
    boundary_names = (:bottom, :top),
)

column_mesh = ClimaCore.Meshes.IntervalMesh(column_domain, nelems = 50)

column_center_space = ClimaCore.Spaces.CenterFiniteDifferenceSpace(column_mesh)
# construct the face space from the center one
column_face_space =
    ClimaCore.Spaces.FaceFiniteDifferenceSpace(column_center_space)

coord = ClimaCore.Fields.coordinate_field(column_center_space)

gw_ncval = Val(501)
ᶜz = coord.z
ᶜρ = copy(ᶜz)
ᶜu = copy(ᶜz)
ᶜv = copy(ᶜz)
ᶜbf = copy(ᶜz)
ᶜlevel = similar(ᶜρ, FT)
u_waveforcing = similar(ᶜu)
v_waveforcing = similar(ᶜv)
for i in 1:Spaces.nlevels(axes(ᶜρ))
    fill!(Fields.level(ᶜlevel, i), i)
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

scratch = (;
    ᶜtemp_scalar = similar(ᶜz, FT),
    ᶜtemp_scalar_2 = similar(ᶜz, FT),
    ᶜtemp_scalar_3 = similar(ᶜz, FT),
    ᶜtemp_scalar_4 = similar(ᶜz, FT),
    ᶜtemp_scalar_5 = similar(ᶜz, FT),
    temp_field_level = similar(Fields.level(ᶜz, 1), FT),
)

params = (; non_orographic_gravity_wave, scratch)

# Jan
Jan_u = mean(center_u_mean[:, month .== 1], dims = 2)[:, 1]
Jan_bf = mean(center_bf_mean[:, month .== 1], dims = 2)[:, 1]
Jan_ρ = mean(center_ρ_mean[:, month .== 1], dims = 2)[:, 1]
Base.parent(ᶜρ) .= Jan_ρ
ᶜv = ᶜu
Base.parent(ᶜu) .= Jan_u
Base.parent(ᶜbf) .= Jan_bf
ᶜρ_source = Fields.level(ᶜρ, source_level)
ᶜu_source = Fields.level(ᶜu, source_level)
ᶜv_source = Fields.level(ᶜv, source_level)
Jan_uforcing = similar(ᶜρ, FT)
Jan_uforcing .= 0
Jan_vforcing = similar(ᶜρ, FT)
Jan_vforcing .= 0

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
    Jan_uforcing,
    Jan_vforcing,
    gw_ncval,
    u_waveforcing,
    v_waveforcing,
    params,
)
Jan_uforcing = Base.parent(Jan_uforcing)
fig = generate_empty_figure();
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
Base.parent(ᶜρ) .= April_ρ
ᶜv = ᶜu
Base.parent(ᶜu) .= April_u
Base.parent(ᶜbf) .= April_bf
April_uforcing = similar(ᶜρ, FT)
April_uforcing .= 0
April_vforcing = similar(ᶜρ, FT)
April_vforcing .= 0

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
    April_uforcing,
    April_vforcing,
    gw_ncval,
    u_waveforcing,
    v_waveforcing,
    params,
)
April_uforcing = Base.parent(April_uforcing)
fig = generate_empty_figure();
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
Base.parent(ᶜρ) .= July_ρ
ᶜv = ᶜu
Base.parent(ᶜu) .= July_u
Base.parent(ᶜbf) .= July_bf
July_uforcing = similar(ᶜρ, FT)
July_uforcing .= 0
July_vforcing = similar(ᶜρ, FT)
July_vforcing .= 0

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
    July_uforcing,
    July_vforcing,
    gw_ncval,
    u_waveforcing,
    v_waveforcing,
    params,
)
July_uforcing = Base.parent(July_uforcing)

fig = generate_empty_figure();
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
Base.parent(ᶜρ) .= Oct_ρ
ᶜv = ᶜu
Base.parent(ᶜu) .= Oct_u
Base.parent(ᶜbf) .= Oct_bf
Oct_uforcing = similar(ᶜρ, FT)
Oct_uforcing .= 0
Oct_vforcing = similar(ᶜρ, FT)
Oct_vforcing .= 0

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
    Oct_uforcing,
    Oct_vforcing,
    gw_ncval,
    u_waveforcing,
    v_waveforcing,
    params,
)

Oct_uforcing = Base.parent(Oct_uforcing)
fig = generate_empty_figure();
create_plot!(
    fig;
    X = Oct_uforcing[source_level:(end - 1)] * 86400,
    Y = center_z[source_level:(end - 1)],
    label = "Oct",
)
CairoMakie.save(joinpath(output_dir, "fig6oct.png"), fig)
