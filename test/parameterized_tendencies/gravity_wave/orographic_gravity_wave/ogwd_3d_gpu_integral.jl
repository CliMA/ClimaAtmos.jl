using ClimaCore:
    Fields, Geometry, Domains, Meshes, Topologies, Spaces, Operators, DataLayouts, Utilities
using ClimaCore
using ClimaCore.CommonSpaces
using NCDatasets
import ClimaAtmos
import ClimaAtmos as CA
import ClimaComms
import ClimaAtmos: AtmosArtifacts as AA
import ClimaUtilities: SpaceVaryingInputs.SpaceVaryingInput
using ClimaCoreTempestRemap

import Interpolations

using CUDA
using Dates

const FT = Float64
include(
    joinpath(pkgdir(ClimaAtmos), "post_processing/remap", "remap_helpers.jl"),
)
include("../gw_plotutils.jl")

comms_ctx = ClimaComms.SingletonCommsContext()
@show CUDA.functional()
@show ClimaComms.device(comms_ctx)

(; config_file, job_id) = CA.commandline_kwargs()
config = CA.AtmosConfig(config_file; job_id, comms_ctx)

config.parsed_args["topography"] = "Earth";
config.parsed_args["topo_smoothing"] = false;
config.parsed_args["mesh_warp_type"] = "Linear";
(; parsed_args) = config

# load gfdl data
include(joinpath(@__DIR__, "../../../artifact_funcs.jl"))
ncfile = joinpath(gfdl_ogw_data_path(), "gfdl_ogw.nc")
nt = NCDataset(ncfile) do ds
    lon = Array(ds["lon"])
    lat = Array(ds["lat"])
    z_full = Array(ds["z_full"])
    temp = Array(ds["temp"])
    ucomp = Array(ds["ucomp"])
    vcomp = Array(ds["vcomp"])
    udt_topo = Array(ds["udt_topo"])
    vdt_topo = Array(ds["vdt_topo"])
    sphum = Array(ds["sphum"])
    ps = Array(ds["ps"])
    pk = Array(ds["pk"])
    bk = Array(ds["bk"])
    (;
        lon,
        lat,
        z_full,
        temp,
        ucomp,
        vcomp,
        udt_topo,
        vdt_topo,
        sphum,
        ps,
        pk,
        bk,
    )
end
(;
    lon,
    lat,
    z_full,
    temp,
    ucomp,
    vcomp,
    udt_topo,
    vdt_topo,
    sphum,
    ps,
    pk,
    bk,
) = nt

# compute pressure
p_half = zeros((size(ps)[1], size(ps)[2], length(bk)))
for k in 1:size(bk)[1]
    p_half[:, :, k, :] = bk[k] * ps .+ pk[k]
end
p_center = 0.5 * (p_half[:, :, 1:(end - 1), :] .+ p_half[:, :, 2:end, :])

# Create meshes and spaces
h_elem = 16
nh_poly = 3
z_max = 42e3
z_elem = 33
dz_bottom = 300.0
radius = 6.371229e6

quad = Quadratures.GLL{nh_poly + 1}()
horizontal_mesh = CA.cubed_sphere_mesh(; radius, h_elem)
h_space = CA.make_horizontal_space(horizontal_mesh, quad, comms_ctx, false)
z_stretch = Meshes.HyperbolicTangentStretching(dz_bottom)
ᶜspace, ᶠspace =
    CA.make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; parsed_args)

ᶜlocal_geometry = Fields.local_geometry_field(ᶜspace)
ᶠlocal_geometry = Fields.local_geometry_field(ᶠspace)

# interpolation functions
function ᶜinterp_latlon2cg(lon, lat, datain, ᶜlocal_geometry)
    # create interpolcation obj
    li_obj = []
    for k in 1:size(datain)[3]
        push!(
            li_obj,
            Interpolations.linear_interpolation(
                (lon, lat),
                datain[:, :, k],
                extrapolation_bc = (
                    Interpolations.Periodic(),
                    Interpolations.Flat(),
                ),
            ),
        )
    end

    # interpolate onto cg grid and return Fields.Field
    gfdl_data = Fields.Field(FT, axes(ᶜlocal_geometry))
    Fields.bycolumn(axes(ᶜlocal_geometry)) do colidx
        lg = ᶜlocal_geometry[colidx]
        lat = lg.coordinates.lat
        lon = lg.coordinates.long
        data_center = zeros(size(datain)[3])
        for k in 1:size(z_full)[3]
            data_center[k] = FT(li_obj[k](parent(lon)[k], parent(lat)[k]))
        end
        parent(gfdl_data[colidx]) .= data_center[end:-1:1]
    end
    return gfdl_data
end

function ᶜinterp2CAlevels(gfdl_z_full, gfdl_data, ᶜlocal_geometry)
    gfdl_ca_data = Fields.Field(FT, axes(ᶜlocal_geometry))
    Fields.bycolumn(axes(ᶜlocal_geometry)) do colidx
        li = Interpolations.linear_interpolation(
            parent(gfdl_z_full[colidx])[:],
            parent(gfdl_data[colidx])[:],
            extrapolation_bc = Interpolations.Line(),
        )
        parent(gfdl_ca_data[colidx]) .=
            li.(parent(ᶜlocal_geometry.coordinates.z[colidx]))
    end
    return gfdl_ca_data
end

# interpolate to cg grid and assign to Fields
gfdl_z_full = ᶜinterp_latlon2cg(lon, lat, z_full, ᶜlocal_geometry)
gfdl_temp = ᶜinterp_latlon2cg(lon, lat, temp, ᶜlocal_geometry)
gfdl_ucomp = ᶜinterp_latlon2cg(lon, lat, ucomp, ᶜlocal_geometry)
gfdl_vcomp = ᶜinterp_latlon2cg(lon, lat, vcomp, ᶜlocal_geometry)
gfdl_udt_topo = ᶜinterp_latlon2cg(lon, lat, udt_topo, ᶜlocal_geometry)
gfdl_vdt_topo = ᶜinterp_latlon2cg(lon, lat, vdt_topo, ᶜlocal_geometry)
gfdl_sphum = ᶜinterp_latlon2cg(lon, lat, sphum, ᶜlocal_geometry)
gfdl_p = ᶜinterp_latlon2cg(lon, lat, p_center, ᶜlocal_geometry)

# interpolate gfdl data (cg) onto our model levels
gfdl_ca_temp = ᶜinterp2CAlevels(gfdl_z_full, gfdl_temp, ᶜlocal_geometry)
gfdl_ca_ucomp = ᶜinterp2CAlevels(gfdl_z_full, gfdl_ucomp, ᶜlocal_geometry)
gfdl_ca_vcomp = ᶜinterp2CAlevels(gfdl_z_full, gfdl_vcomp, ᶜlocal_geometry)
gfdl_ca_udt_topo = ᶜinterp2CAlevels(gfdl_z_full, gfdl_udt_topo, ᶜlocal_geometry)
gfdl_ca_vdt_topo = ᶜinterp2CAlevels(gfdl_z_full, gfdl_vdt_topo, ᶜlocal_geometry)
gfdl_ca_sphum = ᶜinterp2CAlevels(gfdl_z_full, gfdl_sphum, ᶜlocal_geometry)
gfdl_ca_p = ᶜinterp2CAlevels(gfdl_z_full, gfdl_p, ᶜlocal_geometry)

# create Y
Yc = map(ᶜlocal_geometry) do lg
    return (; 
    ρ = FT(1.0), 
    uₕ = Geometry.Covariant12Vector(Geometry.UVVector(FT(0), FT(0)), lg), 
    T = FT(0), 
    qt = FT(0)
    )
    end
Yc.uₕ .= Geometry.Covariant12Vector.(Geometry.UVVector.(gfdl_ca_ucomp, gfdl_ca_vcomp))
# Yc.uₕ.components.data.:2 .= Geometry.Covariant12Vector.(gfdl_ca_vcomp)
Yc.T .= gfdl_ca_temp
Yc.qt .= gfdl_ca_sphum
Yf = map(ᶠlocal_geometry) do lg
    return (; u₃ = Geometry.Covariant3Vector(FT(0), lg))
end
Y = Fields.FieldVector(c = Yc, f = Yf)

# compute density from temperature, humidiry, pressure
R_d = 287.0
grav = 9.8
cp_d = 1004.0
epsilon = 0.622
@. Y.c.ρ = gfdl_ca_p / Y.c.T / R_d / (1 - Y.c.qt + Y.c.qt / epsilon)

# Initialize cache vars for orographic gravity wave
ogw = CA.FullOrographicGravityWave{FT, String}()

# @Main.infiltrate
topo_info = CA.get_topo_info(Y, ogw)

Y = ClimaCore.to_device(ClimaComms.CUDADevice(), copy(Y))

# pre-compute thermal vars
thermo_params = CA.TD.Parameters.ThermodynamicsParameters(FT)
thermo_params = ClimaCore.to_device(ClimaComms.CUDADevice(), thermo_params)

ᶜT_cpu = gfdl_ca_temp
ᶜp_cpu = gfdl_ca_p

ᶜz = Fields.coordinate_field(Y.c).z
ᶜp = similar(Y.c.T)
ᶜT = similar(Y.c.T)

parent(ᶜp) .= ClimaCore.to_device(ClimaComms.CUDADevice(), copy(parent(ᶜp_cpu)))
parent(ᶜT) .= ClimaCore.to_device(ClimaComms.CUDADevice(), copy(parent(ᶜT_cpu)))

topo_info = CA.move_topo_info_to_gpu(Y, topo_info)

cp_m_out = similar(Y.c)
ᶜts = similar(Y.c, CA.TD.PhaseEquil{FT})
@. ᶜts = CA.TD.PhaseEquil_ρpq(thermo_params, Y.c.ρ, ᶜp, Y.c.qt)

# emulate `p` from a ClimaAtmos run
atmos = (; turbconv_model = nothing)
atmos = ClimaCore.to_device(ClimaComms.CUDADevice(), atmos)
p = (; 
    orographic_gravity_wave = CA.orographic_gravity_wave_cache(Y, ogw, topo_info),
    scratch = CA.temporary_quantities(Y, atmos),
    precomputed = (; ᶜts = ᶜts, ᶜp = ᶜp, thermo_params = thermo_params),
    params = CA.ClimaAtmosParameters(config)
    )

(; topo_ᶜz_pbl, topo_ᶠz_pbl, topo_τ_x, topo_τ_y, topo_τ_l, topo_τ_p, topo_τ_np) =
    p.orographic_gravity_wave
(; topo_ᶜτ_sat, topo_ᶠτ_sat) = p.orographic_gravity_wave
(; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) =
    p.orographic_gravity_wave
(; topo_ᶠVτ, topo_k_pbl_values) =
    p.orographic_gravity_wave
(; ᶜdTdz) = p.orographic_gravity_wave
(; ᶜuforcing, ᶜvforcing) = p.orographic_gravity_wave

# Extract parameters
ogw_params = p.orographic_gravity_wave.ogw_params

CA.orographic_gravity_wave_compute_tendency!(Y, p, ogw)

# (; ᶜuforcing, ᶜvforcing) = p.orographic_gravity_wave

# Move GPU arrays back to CPU for plotting
uforcing_cpu = ClimaCore.to_cpu(ᶜuforcing)
vforcing_cpu = ClimaCore.to_cpu(ᶜvforcing)
gfdl_ca_udt_topo_cpu = ClimaCore.to_cpu(gfdl_ca_udt_topo)
gfdl_ca_vdt_topo_cpu = ClimaCore.to_cpu(gfdl_ca_vdt_topo)
ᶜz_cpu = ClimaCore.to_cpu(ᶜz)
Y_cpu  = ClimaCore.to_cpu(Y)

##################
# plotting!!!!
##################
ENV["GKSwstype"] = "nul"
output_dir = "orographic_gravity_wave_test_3d"
mkpath(output_dir)

# remap uforcing and vforcing to regular lat/lon grid
REMAP_DIR = joinpath(@__DIR__, "ogwd_3d", "remap_data/")
if !isdir(REMAP_DIR)
    mkpath(REMAP_DIR)
end
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
datafile_cg = joinpath(REMAP_DIR, "data_cg_$(timestamp).nc")
nc = NCDataset(datafile_cg, "c")
def_space_coord(nc, ᶜspace, type = "cgll")
nc_time = def_time_coord(nc)
nc_ogwd_uforcing = defVar(nc, "ogwd_u", FT, ᶜspace, ("time",))
nc_ogwd_vforcing = defVar(nc, "ogwd_v", FT, ᶜspace, ("time",))
nc_gfdl_udt_topo = defVar(nc, "gfdl_udt_topo", FT, ᶜspace, ("time",))
nc_gfdl_vdt_topo = defVar(nc, "gfdl_vdt_topo", FT, ᶜspace, ("time",))
nc_z_3d = defVar(nc, "z_3d", FT, ᶜspace, ("time",))
nc_time[1] = 1
nc_ogwd_uforcing[:, 1] = uforcing_cpu
nc_ogwd_vforcing[:, 1] = vforcing_cpu
nc_gfdl_udt_topo[:, 1] = gfdl_ca_udt_topo_cpu
nc_gfdl_vdt_topo[:, 1] = gfdl_ca_vdt_topo_cpu
nc_z_3d[:, 1] = ᶜz_cpu
close(nc)

nlat = 90
nlon = 180
weightfile = joinpath(REMAP_DIR, "remap_weights.nc")
create_weightfile(weightfile, axes(Y_cpu.c), axes(Y_cpu.f), nlat, nlon, mono = true)

datafile_rll = joinpath(REMAP_DIR, "data_rll.nc")
apply_remap(
    datafile_rll,
    datafile_cg,
    weightfile,
    ["ogwd_u", "ogwd_v", "gfdl_udt_topo", "gfdl_vdt_topo", "z_3d"],
)

# Plot the zonal and meridional components of the base flux
nt = NCDataset(datafile_rll) do ds
    lon = Array(ds["lon"])
    lat = Array(ds["lat"])
    z_coord = Array(ds["z"])
    z_center = Array(ds["z_3d"])
    ogwd_u = Array(ds["ogwd_u"])
    ogwd_v = Array(ds["ogwd_v"])
    gfdl_udt_topo = Array(ds["gfdl_udt_topo"])
    gfdl_vdt_topo = Array(ds["gfdl_vdt_topo"])
    (;
        lon,
        lat,
        z_coord,
        z_center,
        ogwd_u,
        ogwd_v,
        gfdl_udt_topo,
        gfdl_vdt_topo,
    )
end
(; lon, lat, z_coord, z_center, ogwd_u, ogwd_v, gfdl_udt_topo, gfdl_vdt_topo) =
    nt

# Plot on lat-lon grid
for k in [21, 31]
    fig = generate_empty_figure()
    title = "climaatmos at z = " * string(z_coord[k])
    create_plot!(
        fig;
        X = lon,
        Y = lat,
        Z = ogwd_u[:, :, k, 1] .* 86400,
        levels = range(-10, 10; length = 20),
        title,
        p_loc = (1, 1),
        yreversed = false,
    )

    title = "gfdl at z = " * string(z_coord[k])
    create_plot!(
        fig;
        X = lon,
        Y = lat,
        Z = gfdl_udt_topo[:, :, k, 1] .* 86400,
        levels = range(-10, 10; length = 20),
        title,
        p_loc = (2, 1),
        yreversed = false,
    )
    CairoMakie.save(joinpath(output_dir, "uforcing_$k.png"), fig)
end

for k in [21, 31]
    fig = generate_empty_figure()
    title = "climaatmos at z = " * string(z_coord[k])
    create_plot!(
        fig;
        X = lon,
        Y = lat,
        Z = ogwd_v[:, :, k, 1] .* 86400,
        levels = range(-10, 10; length = 20),
        title,
        p_loc = (1, 1),
        yreversed = false,
    )

    title = "gfdl at z = " * string(z_coord[k])
    create_plot!(
        fig;
        X = lon,
        Y = lat,
        Z = gfdl_vdt_topo[:, :, k, 1] .* 86400,
        levels = range(-10, 10; length = 20),
        title,
        p_loc = (2, 1),
        yreversed = false,
    )
    CairoMakie.save(joinpath(output_dir, "vforcing_$k.png"), fig)
end
