using ClimaCore:
    Fields, Geometry, Domains, Meshes, Topologies, Spaces, Operators
using NCDatasets
import ClimaAtmos
import ClimaAtmos as CA
import Thermodynamics as TD
import CLIMAParameters as CP
import ClimaComms
using ClimaCoreTempestRemap

using ImageFiltering
using Interpolations

using Plots
using ClimaCorePlots

const FT = Float64

include("../../post_processing/remap/remap_helpers.jl")

comms_ctx = ClimaComms.SingletonCommsContext()

# load gfdl data
include(joinpath(pkgdir(ClimaAtmos), "artifacts", "artifact_funcs.jl"))
ncfile = joinpath(gfdl_ogw_data_path(), "gfdl_ogw.nc")
nt = NCDataset(ncfile) do ds
    lon = ds["lon"][:]
    lat = ds["lat"][:]
    z_full = ds["z_full"][:]
    temp = ds["temp"][:]
    ucomp = ds["ucomp"][:]
    vcomp = ds["vcomp"][:]
    udt_topo = ds["udt_topo"][:]
    vdt_topo = ds["vdt_topo"][:]
    sphum = ds["sphum"][:]
    ps = ds["ps"][:]
    pk = ds["pk"][:]
    bk = ds["bk"][:]
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

# earth warp
include(joinpath(pkgdir(ClimaAtmos), "artifacts", "artifact_funcs.jl"))
data_path = joinpath(topo_elev_dataset_path(), "ETOPO1_coarse.nc")
earth_spline = NCDataset(data_path) do data
    zlevels = data["elevation"][:]
    lon = data["longitude"][:]
    lat = data["latitude"][:]
    # Apply Smoothing
    smooth_degree = 15
    esmth = imfilter(zlevels, Kernel.gaussian(smooth_degree))
    linear_interpolation(
        (lon, lat),
        zlevels,
        extrapolation_bc = (Periodic(), Flat()),
    )
end
@info "Generated interpolation stencil"
warp_function = CA.generate_topography_warp(earth_spline)

# Create meshes and spaces
h_elem = 16
nh_poly = 3
z_max = 45e3
z_elem = 33
dz_bottom = 300.0
dz_top = 5000.0
radius = 6.371229e6

quad = Spaces.Quadratures.GLL{nh_poly + 1}()
horizontal_mesh = CA.cubed_sphere_mesh(; radius, h_elem)
h_space = CA.make_horizontal_space(horizontal_mesh, quad, comms_ctx, false)

z_stretch = Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
center_space, face_space = CA.make_hybrid_spaces(
    h_space,
    z_max,
    z_elem,
    z_stretch;
    surface_warp = warp_function,
)

ᶜlocal_geometry = Fields.local_geometry_field(center_space)
ᶠlocal_geometry = Fields.local_geometry_field(face_space)

# interpolation functions
function ᶜinterp_latlon2cg(lon, lat, datain, ᶜlocal_geometry)
    # create interpolcation obj
    li_obj = []
    for k in 1:size(datain)[3]
        push!(
            li_obj,
            linear_interpolation(
                (lon, lat),
                datain[:, :, k],
                extrapolation_bc = (Periodic(), Flat()),
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
        li = linear_interpolation(
            parent(gfdl_z_full[colidx])[:],
            parent(gfdl_data[colidx])[:],
            extrapolation_bc = Line(),
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
    return (; ρ = FT(1.0), u_phy = FT(0), v_phy = FT(0), T = FT(0), qt = FT(0))
end
Yc.u_phy .= gfdl_ca_ucomp
Yc.v_phy .= gfdl_ca_vcomp
Yc.T .= gfdl_ca_temp
Yc.qt .= gfdl_ca_sphum
Yf = map(ᶠlocal_geometry) do lg
    return (; w = Geometry.Covariant3Vector(FT(0), lg))
end
Y = Fields.FieldVector(c = Yc, f = Yf)

# compute density from temperature, humidiry, pressure
R_d = 287.0
grav = 9.8
cp_d = 1004.0
epsilon = 0.622
@. Y.c.ρ = gfdl_ca_p / Y.c.T / R_d / (1 - Y.c.qt + Y.c.qt / epsilon)

# Initialize cache vars for orographic gravity wave
ogw = CA.OrographicGravityWave{FT, String}()
p = CA.orographic_gravity_wave_cache(ogw, Y, FT(radius))

(; topo_k_pbl, topo_τ_x, topo_τ_y, topo_τ_l, topo_τ_p, topo_τ_np) = p
(; topo_ᶠτ_sat, topo_ᶠVτ) = p
(; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) = p
(; hmax, hmin, t11, t12, t21, t22) = p.topo_info
(; ᶜdTdz) = p

# pre-compute thermal vars
aliases = string.(fieldnames(TD.Parameters.ThermodynamicsParameters))
toml_dict = CP.create_toml_dict(FT; dict_type = "alias")
pairs = CP.get_parameter_values!(toml_dict, aliases, "Thermodynamics")
thermo_params = TD.Parameters.ThermodynamicsParameters{FT}(; pairs...)

ᶜT = gfdl_ca_temp
ᶜp = gfdl_ca_p
ᶜts = similar(Y.c, TD.PhaseEquil{FT})
@. ᶜts = TD.PhaseEquil_ρpq(thermo_params, Y.c.ρ, ᶜp, Y.c.qt)

# operators
ᶜgradᵥ = Operators.GradientF2C()
ᶠinterp = Operators.InterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
ᶠgradᵥ = Operators.GradientC2F(
    bottom = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
    top = Operators.SetGradient(Geometry.Covariant3Vector(FT(0))),
)
operators = (; ᶜgradᵥ, ᶠgradᵥ, ᶠinterp)
p = merge(p, (; operators))

# z 
ᶜz = Fields.coordinate_field(Y.c).z
ᶠz = Fields.coordinate_field(Y.f).z

# get PBL info
Fields.bycolumn(axes(Y.c.ρ)) do colidx
    parent(topo_k_pbl[colidx]) .=
        CA.get_pbl(ᶜp[colidx], ᶜT[colidx], ᶜz[colidx], grav, cp_d)
end

# buoyancy frequency at cell centers
parent(ᶜdTdz) .= parent(Geometry.WVector.(ᶜgradᵥ.(ᶠinterp.(ᶜT))))
ᶜN = @. (grav / ᶜT) * (ᶜdTdz + grav / TD.cp_m(thermo_params, ᶜts)) # this is actually ᶜN^2
ᶜN = @. ifelse(ᶜN < eps(FT), sqrt(eps(FT)), sqrt(abs(ᶜN))) # to avoid small numbers

# prepare physical uv input variables for gravity_wave_forcing()
u_phy = Y.c.u_phy
v_phy = Y.c.v_phy

# compute base flux at k_pbl
Fields.bycolumn(axes(Y.c.ρ)) do colidx
    CA.calc_base_flux!(
        topo_τ_x[colidx],
        topo_τ_y[colidx],
        topo_τ_l[colidx],
        topo_τ_p[colidx],
        topo_τ_np[colidx],
        topo_U_sat[colidx],
        topo_FrU_sat[colidx],
        topo_FrU_max[colidx],
        topo_FrU_min[colidx],
        topo_FrU_clp[colidx],
        p,
        max(FT(0), parent(hmax[colidx])[1]),
        max(FT(0), parent(hmin[colidx])[1]),
        parent(t11[colidx])[1],
        parent(t12[colidx])[1],
        parent(t21[colidx])[1],
        parent(t22[colidx])[1],
        parent(Y.c.ρ[colidx]),
        parent(u_phy[colidx]),
        parent(v_phy[colidx]),
        parent(ᶜN[colidx]),
        Int(parent(topo_k_pbl[colidx])[1]),
    )
end

# buoyancy frequency at cell faces
ᶠN = ᶠinterp.(ᶜN) # alternatively, can be computed from ᶠT and ᶠdTdz

# compute saturation profile
Fields.bycolumn(axes(Y.c.ρ)) do colidx
    CA.calc_saturation_profile!(
        topo_ᶠτ_sat[colidx],
        topo_U_sat[colidx],
        topo_FrU_sat[colidx],
        topo_FrU_clp[colidx],
        topo_ᶠVτ[colidx],
        p,
        topo_FrU_max[colidx],
        topo_FrU_min[colidx],
        ᶠN[colidx],
        topo_τ_x[colidx],
        topo_τ_y[colidx],
        topo_τ_p[colidx],
        u_phy[colidx],
        v_phy[colidx],
        Y.c.ρ[colidx],
        ᶜp[colidx],
        Int(parent(topo_k_pbl[colidx])[1]),
    )
end

# a place holder to store physical forcing on uv
uforcing = zeros(axes(u_phy))
vforcing = zeros(axes(v_phy))

# compute drag tendencies due to propagating part
Fields.bycolumn(axes(Y.c.ρ)) do colidx
    CA.calc_propagate_forcing!(
        uforcing[colidx],
        vforcing[colidx],
        p,
        topo_τ_x[colidx],
        topo_τ_y[colidx],
        topo_τ_l[colidx],
        topo_ᶠτ_sat[colidx],
        Y.c.ρ[colidx],
    )
end

# compute drag tendencies due to non-propagating part
Fields.bycolumn(axes(Y.c.ρ)) do colidx
    CA.calc_nonpropagating_forcing!(
        uforcing[colidx],
        vforcing[colidx],
        p,
        ᶠN[colidx],
        topo_ᶠVτ[colidx],
        ᶜp[colidx],
        topo_τ_x[colidx],
        topo_τ_y[colidx],
        topo_τ_l[colidx],
        topo_τ_np[colidx],
        ᶠz[colidx],
        ᶜz[colidx],
        Int(parent(topo_k_pbl[colidx])[1]),
        grav,
    )
end

# constrain forcing
@. uforcing = max(FT(-3e-3), min(FT(3e-3), uforcing))
@. vforcing = max(FT(-3e-3), min(FT(3e-3), vforcing))

##################
# plotting!!!!
##################
ENV["GKSwstype"] = "nul"
output_dir = "orographic_gravity_wave_test_3d"
mkpath(output_dir)

# remap uforcing and vforcing to regular lat/lon grid
REMAP_DIR = joinpath(@__DIR__, "remap_data/")
if !isdir(REMAP_DIR)
    mkdir(REMAP_DIR)
end
datafile_cg = joinpath(REMAP_DIR, "data_cg.nc")
nc = NCDataset(datafile_cg, "c")
def_space_coord(nc, center_space, type = "cgll")
nc_time = def_time_coord(nc)
nc_ogwd_uforcing = defVar(nc, "ogwd_u", FT, center_space, ("time",))
nc_ogwd_vforcing = defVar(nc, "ogwd_v", FT, center_space, ("time",))
nc_gfdl_udt_topo = defVar(nc, "gfdl_udt_topo", FT, center_space, ("time",))
nc_gfdl_vdt_topo = defVar(nc, "gfdl_vdt_topo", FT, center_space, ("time",))
nc_z_3d = defVar(nc, "z_3d", FT, center_space, ("time",))
nc_time[1] = 1
nc_ogwd_uforcing[:, 1] = uforcing
nc_ogwd_vforcing[:, 1] = vforcing
nc_gfdl_udt_topo[:, 1] = gfdl_ca_udt_topo
nc_gfdl_vdt_topo[:, 1] = gfdl_ca_vdt_topo
nc_z_3d[:, 1] = ᶜz
close(nc)

nlat = 90
nlon = 180
weightfile = joinpath(REMAP_DIR, "remap_weights.nc")
create_weightfile(weightfile, axes(Y.c), axes(Y.f), nlat, nlon, mono = true)

datafile_rll = joinpath(REMAP_DIR, "data_rll.nc")
apply_remap(
    datafile_rll,
    datafile_cg,
    weightfile,
    ["ogwd_u", "ogwd_v", "gfdl_udt_topo", "gfdl_vdt_topo", "z_3d"],
)

# Plot the zonal and meridional components of the base flux
nt = NCDataset(datafile_rll) do ds
    lon = ds["lon"][:]
    lat = ds["lat"][:]
    z_coord = ds["z"][:]
    z_center = ds["z_3d"][:]
    ogwd_u = ds["ogwd_u"][:]
    ogwd_v = ds["ogwd_v"][:]
    gfdl_udt_topo = ds["gfdl_udt_topo"][:]
    gfdl_vdt_topo = ds["gfdl_vdt_topo"][:]
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
    p1 = contourf(
        lon,
        lat,
        ogwd_u[:, :, k, 1]' .* 86400,
        color = :balance,
        clim = (-10, 10),
        title = "climaatmos at z = " * string(z_coord[k]),
    )
    p2 = contourf(
        lon,
        lat,
        gfdl_udt_topo[:, :, k, 1]' .* 86400,
        color = :balance,
        clim = (-10, 10),
        title = "gfdl at z = " * string(z_coord[k]),
    )
    uplot = plot(p1, p2, layout = (2, 1))
    png(uplot, joinpath(output_dir, "uforcing_" * string(k) * ".png"))
end

for k in [21, 31]
    p1 = contourf(
        lon,
        lat,
        ogwd_v[:, :, k, 1]' .* 86400,
        color = :balance,
        clim = (-10, 10),
        title = "climaatmos at z = " * string(z_coord[k]),
    )
    p2 = contourf(
        lon,
        lat,
        gfdl_vdt_topo[:, :, k, 1]' .* 86400,
        color = :balance,
        clim = (-10, 10),
        title = "gfdl at z = " * string(z_coord[k]),
    )
    vplot = plot(p1, p2, layout = (2, 1))
    png(vplot, joinpath(output_dir, "vforcing_" * string(k) * ".png"))
end
