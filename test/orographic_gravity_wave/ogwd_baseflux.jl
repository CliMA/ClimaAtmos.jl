using NCDatasets
import ClimaAtmos
import ClimaAtmos as CA
using ClimaCore: Fields, Domains, Meshes, Topologies, Spaces, Geometry
import ClimaComms, Logging
using Plots
using Interpolations
using ClimaCoreTempestRemap
const FT = Float64

include("../../post_processing/remap/remap_helpers.jl")

comms_ctx = ClimaComms.SingletonCommsContext()

# Create meshes and spaces
h_elem = 6
nh_poly = 3
z_max = 30e3
z_elem = 1
radius = 6.371229e6

quad = Spaces.Quadratures.GLL{nh_poly + 1}()
horizontal_mesh = CA.cubed_sphere_mesh(; radius, h_elem)
h_space = CA.make_horizontal_space(horizontal_mesh, quad, comms_ctx, false)
z_stretch = Meshes.Uniform()
center_space, face_space =
    CA.make_hybrid_spaces(h_space, z_max, z_elem, z_stretch)

ᶜlocal_geometry = Fields.local_geometry_field(center_space)
ᶠlocal_geometry = Fields.local_geometry_field(face_space)

# Initialize Y with prescribed wind, density, and buoyancy frequency
# Figure 1 in Garner05
Yc = map(ᶜlocal_geometry) do lg
    lat = lg.coordinates.lat
    if abs(lat) <= 23.5
        uₕ = Geometry.Covariant12Vector(Geometry.UVVector(FT(-7), FT(0)), lg)
    else
        uₕ = Geometry.Covariant12Vector(Geometry.UVVector(FT(13), FT(0)), lg)
    end
    return (; ρ = FT(1.0), uₕ = uₕ, N = 0.01)
end
Yf = map(ᶠlocal_geometry) do lg
    return (; w = Geometry.Covariant3Vector(FT(0), lg))
end
Y = Fields.FieldVector(c = Yc, f = Yf)

# Initialize cache vars for orographic gravity wave
ogw = CA.OrographicGravityWave{FT, String}()
p = CA.orographic_gravity_wave_cache(ogw, Y, FT(radius))

# Unpack cache vars
(; topo_τ_x, topo_τ_y, topo_τ_l, topo_τ_p, topo_τ_np) = p
(; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) = p
(; hmax, hmin, t11, t12, t21, t22) = p.topo_info

u_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:1
v_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:2

# Compute base flux
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
        parent(Y.c.N[colidx]),
        1,
    )
end

# Remap base flux to regular lat/lon grid for visualization
TOPO_DIR = joinpath(@__DIR__, "remap_data/")
if !isdir(TOPO_DIR)
    mkdir(TOPO_DIR)
end
datafile_cg = joinpath(TOPO_DIR, "data_cg.nc")
nc = NCDataset(datafile_cg, "c")
def_space_coord(nc, center_space, type = "cgll")
nc_time = def_time_coord(nc)
nc_tau_x = defVar(nc, "tau_x", FT, h_space, ("time",))
nc_tau_y = defVar(nc, "tau_y", FT, h_space, ("time",))
nc_time[1] = 1
nc_tau_x[:, 1] = topo_τ_x
nc_tau_y[:, 1] = topo_τ_y
close(nc)

nlat = 90
nlon = 180
weightfile = joinpath(TOPO_DIR, "remap_weights.nc")
create_weightfile(weightfile, axes(Y.c), axes(Y.f), nlat, nlon)

datafile_rll = joinpath(TOPO_DIR, "data_rll.nc")
apply_remap(datafile_rll, datafile_cg, weightfile, ["tau_x", "tau_y"])

# Plot the zonal and meridional components of the base flux
nt = NCDataset(datafile_rll) do ds
    lon = ds["lon"][:]
    lat = ds["lat"][:]
    tau_x = ds["tau_x"][:]
    tau_y = ds["tau_y"][:]
    (; lon, lat, tau_x, tau_y)
end
(; lon, lat, tau_x, tau_y) = nt

ENV["GKSwstype"] = "nul"
output_dir = "orographic_gravity_wave_test_baseflux"
mkpath(output_dir)

c1 = contourf(
    lon,
    lat,
    tau_x[:, :, 1]',
    color = :balance,
    clim = (-5, 5),
    title = "ogwd clima (zonal)",
)
c2 = contourf(
    lon,
    lat,
    tau_y[:, :, 1]',
    color = :balance,
    clim = (-1, 1),
    title = "ogwd clima (meridional)",
)

p = plot(c1, c2, layout = (1, 2), size = (1500, 800))
png(p, joinpath(output_dir, "baseflux.png"))

rm(TOPO_DIR; recursive = true)
