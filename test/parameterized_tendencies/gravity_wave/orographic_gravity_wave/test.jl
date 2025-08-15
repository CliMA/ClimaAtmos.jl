using NCDatasets
import ClimaAtmos as CA
using ClimaUtilities.ClimaArtifacts
import ClimaAtmos.AtmosArtifacts as AA
using Plots
using Statistics: mean

using ImageFiltering
using Interpolations

using ClimaComms
const comms_ctx = ClimaComms.context(ClimaComms.CPUSingleThreaded())

const FT = Float32
###
### load gfdl restart 
###
# orographic_info_rll = joinpath(topo_res_path(), "topo_drag.res.nc")
topo_path = @clima_artifact("topo-info", comms_ctx)
orographic_info_rll = joinpath(topo_path, "topo_drag.res.nc")

skip_pt = 1
nt = NCDataset(orographic_info_rll, "r") do ds
	gfdl_lon = ds["lon"][1:skip_pt:end]
	gfdl_lat = ds["lat"][1:skip_pt:end]
	gfdl_hmax = ds["hmax"][1:skip_pt:end, 1:skip_pt:end]
	gfdl_hmin = ds["hmin"][1:skip_pt:end, 1:skip_pt:end]
	gfdl_t11 = ds["t11"][1:skip_pt:end, 1:skip_pt:end]
	gfdl_t12 = ds["t12"][1:skip_pt:end, 1:skip_pt:end]
	gfdl_t21 = ds["t21"][1:skip_pt:end, 1:skip_pt:end]
	gfdl_t22 = ds["t22"][1:skip_pt:end, 1:skip_pt:end]
	(;gfdl_lon, gfdl_lat, gfdl_hmax, gfdl_hmin, gfdl_t11, gfdl_t12, gfdl_t21, gfdl_t22)
end
(;gfdl_lon, gfdl_lat, gfdl_hmax, gfdl_hmin, gfdl_t11, gfdl_t12, gfdl_t21, gfdl_t22) = nt

###
### compute from high resolution elevation
###
elev_data = "/home/chew/ETOPO1_Ice_g_gmt4.grd"
# elev_data = AA.earth_orography_file_path(;
#                 context = comms_ctx,
#             )

nt = NCDataset(elev_data, "r") do ds
	hr_lon = ds["x"][:]
	hr_lat = ds["y"][:]
	hr_z = FT.( ds["z"][:] )
	(; hr_lon, hr_lat, hr_z)
end
(; hr_lon, hr_lat, hr_z) = nt;
 
skip_pt = 100
hr_z = reshape(hr_z, length(hr_lon), length(hr_lat))
hr_z = hr_z[1:skip_pt:end, 1:skip_pt:end]
hr_lon = hr_lon[1:skip_pt:end]
hr_lat = hr_lat[1:skip_pt:end]

latidx = abs.(hr_lat).!=FT(90)
hr_lat = hr_lat[latidx]
hr_z = hr_z[:,latidx]

earth_radius = FT(6.371229e6)
hr_hmax = CA.calc_hpoz_latlon(hr_z, hr_lon, hr_lat, earth_radius);
print("done")
# interpolate hmax to gfdl lat/lon grid
li_hmax = linear_interpolation( (hr_lon, hr_lat), hr_hmax, extrapolation_bc = (Periodic(), Flat()), )
hr_hmax_interp = zeros(size(gfdl_hmax))
for i in 1:length(gfdl_lon)
	for j in 1:length(gfdl_lat)
		if gfdl_lon[i] > 180.0 
			hr_hmax_interp[i,j] = li_hmax(gfdl_lon[i]-360.0, gfdl_lat[j])
		else
			hr_hmax_interp[i,j] = li_hmax(gfdl_lon[i], gfdl_lat[j])
		end 
	end
end

ENV["GKSwstype"] = "nul"
p1 = contourf(gfdl_lon, gfdl_lat, gfdl_hmax[:,:,1]', clim=(-2500,2500), color=:balance);
p2 = contourf(gfdl_lon, gfdl_lat, hr_hmax_interp[:,:,1]', clim=(-2500,2500), color=:balance);
p = plot(p1,p2, layout = (2,1), size = (1500,1600));
png(p, "./hr_hmax_4min_100e3.png")

# ## save hmax to nc
# ds = NCDataset("./oro_test/hr_hmax.nc", "c")
# defDim(ds, "lon", length(hr_lon))
# defDim(ds, "lat", length(hr_lat))
# lon = defVar(ds, "lon", Float64, ("lon",))
# lat = defVar(ds, "lat", Float64, ("lat",))
# lon[:] = hr_lon[:]
# lat[:] = hr_lat[:]
# hmax = defVar(ds, "hmax", Float64, ("lon","lat"))
# hmax[:,:] = hr_hmax[:,:]
# close(ds)



# using coarse resolution elevation data
elev_data = AA.earth_orography_file_path(; context = comms_ctx)
# elev_data = joinpath(topo_elev_dataset_path(), "ETOPO1_coarse.nc")

skip_pt = 10
nt = NCDataset(elev_data, "r") do ds
	lon = FT.(Array(ds["lon"]))[1:skip_pt:end]
	lat = FT.(Array(ds["lat"]))[1:skip_pt:end]
	elev = FT.(Array(ds["z"]))[1:skip_pt:end, 1:skip_pt:end]
	(; lon, lat, elev)
end
(; lon, lat, elev) = nt

# @Main.infiltrate

latidx = FT(-75) .< lat .< FT(-70)  # Exclude latitudes beyond ±80° 
lonidx = FT(-100) .< lon .< FT(-50)  # Exclude longitudes beyond ±180°
lat = lat[latidx]
lon = lon[lonidx]
elev = elev[lonidx,latidx]

# latidx = abs.(lat) .!= FT(90)  # Exclude poles
# lat = lat[latidx]
# # lon = lon[latidx]
# elev = elev[:,latidx]


# latidx = FT(-85) .< gfdl_lat .< FT(-80)  # Exclude latitudes beyond ±80° 
# lonidx = FT(50) .< gfdl_lon .< FT(55)  # Exclude longitudes beyond ±180°
# gfdl_lat = gfdl_lat[latidx]
# gfdl_lon = gfdl_lon[lonidx]
# gfdl_t11 = gfdl_t11[lonidx,latidx]
# gfdl_t12 = gfdl_t12[lonidx,latidx]
# gfdl_t21 = gfdl_t21[lonidx,latidx]
# gfdl_t22 = gfdl_t22[lonidx,latidx]


# @Main.infiltrate

earth_radius = FT(6.371229e6)
# lr_hmax = CA.calc_hpoz_latlon(elev, lon, lat, earth_radius);

# ENV["GKSwstype"] = "nul"
# p1 = contourf(gfdl_lon, gfdl_lat, gfdl_hmax[:,:,1]', clim=(-2000,2000), color=:balance);
# p2 = contourf(lr_lon, lr_lat, lr_hmax', clim = (-2000, 2000),  color=:balance);
# p = plot(p1,p2, layout = (2,1), size = (1500,1600));
# png(p, "./oro_test/lr_hmax.png")


# p1 = contourf(gfdl_lon[260 .< gfdl_lon .< 300], gfdl_lat, gfdl_hmax[260 .< gfdl_lon .< 300,:,1]', clim=(-2000,2000), color=:balance);
# p2 = contourf(lr_lon[-100 .< lr_lon .< -60], lr_lat, lr_hmax[-100 .< lr_lon .< -60,:]', clim = (-2000, 2000),  color=:balance);
# p = plot(p1,p2, layout = (2,1), size = (1500,1600));
# png(p, "./oro_test/us_hmax.png")

# 

χ = CA.calc_velocity_potential(elev, lon, lat, earth_radius);
hr_t11, hr_t21, hr_t12, hr_t22 = CA.calc_orographic_tensor(elev, χ, lon, lat, earth_radius);

# %% Calculate color limits once
ENV["GKSwstype"] = "nul"

# Pre-calculate all color limits
val = 1.0
max_abs_gfdl_t11 = maximum(abs.(gfdl_t11))*val
max_abs_gfdl_t12 = maximum(abs.(gfdl_t12))*val
max_abs_gfdl_t21 = maximum(abs.(gfdl_t21))*val
max_abs_gfdl_t22 = maximum(abs.(gfdl_t22))*val

max_abs_hr_t11 = maximum(abs.(hr_t11))*val
max_abs_hr_t12 = maximum(abs.(hr_t12))*val
max_abs_hr_t21 = maximum(abs.(hr_t21))*val
max_abs_hr_t22 = maximum(abs.(hr_t22))*val


# # %% Debug: Check for NaN values
# println("hr_t11 size: ", size(hr_t11))
# println("Number of NaN values in hr_t11: ", sum(isnan.(hr_t11)))
# println("hr_t11 range: ", extrema(hr_t11[.!isnan.(hr_t11)]))

# # %% Plot t11 comparison  
# idx_min = 10                    # Start from 10th element
# idx_max = 30

# # Find corresponding GFDL indices for the same lat/lon range
hr_lat_range = (minimum(lat), maximum(lat))
hr_lon_range = (minimum(lon .+ 360.0), maximum(lon .+ 360.0))

gfdl_lat_idx = findall(lat_val -> hr_lat_range[1] <= lat_val <= hr_lat_range[2], gfdl_lat)
gfdl_lon_idx = findall(lon_val -> hr_lon_range[1] <= lon_val <= hr_lon_range[2], gfdl_lon)

p1 = contourf(gfdl_lon[gfdl_lon_idx], gfdl_lat[gfdl_lat_idx], gfdl_t11'[gfdl_lat_idx, gfdl_lon_idx], color=:balance, title="GFDL t11");
p2 = contourf(lon, lat, hr_t11', color=:balance, title="HR t11");
p = plot(p1,p2, layout = (2,1), size = (1500,1600));
png(p, "./cut.png")


# %% Plot t11 comparison
p1 = contourf(gfdl_lon, gfdl_lat, gfdl_t11', clim=(-max_abs_gfdl_t11, max_abs_gfdl_t11), color=:balance, title="GFDL t11");
p2 = contourf(lon, lat, hr_t11', clim=(-max_abs_hr_t11, max_abs_hr_t11), color=:balance, title="HR t11");
p = plot(p1,p2, layout = (2,1), size = (1500,1600));
png(p, "./t11_comparison.png")

# %% Plot t12 comparison
p1 = contourf(gfdl_lon, gfdl_lat, gfdl_t12', clim=(-max_abs_gfdl_t12, max_abs_gfdl_t12), color=:balance, title="GFDL t12");
p2 = contourf(lon, lat, hr_t12', clim=(-max_abs_hr_t12, max_abs_hr_t12), color=:balance, title="HR t12");
p = plot(p1,p2, layout = (2,1), size = (1500,1600));
png(p, "./t12_comparison.png")

# %% Plot t21 comparison
p1 = contourf(gfdl_lon, gfdl_lat, gfdl_t21', clim=(-max_abs_gfdl_t21, max_abs_gfdl_t21), color=:balance, title="GFDL t21");
p2 = contourf(lon, lat, hr_t21', clim=(-max_abs_hr_t21, max_abs_hr_t21), color=:balance, title="HR t21");
p = plot(p1,p2, layout = (2,1), size = (1500,1600));
png(p, "./t21_comparison.png")

# %% Plot t22 comparison
p1 = contourf(gfdl_lon, gfdl_lat, gfdl_t22', clim=(-max_abs_gfdl_t22, max_abs_gfdl_t22), color=:balance, title="GFDL t22");
p2 = contourf(lon, lat, hr_t22', clim=(-max_abs_hr_t22, max_abs_hr_t22), color=:balance, title="HR t22");
p = plot(p1,p2, layout = (2,1), size = (1500,1600));
png(p, "./t22_comparison.png")

# using coarse resolution elevation data
# elev_data = joinpath(topo_elev_dataset_path(), "ETOPO1_coarse.nc")

# nt = NCDataset(elev_data, "r") do ds
# 	lr_lon = ds["longitude"][:]
# 	lr_lat = ds["latitude"][:]
# 	lr_z = FT.( ds["elevation"][:] )
# 	(; lr_lon, lr_lat, lr_z)
# end
# (; lr_lon, lr_lat, lr_z) = nt;

# latidx = abs.(lr_lat).!=FT(90) 
# lr_lat = lr_lat[latidx]
# lr_z = lr_z[:,latidx]

# earth_radius = FT(6.371229e6)
# χ = CA.calc_velocity_potential(lr_z, lr_lon, lr_lat, earth_radius);
# lr_t11, lr_t21, lr_t12, lr_t22 = CA.calc_orographic_tensor(lr_z, χ, lr_lon, lr_lat, earth_radius);