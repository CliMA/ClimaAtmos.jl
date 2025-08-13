using NCDatasets
import ClimaAtmos as CA
import ClimaComms
using Plots
using Statistics: mean

using ImageFiltering
using Interpolations

include("/Users/jiahe/work/CliMA/ClimaAtmos.jl/artifacts/artifact_funcs.jl")

const FT = Float32
###
### load gfdl restart 
###
# orographic_info_rll = joinpath(topo_res_path(), "topo_drag.res.nc")
topo_path = @clima_artifact("topo_drag", ClimaComms.context(Y.c))
orographic_info_rll = joinpath(topo_path, "topo_drag.res.nc")
nt = NCDataset(orographic_info_rll, "r") do ds
	gfdl_lon = ds["lon"][:]
	gfdl_lat = ds["lat"][:]
	gfdl_hmax = ds["hmax"][:]
	gfdl_hmin = ds["hmin"][:]
	gfdl_t11 = ds["t11"][:]
	gfdl_t12 = ds["t12"][:]
	gfdl_t21 = ds["t21"][:]
	gfdl_t22 = ds["t22"][:]
	(;gfdl_lon, gfdl_lat, gfdl_hmax, gfdl_hmin, gfdl_t11, gfdl_t12, gfdl_t21, gfdl_t22)
end
(;gfdl_lon, gfdl_lat, gfdl_hmax, gfdl_hmin, gfdl_t11, gfdl_t12, gfdl_t21, gfdl_t22) = nt


###
### compute from high resolution elevation
###
# elev_data = "/Users/jiahe/work/CliMA/gravity_wave/OGW/ETOPO1_Ice_g_gmt4.grd"
elev_data = "/Users/jiahe/work/CliMA/ClimaAtmos.jl/oro_test/ETOPO1_Ice_g_gmt4_4min.nc"

nt = NCDataset(elev_data, "r") do ds
	hr_lon = ds["x"][:]
	hr_lat = ds["y"][:]
	hr_z = FT.( ds["z"][:] )
	(; hr_lon, hr_lat, hr_z)
end
(; hr_lon, hr_lat, hr_z) = nt;

latidx = abs.(hr_lat).!=FT(90) 
hr_lat = hr_lat[latidx]
hr_z = hr_z[:,latidx]

earth_radius = FT(6.371229e6)
hr_hmax = CA.calc_hmax_latlon(hr_z, hr_lon, hr_lat, earth_radius);

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
png(p, "./oro_test/hr_hmax_4min_100e3.png")

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
elev_data = joinpath(topo_elev_dataset_path(), "ETOPO1_coarse.nc")

nt = NCDataset(elev_data, "r") do ds
	lr_lon = ds["longitude"][:]
	lr_lat = ds["latitude"][:]
	lr_z = FT.( ds["elevation"][:] )
	(; lr_lon, lr_lat, lr_z)
end
(; lr_lon, lr_lat, lr_z) = nt;

latidx = abs.(lr_lat).!=FT(90) 
lr_lat = lr_lat[latidx]
lr_z = lr_z[:,latidx]

earth_radius = FT(6.371229e6)
lr_hmax = CA.calc_hmax_latlon(lr_z, lr_lon, lr_lat, earth_radius);

ENV["GKSwstype"] = "nul"
p1 = contourf(gfdl_lon, gfdl_lat, gfdl_hmax[:,:,1]', clim=(-2000,2000), color=:balance);
p2 = contourf(lr_lon, lr_lat, lr_hmax', clim = (-2000, 2000),  color=:balance);
p = plot(p1,p2, layout = (2,1), size = (1500,1600));
png(p, "./oro_test/lr_hmax.png")


p1 = contourf(gfdl_lon[260 .< gfdl_lon .< 300], gfdl_lat, gfdl_hmax[260 .< gfdl_lon .< 300,:,1]', clim=(-2000,2000), color=:balance);
p2 = contourf(lr_lon[-100 .< lr_lon .< -60], lr_lat, lr_hmax[-100 .< lr_lon .< -60,:]', clim = (-2000, 2000),  color=:balance);
p = plot(p1,p2, layout = (2,1), size = (1500,1600));
png(p, "./oro_test/us_hmax.png")

# 
χ = CA.calc_velocity_potential(hr_z, hr_lon, hr_lat, earth_radius);
hr_t11, hr_t21, hr_t12, hr_t22 = CA.calc_orographic_tensor(hr_z, χ, hr_lon, hr_lat, earth_radius);

# using coarse resolution elevation data
elev_data = joinpath(topo_elev_dataset_path(), "ETOPO1_coarse.nc")

nt = NCDataset(elev_data, "r") do ds
	lr_lon = ds["longitude"][:]
	lr_lat = ds["latitude"][:]
	lr_z = FT.( ds["elevation"][:] )
	(; lr_lon, lr_lat, lr_z)
end
(; lr_lon, lr_lat, lr_z) = nt;

latidx = abs.(lr_lat).!=FT(90) 
lr_lat = lr_lat[latidx]
lr_z = lr_z[:,latidx]

earth_radius = FT(6.371229e6)
χ = CA.calc_velocity_potential(lr_z, lr_lon, lr_lat, earth_radius);
lr_t11, lr_t21, lr_t12, lr_t22 = CA.calc_orographic_tensor(lr_z, χ, lr_lon, lr_lat, earth_radius);