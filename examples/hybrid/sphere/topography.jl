using NCDatasets
using DelimitedFiles
using Dierckx
using Interpolations

# Load ETOPO1 ice-sheet surface data
# Ocean values are considered 0
data = NCDataset("/groups/esm/asridhar/ETOPO1_Ice_g_gdal.grd")
# Unpack information
x_range = data["x_range"][:] 
y_range = data["y_range"][:]
spacing = data["spacing"][:]
dimension = data["dimension"][:]
elevation = data["z"][:]
lon = collect(x_range[1]:spacing[1]:x_range[2])
lat = collect(y_range[1]:spacing[2]:y_range[2])
nlon = dimension[1]
nlat = dimension[2]
zlevels = reshape(elevation, (nlon, nlat))

# Construct Elevation Matrix
function coarsen(X::Vector; factor = 100)
  return X[1:factor:end]
end
function coarsen(X::AbstractArray; factor = 100)
  return X[1:factor:end, 1:factor:end]
end
map_source = zlevels
map_source[map_source .< 0.0] .= 0.0

using Interpolations, Plots
elev = reverse(map_source, dims=2)
const spline_2d = LinearInterpolation((lon, lat), elev; extrapolation_bc=(Periodic(), Flat()))

"""
  earth_orography(λ,Φ)
λ = Longitude (degrees)
ϕ = Latitude (degrees)

Simple spline interpolant representation of Earth's surface
orography. Data sourced from ETOPO1 datasets (grid-referenced),
sea-ice surface information. 

"""
function earth_orography(coords)
  λ, ϕ = coords.long, coords.lat # Unpack longitude
  FT = eltype(λ)
  zₛ = FT.(spline_2d.(λ,ϕ))
  zₛ = @. zₛ < eltype(λ)(0) ? FT(0) : zₛ
  return zₛ .* FT(0.60)
end

"""
  dcmip_orography(λ,ϕ)
λ = Longitude (degrees)
ϕ = Latitude (degrees)

Given horizontal coordinates in lat-long space, 
compute and return the local elevation of the 
surface subject to warping consistent with the 
DCMIP 2-0-0.
"""
function dcmip200_orography(coords)
  λ, ϕ = coords.long, coords.lat 
  FT = eltype(λ)
  ϕₘ = FT(0) # degrees (equator)
  λₘ = FT(3/2 * 180)  # degrees
  rₘ = @. FT(acos(sind(ϕₘ)*sind(ϕ) + cosd(ϕₘ)*cosd(ϕ)*cosd(λ-λₘ))) # Great circle distance (rads)
  Rₘ = FT(3π/4) # Moutain radius
  ζₘ = FT(π/16) # Mountain oscillation half-width
  h₀ = FT(2000)
  if rₘ < Rₘ
    zₛ = FT(h₀/2) * (1 + cospi(rₘ/Rₘ)) * (cospi(rₘ/ζₘ))^2
  else
    zₛ = @. FT(0)
  end
  return zₛ
end

function schar_orography(coords)
    λ = coords.long
    FT = eltype(λ)
    return FT(0)
end

function no_orography(coords)
    λ = coords.long
    FT = eltype(λ)
    return FT(0)
end

function lift_surface(coords)
  zₛ = @. FT(500.0)
  return zₛ
end

function smooth_peak(coords)
  λ, ϕ = coords.long, coords.lat # Unpack longitude, latitude (degrees # Unpack longitude, latitude (degrees # Unpack longitude, latitude (degrees # Unpack longitude, latitude (degrees))))
  FT = eltype(λ)
  ϕₘ = FT(0) # degrees (equator)
  λₘ = FT(3/2 * 180)  # degrees
  rₘ = @. FT(acos(sind(ϕₘ)*sind(ϕ) + cosd(ϕₘ)*cosd(ϕ)*cosd(λ-λₘ))) # Great circle distance (rads)
  Rₘ = FT(3π/4) # Moutain radius
  ζₘ = FT(π/16) # Mountain oscillation half-width
  h₀ = FT(6000)
  if rₘ < Rₘ
    zₛ = FT(h₀/2) * (1 + cospi(rₘ/Rₘ)) 
  else
    zₛ = @. FT(0)
  end
  return zₛ
end


