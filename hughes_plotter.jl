using Revise
using ClimaAnalysis
using CairoMakie
import ClimaAnalysis.Utils: kwargs as ca_kwargs
import ClimaAnalysis: select, view_select, Index, NearestValue, MatchValue
using ClimaInterpolations
using Colors, ColorSchemes
import ClimaInterpolations.Interpolation1D: Linear, interpolate1d!, Flat

### USER ENTERS DIRECTORY
#dir = "/Users/akshaysridhar/Research/Data/paper_data/test_hughes/moist/output_0003"
dir_old_pgf = "./output/baroclinic_wave_hughes2023/output_00xx"
dir_exner_pgf = "./output/baroclinic_wave_hughes2023/output_0089"
dir_exner_ref_pgf = "./output/baroclinic_wave_hughes2023/output_0088"

function topography_hughes2023(lat, long)
    λ, ϕ = long, lat
    FT = eltype(λ)
    h₀ = FT(2e3)
    # Angles in degrees
    ϕ₁ = FT(45)
    ϕ₂ = FT(45)
    λ_min = minimum(λ)
    λ₁ = FT(72)
    λ₂ = FT(140)
    λₘ = FT(7)
    ϕₘ = FT(40)
    d = ϕₘ / 2 * (-log(0.1))^(-1 / 6)
    c = λₘ / 2 * (-log(0.1))^(-1 / 2)
    d₁ = (λ - λ_min) - λ₁
    d₂ = (λ - λ_min) - λ₂
    l₁ = λ - λ₁
    l₂ = λ - λ₂
    zₛ = FT(
        h₀ * (
            exp(-(((ϕ - ϕ₁) / d)^6 + (l₁ / c)^2)) +
            exp(-(((ϕ - ϕ₂) / d)^6 + (l₂ / c)^2))
        ),
    )
end

simdir = ClimaAnalysis.SimDir(dir)
pfull = get(simdir; short_name = "pfull", reduction = "inst");
# @Main.infiltrate
pfull = window(pfull, "time", by = Index(), left = 8, right = 21);
pfull0 = average_time(window(pfull, "time", by = Index(), right = 1));
pfull = average_time(pfull);
zg = get(simdir; short_name = "zg", reduction = "average");
pressure = pfull.data - pfull0.data

function offset_longitudes(longitude)
    new_longitude = mod.(longitude .+ 360, 360)
    return circshift(new_longitude, 720)
end

function offset_data(data)
    new_data = circshift(data, (720, 0))
    return new_data
end

# @Main.infiltrate

#Data

pressure = permutedims(pressure, (3, 1, 2))

#SpatialData
lat = pfull.dims["lat"]; # Index 540 for 45 degree north
lon = pfull.dims["lon"];
xvariable = lon
yvariable = lat

fontsize = 20

###### PRESSURE DIFFERENCE (SURFACE)
@show "Plotting Pressure Difference"
figsize = (1500, 400);
vertind = 49;
mycolors = reverse(colormap("RdBu"; logscale = false, mid = 0.5))
pressurelevels = -20:1:20
pressurelevels = filter(!=(0), pressurelevels)
pressureticks = -20:1:20
fig = Figure(; size = figsize)
ax = Axis(
    fig[1, 1],
    limits = (0, 245, 0, 90),
    title = "Day 4",
    xlabel = "Longitude (°E)",
    ylabel = "Latitude (°N)",
    titlesize = fontsize,
    xlabelsize = 20,
    ylabelsize = 20,
    xticklabelsize = 20,
    yticklabelsize = 20,
    xgridvisible = false,
    ygridvisible = false,
)
# @Main.infiltrate
hm1 = CairoMakie.contourf!(
    offset_longitudes(xvariable),
    yvariable,
    offset_data(pressure[1, :, :] ./ 100),
    # colormap = mycolors,
    # levels = pressurelevels,
)
CairoMakie.contour!(
    xvariable,
    yvariable,
    zg.data[1, :, :, 1] .- 15.0,
    alpha = 1,
    levels = 2,
    color = :gray,
)
Colorbar(
    fig[1, 3],
    hm1,
    label = "Sfc Pressure Perturbation (hPa)",
    labelsize = fontsize,
    ticklabelsize = fontsize,
    # ticks = pressureticks,
)
fig
CairoMakie.save(
    "paperrevision_mbw_hughes_air_pressure.pdf",
    fig,
)