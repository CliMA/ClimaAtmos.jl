using ClimaAnalysis
import ClimaAnalysis.Visualize as viz
import CairoMakie
import StatsBase: autocor

output_dir = joinpath(
    "calibration",
    "experiments",
    "aquaplanet_rhoe_equil_precip_toa",
    "generate_observations",
    "output_active",
)
simdir = SimDir(output_dir)


vars = ("rlut", "rsut", "pr")
for short_name in vars
    f = CairoMakie.Figure()
    output_var = get(simdir; short_name, period = "60d")
    zonal_avg = average_lat(average_lon(output_var))
    viz.plot!(f, zonal_avg)
    CairoMakie.save("$(short_name)_$period.png", f)

    autocorrelation = autocor(zonal_avg.data)
    @show autocorrelation

    f = CairoMakie.Figure()
    ax = CairoMakie.Axis(f[1, 1])
    CairoMakie.lines!(ax, 1:length(autocorrelation), autocorrelation)
    CairoMakie.save("$(short_name)_$(period)_autocor.png", f)
end

rlut = get(simdir; short_name = "rlut", period = "60d")
rsut = get(simdir; short_name = "rsut", period = "60d")
pr = get(simdir; short_name = "pr", period = "60d")
