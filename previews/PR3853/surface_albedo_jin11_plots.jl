# visual inspection of test/surface_albedo.jl to compare with Jin et al 2011
using ClimaAtmos
using CairoMakie

FT = Float32

# DIRECT
# values matching the paper
n_list = Array{FT}([1.34, 1.2])
θ_list = Array{FT}(collect(0:(π / 20):(π / 2)))
u_list = Array{FT}([0, 3, 12])

fig = Figure()
ax = Axis(
    fig[1, 1],
    xlabel = "θ (deg)",
    ylabel = "α_dir",
    yticks = ([0.001, 0.01, 0.1, 1], ["0.001", "0.01", "0.1", "1"]),
    yscale = log,
)
ylims!(ax, 0.001, 1.0)
xlims!(ax, -2, 92)
for n in n_list
    for wind in u_list
        α_dir_list = Array{FT}([])
        for θ_rad in θ_list
            albedo_model = ClimaAtmos.RegressionFunctionAlbedo{FT}(n = n)
            push!(
                α_dir_list,
                ClimaAtmos.surface_albedo_direct(albedo_model)(
                    FT(0),
                    cos(θ_rad),
                    FT(wind),
                ),
            )
        end
        lines!(ax, θ_list * 180 / π, α_dir_list, label = "n = $n, wind = $wind")
    end
end
axislegend(ax, position = :lt)
save("assets/direct_albedo_fig2.png", fig.scene)

# DIFFUSE
# values matching the paper
n_list = Array{FT}([1.2, 1.34, 1.45])
u_list = Array{FT}(collect(0:5:25))

fig = Figure()
ax = Axis(
    fig[1, 1],
    xlabel = "wind (m/s)",
    ylabel = "α_diff",
    yticks = (collect(0:0.02:0.1), string.(collect(0:0.02:0.1))),
)
ylims!(ax, 0.001, 0.1)
xlims!(ax, -2, 27)
for n in n_list
    α_diff_list = Array{FT}([])
    for wind in u_list
        albedo_model = ClimaAtmos.RegressionFunctionAlbedo{FT}(n = n)
        push!(
            α_diff_list,
            ClimaAtmos.surface_albedo_diffuse(albedo_model)(
                FT(0),
                FT(cos(30 * π / 180)),
                FT(wind),
            ),
        )

    end
    lines!(ax, u_list, α_diff_list, label = "n = $n")
end
axislegend(ax, position = :rt)
save("assets/diffuse_albedo_fig4.png", fig.scene)
