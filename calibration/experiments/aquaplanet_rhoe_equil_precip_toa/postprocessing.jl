import CairoMakie
import EnsembleKalmanProcesses as EKP

function phi_across_iterations(prior, eki)
    
    params = EKP.get_ϕ(prior, eki)
    n_params = length(params[1][:,1])
    n_iters = length(params)

    f = CairoMakie.Figure(size = (400*n_params, 600))
    for p in 1:n_params
        ax = CairoMakie.Axis(
            f[p, 1],
            ylabel = "$(prior.name[p])",
            xlabel = "Iteration",
        )
        for i in 1:n_iters
            param = params[i][p,:]
            CairoMakie.scatter!(ax, fill(i-1, length(param)), vec(param))
        end
    end
    output = joinpath("param_vs_iter.png")
    CairoMakie.save(output, f)
    return output
end

phi_across_iterations(prior, eki)


f = CairoMakie.Figure(resolution = (800, 600))
ax = CairoMakie.Axis(
    f[1, 1],
    ylabel = "Parameter Value",
    xlabel = "G output",
)

g = EKP.get_g(eki; return_array = true)
precip_extremes = [i[3,:] for i in g]
params = vec.(EKP.get_ϕ(prior, eki))

for (gg, uu) in zip(precip_extremes, params)
    CairoMakie.scatter!(ax, gg, uu)
end

output = joinpath(output_dir, "scatter.png")
CairoMakie.save(output, f)
