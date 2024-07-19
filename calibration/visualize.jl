import EnsembleKalmanProcesses as EKP
using JLD2
using ClimaAnalysis
import ClimaCalibrate
using CairoMakie


function extract_array(params, idx, n_iters, ens_size)
    arr = Array{Float64}(undef, ens_size, n_iters+1)
    for i in 1:(n_iters+1)
        arr[:, i] = params[i][idx, :]
    end
    arr
end

function get_results(job_id, nvars, n_iters, ens_size)
    #eki_objs = map(i -> load_object(joinpath(ClimaCalibrate.path_to_iteration("output/$job_id", i), "eki_file.jld2")), 0:(n_iters+1))
    eki_obj = load_object(joinpath(ClimaCalibrate.path_to_iteration("output/$job_id", n_iters), "eki_file.jld2"))
    # load parameters
    prior = ClimaCalibrate.get_prior("prior.toml")
    params = EKP.transform_unconstrained_to_constrained(prior, EKP.get_u(eki_obj))

    # reshape params 
    ar = []
    for i in 1:nvars
        push!(ar, extract_array(params, i, n_iters, ens_size))
    end
    ar 
end

function plot_results(job_id, nvars, n_iters, ens_size)
    ar = get_results(job_id, nvars, n_iters, ens_size)
    f = Figure(size = (1000, 500))

    for var in 1:nvars
        ind = 1 
        if var > 3 
            ind = 2
        ax = Axis(f[1,var],
            title = "Entrainment Coefficient Calibration",
            xlabel = "Iteration",
            ylabel = "Entrainment Coefficient")
        
        for i in 1:ens_size
            lines!(ax, 0:n_iters, ar[var][i, :], color = (:red, .3))
        end
    end


    # ax1 = Axis(f[1,1],
    #     title = "Entrainment Coefficient Calibration",
    #     xlabel = "Iteration",
    #     ylabel = "Entrainment Coefficient")

    # hlines!(ax1, .3, color = :blue, label="True Entrainment Rate")

    # ax2 = Axis(f[1,2],
    #     title = "Eddy Viscosity Calibration",
    #     xlabel = "Iteration",
    #     ylabel = "Eddy Viscosity")

    # hlines!(ax2, .14, color = :blue, label="True Eddy Viscosity")

    # for i in 1:ens_size
    #     lines!(ax1, 0:n_iters, ar[1][i, :], color = (:red, 0.3))
    #     lines!(ax2, 0:n_iters, ar[2][i, :], color = (:red, 0.3))
    # end
    # save in two locations to ensure no overwriting
    save("plots/2param_calibration.png", f)
    save("output/$job_id/2param_calibration.png", f)
    jldsave("output/$job_id/2param_calibration.jld2"; ar)
    f
end