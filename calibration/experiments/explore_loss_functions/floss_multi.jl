function H_perf(dir::String, short_names; noise = 0.01, cutoff=50, output_cov = true)
    """Single Variable Calibration"""
	simdir = ClimaAnalysis.SimDir(joinpath(dir, "output_active")) # allow for flexibility in choosing data directory
    means = []
    for short_name in short_names
        obs = ClimaAnalysis.get(simdir; short_name = short_name, period = "10m")
    
        # compute data
        obs_dat = reshape(mean(obs.data[cutoff:end, :, :, :], dims=(2,3)), size(obs.data)[1]-cutoff+1, size(obs.data)[end])
        # compute time mean
        obs_mean = vec(mean(obs_dat, dims=1))

        # normalize 
        obs_mean = (obs_mean .- mean(obs_mean)) ./ std(obs_mean)
        # compute covariance if required
        append!(means, obs_mean)
    end
    if output_cov 
        obs_cov = noise .* Matrix(I, length(means), length(means))
        return means, obs_cov
    else
        return means
    end
end	