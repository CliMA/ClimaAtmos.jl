function H_perf(dir::String, short_names; noise = 0.01, obs_length = 40, cutoff=50, output_cov = true)
    """Single Variable Calibration"""
	simdir = ClimaAnalysis.SimDir(joinpath(dir, "output_active")) # allow for flexibility in choosing data directory
    means = []
    covs = Vector{AbstractMatrix{Float64}}()
    for short_name in short_names
        obs = ClimaAnalysis.get(simdir; short_name = short_name, period = "10m")
    
        # compute data
        obs_dat = reshape(mean(obs.data[cutoff:end, :, :, :], dims=(2,3)), size(obs.data)[1]-cutoff+1, size(obs.data)[end])
        # compute time mean
        obs_mean = vec(mean(obs_dat, dims=1))

        # normalize 
        #obs_mean = (obs_mean .- mean(obs_mean)) ./ std(obs_mean)
        # compute covariance if required
        append!(means, obs_mean)
        if output_cov
            # chunk the longrun observation by the same number of windows as the simulation (need to hand pick this value based on length)
            obs_cov = cov([vec(mean(obs_dat[i:i+obs_length, :], dims=1)) for i in 1:Integer(floor(size(obs_dat)[1]/obs_length))])
            #obs_cov = cov(obs_dat)
            # ensure positive definiteness
            #if !isposdef(obs_cov)
            obs_cov = obs_cov + (abs(minimum(eigvals(obs_cov)))+maximum(diag(obs_cov))*.001) .* I
            #end
            push!(covs, obs_cov)
        end
            
    end
    if output_cov 
        #obs_cov = Matrix(I, length(means), length(means))
        obs_covs = Matrix(BlockDiagonal(covs))
        return means, obs_covs
    else
        return means
    end
end	

# function H_perf(dir::String, short_names; noise = 0.01, obs_length = 40, cutoff=240, cutoff_max = 264, output_cov = true)
#     """Single Variable Calibration"""
# 	simdir = ClimaAnalysis.SimDir(joinpath(dir, "output_active")) # allow for flexibility in choosing data directory
#     means = []
#     covs = Vector{AbstractMatrix{Float64}}()
#     for short_name in short_names
#         obs = ClimaAnalysis.get(simdir; short_name = short_name, period = "10m")
    
#         # compute data
#         #return mean(obs.data[cutoff:cutoff_max, :, :, :], dims=(2,3))
#         obs_dat = reshape(mean(obs.data[cutoff:cutoff_max, :, :, :], dims=(2,3)), cutoff_max - cutoff+1, size(obs.data)[end])
#         # compute time mean
#         obs_mean = vec(mean(obs_dat, dims=1))

#         # normalize 
#         #obs_mean = (obs_mean .- mean(obs_mean)) ./ std(obs_mean)
#         # compute covariance if required
#         append!(means, obs_mean)
#         if output_cov
#             # chunk the longrun observation by the same number of windows as the simulation (need to hand pick this value based on length)
#             #obs_cov = cov([vec(mean(obs_dat[i:i+obs_length, :], dims=1)) for i in 1:Integer(floor(size(obs_dat)[1]/obs_length))])
#             obs_cov = cov(obs_dat)
#             # ensure positive definiteness
#             #if !isposdef(obs_cov)
#             obs_cov = obs_cov + (abs(minimum(eigvals(obs_cov)))+maximum(diag(obs_cov))*.001) .* I
#             #end
#             push!(covs, obs_cov)
#         end
            
#     end
#     if output_cov 
#         #obs_cov = Matrix(I, length(means), length(means))
#         obs_covs = Matrix(BlockDiagonal(covs))
#         return means, obs_covs
#     else
#         return means
#     end
# end	

# function H_perf2(dir::String, short_names; noise = 0.01, obs_length = 40, cutoff=240, cutoff_max = 264, output_cov = true)
#     """Single Variable Calibration"""
# 	simdir = ClimaAnalysis.SimDir(joinpath(dir, "output_active")) # allow for flexibility in choosing data directory
#     means = []
#     covs = Vector{AbstractMatrix{Float64}}()
#     for short_name in short_names
#         obs = ClimaAnalysis.get(simdir; short_name = short_name, period = "10m")
    
#         # compute data
#         #return mean(obs.data[cutoff:cutoff_max, :, :, :], dims=(2,3))
#         obs_dat = reshape(mean(obs.data[:, :, :, :], dims=(2,3)), size(obs.data)[1], size(obs.data)[end])
#         # compute time mean
#         obs_mean = vec(mean(obs_dat, dims=1))

#         # normalize 
#         #obs_mean = (obs_mean .- mean(obs_mean)) ./ std(obs_mean)
#         # compute covariance if required
#         append!(means, obs_mean)
#         if output_cov
#             # chunk the longrun observation by the same number of windows as the simulation (need to hand pick this value based on length)
#             #obs_cov = cov([vec(mean(obs_dat[i:i+obs_length, :], dims=1)) for i in 1:Integer(floor(size(obs_dat)[1]/obs_length))])
#             obs_cov = cov(obs_dat)
#             # ensure positive definiteness
#             #if !isposdef(obs_cov)
#             obs_cov = obs_cov + (abs(minimum(eigvals(obs_cov)))+maximum(diag(obs_cov))*.001) .* I
#             #end
#             push!(covs, obs_cov)
#         end
            
#     end
#     if output_cov 
#         #obs_cov = Matrix(I, length(means), length(means))
#         obs_covs = Matrix(BlockDiagonal(covs))
#         return means, obs_covs
#     else
#         return means
#     end
# end	