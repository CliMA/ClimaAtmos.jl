# can run ts, ta, ua, va, wa, hur, hus, cl, clw, entr, detr # e.g., anything from diagnostics 
function H_perf(dir::String, short_name; cutoff=50, output_cov = true)
    """Single Variable Calibration"""
	simdir = ClimaAnalysis.SimDir(joinpath(dir, "output_active")) # allow for flexibility in choosing data directory
	obs = ClimaAnalysis.get(simdir; short_name = short_name, period = "10m")
    
	# compute data
	obs_dat = reshape(mean(obs.data[cutoff:end, :, :, :], dims=(2,3)), size(obs.data)[1]-cutoff+1, size(obs.data)[end])
	# compute means
	obs_mean = vec(mean(obs_dat, dims=1))

    # normalize 
    obs_mean = obs_mean ./ mean(obs_mean)
	# compute covariance if required
    if output_cov
        # add diagonal to covariance matrix to ensure positive definiteness
        obs_cov = .01 .* Matrix(I, length(obs_mean), length(obs_mean)) #cov(obs_dat)
        # print shape of obs_cov and obs_mean

        return 	obs_mean, obs_cov
    else
        return obs_mean
    end
end	
# function H_perf(dir::String, short_name; cutoff=50, output_cov = true)
#     """Single Variable Calibration"""
# 	simdir = ClimaAnalysis.SimDir(joinpath(dir, "output_active")) # allow for flexibility in choosing data directory
	
# 	obs = ClimaAnalysis.get(simdir; short_name = short_name, period = "10m")

# 	# compute data
# 	obs_dat = reshape(mean(obs.data[cutoff:end, :, :, :], dims=(2,3)), size(obs.data)[1]-cutoff+1, size(obs.data)[end])

# 	# compute means
# 	obs_mean = vec(mean(obs_dat, dims=1))

# 	# compute covariance if required
#     if output_cov
#         # add diagonal to covariance matrix to ensure positive definiteness
#         obs_cov = cov(obs_dat) 
#         obs_cov = obs_cov + .01 * mean(obs_cov) * I
#         return 	obs_mean, obs_cov
#     else
#         return obs_mean
#     end
# end	

# covariance between sst and lwp. ts and lwp. 
# seasonal cycle 
# 

# perfect model observation map for ta and hus 2 parameter calibration 
# function H_perf(dir::String; cutoff=50, output_cov = true)
# 	simdir = SimDir(dir) #joinpath(dir, "output_active") # allow for flexibility in choosing data directory
	
# 	ta = ClimaAnalysis.get(simdir; short_name = "ta", period = "10m")
# 	hus = ClimaAnalysis.get(simdir; short_name = "hus", period = "10m")

# 	# compute data
# 	ta_dat = reshape(mean(ta.data[cutoff:end, :, :, :], dims=(2,3)), size(ta.data)[1]-cutoff+1, size(ta.data)[end])

# 	hus_dat = reshape(mean(hus.data[cutoff:end, :, :, :], dims=(2,3)), size(hus.data)[1]-cutoff+1, size(hus.data)[end]) 

# 	# compute means
# 	ta_mean = mean(ta_dat, dims=1)
# 	hus_mean = mean(hus_dat, dims = 1)

# 	# scale hus to be same order of magnitude
# 	hus_mean = hus_mean * 3e4 #mean(ta_mean) / mean(hus_mean)

# 	# append
# 	vec_means = [vec(ta_mean); vec(hus_mean)]

# 	# compute covariance if required
#     if output_cov
#         ta_cov = cov(ta_dat)
#         hus_cov = cov(hus_dat) * 5e5#* norm(ta_cov) /norm(cov(hus_dat))
#         cov_mat = BlockDiagonal([ta_cov, hus_cov]) + 0.001 * I
#         return 	vec_means, cov_mat
#     else
#         return vec_means
#     end
# end	