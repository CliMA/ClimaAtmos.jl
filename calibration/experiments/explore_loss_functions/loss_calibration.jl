# load packages and functions
includet("functions.jl")
includet("floss.jl")

simulation_id = "obs_" * string(ARGS[1])
println("Running simulation $simulation_id...")
# simulation_id = "obs_ta"
# simulation_id = "obs_hus"
# simulation_id = "obs_clw"

# set ensemble size and number of iterations
ens_size= 15
n_iters = 5

##### run perfect model simulation #####
# config = "configs/perfect_prog_bomex.yml"
# job_id = "prognostic_edmfx_bomex_column"
# edmf_sim = gen_perfect(config, job_id)

##### LOAD or COMPUTE observational mean and covariances #####
# load observational mean and covariances 
# obs_mean = Float64.(JLD2.load_object("obs_mean.jld2"))
# obs_cov = Float64.(JLD2.load_object("obs_noise_cov.jld2"))

# compute observational mean and covariances directly with H_perf
obs_mean, obs_cov = H_perf("output/prognostic_edmfx_bomex_column", string(ARGS[1]))

@assert isposdef(obs_cov) # make sure positive definite or calibration will not work


# set prior
prior_entr = EKP.constrained_gaussian("entr_coeff", .4, .1, 0, 1)
prior_visc = EKP.constrained_gaussian("mixing_length_eddy_viscosity_coefficient", .2, .1, 0, 1)
prior = EKP.combine_distributions([prior_entr, prior_visc])



# ClimaCalibrate.initialize(ens_size, load_obs_mean, load_obs_cov, prior, "output/ensemble")

# set the scheduler to adaptively control EKP stepping by tuning the noise and choose a job id
#scheduler=EKP.DataMisfitController(terminate_at=1)
scheduler =  EKP.DefaultScheduler(1)

calibrate(n_iters, ens_size, obs_mean, obs_cov, prior, scheduler, simulation_id, string(ARGS[1]))

# run output analysis
plot_2param(simulation_id)


if false
    ens_size = 7
    n_iters = 3
    id = "ta"
    simulation_id = "obs_" * id
    obs_mean, obs_cov = H_perf("output/prognostic_edmfx_bomex_column", id)
    @assert isposdef(obs_cov) # make sure positive definite or calibration will not work
    calibrate(n_iters, ens_size, obs_mean, obs_cov, prior, scheduler, simulation_id, id)
    plot_2param(simulation_id)
end