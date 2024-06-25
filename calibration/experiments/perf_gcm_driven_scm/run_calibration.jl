# load packages and functions
includet("functions.jl")
includet("floss.jl")


simulation_id = "obs_ta"

# set ensemble size and number of iterations
ens_size= 20
n_iters = 3

##### run perfect model simulation #####
config = "configs/perfect_gcm_driven.yml"
job_id = "gcm_driven_scm"
edmf_sim = gen_perfect(config, job_id)

##### LOAD or COMPUTE observational mean and covariances #####
# load observational mean and covariances 
# obs_mean = Float64.(JLD2.load_object("obs_mean.jld2"))
# obs_cov = Float64.(JLD2.load_object("obs_noise_cov.jld2"))

# compute observational mean and covariances directly with H_perf
obs_mean, obs_cov = H_perf("output/prognostic_edmfx_bomex_column/output_active")

@assert isposdef(obs_cov) # make sure positive definite or calibration will not work


# set prior
prior_entr = EKP.constrained_gaussian("entr_coeff", .4, .1, 0, 1)
prior_visc = EKP.constrained_gaussian("mixing_length_eddy_viscosity_coefficient", .2, .1, 0, 1)
prior = EKP.combine_distributions([prior_entr, prior_visc])



# ClimaCalibrate.initialize(ens_size, load_obs_mean, load_obs_cov, prior, "output/ensemble")

# set the scheduler to adaptively control EKP stepping by tuning the noise and choose a job id
scheduler=EKP.DataMisfitController(terminate_at=1)

calibrate(n_iters, ens_size, obs_mean, obs_cov, prior, scheduler, simulation_id)

# run output analysis
plot_2param(simulation_id)