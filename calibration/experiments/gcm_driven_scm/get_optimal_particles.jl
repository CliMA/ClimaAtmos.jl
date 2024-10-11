import TOML, YAML
import JLD2
using Distributions
import EnsembleKalmanProcesses as EKP
using EnsembleKalmanProcesses.ParameterDistributions
using EnsembleKalmanProcesses.TOMLInterface
import ClimaCalibrate as CAL
using Plots
using LinearAlgebra
using DataFrames
using Statistics

# output_dir = "output/exp_1"
iterations = nothing
output_dir = "output_deep_conv_etki_exp/exp_13" # output directory
iteration = 6 # iteration to get optimal particles
final_iter = iteration + 1

n_lowest = 15

include("helper_funcs.jl")

prior_dict = TOML.parsefile(joinpath(output_dir, "configs", "prior.toml"))

function serialize_dict(dict)
    serialized = Dict{String, Any}()
    for (key, value) in dict
        if isa(value, Dict)
            serialized[key] = serialize_dict(value)
        elseif isa(value, Vector)
            serialized[key] = [isa(v, Union{Int, Float64, String, Bool, Array}) ? v : string(v) for v in value]
        elseif isa(value, Union{Int, Float64, String, Bool})
            serialized[key] = value
        else
            serialized[key] = string(value)
        end
    end
    return serialized
end

serializable_dict = serialize_dict(prior_dict)

# Open a file and write the serialized TOML dictionary
open("tessy_prior.toml", "w") do io
    TOML.print(io, serializable_dict)
end




# TOML.write("tessy_prior.toml", prior_dict)


const prior = CAL.get_prior(joinpath(output_dir, "configs", "prior.toml"))


### print best particles and loss
@info "Best particle in final iteration and loss"
@info "Final Iteration: $final_iter"

iter_path = CAL.path_to_iteration(output_dir, iteration)
eki = JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))

iter_path_p1 = CAL.path_to_iteration(output_dir, iteration + 1)
eki_p1 = JLD2.load_object(joinpath(iter_path_p1, "eki_file.jld2"))

Δt = EKP.get_Δt(eki)
y_i = EKP.get_obs(eki)
gamma_i = EKP.get_obs_noise_cov(eki)

g = EKP.get_g_final(eki_p1)

nan_columns_count = sum(any(isnan, g, dims = 1))

# Find successful simulations
non_nan_columns_indices = findall(x -> !x, vec(any(isnan, g, dims = 1)))
g_non_nan = g[:, non_nan_columns_indices]

y_diff = y_i .- g_non_nan

u = EKP.get_u_final(eki)
N_obs = size(g, 1)
cov_init = cov(u, dims = 2)

loss_min_inds, loss_min_vals = get_loss_min(output_dir, iteration, n_lowest = n_lowest)

u_best = u[:,loss_min_inds]

df = DataFrame(ParticleNumber = loss_min_inds, Loss = loss_min_vals)
display(df)

u_best_mean = mean(u_best, dims = 2)
u_best_std = std(u_best, dims = 2)

phi_best = EKP.transform_unconstrained_to_constrained(prior, u_best)

phi_best_mean = mean(phi_best, dims = 2)
phi_best_std = std(phi_best, dims = 2)

 
### find ensemble member nearest to be mean 

mean_diff = sum(abs, u .- u_best_mean, dims = 1)
nearest_mean_index = argmin(mean_diff)  
col_index = nearest_mean_index[2]

u_nearest = u[:, col_index]
phi_nearest = EKP.transform_unconstrained_to_constrained(prior, u_nearest)
g_nearest = g[:, col_index]

@info "Ensemble member nearest to the mean:"
@info "Particle Number: $col_index"
@info "u values: $u_nearest"
@info "phi values: $phi_nearest"




names = []
for i in 1:length(prior.name)
    if isa(prior.distribution[i], EKP.Parameterized)
        push!(names, prior.name[i])
    elseif isa(prior.distribution[i], EKP.VectorOfParameterized)
        for j in 1:length(prior.distribution[i].distribution)
            push!(names, prior.name[i] * "_$j")
        end
    end
end


# println("Best Particle Parameters:")
# for ii = 1:length(names)
#     println("Parameter: $(names[ii])")
#     if occursin("turb_entr_param", names[ii])
#         println("Mean: $(u_best_mean[ii])")
#         println("Std: $(u_best_std[ii])")
#     end
#     println("Mean: $(phi_best_mean[ii])")
#     println("Std: $(phi_best_std[ii])")
# end


println("Nearest Particle Parameters:")
for ii = 1:length(names)
    println("Parameter: $(names[ii])")
    if occursin("turb_entr_param", names[ii])
        println("Mean: $(u_nearest[ii])")
        # println("Std: $(u_best_std[ii])")
    end
    println("Mean: $(phi_nearest[ii])")
    # println("Std: $(phi_best_std[ii])")
end


const prior_config = TOML.parsefile(joinpath(output_dir, "configs", "prior.toml"))

prior_config_out = deepcopy(prior_config)
for (name_i, prior_i) in pairs(prior_config)
    # println("Key: $name_i, Value: $prior")
    if occursin("VectorOfParameterized", prior_i["prior"])
        contraint_i = get(prior_i, "constraint", nothing)
        # @show prior
        # @show contraint_i
    end
    # else occursin("constrained_gaussian", prior_i["prior"])
    # @show prior_i["prior"]
end

# for prior_param_name in names
#     prior_config[prior_param_name]
# end



# if !isnothing(contraint_i)
#     if occursin("no_contraint", contraint_i)
#         println("Constrained Gaussian")
#     end
# end





# param_dict = CAL.get_param_dict(prior)
# save_parameter_ensemble(
#     u_best_mean,
#     prior,
#     param_dict,
#     output_dir,
#     "test_write_parameters.toml",
#     iteration)

# param_dict = get_param_dict(prior)
# save_parameter_ensemble(
#     EKP.get_u_final(eki),
#     prior,
#     param_dict,
#     output_dir,
#     "test_write_parameters.toml",
#     iteration,
# )