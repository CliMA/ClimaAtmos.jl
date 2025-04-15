### to go in here:
# - generate synthetic observation
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import Random
Random.seed!(1234)

using NCDatasets
using Dates
using Statistics
using LinearAlgebra
using Distributions
using Random
import EnsembleKalmanProcesses as EKP
using ClimaCalibrate
import ClimaAnalysis
import ClimaAnalysis: Visualize as viz

FT = Float64

"""
Generate perfect model simulation. In initial test case, run baroclinic wave
w/equil microphysics parameterization. Will have pow_ice = 1 be our
input parameter.
"""
function generate_bw(config_file, job_id)
    # to do:
    #   - change toml to the updated parameter value (???)
    #   - otherwise looks mostly correct?

    config = CA.AtmosConfig(config_file; job_id)
    simulation = CA.AtmosSimulation(config)
    sol_res = CA.solve_atmos!(simulation)

    # return sol_res?

end


"""
Add noise to model truth
"""
function synthetic_observed_y(sim_path, reduction)

    # - load the model output

    simdir = ClimaAnalysis.SimDir(sim_path)  
    cli = get(simdir; short_name = "cli", reduction = reduction)
    clw = get(simdir; short_name = "clw", reduction = reduction)

    # liquid fraction loss function
    y = clw / (clw + cli)

    # - create noise (I need to remember how to make this Gamma.)
    Γ = FT(0.003)^2 * I * (maximum(y.data) - minimum(y.data))

    noise_dist = MvNormal(zeros(1), Γ)
    #rand(noise_dist)
    apply_noise!(y, noise_dist) = y.data + rand(noise_dist)[1]
    # broadcast the noise to each element of y
    y_noisy = apply_noise!(y, noise_dist)

    # my old version of this
    #Γy = 0.01 * (maximum(y_t) - minimum(y_t)) + 0.0001# * sol[2,:] .+ I(size(sol[2,:], 1))*0.001
    #Γy = convert(Array, LinearAlgebra.Diagonal([0.01 * sol[1], 0.01 * sol[2]]))
    #μ = zeros(length(sol))

    #for i in 1:n_samples

        # do for each t in y????

    #    for t in 1:length(sol)

    #        y_t[t,i] = sol[t] + rand(Distributions.Normal(0, Γy))

    #    end

        #Γy = 0.01 * sol[2,:]
        # generate a bunch of noise
        #Distributions.MvNormal(μ, Γy)
        #noise = randn(μ, Γy)
        #y = sol[2,:] + noise
        #push!(y_t,y)
    #end

    # - save noisey output
    JLD2.save_object(
        joinpath(sim_path, "bw_equil_slf_noisy.jld2"), # probably dont save that way
        y_noisy,
    )
    return y, y_noisy

end

"""
write script to run the baroclinic wave model and generate noise
"""

#config_file = "baroclinic_wave_equil.yml"
#job_id = "baroclinic_wave_equil"

#generate_bw(config_file, job_id)

sim_path = "/Users/oliviaalcabes/Documents/research/microphysics/bw_sims/output_0000"
y, y_noisy = synthetic_observed_y(sim_path, "average")