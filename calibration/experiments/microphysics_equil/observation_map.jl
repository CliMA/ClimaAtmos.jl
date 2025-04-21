### to go in here:
# - observation_map function



"""
version from the TUTORIAL
"""
const days = 86_400
function CAL.observation_map(iteration)
    single_member_dims = (1,)
    G_ensemble = Array{Float64}(undef, single_member_dims..., ensemble_size)

    for m in 1:ensemble_size
        member_path = CAL.path_to_ensemble_member(output_dir, iteration, m)
        simdir_path = joinpath(member_path, "output_active")
        if isdir(simdir_path)
            simdir = SimDir(simdir_path)
            G_ensemble[:, m] .= process_member_data(simdir)
        else
            G_ensemble[:, m] .= NaN
        end
    end
    return G_ensemble
end

# make the loss function daily averaged liquid fraction
function process_member_data(simdir::SimDir)
    isempty(simdir.vars) && return NaN
    cli =
        get(simdir; short_name = "cli", reduction = "average", period = "1days")
    clw = 
        get(simdir; short_name = "clw", reduction = "average", period = "1days")

    liquid_fraction = clw / (cli + clw)

    return slice(average_xy(liquid_fraction); time = 1days).data
end
