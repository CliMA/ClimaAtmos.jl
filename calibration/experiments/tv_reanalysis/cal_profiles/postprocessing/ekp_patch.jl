"""
$(TYPEDSIGNATURES)

if `build=true` then gets the observed sample, stacked over the minibatch at `iteration`. `build=false` lists the `samples` for all observations. If `isnothing(iteration)` or not defined then the current iteration is used.
"""
function get_obs(os::OS, iteration::IorN; build = true) where {OS <: ObservationSeries, IorN <: Union{Int, Nothing}}
    minibatch = get_minibatch(os, iteration)
    minibatch_length = length(minibatch)
    observations_vec = get_observations(os)[minibatch] # gives observation objects
    if minibatch_length == 1
        return get_obs(observations_vec[1], build = build)
    end

    if !build # return y as vec of vecs
        return get_obs.(observations_vec, build = false)
    else # stack y
        sample_lengths = [length(get_obs(ov, build = true)) for ov in observations_vec]
        minibatch_samples = zeros(sum(sample_lengths))
        for (i, observation) in enumerate(observations_vec)
            idx = (sum(sample_lengths[1:(i - 1)]) + 1):sum(sample_lengths[1:i])
            minibatch_samples[idx] = get_obs(observation, build = true)
        end
        return minibatch_samples
    end

end


"""
$(TYPEDSIGNATURES)

if `build=true` then gets the observed sample, stacked over the current minibatch. `build=false` lists the `samples` for all observations. 
"""
get_obs(os::OS; kwargs...) where {OS <: ObservationSeries} = get_obs(os, nothing; kwargs...)

"""
$(TYPEDSIGNATURES)

get the minibatch for a given minibatch index (`Dict("epoch"=> x, "minibatch" => y)`), or iteration `Int`. If `nothing` is provided as an iteration then the current minibatch is returned
"""
function get_minibatch(os::OS, it_or_mbi::IorDorN) where {OS <: ObservationSeries, IorDorN <: Union{Int, Dict, Nothing}}
    if isnothing(it_or_mbi)
        return get_current_minibatch(os)
    else
        index = isa(it_or_mbi, Dict) ? it_or_mbi : get_minibatch_index(os, it_or_mbi)
        minibatches = get_minibatches(os)
        epoch = index["epoch"]
        mini = index["minibatch"]
        return minibatches[epoch][mini]
    end
end


"""
$(TYPEDSIGNATURES)

returns the minibatch_index `Dict("epoch"=> x, "minibatch" => y)`, for a given `iteration` 
"""
function get_minibatch_index(os::OS, iteration::Int) where {OS <: ObservationSeries}
    len_epoch = get_length_epoch(os)
    return Dict("epoch" => ((iteration - 1) ÷ len_epoch) + 1, "minibatch" => ((iteration - 1) % len_epoch) + 1)
end

"""
$(TYPEDSIGNATURES)

gets the `minibatches` field from the `FixedMinibatcher` object
"""
get_minibatches(m::FM) where {FM <: FixedMinibatcher} = m.minibatches

"""
$(TYPEDSIGNATURES)

gets the number of minibatches in an epoch
"""
function get_length_epoch(os::OS) where {OS <: ObservationSeries}
    return length(get_minibatches(os)[1])
end