using MPI
import ClimaCalibrate as CAL
import EnsembleKalmanProcesses.ParameterDistributions: constrained_gaussian
import ClimaAnalysis: SimDir, get, slice, average_xy
import JLD2

function initialize_mpi()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    return comm, rank, size
end

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

function process_member_data(simdir::SimDir)
    isempty(simdir.vars) && return NaN
    rsut =
        get(simdir; short_name = "rsut", reduction = "average", period = "30d")
    return slice(average_xy(rsut); time = 30).data
end


function get_rank_members(ensemble_size, rank, size)
    # Number of members per rank, rounded up
    members_per_rank = ceil(Int, ensemble_size / size)
    start_idx = rank * members_per_rank + 1
    end_idx = min((rank + 1) * members_per_rank, ensemble_size)
    return start_idx:end_idx
end

function run_ensemble_member(m, iter, config)
    try
        model_config = CAL.set_up_forward_model(m, iter, config)
        return CAL.run_forward_model(model_config)
    catch e
        @error "Error running member $m" exception=e
        return e
    end
end

function run_iteration(config, iter, comm, rank, size)
    # Get members assigned to this rank
    local_members = get_rank_members(config.ensemble_size, rank, size)
    
    # Run assigned members
    local_results = Dict{Int, Any}()
    for m in local_members
        if rank == 0
            @info "Running member $m on rank $rank ($(length(local_members)) members total)"
        end
        result = run_ensemble_member(m, iter, config)
        local_results[m] = result
    end
    
    # Gather results from all ranks
    if rank == 0
        # Root process collects results
        ensemble_results = Dict{Int, Any}()
        # Add local results first
        merge!(ensemble_results, local_results)
        
        # Receive results from other ranks
        for src in 1:(size-1)
            remote_results, status = MPI.recv(src, 0, comm)
            merge!(ensemble_results, remote_results)
        end
        
        # Verify we received all members
        @assert length(ensemble_results) == config.ensemble_size "Missing ensemble members: expected $(config.ensemble_size), got $(length(ensemble_results))"
        
        # Check for failures
        results = values(ensemble_results)
        if all(isa.(results, Exception))
            error("Full ensemble for iter $iter failed")
        end
        
        # Process results
        G_ensemble = CAL.observation_map(iter)
        CAL.save_G_ensemble(config, iter, G_ensemble)
        CAL.update_ensemble(config, iter)
        
        # Load and return EKI file
        iter_path = CAL.path_to_iteration(config.output_dir, iter)
        return JLD2.load_object(joinpath(iter_path, "eki_file.jld2"))
    else
        # Worker processes send their results to root
        MPI.send(local_results, 0, 0, comm)
        return nothing
    end
end

function calibrate(config)
    comm, rank, size = initialize_mpi()
    
    if rank == 0
        CAL.initialize(config)
        @info "Starting calibration with $size MPI ranks for $(config.ensemble_size) ensemble members"
    end
    MPI.Barrier(comm)
    
    for iter in 0:config.n_iterations
        if rank == 0
            @info "Starting iteration $iter"
        end
        
        if rank == 0
            for r in 0:(size-1)
                r_members = get_rank_members(config.ensemble_size, r, size)
                @info "Rank $r will process members $r_members"
            end
        end
        
        result = run_iteration(config, iter, comm, rank, size)
        
        if rank == 0
            @info "Completed iteration $iter"
        end
        MPI.Barrier(comm)
    end
    
    MPI.Finalize()
end

# Main program
if abspath(PROGRAM_FILE) == @__FILE__
    import ClimaAtmos as CA
    import LinearAlgebra: I
    
    # Setup paths and configuration
    experiment_dir = joinpath(pkgdir(CA), "calibration", "test")
    model_interface = joinpath(pkgdir(CA), "calibration", "model_interface.jl")
    output_dir = "calibration_end_to_end_test"
    include(model_interface)
    
    comm, rank, size = initialize_mpi()
    
    if rank == 0
        # Generate observations if needed
        obs_path = joinpath(experiment_dir, "observations.jld2")
        if !isfile(obs_path)
            @info "Generating observations"
            atmos_config = CA.AtmosConfig(joinpath(experiment_dir, "model_config.yml"); comms_ctx = ClimaComms.SingletonCommsContext())
            simulation = CA.get_simulation(atmos_config)
            CA.solve_atmos!(simulation)
            observations = Vector{Float64}(undef, 1)
            observations .= process_member_data(SimDir(simulation.output_dir))
            JLD2.save_object(obs_path, observations)
        end
    end
    MPI.Barrier(comm)
    
    # Load configuration
    ensemble_size = 50
    # Prevent all ranks trying to load object
    observations = rank == 0 ? JLD2.load_object(obs_path) : []
    observations = MPI.bcast(observations, 0, comm)
    noise = 0.1 * I
    n_iterations = 10
    prior = constrained_gaussian("astronomical_unit", 60_000_000_000, 100_000_000_000, 200_000, Inf)
    
    config = CAL.ExperimentConfig(;
        n_iterations,
        ensemble_size,
        observations,
        noise,
        output_dir,
        prior,
    )
    
    calibrate(config)
end
