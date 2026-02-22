"""
Shared test utilities for OGW (Orographic Gravity Wave) tests.

Provides:
- `load_computed_drag`: Load computed drag data (local file → artifact fallback)
- `create_ogw_simulation`: Create OGW simulation (raw_topo → gfdl_restart fallback)

Prerequisites: callers must have already:
- `import ClimaAtmos as CA`
- `import ClimaCore: InputOutput` (or equivalent)
"""

import ClimaAtmos: AtmosArtifacts as AA

"""
    load_computed_drag(parsed_args, comms_ctx)

Load computed drag data: try local file first, fall back to ArtifactWrappers artifact.
Returns a ClimaCore Field with (hmax, hmin, t11, t12, t21, t22).
"""
function load_computed_drag(parsed_args, comms_ctx)
    (; output_filename) = CA.gen_fn(parsed_args)
    h_elem = parsed_args["h_elem"]

    # Try local file first
    local_path = joinpath(pkgdir(CA), "$(output_filename).hdf5")
    if isfile(local_path)
        @info "Loading computed drag from local file: $(local_path)"
        return CA.load_preprocessed_topography(parsed_args)
    end

    # Fall back to ClimaArtifacts
    @info "Local file not found, loading from ClimaArtifacts..."
    artifact_path = AA.ogwd_computed_drag_file_path(; h_elem, context = comms_ctx)
    @info "Loading from: $(artifact_path)"
    reader = InputOutput.HDF5Reader(artifact_path, comms_ctx)
    drag = InputOutput.read_field(reader, "computed_drag")
    Base.close(reader)
    return drag
end

"""
    create_ogw_simulation(config_file, job_id, comms_ctx; extra_parsed_args)

Create a simulation with OGW raw_topo data.
Try raw_topo (local file) first; fall back to gfdl_restart + artifact overwrite.
Returns (simulation, config).
"""
function create_ogw_simulation(
    config_file,
    job_id,
    comms_ctx;
    h_elem = 8,
    extra_parsed_args = Dict{String, Any}(),
)
    # Try raw_topo (local file) first
    try
        @info "Trying raw_topo (local file)..."
        config = CA.AtmosConfig(config_file; job_id, comms_ctx)
        config.parsed_args["h_elem"] = h_elem
        config.parsed_args["orographic_gravity_wave"] = "raw_topo"
        config.parsed_args["topography"] = "Earth"
        config.parsed_args["topo_smoothing"] = false
        config.parsed_args["topography_damping_factor"] = 1
        for (k, v) in extra_parsed_args
            config.parsed_args[k] = v
        end
        simulation = CA.get_simulation(config)
        return simulation, config
    catch e
        @warn "raw_topo failed, falling back to artifact" exception =
            (e, catch_backtrace())
    end

    # Fallback: gfdl_restart + overwrite topo_info from artifact
    config = CA.AtmosConfig(config_file; job_id, comms_ctx)
    config.parsed_args["h_elem"] = h_elem
    config.parsed_args["orographic_gravity_wave"] = "gfdl_restart"
    config.parsed_args["topography"] = "Earth"
    config.parsed_args["topo_smoothing"] = false
    config.parsed_args["topography_damping_factor"] = 1
    for (k, v) in extra_parsed_args
        config.parsed_args[k] = v
    end
    simulation = CA.get_simulation(config)

    # Load computed_drag from artifact and overwrite topo_info
    computed_drag = load_computed_drag(config.parsed_args, comms_ctx)
    topo_info = simulation.integrator.p.orographic_gravity_wave.topo_info
    parent(topo_info.hmax) .= parent(computed_drag.hmax)
    parent(topo_info.hmin) .= parent(computed_drag.hmin)
    parent(topo_info.t11) .= parent(computed_drag.t11)
    parent(topo_info.t12) .= parent(computed_drag.t12)
    parent(topo_info.t21) .= parent(computed_drag.t21)
    parent(topo_info.t22) .= parent(computed_drag.t22)

    return simulation, config
end
