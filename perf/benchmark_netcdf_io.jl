import ClimaAtmos as CA
import ClimaAtmos.Diagnostics as CAD
import ClimaComms
import Profile, ProfileCanvas
import NCDatasets
import Base: rm
using BenchmarkTools

# This script runs our NetCDF writer and compares its performance with
# NCDatasets. It also produces a flamegraph for IO.

# Number of target interpolation point along each dimension
const NUM = 100

timings = Dict{ClimaComms.AbstractDevice, Any}(
    ClimaComms.CPUSingleThreaded() => missing,
)
# If a GPU is available, runs on CPU and GPU
if ClimaComms.device() isa ClimaComms.CUDADevice
    timings[ClimaComms.CUDADevice()] = missing
end

timings_ncdataset = copy(timings)

function add_nc(nc, outarray)
    v = nc["rhoa"]
    temporal_size, spatial_size... = size(v)
    time_index = temporal_size + 1
    nc["time"][time_index] = 10.0 * time_index
    v[time_index, :, :, :] = outarray
end

for device in keys(timings)
    device_name = nameof(typeof(device))
    config = CA.AtmosConfig(; comms_ctx = ClimaComms.context(device))
    config.parsed_args["diagnostics"] =
        [Dict("short_name" => "rhoa", "period" => config.parsed_args["dt"])]
    config.parsed_args["netcdf_interpolation_num_points"] = [NUM, NUM, NUM]
    config.parsed_args["job_id"] = "flame_perf_io"

    simulation = CA.get_simulation(config)

    # Cleanup pre-existing NC files
    foreach(
        rm,
        filter(endswith(".nc"), readdir(simulation.output_dir, join = true)),
    )

    atmos_model = simulation.integrator.p.atmos
    cspace = axes(simulation.integrator.u.c)
    diagnostics, _ = CA.get_diagnostics(config.parsed_args, atmos_model, cspace)
    rhoa_diag = diagnostics[end]

    netcdf_writer = simulation.output_writers[2]
    field = simulation.integrator.u.c.ρ

    integrator = simulation.integrator

    # Run once to create the file
    CAD.write_field!(
        netcdf_writer,
        field,
        rhoa_diag,
        integrator.u,
        integrator.p,
        integrator.t,
        simulation.output_dir,
    )
    output_path = CAD.outpath_name(simulation.output_dir, rhoa_diag)
    NCDatasets.sync(netcdf_writer.open_files[output_path])
    # Now, profile
    @info "Profiling ($device_name)"
    prof = Profile.@profile CAD.save_diagnostic_to_disk!(
        netcdf_writer,
        field,
        rhoa_diag,
        integrator.u,
        integrator.p,
        integrator.t,
        simulation.output_dir,
    )
    results = Profile.fetch()
    flame_path = joinpath(simulation.output_dir, "flame_$device_name.html")
    ProfileCanvas.html_file(flame_path, results)
    @info "Flame saved in $flame_path"

    @info "Benchmarking our NetCDF writer (only IO) ($device_name)"
    timings[device] =
        @benchmark ClimaComms.@cuda_sync $device CAD.save_diagnostic_to_disk!(
            $netcdf_writer,
            $field,
            $rhoa_diag,
            $(integrator.u),
            $(integrator.p),
            $(integrator.t),
            $(simulation.output_dir),
        )

    @info "Benchmarking NCDatasets ($device_name)"

    output_path = joinpath(simulation.output_dir, "clean_netcdf.nc")
    nc = NCDatasets.NCDataset(output_path, "c")
    NCDatasets.defDim(nc, "time", Inf)
    NCDatasets.defVar(nc, "time", Float32, ("time",))
    NCDatasets.defDim(nc, "x", NUM)
    NCDatasets.defDim(nc, "y", NUM)
    NCDatasets.defDim(nc, "z", NUM)
    v = NCDatasets.defVar(nc, "rhoa", Float64, ("time", "x", "y", "z"))
    outarray = Array(netcdf_writer.remappers["rhoa"]._interpolated_values)
    v[1, :, :, :] = outarray

    timings_ncdataset[device] = @benchmark $add_nc($nc, $outarray)
end

for device in keys(timings)
    println("DEVICE: ", device)
    println("Our writer")
    show(stdout, MIME"text/plain"(), timings[device])
    println()
    println("NCDatasets")
    show(stdout, MIME"text/plain"(), timings_ncdataset[device])
    println()
end
