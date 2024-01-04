#=
Sbatch script example:
# Serial remapping
```
#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=remap
#SBATCH --time=01:00:00
#SBATCH --output=remap.log
module purge
module load julia/1.9.4
export JULIA_CUDA_USE_BINARYBUILDER=false
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}
export HDF5_DIR=/central/groups/esm/{username}/ClimaAtmos_remap/
CA_EXAMPLE=$HOME'/Documents/ClimaAtmos.jl/examples/'
DRIVER=$CA_EXAMPLE'post_processing/remap/remap_pipeline.jl'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.instantiate()'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.build("HDF5")'
julia --project=$CA_EXAMPLE -e 'using Pkg; Pkg.API.precompile()'
julia --project=$CA_EXAMPLE $DRIVER
```
=#

#=
Here's a sketch of what this pipeline does:

Inputs:
 - data_dir/day0.0.1.hdf5
 - data_dir/day0.2.0.hdf5
 - data_dir/day0.3.0.hdf5
 - ...
 - tmpdir/weightsfile.nc
Apply remap pipeline:
 - tmpdir[/myid()]/test.nc -> remap -> out_dir/day0.0.1.nc, rm(tmpdir[/myid()]/test.nc)
 - tmpdir[/myid()]/test.nc -> remap -> out_dir/day0.2.0.nc, rm(tmpdir[/myid()]/test.nc)
 - tmpdir[/myid()]/test.nc -> remap -> out_dir/day0.3.0.nc, rm(tmpdir[/myid()]/test.nc)
rm(tmpdir/weightsfile.nc)
=#

import ClimaCore
import ClimaAtmos
using ClimaCore:
    Geometry, Meshes, Domains, Topologies, Spaces, Operators, InputOutput
using NCDatasets
using ClimaCoreTempestRemap

include(joinpath(@__DIR__, "remap_helpers.jl"))

import ArgParse
function parse_commandline()
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--data_dir"
        help = "Data directory"
        arg_type = String
        "--out_dir"
        help = "Output data directory"
        arg_type = String
        "--nlat"
        help = "Number of latitude points"
        arg_type = Int
        default = 90
        "--nlon"
        help = "Number of longitude points"
        arg_type = Int
        default = 180
    end
    parsed_args = ArgParse.parse_args(ARGS, s)
    return (s, parsed_args)
end

function get_params()
    (s, parsed_args) = parse_commandline()
    data_dir = parsed_args["data_dir"]
    out_dir = parsed_args["out_dir"]
    nlat = parsed_args["nlat"]
    nlon = parsed_args["nlon"]
    if isnothing(out_dir)
        out_dir = joinpath(data_dir, "remap")
    end
    mkpath(out_dir)

    data_files =
        filter(x -> endswith(x, ".hdf5"), readdir(data_dir, join = true))

    remap_tmpdir = joinpath(out_dir, "remaptmp")
    mkpath(remap_tmpdir)
    return (; remap_tmpdir, data_files, out_dir, nlat, nlon)
end

# Only run this in serial mode:
parallel_mode = @isdefined pmap
if !parallel_mode
    (; remap_tmpdir, data_files, out_dir, nlat, nlon) = get_params()
    weightfile = create_weightfile(data_files[1], remap_tmpdir, nlat, nlon)
    map(data_files) do data_file
        remap2latlon(data_file, out_dir, remap_tmpdir, weightfile, nlat, nlon)
    end
    rm(remap_tmpdir; recursive = true)
end
