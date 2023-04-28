#=
Run with, for example:
```
julia -p 4 remap_pipeline_parallel.jl
```
=#
using Distributed
@everywhere using Pkg
@everywhere begin
    ca_dir = joinpath(@__DIR__, "..", "..")
    Pkg.activate(joinpath(ca_dir, "examples"))
    include(joinpath(@__DIR__, "remap_pipeline.jl"))
end
(; remap_tmpdir, data_files, out_dir, nlat, nlon) = get_params()

weightfile_0 = create_weightfile(data_files[1], remap_tmpdir, nlat, nlon)

@everywhere begin
    weightfile = $weightfile_0
    remap_tmpdir = $remap_tmpdir
    out_dir = $out_dir
    nlat = $nlat
    nlon = $nlon
end
pmap(data_files) do data_file
    @info "Processor $(myid()) processing: $data_file"
    remap2latlon(data_file, out_dir, remap_tmpdir, weightfile, nlat, nlon)
end
rm(remap_tmpdir; recursive = true)
