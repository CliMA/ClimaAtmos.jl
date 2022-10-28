using Distributed
@everywhere using Pkg
@everywhere begin
    ca_dir = joinpath(@__DIR__, "..", "..")
    Pkg.activate(ca_dir)
    include(joinpath(ca_dir, "hybrid", "sphere", "remap_pipeline.jl"))
end
@everywhere begin
    weightfile_0 = create_weightfile(data_files[1], nc_dir, nlat, nlon)
end
@everywhere begin
    weightfile = weightfile_0
end
pmap(data_files) do data_file
    remap2latlon(data_file, nc_dir, weightfile, nlat, nlon)
end
remove_tmpdir(nc_dir)
