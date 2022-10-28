import ClimaCoreTempestRemap
import ClimaCore.Spaces as Spaces

"""
    create_weightfile(
        weightfile::String,
        cspace::Spaces.AbstractSpace,
        fspace::Spaces.AbstractSpace,
        nlat::Int,
        nlon::Int
    )

A helper for creating weights for ClimaCoreTempestRemap. Example:

```julia
create_weightfile(weightfile, cspace, fspace, nlat, nlon)
```
TODO: should this live in ClimaCoreTempestRemap?
"""
function create_weightfile(
    weightfile::String,
    cspace::Spaces.AbstractSpace,
    fspace::Spaces.AbstractSpace,
    nlat::Int,
    nlon::Int,
)
    # space info to generate nc raw data
    hspace = cspace.horizontal_space
    Nq = Spaces.Quadratures.degrees_of_freedom(
        cspace.horizontal_space.quadrature_style,
    )
    # create a temporary dir for intermediate data
    mktempdir() do tmp
        mkpath(tmp)
        # write out our cubed sphere mesh
        meshfile_cc = joinpath(tmp, "mesh_cubedsphere.g")
        ClimaCoreTempestRemap.write_exodus(meshfile_cc, hspace.topology)
        meshfile_rll = joinpath(tmp, "mesh_rll.g")
        ClimaCoreTempestRemap.rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)
        meshfile_overlap = joinpath(tmp, "mesh_overlap.g")
        ClimaCoreTempestRemap.overlap_mesh(
            meshfile_overlap,
            meshfile_cc,
            meshfile_rll,
        )
        tmp_weightfile = joinpath(tmp, "remap_weights.nc")
        ClimaCoreTempestRemap.remap_weights(
            tmp_weightfile,
            meshfile_cc,
            meshfile_rll,
            meshfile_overlap;
            in_type = "cgll",
            in_np = Nq,
        )
        mv(tmp_weightfile, weightfile; force = true)
    end
end
