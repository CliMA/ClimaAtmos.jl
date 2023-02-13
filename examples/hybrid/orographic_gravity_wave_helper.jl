using ClimaCoreTempestRemap
using NCDatasets

function get_topo_info(Y, topo_dir, datafile_rll, comms_ctx)
    # model grid/space information
    cspace = axes(Y.c)
    hspace = cspace.horizontal_space

    # generate weight file for regridding
    weightfile = create_regrid_weight_files(
        comms_ctx,
        topo_dir,
        datafile_rll,
        hspace;
        hd_outfile_root = "topo",
        mono = false,
    )

    varnames = ["t11", "t12", "t21", "t22", "hmin", "hmax"]

    topo_info = FieldFromNamedTuple(
        axes(Fields.level(Y.c.ρ, 1)),
        (;
            t11 = FT(0),
            t12 = FT(0),
            t21 = FT(0),
            t22 = FT(0),
            hmin = FT(0),
            hmax = FT(0),
        ),
    )


    for varname in varnames
        datafile_cgll = joinpath(topo_dir, "topo_" * varname * ".g")
        apply_remap(datafile_cgll, datafile_rll, weightfile, [varname])
        parent(getproperty(topo_info, Symbol(varname))) .=
            parent(convert_g2field(datafile_cgll, weightfile, hspace, varname))
    end

    # write to hdf5
    hdfwriter = InputOutput.HDF5Writer(joinpath(topo_dir, "topo_info.hdf5"))
    InputOutput.write!(hdfwriter, topo_info, "topo_info")
    Base.close(hdfwriter)

    return
end

function FieldFromNamedTuple(space, nt::NamedTuple)
    cmv(z) = nt
    return cmv.(Fields.coordinate_field(space))
end

function reshape_cgll_sparse_to_field!(field::Fields.Field, in_array::Array, R)
    field_array = parent(field)

    fill!(field_array, zero(eltype(field_array)))
    Nf = size(field_array, 3)

    f = 1
    for (n, row) in enumerate(R.row_indices)
        it, jt, et = (
            view(R.target_idxs[1], n),
            view(R.target_idxs[2], n),
            view(R.target_idxs[3], n),
        )
        for f in 1:Nf
            field_array[it, jt, f, et] .= in_array[row]
        end
    end
    # broadcast to the redundant nodes using unweighted dss
    topology = Spaces.topology(axes(field))
    hspace = Spaces.horizontal_space(axes(field))
    quadrature_style = hspace.quadrature_style
    Spaces.dss2!(Fields.field_values(field), topology, quadrature_style)
    return field
end

function create_regrid_weight_files(
    comms_ctx,
    REGRID_DIR,
    datafile_rll,
    space;
    hd_outfile_root = "data_cgll",
    mono = false,
)

    out_type = "cgll"

    outfile = hd_outfile_root * ".nc"
    outfile_root = mono ? outfile[1:(end - 3)] * "_mono" : outfile[1:(end - 3)]

    meshfile_rll = joinpath(REGRID_DIR, outfile_root * "_mesh_rll.g")
    meshfile_cgll = joinpath(REGRID_DIR, outfile_root * "_mesh_cgll.g")
    meshfile_overlap = joinpath(REGRID_DIR, outfile_root * "_mesh_overlap.g")
    weightfile = joinpath(REGRID_DIR, outfile_root * "_remap_weights.nc")

    topology = Topologies.Topology2D(
        space.topology.mesh,
        Topologies.spacefillingcurve(space.topology.mesh),
    )
    Nq = Spaces.Quadratures.polynomial_degree(space.quadrature_style) + 1

    isdir(REGRID_DIR) ? nothing : mkpath(REGRID_DIR)

    nlat, nlon = NCDataset(datafile_rll) do ds
        (ds.dim["lat"], ds.dim["lon"])
    end
    # write lat-lon mesh
    rll_mesh(meshfile_rll; nlat = nlat, nlon = nlon)

    # write cgll mesh, overlap mesh and weight file 
    write_exodus(meshfile_cgll, topology)
    overlap_mesh(meshfile_overlap, meshfile_rll, meshfile_cgll)

    # 'in_np = 1' and 'mono = true' arguments ensure mapping is conservative and monotone
    # Note: for a kwarg not followed by a value, set it to true here (i.e. pass 'mono = true' to produce '--mono')
    # Note: out_np = degrees of freedom = polynomial degree + 1

    kwargs = (; out_type = out_type, out_np = Nq)
    kwargs =
        mono ? (; (kwargs)..., in_np = mono ? 1 : false, mono = mono) : kwargs
    remap_weights(
        weightfile,
        meshfile_rll,
        meshfile_cgll,
        meshfile_overlap;
        kwargs...,
    )

    return weightfile

end

function convert_g2field(
    datafile_cgll,
    weightfile,
    space,
    varname;
    out_type = "cgll",
)
    FT = Spaces.undertype(space)
    # read the remapped file with sparse matrices
    offline_outvector = NCDataset(datafile_cgll, "r") do ds_wt
        ds_wt[varname][:]
    end

    # weightfile info needed to populate all nodes and save into fields with sparse matrices
    _, _, row_indices = NCDataset(weightfile, "r") do ds_wt
        (Array(ds_wt["S"]), Array(ds_wt["col"]), Array(ds_wt["row"]))
    end

    topology = Topologies.Topology2D(
        space.topology.mesh,
        Topologies.spacefillingcurve(space.topology.mesh),
    )
    Nq = Spaces.Quadratures.polynomial_degree(space.quadrature_style) + 1
    space_undistributed =
        Spaces.SpectralElementSpace2D(topology, Spaces.Quadratures.GLL{Nq}())

    target_unique_idxs =
        out_type == "cgll" ? collect(Spaces.unique_nodes(space_undistributed)) :
        collect(Spaces.all_nodes(space_undistributed))
    target_unique_idxs_i =
        map(row -> target_unique_idxs[row][1][1], row_indices)
    target_unique_idxs_j =
        map(row -> target_unique_idxs[row][1][2], row_indices)
    target_unique_idxs_e = map(row -> target_unique_idxs[row][2], row_indices)
    target_unique_idxs =
        (target_unique_idxs_i, target_unique_idxs_j, target_unique_idxs_e)

    R = (; target_idxs = target_unique_idxs, row_indices = row_indices)

    offline_field = Fields.zeros(FT, space_undistributed)

    field = reshape_cgll_sparse_to_field!(offline_field, offline_outvector, R)

    return field
end
