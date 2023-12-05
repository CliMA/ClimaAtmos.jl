# netcdf_writer.jl
#
# The flow of the code is:
#
# - We define a generic NetCDF struct with the NetCDF constructor. In doing this, we check
#   what we have to do for topography. We might have to interpolate it or not so that we can
#   later provide the correct elevation profile.

# - The first time write_fields! is called with the writer on a new field, we add the
#   dimensions to the NetCDF file. This requires understanding what coordinates we have to
#   interpolate on and potentially handle topography.
#
#   The functions to do so are for the most part all methods of the same functions, so that
#   we can achieve the same behavior for different types of configurations (planes/spheres,
#   ...), but also different types of fields (horizontal ones, 3D ones, ...)
#

################
# NetCDFWriter #
################
"""
    add_dimension_maybe!(nc::NCDatasets.NCDataset,
                         name::String,
                         points;
                         depending_on_dimensions = (),
                         kwargs...)


Add dimension identified by `name` in the given `nc` file and fill it with the given
`points`. If the dimension already exists, check if it is consistent with the new one.
Optionally, add all the keyword arguments as attributes.

`depending_on_dimensions` identifies the dimensions upon which the current one depends on
(excluding itself). In pretty much all cases, the dimensions depend only on themselves
(e.g., `lat` is a variable only defined on the latitudes.), and `depending_on_dimensions`
should be an empty tuple. The only case in which this is not what happens is with `z` with
topography. With topography, the altitude will depend on the spatial coordinates. So,
`depending_on_dimensions` might be `("lon", "lat)`, or similar.

"""

function add_dimension_maybe!(
    nc::NCDatasets.NCDataset,
    name::String,
    points;
    depending_on_dimensions = (),
    kwargs...,
)
    FT = eltype(points)

    if haskey(nc, name)
        # dimension already exists: check correct size
        if size(nc[name]) != size(points)
            error("Incompatible $name dimension already exists")
        end
    else
        # `points` is a 1-3D array. It is a 1D vector in pretty much all the cases except
        # when we have topography. If we have topography, `points` will be 2D or 3D,
        # depending if we have a plane or box/sphere. In all these cases, we can always
        # assume that the new dimension that we want to add has length of the last dimension
        # of points. This is because it is the only dimension if points is 1D, and it is the
        # altitude z if points is 2D/3D.

        NCDatasets.defDim(nc, name, size(points)[end])

        dim =
            NCDatasets.defVar(nc, name, FT, (depending_on_dimensions..., name))
        for (k, v) in kwargs
            dim.attrib[String(k)] = v
        end

        if length(size(points)) == 1
            dim[:] = points
        else
            # We have topography
            dim[:, :, :] = points
        end
    end
    return nothing
end

"""
    add_time_maybe!(nc::NCDatasets.NCDataset,
                    float_type::DataType;
                    kwargs...)


Add the `time` dimension (with infinite size) to the given NetCDF file if not already there.
Optionally, add all the keyword arguments as attributes.
"""
function add_time_maybe!(
    nc::NCDatasets.NCDataset,
    float_type::DataType;
    kwargs...,
)

    # If we already have time, do nothing
    haskey(nc, "time") && return nothing

    NCDatasets.defDim(nc, "time", Inf)
    dim = NCDatasets.defVar(nc, "time", float_type, ("time",))
    for (k, v) in kwargs
        dim.attrib[String(k)] = v
    end
    return nothing
end

"""
    add_space_coordinates_maybe!(nc::NCDatasets.NCDataset,
                                 space::Spaces.AbstractSpace,
                                 num_points)

Add dimensions relevant to the `space` to the given `nc` NetCDF file. The range is
automatically determined and the number of points is set with `num_points`, which has to be
an iterable of size N, where N is the number of dimensions of the space. For instance, 3 for
a cubed sphere, 2 for a surface, 1 for a column.

The function returns an array with the names of the relevant dimensions. (We want arrays
because we want to preserve the order to match the one in num_points).
"""
function add_space_coordinates_maybe! end

"""
    target_coordinates!(space::Spaces.AbstractSpace,
                        num_points)

Return the range of interpolation coordinates. The range is automatically determined and the
number of points is set with `num_points`, which has to be an iterable of size N, where N is
the number of dimensions of the space. For instance, 3 for a cubed sphere, 2 for a surface,
1 for a column.
"""
function target_coordinates(space, num_points) end

function target_coordinates(
    space::S,
    num_points,
) where {
    S <:
    Union{Spaces.CenterFiniteDifferenceSpace, Spaces.FaceFiniteDifferenceSpace},
}
    num_points_z = num_points[]
    FT = Spaces.undertype(space)
    vert_domain = space.topology.mesh.domain
    z_min, z_max = vert_domain.coord_min.z, vert_domain.coord_max.z
    return collect(range(FT(z_min), FT(z_max), num_points_z))
end

# Column
function add_space_coordinates_maybe!(
    nc::NCDatasets.NCDataset,
    space::Spaces.FiniteDifferenceSpace,
    num_points_z,
)
    name = "z"
    zpts = target_coordinates(space, num_points_z)
    add_dimension_maybe!(nc, "z", zpts, units = "m")
    return [name]
end

add_space_coordinates_maybe!(
    nc::NCDatasets.NCDataset,
    space::Spaces.AbstractSpectralElementSpace,
    num_points,
) = add_space_coordinates_maybe!(
    nc,
    space,
    num_points,
    Meshes.domain(space.topology),
)


# For the horizontal space, we also have to look at the domain, so we define another set of
# functions that dispatches over the domain
target_coordinates(space::Spaces.AbstractSpectralElementSpace, num_points) =
    target_coordinates(space, num_points, Meshes.domain(space.topology))

# Box
function target_coordinates(
    space::Spaces.SpectralElementSpace2D,
    num_points,
    domain::Domains.RectangleDomain,
)
    num_points_x, num_points_y = num_points
    FT = Spaces.undertype(space)
    xmin = FT(domain.interval1.coord_min.x)
    xmax = FT(domain.interval1.coord_max.x)
    ymin = FT(domain.interval2.coord_min.y)
    ymax = FT(domain.interval2.coord_max.y)
    xpts = collect(range(xmin, xmax, num_points_x))
    ypts = collect(range(ymin, ymax, num_points_y))
    return (xpts, ypts)
end

# Plane
function target_coordinates(
    space::Spaces.SpectralElementSpace1D,
    num_points,
    domain::Domains.IntervalDomain,
)
    num_points_x, _... = num_points
    FT = Spaces.undertype(space)
    xmin = FT(domain.coord_min.x)
    xmax = FT(domain.coord_max.x)
    xpts = collect(range(xmin, xmax, num_points_x))
    return (xpts)
end

# Cubed sphere
function target_coordinates(
    space::Spaces.SpectralElementSpace2D,
    num_points,
    ::Domains.SphereDomain,
)
    num_points_long, num_points_lat = num_points
    FT = Spaces.undertype(space)
    longpts = collect(range(FT(-180), FT(180), num_points_long))
    latpts = collect(range(FT(-80), FT(80), num_points_lat))

    return (longpts, latpts)
end

# Box
function add_space_coordinates_maybe!(
    nc::NCDatasets.NCDataset,
    space::Spaces.SpectralElementSpace2D,
    num_points,
    ::Domains.RectangleDomain,
)
    xname, yname = ("x", "y")
    xpts, ypts = target_coordinates(space, num_points)
    add_dimension_maybe!(nc, "x", xpts; units = "m")
    add_dimension_maybe!(nc, "y", ypts; units = "m")
    return [xname, yname]
end

# Plane
function add_space_coordinates_maybe!(
    nc::NCDatasets.NCDataset,
    space::Spaces.SpectralElementSpace1D,
    num_points,
    ::Domains.IntervalDomain,
)
    xname = "x"
    xpts = target_coordinates(space, num_points)
    add_dimension_maybe!(nc, "x", xpts; units = "m")
    return [xname]
end

# Cubed sphere
function add_space_coordinates_maybe!(
    nc::NCDatasets.NCDataset,
    space::Spaces.SpectralElementSpace2D,
    num_points,
    ::Domains.SphereDomain,
)
    longname, latname = ("lon", "lat")
    longpts, latpts = target_coordinates(space, num_points)
    add_dimension_maybe!(nc, "lon", longpts; units = "degrees_east")
    add_dimension_maybe!(nc, "lat", latpts; units = "degrees_north")
    return [longname, latname]
end

# General hybrid space. This calls both the vertical and horizontal add_space_coordinates_maybe!
# and combines the resulting dictionaries
function add_space_coordinates_maybe!(
    nc::NCDatasets.NCDataset,
    space::Spaces.ExtrudedFiniteDifferenceSpace,
    num_points;
    interpolated_surface = nothing,
)

    hdims_names = vdims_names = []

    num_points_horiz..., num_points_vertic = num_points

    # Being an Extruded space, we can assume that we have an horizontal and a vertical space.
    # We can also assume that the vertical space has dimension 1
    horizontal_space = Spaces.horizontal_space(space)

    hdims_names =
        add_space_coordinates_maybe!(nc, horizontal_space, num_points_horiz)

    vertical_space = Spaces.FiniteDifferenceSpace(
        Spaces.vertical_topology(space),
        Spaces.staggering(space),
    )

    if isnothing(interpolated_surface)
        vdims_names =
            add_space_coordinates_maybe!(nc, vertical_space, num_points_vertic)
    else
        vdims_names = add_space_coordinates_maybe!(
            nc,
            vertical_space,
            num_points_vertic,
            interpolated_surface;
            depending_on_dimensions = hdims_names,
        )
    end

    return (hdims_names..., vdims_names...)
end

# Ignore the interpolated_surface keyword in the general case (we only case about the
# specialized one for extruded spaces)
add_space_coordinates_maybe!(
    nc::NCDatasets.NCDataset,
    space,
    num_points;
    interpolated_surface = nothing,
) = add_space_coordinates_maybe!(nc::NCDatasets.NCDataset, space, num_points)

# Elevation with topography
function add_space_coordinates_maybe!(
    nc::NCDatasets.NCDataset,
    space::Spaces.FiniteDifferenceSpace,
    num_points,
    interpolated_surface;
    depending_on_dimensions,
)
    num_points_z = num_points
    name = "z"

    # Implement the LinearAdaption hypsography
    reference_altitudes = target_coordinates(space, num_points_z)
    z_top = space.topology.mesh.domain.coord_max.z

    # Prepare output array
    desired_shape = (size(interpolated_surface)..., num_points_z)
    zpts = zeros(eltype(interpolated_surface), desired_shape)

    for i in CartesianIndices(interpolated_surface)
        z_surface = interpolated_surface[i]
        @. zpts[i, :] +=
            reference_altitudes + (1 .- reference_altitudes / z_top) * z_surface
    end

    add_dimension_maybe!(nc, name, zpts; depending_on_dimensions, units = "m")
    return [name]
end

# General hybrid space. This calls both the vertical and horizontal add_space_coordinates_maybe!
# and combines the resulting dictionaries
function target_coordinates(
    space::Spaces.ExtrudedFiniteDifferenceSpace,
    num_points,
)

    hcoords = vcoords = ()

    num_points_horiz..., num_points_vertic = num_points

    hcoords =
        target_coordinates(Spaces.horizontal_space(space), num_points_horiz)

    vertical_space = Spaces.FiniteDifferenceSpace(
        Spaces.vertical_topology(space),
        Spaces.staggering(space),
    )
    vcoords = target_coordinates(vertical_space, num_points_vertic)

    hcoords == vcoords == () && error("Found empty space")

    return hcoords, vcoords
end

function hcoords_from_horizontal_space(
    space::Spaces.SpectralElementSpace2D,
    domain::Domains.SphereDomain,
    hpts,
)
    # Notice LatLong not LongLat!
    return [Geometry.LatLongPoint(hc2, hc1) for hc1 in hpts[1], hc2 in hpts[2]]
end

function hcoords_from_horizontal_space(
    space::Spaces.SpectralElementSpace2D,
    domain::Domains.RectangleDomain,
    hpts,
)
    return [Geometry.XYPoint(hc1, hc2) for hc1 in hpts[1], hc2 in hpts[2]]
end

function hcoords_from_horizontal_space(
    space::Spaces.SpectralElementSpace1D,
    domain::Domains.IntervalDomain,
    hpts,
)
    return [Geometry.XPoint(hc1) for hc1 in hpts]
end

"""
    hcoords_from_horizontal_space(space, domain, hpts)

Prepare the matrix of horizontal coordinates with the correct type according to the given `space`
and `domain` (e.g., `ClimaCore.Geometry.LatLongPoint`s).
"""
function hcoords_from_horizontal_space(space, domain, hpts) end

struct NetCDFWriter{T, TS}

    # TODO: At the moment, each variable gets its remapper. This is a little bit of a waste
    # because we probably only need a handful of remappers since the same remapper can be
    # used for multiple fields as long as they are all defined on the same space. We need
    # just a few remappers because realistically we need to support fields defined on the
    # entire space and fields defined on 2D slices. However, handling this distinction at
    # construction time is quite difficult.
    remappers::Dict{String, Remapper}

    # Tuple/Array of integers that identifies how many points to use for interpolation along
    # the various dimensions. It has to have the same size as the target interpolation
    # space.
    num_points::T

    # How much to compress the data in the final NetCDF file: 0 no compression, 9 max
    # compression.
    compression_level::Int

    # nothing, if z is the altitude from the sea level (ie, no topography, or topography
    # already interpolated). Otherwise, a 1/2D array that describes the surface elevation
    # interpolated on the horizontal points.
    interpolated_surface::TS

    # NetCDF files that are currently open. Only the root process uses this field.
    open_files::Dict{String, NCDatasets.NCDataset}

    # Whether to treat z as altitude over the surface or the sea level or over the surface
    interpolate_z_over_msl::Bool
end

"""
    close(writer::NetCDFWriter)


Close all the files open in `writer`.
"""
close(writer::NetCDFWriter) = map(NCDatasets.close, values(writer.open_files))

"""
    NetCDFWriter()


Save a `ScheduledDiagnostic` to a NetCDF file inside the `output_dir` of the simulation by
performing a pointwise (non-conservative) remapping first.

Keyword arguments
==================

- `num_points`: How many points to use along the different dimensions to interpolate the
                fields. This is a tuple of integers, typically having meaning Long-Lat-Z,
                or X-Y-Z (the details depend on the configuration being simulated).
- `interpolate_z_over_msl`: When `true`, the variable `z` is intended to be altitude from
                            the sea level (in meters). Values of `z` that are below the
                            surface are filled with `NaN`s. When `false`, `z` is intended to
                            be altitude from the surface (in meters). `z` becomes a
                            multidimensional array that returns the altitude for the given
                            horizontal coordinates.
- `compression_level`: How much to compress the output NetCDF file (0 is no compression, 9
  is maximum compression).

"""
function NetCDFWriter(;
    hypsography,
    num_points = (180, 80, 10),
    interpolate_z_over_msl = false,
    compression_level = 9,
)

    # We have to deal with the pesky topography. This is a little annoying to deal with
    # because it couples the horizontal and the vertical dimensions, so it doesn't fit the
    # common paradigm for all the other dimensions. Moreover, with topography, we need need
    # to perform an interpolation for the surface.

    # We have to deal with the surface only if we are not interpolating the topography and
    # if our topography is non-trivial
    if hypsography isa Grids.Flat || interpolate_z_over_msl
        interpolated_surface = nothing
    elseif hypsography isa Hypsography.LinearAdaption
        horizontal_space = axes(hypsography.surface)
        hpts = target_coordinates(horizontal_space, num_points)
        hcoords = hcoords_from_horizontal_space(
            horizontal_space,
            horizontal_space.topology.mesh.domain,
            hpts,
        )
        vcoords = []
        remapper = Remapper(hcoords, vcoords, horizontal_space)
        interpolated_surface =
            interpolate(remapper, hypsography.surface, physical_z = false)
    else
        error("Cannot process hysography $hypsography")
    end

    return NetCDFWriter{typeof(num_points), typeof(interpolated_surface)}(
        Dict(),
        num_points,
        compression_level,
        interpolated_surface,
        Dict(),
        interpolate_z_over_msl,
    )
end

function write_field!(
    writer::NetCDFWriter,
    field,
    diagnostic,
    integrator,
    output_dir,
)

    var = diagnostic.variable

    # If we have a face-valued field, we interpolate it to the centers
    if axes(field) isa Spaces.FaceExtrudedFiniteDifferenceSpace
        field = ᶜinterp.(field)
    end

    space = axes(field)
    FT = Spaces.undertype(space)

    horizontal_space = Spaces.horizontal_space(space)

    # We have to deal with to cases: when we have an horizontal slice (e.g., the
    # surface), and when we have a full space. We distinguish these cases by checking if
    # the given space has the horizontal_space attribute. If not, it is going to be a
    # SpectralElementSpace2D and we don't have to deal with the z coordinates.
    is_horizontal_space = horizontal_space == space

    # Prepare the remapper if we don't have one for the given variable. We need one remapper
    # per variable (not one per diagnostic since all the time reductions return the same
    # type of space).

    # TODO: Expand this once we support spatial reductions
    if !haskey(writer.remappers, var.short_name)
        if is_horizontal_space
            hpts = target_coordinates(space, writer.num_points)
            vpts = []
        else
            hpts, vpts = target_coordinates(space, writer.num_points)
        end

        # zcoords is going to be empty for a 2D horizontal slice
        zcoords = [Geometry.ZPoint(p) for p in vpts]
        hcoords = hcoords_from_horizontal_space(
            horizontal_space,
            Meshes.domain(horizontal_space.topology),
            hpts,
        )

        writer.remappers[var.short_name] = Remapper(hcoords, zcoords, space)
    end

    remapper = writer.remappers[var.short_name]

    # Now we can interpolate onto the target points
    # There's an MPI call in here (to aggregate the results)
    interpolated_field =
        interpolate(remapper, field; physical_z = writer.interpolate_z_over_msl)

    # Only the root process has to write
    ClimaComms.iamroot(ClimaComms.context(field)) || return

    output_path = joinpath(output_dir, "$(diagnostic.output_short_name).nc")

    if !haskey(writer.open_files, output_path)
        # Append or write a new file
        open_mode = isfile(output_path) ? "a" : "c"
        writer.open_files[output_path] =
            NCDatasets.Dataset(output_path, open_mode)
    end

    nc = writer.open_files[output_path]

    # Define time coordinate
    add_time_maybe!(nc, FT; units = "s", axis = "T")

    dim_names = add_space_coordinates_maybe!(
        nc,
        space,
        writer.num_points;
        writer.interpolated_surface,
    )

    if haskey(nc, "$(var.short_name)")
        # We already have something in the file
        v = nc["$(var.short_name)"]
        temporal_size, spatial_size... = size(v)
        spatial_size == size(interpolated_field) ||
            error("incompatible dimensions for $(var.short_name)")
    else
        v = NCDatasets.defVar(
            nc,
            "$(var.short_name)",
            FT,
            ("time", dim_names...),
            deflatelevel = writer.compression_level,
        )
        v.attrib["long_name"] = diagnostic.output_long_name
        v.attrib["units"] = var.units
        v.attrib["comments"] = var.comments

        temporal_size = 0
    end

    # We need to write to the next position after what we read from the data (or the first
    # position ever if we are writing the file for the first time)
    time_index = temporal_size + 1

    nc["time"][time_index] = integrator.t

    # TODO: It would be nice to find a cleaner way to do this
    if length(dim_names) == 3
        v[time_index, :, :, :] = interpolated_field
    elseif length(dim_names) == 2
        v[time_index, :, :] = interpolated_field
    elseif length(dim_names) == 1
        v[time_index, :] = interpolated_field
    end
end
