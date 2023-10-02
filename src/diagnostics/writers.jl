# Writers.jl
#
# This file defines generic writers for diagnostics with opinionated defaults.

import ClimaCore.Remapping: Remapper, interpolate, interpolate_array

import NCDatasets

##############
# HDF5Writer #
##############

"""
    HDF5Writer()


Save a `ScheduledDiagnostic` to a HDF5 file inside the `output_dir` of the simulation.


TODO: This is a very barebone HDF5Writer. Do not consider this implementation as the "final
word".

We need to implement the following features/options:
- Toggle for write new files/append
- Checks for existing files
- Check for new subfolders that have to be created
- More meaningful naming conventions (keeping in mind that we can have multiple variables
  with different reductions)
- All variables in one file/each variable in its own file
- All timesteps in one file/each timestep in its own file
- Writing the correct attributes
- Overriding simulation.output_dir (e.g., if the path starts with /)
- ...more features/options

"""
struct HDF5Writer end


"""
    close(writer::HDF5Writer)

Close all the files open in `writer`. (Currently no-op.)
"""
close(writer::HDF5Writer) = nothing

function write_field!(writer::HDF5Writer, field, diagnostic, integrator)
    var = diagnostic.variable
    time = integrator.t

    output_path = joinpath(
        integrator.p.simulation.output_dir,
        "$(diagnostic.output_short_name)_$(time).h5",
    )

    hdfwriter = InputOutput.HDF5Writer(output_path, integrator.p.comms_ctx)
    InputOutput.write!(hdfwriter, field, "$(diagnostic.output_short_name)")
    attributes = Dict(
        "time" => time,
        "long_name" => diagnostic.output_long_name,
        "variable_units" => var.units,
        "standard_variable_name" => var.standard_name,
    )

    # TODO: Use directly InputOutput functions
    InputOutput.HDF5.h5writeattr(
        hdfwriter.file.filename,
        "fields/$(diagnostic.output_short_name)",
        attributes,
    )

    Base.close(hdfwriter)
    return nothing
end

################
# NetCDFWriter #
################
"""
    add_dimension_maybe!(nc::NCDatasets.NCDataset,
                         name::String,
                         points::Vector{FT};
                         kwargs...)


Add dimension identified by `name` in the given `nc` file and fill it with the given
`points`. If the dimension already exists, check if it is consistent with the new one.
Optionally, add all the keyword arguments as attributes.
"""
function add_dimension_maybe!(
    nc::NCDatasets.NCDataset,
    name::String,
    points::Vector{FT};
    kwargs...,
) where {FT <: Real}

    if haskey(nc, "$name")
        # dimension already exists: check correct size
        if size(nc["$name"]) != size(points)
            error("Incompatible $name dimension already exists")
        end
    else
        NCDatasets.defDim(nc, "$name", length(points))
        dim = NCDatasets.defVar(nc, "$name", FT, ("$name",))
        for (k, v) in kwargs
            dim.attrib[String(k)] = v
        end
        dim[:] = points
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

# For the horizontal space, we also have to look at the domain, so we define another set of
# functions that dispatches over the domain
target_coordinates(space::Spaces.AbstractSpectralElementSpace, num_points) =
    target_coordinates(space, num_points, space.topology.mesh.domain)

add_space_coordinates_maybe!(
    nc::NCDatasets.NCDataset,
    space::Spaces.AbstractSpectralElementSpace,
    num_points,
) = add_space_coordinates_maybe!(
    nc,
    space,
    num_points,
    space.topology.mesh.domain,
)

# Box
function target_coordinates(
    space::Spaces.SpectralElementSpace2D,
    num_points,
    domain::Domains.RectangleDomain,
)
    num_points_x, num_points_y = num_points
    FT = Spaces.undertype(space)
    xmin = FT(domain.coord_min.x)
    xmax = FT(domain.coord_max.x)
    ymin = FT(domain.coord_min.y)
    ymax = FT(domain.coord_max.y)
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
    space::Spaces.ExtrudedFiniteDifferenceSpace{S},
    num_points,
) where {S <: Spaces.Staggering}

    hdims_names = vdims_names = []

    num_points_horiz..., num_points_vertic = num_points

    # Being an Extruded space, we can assume that we have an horizontal and a vertical space.
    # We can also assume that the vertical space has dimension 1
    hdims_names = add_space_coordinates_maybe!(
        nc,
        Spaces.horizontal_space(space),
        num_points_horiz,
    )

    vertical_space = Spaces.FiniteDifferenceSpace{S}(space.vertical_topology)
    vdims_names =
        add_space_coordinates_maybe!(nc, vertical_space, num_points_vertic)

    hdims_names == vdims_names == [] && error("Found empty space")

    return vcat(hdims_names, vdims_names)
end

# General hybrid space. This calls both the vertical and horizontal add_space_coordinates_maybe!
# and combines the resulting dictionaries
function target_coordinates(
    space::Spaces.ExtrudedFiniteDifferenceSpace{S},
    num_points,
) where {S <: Spaces.Staggering}

    hcoords = vcoords = ()

    num_points_horiz..., num_points_vertic = num_points

    hcoords =
        target_coordinates(Spaces.horizontal_space(space), num_points_horiz)

    vertical_space = Spaces.FiniteDifferenceSpace{S}(space.vertical_topology)
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
    htps,
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

struct NetCDFWriter{T}

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

    # NetCDF files that are currently open. Only the root process uses this field.
    open_files::Dict{String, NCDatasets.NCDataset}
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

- `num_points`: Long-Lat-Z, X-Y-Z.

"""
function NetCDFWriter(; num_points = (180, 80, 10), compression_level = 9)
    return NetCDFWriter{typeof(num_points)}(
        Dict(),
        num_points,
        compression_level,
        Dict(),
    )
end

function write_field!(writer::NetCDFWriter, field, diagnostic, integrator)

    var = diagnostic.variable
    space = axes(field)
    FT = Spaces.undertype(space)
    # We have to deal with to cases: when we have an horizontal slice (e.g., the
    # surface), and when we have a full space. We distinguish these cases by checking if
    # the given space has the horizontal_space attribute. If not, it is going to be a
    # SpectralElementSpace2D and we don't have to deal with the z coordinates.
    is_horizontal_space = !hasproperty(space, :horizontal_space)

    if !is_horizontal_space
        horizontal_space = Spaces.horizontal_space(space)
        is_plane = typeof(horizontal_space) <: Spaces.SpectralElementSpace1D
    else
        horizontal_space = space
        is_plane = false
    end
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
            horizontal_space.topology.mesh.domain,
            hpts,
        )

        writer.remappers[var.short_name] = Remapper(hcoords, zcoords, space)
    end

    remapper = writer.remappers[var.short_name]

    # If we have a face-valued field, we interpolate it to the centers
    if axes(field) isa Spaces.FaceExtrudedFiniteDifferenceSpace
        field = ᶜinterp(field)
    end

    # Now we can interpolate onto the target points
    # There's an MPI call in here (to aggregate the results)
    interpolated_field = interpolate(remapper, field)

    # Only the root process has to write
    ClimaComms.iamroot(ClimaComms.context(field)) || return

    output_path = joinpath(
        integrator.p.simulation.output_dir,
        "$(diagnostic.output_short_name).nc",
    )

    if !haskey(writer.open_files, output_path)
        # Append or write a new file
        open_mode = isfile(output_path) ? "a" : "c"
        writer.open_files[output_path] =
            NCDatasets.Dataset(output_path, open_mode)
    end

    nc = writer.open_files[output_path]

    # Define time coordinate
    add_time_maybe!(nc, FT; units = "s", axis = "T")

    dim_names = add_space_coordinates_maybe!(nc, space, writer.num_points)

    if haskey(nc, "$(var.short_name)")
        # We already have something in the file
        v = nc["$(var.short_name)"]
        spatial_size..., temporal_size = size(v)
        spatial_size == size(interpolated_field) ||
            error("incompatible dimensions for $(var.short_name)")
    else
        v = NCDatasets.defVar(
            nc,
            "$(var.short_name)",
            FT,
            (dim_names..., "time"),
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

    # selectdim(v, 1, time_index) is equivalent to v[time_index, :, :] or
    # v[time_index, :, :, :] in 3/4 dimensions
    selectdim(v, 1, time_index) .= interpolated_field
end
