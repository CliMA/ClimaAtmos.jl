# This file contains the definitions of common AbstractAtmosGrids.
# - ColumnGrid (with UniformColumnGrid and StretchedColumnGrid)
#
# We also provide convenience functions to build these grids.
include("atmos_grids_makers.jl")

Base.@kwdef struct ColumnGrid{
    CS <: Spaces.ExtrudedFiniteDifferenceSpace,
    FS <: Spaces.ExtrudedFiniteDifferenceSpace,
    I <: Integer,
    FT <: Real,
    SR <: Meshes.StretchingRule,
} <: AbstractAtmosGrid
    center_space::CS
    face_space::FS

    z_elem::I
    z_max::FT

    z_stretch::SR
end

function Base.summary(io::IO, grid::ColumnGrid)
    println(io, "Grid type: $(nameof(typeof(grid)))")
    println(io, "Number of elements: $(grid.z_elem)")
    println(io, "Height: $(grid.z_max) meters")
    println(io, "Grid stretching: $(nameof(typeof(grid.z_stretch)))")
    # Add information about the stretching, if any
    for field in fieldnames(typeof(grid.z_stretch))
        println(io, "  with: $(field): $(getproperty(grid.z_stretch, field))")
    end
end


"""
function UniformColumnGrid(; z_elem,
                             z_max,
                             comms_ctx = ClimaComms.context(),
                             float_type = Float64)

Construct an `ColumnGrid` for a column with uniform resolution.

Keyword arguments
=================

- `z_elem`: Number of elements.
- `z_max`: Height of the column (in meters).
- `comms_ctx`: Context of the computational environment where the simulation should be run,
               as defined in ClimaComms. By default, the CLIMACOMMS_DEVICE environment
               variable is read for one of "CPU", "CPUSingleThreaded", "CPUMultiThreaded",
               "CUDA". If none is found, the fallback is to use a GPU (if available), or a
               single threaded CPU (if not).
- `float_type`: Floating point type. Typically, Float32 or Float64 (default).

"""
function UniformColumnGrid(;
    z_elem,
    z_max,
    comms_ctx = ClimaComms.context(),
    float_type = Float64,
)
    isa(comms_ctx, ClimaComms.SingletonCommsContext) ||
        error("ColumnGrids are incompatible with MPI")

    # Promote types
    z_max = float_type(z_max)

    # Vertical space
    z_stretch = Meshes.Uniform()
    z_space =
        make_vertical_space(; z_elem, z_max, z_stretch, comms_ctx, float_type)

    # Horizontal space
    h_space = make_trivial_horizontal_space(; comms_ctx, float_type)

    # 3D space
    center_space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    return ColumnGrid(; center_space, face_space, z_elem, z_max, z_stretch)
end


"""
function StretchedColumnGrid(; z_elem,
                               dz_bottom,
                               dz_top,
                               z_max,
                               comms_ctx = ClimaComms.context(),
                               float_type = Float64)

Construct an `AbstractAtmosGrid` for a column with non-uniform resolution (as
prescribed by `ClimaCore.Meshes.GeneralizedExponentialStretching`).

Keyword arguments
=================

- `z_elem`: Number of elements.
- `dz_bottom`: Resolution at the lower end of the column (in meters).
- `dz_top`: Resolution at the top end of the column (in meters).
- `z_max`: Height of the column (in meters).
- `comms_ctx`: Context of the computational environment where the simulation should be run,
               as defined in ClimaComms. By default, the CLIMACOMMS_DEVICE environment
               variable is read for one of "CPU", "CPUSingleThreaded", "CPUMultiThreaded",
               "CUDA". If none is found, the fallback is to use a GPU (if available), or a
               single threaded CPU (if not).
- `float_type`: Floating point type. Typically, Float32 or Float64 (default).

"""
function StretchedColumnGrid(;
    z_elem,
    z_max,
    dz_bottom,
    dz_top,
    comms_ctx = ClimaComms.context(),
    float_type = Float64,
)

    isa(comms_ctx, ClimaComms.SingletonCommsContext) ||
        error("ColumnGrids are incompatible with MPI")

    # Promote types
    z_max, dz_bottom, dz_top =
        [float_type(v) for v in [z_max, dz_bottom, dz_top]]

    # Vertical space
    z_stretch = Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
    z_space =
        make_vertical_space(; z_elem, z_max, z_stretch, comms_ctx, float_type)

    # Horizontal space
    h_space = make_trivial_horizontal_space(; comms_ctx, float_type)

    # 3D space
    center_space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    return ColumnGrid(; center_space, face_space, z_max, z_elem, z_stretch)
end
