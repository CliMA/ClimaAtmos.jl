# This file contains the definitions of common AbstractAtmosGrids.
# - ColumnGrid (with UniformColumnGrid and StretchedColumnGrid)
# - BoxGrid (with VerticallyUniformBoxGrid and VerticallyStretchedBoxGrid)
#
# We provide aliases for common grids:
# - Box = VerticallyStretchedBoxGrid
#
# We also provide convenience functions to build these grids.
include("atmos_grids_makers.jl")

##############
# ColumnGrid #
##############

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

###########
# BoxGrid #
###########

Base.@kwdef struct BoxGrid{
    CS <: Spaces.ExtrudedFiniteDifferenceSpace,
    FS <: Spaces.ExtrudedFiniteDifferenceSpace,
    I <: Integer,
    FT <: Real,
    SR <: Meshes.StretchingRule,
} <: AbstractAtmosGrid
    center_space::CS
    face_space::FS

    nh_poly::I

    x_elem::I
    x_max::FT
    y_elem::I
    y_max::FT
    z_elem::I
    z_max::FT

    z_stretch::SR

    enable_bubble::Bool
end

function Base.summary(io::IO, grid::BoxGrid)
    println(io, "Grid type: $(nameof(typeof(grid)))")
    println(io, "Number of vertical elements: $(grid.z_elem)")
    println(io, "Height: $(grid.z_max) meters")
    println(io, "Vertical grid stretching: $(nameof(typeof(grid.z_stretch)))")
    # Add information about the stretching, if any
    for field in fieldnames(typeof(grid.z_stretch))
        println(io, "  with: $(field): $(getproperty(grid.z_stretch, field))")
    end
    FT = float_type(grid)
    println(
        io,
        "Horizontal domain: $x, y ∈ [$(zero(FT)), $(grid.x_max)] × [$(zero(FT)), $(grid.y_max)]",
    )
    println(io, "  with: $(grid.x_elem) × $(grid.y_elem) elements")
    println(io, "  with: $(grid.nh_poly)-degree polynomials")
    println(
        io,
        "  ",
        grid.enable_bubble ? "with" : "without",
        " bubble correction",
    )
end

"""
function VerticallyStretchedBoxGrid(; nh_poly,
                                      x_elem,
                                      x_max,
                                      y_elem,
                                      y_max,
                                      z_elem,
                                      z_max,
                                      dz_bottom,
                                      dz_top,
                                      enable_bubble = false,
                                      comms_ctx = ClimaComms.context(),
                                      float_type = Float64)

Construct an `AbstractAtmosGrid` for a periodic box with columns with non-uniform
resolution (as prescribed by `ClimaCore.Meshes.GeneralizedExponentialStretching`).

Keyword arguments
=================

- `nh_poly`: Horizontal polynomial degree.
- `x_elem`: Number of spectral elements on the x direction.
- `x_max`: Length of the box (in meters) (`x_min` is 0).
- `y_elem`: Number of spectral elements on the y direction.
- `y_max`: Depth of the box (in meters) (`y_min` is 0).
- `z_elem`: Number of spectral elements on the vertical direction.
- `dz_bottom`: Resolution at the lower end of the column (in meters).
- `dz_top`: Resolution at the top end of the column (in meters).
- `z_max`: Height of the column (in meters).
- `enable_bubble`: Enables the `bubble correction` for more accurate element areas.
- `comms_ctx`: Context of the computational environment where the simulation should be run,
               as defined in ClimaComms. By default, the CLIMACOMMS_DEVICE environment
               variable is read for one of "CPU", "CPUSingleThreaded", "CPUMultiThreaded",
               "CUDA". If none is found, the fallback is to use a GPU (if available), or a
               single threaded CPU (if not).
- `float_type`: Floating point type. Typically, Float32 or Float64 (default).

"""
function VerticallyStretchedBoxGrid(;
    nh_poly,
    x_elem,
    x_max,
    y_elem,
    y_max,
    z_elem,
    z_max,
    dz_bottom,
    dz_top,
    enable_bubble = false,
    comms_ctx = ClimaComms.context(),
    float_type = Float64,
)

    # Promote types
    x_max, y_max, z_max, dz_bottom, dz_top =
        [float_type(v) for v in [x_max, y_max, z_max, dz_bottom, dz_top]]

    # Vertical
    z_stretch = Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
    z_space =
        make_vertical_space(; z_elem, z_max, z_stretch, comms_ctx, float_type)

    # Horizontal
    x_domain = Domains.IntervalDomain(
        Geometry.XPoint(zero(x_max)),
        Geometry.XPoint(x_max);
        periodic = true,
    )
    y_domain = Domains.IntervalDomain(
        Geometry.YPoint(zero(y_max)),
        Geometry.YPoint(y_max);
        periodic = true,
    )
    h_domain = Domains.RectangleDomain(x_domain, y_domain)
    h_mesh = Meshes.RectilinearMesh(h_domain, x_elem, y_elem)
    h_space = make_horizontal_space(; nh_poly, h_mesh, comms_ctx, enable_bubble)

    center_space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    return BoxGrid(;
        center_space,
        face_space,
        nh_poly,
        x_elem,
        x_max,
        y_elem,
        y_max,
        z_elem,
        z_max,
        z_stretch,
        enable_bubble,
    )
end

# Alias for a commonly used grid type
const Box = VerticallyStretchedBoxGrid

"""
function VerticallyUniformBoxGrid(; nh_poly,
                                    x_elem,
                                    x_max,
                                    y_elem,
                                    y_max,
                                    z_elem,
                                    z_max,
                                    enable_bubble = false,
                                    comms_ctx = ClimaComms.context(),
                                    float_type = Float64)

Construct an `BoxGrid` for a periodic box with columns with uniform resolution.

Keyword arguments
=================

- `nh_poly`: Horizontal polynomial degree.
- `x_elem`: Number of spectral elements on the x direction.
- `x_max`: Length of the box (in meters) (`x_min` is 0).
- `y_elem`: Number of spectral elements on the y direction.
- `y_max`: Depth of the box (in meters) (`y_min` is 0).
- `z_elem`: Number of spectral elements on the vertical direction.
- `z_max`: Height of the column (in meters).
- `enable_bubble`: Enables the `bubble correction` for more accurate element areas.
- `comms_ctx`: Context of the computational environment where the simulation should be run,
               as defined in ClimaComms. By default, the CLIMACOMMS_DEVICE environment
               variable is read for one of "CPU", "CPUSingleThreaded", "CPUMultiThreaded",
               "CUDA". If none is found, the fallback is to use a GPU (if available), or a
               single threaded CPU (if not).
- `float_type`: Floating point type. Typically, Float32 or Float64 (default).

"""
function VerticallyUniformBoxGrid(;
    nh_poly,
    x_elem,
    x_max,
    y_elem,
    y_max,
    z_elem,
    z_max,
    enable_bubble = false,
    comms_ctx = ClimaComms.context(),
    float_type = Float64,
)

    # Promote types
    x_max, y_max, z_max = [float_type(v) for v in [x_max, y_max, z_max]]

    # Vertical
    z_stretch = Meshes.Uniform()
    z_space =
        make_vertical_space(; z_elem, z_max, z_stretch, comms_ctx, float_type)

    # Horizontal
    x_domain = Domains.IntervalDomain(
        Geometry.XPoint(zero(x_max)),
        Geometry.XPoint(x_max);
        periodic = true,
    )
    y_domain = Domains.IntervalDomain(
        Geometry.YPoint(zero(y_max)),
        Geometry.YPoint(y_max);
        periodic = true,
    )
    h_domain = Domains.RectangleDomain(x_domain, y_domain)
    h_mesh = Meshes.RectilinearMesh(h_domain, x_elem, y_elem)
    h_space = make_horizontal_space(; nh_poly, h_mesh, comms_ctx, enable_bubble)

    center_space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    return BoxGrid(;
        center_space,
        face_space,
        nh_poly,
        x_elem,
        x_max,
        y_elem,
        y_max,
        z_elem,
        z_max,
        z_stretch,
        enable_bubble,
    )
end
