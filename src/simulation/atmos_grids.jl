# This file contains the definitions of common AbstractAtmosGrids.
# - ColumnGrid (with UniformColumnGrid and StretchedColumnGrid)
# - BoxGrid (with VerticallyUniformBoxGrid and VerticallyStretchedBoxGrid)
# - SphereGrid (with VerticallyUniformSphereGrid and VerticallyStretchedSphereGrid)
# - PlaneGrid (with VerticallyUniformPlaneGrid and VerticallyStretchedPlaneGrid)
#
# We provide aliases for common grids:
# - Box = VerticallyStretchedBoxGrid
# - Sphere = VerticallyStretchedSphereGrid
# - Plane = VerticallyStretchedPlaneGrid
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
    # Promote types
    z_max = float_type(z_max)
    z_stretch = Meshes.Uniform()
    return make_column(z_elem, z_max, z_stretch, comms_ctx, float_type)
end


"""
function StretchedColumnGrid(; z_elem,
                               dz_bottom,
                               dz_top,
                               z_max,
                               comms_ctx = ClimaComms.context(),
                               float_type = Float64)

Construct an `ColumnGrid` for a column with non-uniform resolution (as
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
    z_max, dz_bottom, dz_top = map(float_type, (z_max, dz_bottom, dz_top))
    z_stretch = Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
    return make_column(z_elem, z_max, z_stretch, comms_ctx, float_type)
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
    T,
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

    topography::T
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
        ": bubble correction",
    )
    if !isnothing(grid.topography)
        println(io, "  with: $(grid.topography) topography")
    end
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
                                      topography = nothing,
                                      topo_smoothing = false,
                                      enable_bubble = false,
                                      comms_ctx = ClimaComms.context(),
                                      float_type = Float64)

Construct an `BoxGrid` for a periodic box with columns with non-uniform
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
- `topography`: Define the surface elevation profile. Provided as a warp function (or nothing).
- `topo_smoothing`: Whether to order-2 smoothing on the LGL mesh.
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
    topography = nothing,
    topo_smoothing = false,
    enable_bubble = false,
    comms_ctx = ClimaComms.context(),
    float_type = Float64,
)
    x_max, y_max, z_max, dz_bottom, dz_top =
        map(float_type, (x_max, y_max, z_max, dz_bottom, dz_top))
    z_stretch = Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
    return make_box(;
        nh_poly,
        x_elem,
        x_max,
        y_elem,
        y_max,
        z_elem,
        z_max,
        z_stretch,
        enable_bubble,
        topography,
        topo_smoothing,
        comms_ctx,
        float_type,
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
                                    topography = nothing,
                                    topo_smoothing = false,
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
- `topography`: Define the surface elevation profile. Provided as a warp function (or nothing).
- `topo_smoothing`: Whether to order-2 smoothing on the LGL mesh.
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
    topography = nothing,
    topo_smoothing = false,
    enable_bubble = false,
    comms_ctx = ClimaComms.context(),
    float_type = Float64,
)
    # Promote types
    x_max, y_max, z_max = map(float_type, (x_max, y_max, z_max))

    # Vertical
    z_stretch = Meshes.Uniform()
    return make_box(;
        nh_poly,
        x_elem,
        x_max,
        y_elem,
        y_max,
        z_elem,
        z_max,
        z_stretch,
        enable_bubble,
        topography,
        topo_smoothing,
        comms_ctx,
        float_type,
    )
end

##############
# SphereGrid #
##############

Base.@kwdef struct SphereGrid{
    CS <: Spaces.ExtrudedFiniteDifferenceSpace,
    FS <: Spaces.ExtrudedFiniteDifferenceSpace,
    I <: Integer,
    FT <: Real,
    SR <: Meshes.StretchingRule,
    T,
} <: AbstractAtmosGrid
    center_space::CS
    face_space::FS

    nh_poly::I

    h_elem::I
    radius::FT
    z_elem::I
    z_max::FT

    z_stretch::SR

    enable_bubble::Bool

    topography::T
end

function Base.summary(io::IO, grid::SphereGrid)
    println(io, "Grid type: $(nameof(typeof(grid)))")
    println(io, "Number of vertical elements: $(grid.z_elem)")
    println(io, "Height: $(grid.z_max) meters")
    println(io, "Vertical grid stretching: $(nameof(typeof(grid.z_stretch)))")
    # Add information about the stretching, if any
    for field in fieldnames(typeof(grid.z_stretch))
        println(io, "  with: $(field): $(getproperty(grid.z_stretch, field))")
    end
    println(io, "Radius: $(grid.radius) meters")
    println(io, "Horizontal elements per edge: $(grid.h_elem)")
    println(io, "  with: $(grid.nh_poly)-degree polynomials")
    println(
        io,
        "  ",
        grid.enable_bubble ? "with" : "without",
        ": bubble correction",
    )
    if !isnothing(grid.topography)
        println(io, "  with: $(grid.topography) topography")
    end
end

"""
function VerticallyStretchedSphereGrid(; nh_poly,
                                         h_elem,
                                         radius,
                                         z_elem,
                                         z_max,
                                         dz_bottom,
                                         dz_top,
                                         topography = nothing,
                                         topo_smoothing = false,
                                         enable_bubble = false,
                                         comms_ctx = ClimaComms.context(),
                                         float_type = Float64)

Construct an `SphereGrid` for a cubed sphere with columns with non-uniform
resolution (as prescribed by
`ClimaCore.Meshes.GeneralizedExponentialStretching`).

Keyword arguments
=================

- `nh_poly`: Horizontal polynomial degree.
- `radius`: Radius of the sphere (in meters).
- `h_elem`: Number of spectral elements per edge.
- `x_elem`: Number of spectral elements on the x direction.
- `x_max`: Length of the box (in meters) (`x_min` is 0).
- `y_elem`: Number of spectral elements on the y direction.
- `y_max`: Depth of the box (in meters) (`y_min` is 0).
- `z_elem`: Number of spectral elements on the vertical direction.
- `dz_bottom`: Resolution at the lower end of the column (in meters).
- `dz_top`: Resolution at the top end of the column (in meters).
- `z_max`: Height of the column (in meters).
- `topography`: Define the surface elevation profile. Provided as a warp function (or nothing).
- `topo_smoothing`: Whether to order-2 smoothing on the LGL mesh.
- `enable_bubble`: Enables the `bubble correction` for more accurate element areas.
- `comms_ctx`: Context of the computational environment where the simulation should be run,
               as defined in ClimaComms. By default, the CLIMACOMMS_DEVICE environment
               variable is read for one of "CPU", "CPUSingleThreaded", "CPUMultiThreaded",
               "CUDA". If none is found, the fallback is to use a GPU (if available), or a
               single threaded CPU (if not).
- `float_type`: Floating point type. Typically, Float32 or Float64 (default).

"""
function VerticallyStretchedSphereGrid(;
    nh_poly,
    h_elem,
    radius,
    z_elem,
    z_max,
    dz_bottom,
    dz_top,
    topography = nothing,
    topo_smoothing = false,
    enable_bubble = false,
    comms_ctx = ClimaComms.context(),
    float_type = Float64,
)
    radius, dz_bottom, dz_top, z_max =
        map(float_type, (radius, dz_bottom, dz_top, z_max))
    z_stretch = Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
    return make_sphere(;
        h_elem,
        radius,
        nh_poly,
        z_elem,
        z_max,
        z_stretch,
        topography,
        topo_smoothing,
        enable_bubble,
        comms_ctx,
        float_type,
    )
end

# Alias for a commonly used grid type
const Sphere = VerticallyStretchedSphereGrid

"""
function VerticallyUniformSphereGrid(; nh_poly,
                                       h_elem,
                                       radius,
                                       z_elem,
                                       z_max,
                                       topography = nothing,
                                       topo_smoothing = false,
                                       enable_bubble = false,
                                       comms_ctx = ClimaComms.context(),
                                       float_type = Float64)


Construct an `SphereGrid` for a cubed sphere with columns with uniform resolution.

Keyword arguments
=================

- `nh_poly`: Horizontal polynomial degree.
- `radius`: Radius of the sphere (in meters).
- `h_elem`: Number of spectral elements per edge.
- `z_elem`: Number of spectral elements on the vertical direction.
- `z_max`: Height of the column (in meters).
- `topography`: Define the surface elevation profile. Provided as a warp function (or nothing).
- `topo_smoothing`: Whether to order-2 smoothing on the LGL mesh.
- `enable_bubble`: Enables the `bubble correction` for more accurate element areas.
- `comms_ctx`: Context of the computational environment where the simulation should be run,
               as defined in ClimaComms. By default, the CLIMACOMMS_DEVICE environment
               variable is read for one of "CPU", "CPUSingleThreaded", "CPUMultiThreaded",
               "CUDA". If none is found, the fallback is to use a GPU (if available), or a
               single threaded CPU (if not).
- `float_type`: Floating point type. Typically, Float32 or Float64 (default).

"""
function VerticallyUniformSphereGrid(;
    nh_poly,
    h_elem,
    radius,
    z_elem,
    z_max,
    topography = nothing,
    topo_smoothing = false,
    enable_bubble = false,
    comms_ctx = ClimaComms.context(),
    float_type = Float64,
)
    radius, z_max = map(float_type, (radius, z_max))
    z_stretch = Meshes.Uniform()
    return make_sphere(;
        h_elem,
        radius,
        nh_poly,
        z_elem,
        z_max,
        z_stretch,
        topography,
        topo_smoothing,
        enable_bubble,
        comms_ctx,
        float_type,
    )
end

#############
# PlaneGrid #
#############

Base.@kwdef struct PlaneGrid{
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
    z_elem::I
    z_max::FT

    z_stretch::SR
end

function Base.summary(io::IO, grid::PlaneGrid)
    println(io, "Grid type: $(nameof(typeof(grid)))")
    println(io, "Number of vertical elements: $(grid.z_elem)")
    println(io, "Height: $(grid.z_max) meters")
    println(io, "Vertical grid stretching: $(nameof(typeof(grid.z_stretch)))")
    # Add information about the stretching, if any
    for field in fieldnames(typeof(grid.z_stretch))
        println(io, "  with: $(field): $(getproperty(grid.z_stretch, field))")
    end
    FT = float_type(grid)
    println(io, "Horizontal domain: $x∈ [$(zero(FT)), $(grid.x_max)]")
    println(io, "  with: $(grid.x_elem) elements")
    println(io, "  with: $(grid.nh_poly)-degree polynomials")
end

"""
function VerticallyStretchedPlaneGrid(; nh_poly,
                                        x_elem,
                                        x_max,
                                        z_elem,
                                        z_max,
                                        dz_bottom,
                                        dz_top,
                                        comms_ctx = ClimaComms.context(),
                                        float_type = Float64)

Construct a `PlaneGrid` for a periodic linear domain with columns with
non-uniform resolution (as prescribed by
`ClimaCore.Meshes.GeneralizedExponentialStretching`).

Keyword arguments
=================

- `nh_poly`: Horizontal polynomial degree.
- `x_elem`: Number of spectral elements on the x direction.
- `x_max`: Length of the box (in meters) (`x_min` is 0).
- `z_elem`: Number of spectral elements on the vertical direction.
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
function VerticallyStretchedPlaneGrid(;
    nh_poly,
    x_elem,
    x_max,
    z_elem,
    z_max,
    dz_bottom,
    dz_top,
    comms_ctx = ClimaComms.context(),
    float_type = Float64,
)

    # Promote types
    x_max, z_max, dz_bottom, dz_top =
        map(float_type, (x_max, z_max, dz_bottom, dz_top))

    # Vertical
    z_stretch = Meshes.GeneralizedExponentialStretching(dz_bottom, dz_top)
    z_space =
        make_vertical_space(; z_elem, z_max, z_stretch, comms_ctx, float_type)

    # Horizontal
    h_space = make_plane_horizontal_space(; x_max, x_elem, nh_poly, comms_ctx)

    center_space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    return PlaneGrid(;
        center_space,
        face_space,
        nh_poly,
        x_elem,
        x_max,
        z_elem,
        z_max,
        z_stretch,
    )
end

# Alias for a commonly used grid type
const Plane = VerticallyStretchedPlaneGrid

"""
function VerticallyUniformPlaneGrid(; nh_poly,
                                      x_elem,
                                      x_max,
                                      y_elem,
                                      y_max,
                                      z_elem,
                                      z_max,
                                      comms_ctx = ClimaComms.context(),
                                      float_type = Float64)

Construct an `PlaneGrid` for a periodic box with columns with uniform resolution.

Keyword arguments
=================

- `nh_poly`: Horizontal polynomial degree.
- `x_elem`: Number of spectral elements on the x direction.
- `x_max`: Length of the box (in meters) (`x_min` is 0).
- `z_elem`: Number of spectral elements on the vertical direction.
- `z_max`: Height of the column (in meters).
- `comms_ctx`: Context of the computational environment where the simulation should be run,
               as defined in ClimaComms. By default, the CLIMACOMMS_DEVICE environment
               variable is read for one of "CPU", "CPUSingleThreaded", "CPUMultiThreaded",
               "CUDA". If none is found, the fallback is to use a GPU (if available), or a
               single threaded CPU (if not).
- `float_type`: Floating point type. Typically, Float32 or Float64 (default).

"""
function VerticallyUniformPlaneGrid(;
    nh_poly,
    x_elem,
    x_max,
    z_elem,
    z_max,
    comms_ctx = ClimaComms.context(),
    float_type = Float64,
)
    # Promote types
    x_max, z_max = [float_type(v) for v in [x_max, z_max]]

    # Vertical
    z_stretch = Meshes.Uniform()
    z_space =
        make_vertical_space(; z_elem, z_max, z_stretch, comms_ctx, float_type)

    # Horizontal
    h_space = make_plane_horizontal_space(; x_max, x_elem, nh_poly, comms_ctx)

    center_space = Spaces.ExtrudedFiniteDifferenceSpace(h_space, z_space)
    face_space = Spaces.FaceExtrudedFiniteDifferenceSpace(center_space)

    return PlaneGrid(;
        center_space,
        face_space,
        nh_poly,
        x_elem,
        x_max,
        z_elem,
        z_max,
        z_stretch,
    )
end
