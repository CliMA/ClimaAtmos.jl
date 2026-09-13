# Grids

`ClimaAtmos.jl` provides several grid constructors that set up the domain
layout for a simulation. They create the underlying `ClimaCore` meshes,
topologies, and spaces, including optional topography.

## Available grids

### SphereGrid

The [`SphereGrid`](@ref) creates a grid on a cubed-sphere domain, suitable for global atmospheric simulations.

```@example grids
using ClimaAtmos
grid = SphereGrid(
    Float64;
    z_elem = 10,
    radius = 6.371229e6,
    h_elem = 6,
)
```

### BoxGrid

The [`BoxGrid`](@ref) creates a 3D Cartesian box grid.

```@example grids
grid = BoxGrid(
    Float64;
    x_elem = 6,
    x_max = 300000.0,
    y_elem = 6,
    y_max = 300000.0,
    z_elem = 10,
    z_max = 30000.0,
)
```

### ColumnGrid

The [`ColumnGrid`](@ref) creates a single-column grid, used for single-column models (SCM).

```@example grids
grid = ColumnGrid(
    Float64;
    z_elem = 10,
    z_max = 30000.0,
)
```

### PlaneGrid

The [`PlaneGrid`](@ref) creates a 2D (x-z) plane grid.

```@example grids
grid = PlaneGrid(
    Float64;
    x_elem = 6,
    x_max = 300000.0,
    z_elem = 10,
    z_max = 30000.0,
)
```

## Mesh ordering

Elements are numbered along a space-filling curve so that spatial neighbors
are memory neighbors and each MPI rank owns a compact patch; ClimaCore's
[Run distributed with MPI](@extref ClimaCore Run-distributed-with-MPI) shows
the curve and how it is cut across ranks.
