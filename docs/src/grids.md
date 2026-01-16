# Grids

`ClimaAtmos.jl` provides several grid constructors to set up the domain layout for simulations. These grids handle the creation of the underlying `ClimaCore` meshes, topologies, and spaces, including optional topography.

## Available Grids

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

The [`ColumnGrid`](@ref) creates a single column grid, used for Single Column Models (SCM).

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

## Mesh Ordering

When constructing grids, `ClimaAtmos` uses a space-filling curve to order the elements. This improves memory locality.

Here is an example visualizing the space-filling curve for a small `BoxGrid`:

```@setup boxgrid_curve
import ClimaAtmos as CA
CC = CA.CC
using CairoMakie

# Create a small grid for visualization
grid = CA.BoxGrid(Float64; nh_poly=1, x_elem=3, y_elem=6, x_max=100, y_max=100)

# Extract the space-filling curve from the topology's mesh
spacefilling = CC.Topologies.spacefillingcurve(grid.horizontal_grid.topology.mesh)

# Extract coordinates for plotting
coords = tuple.(
   parent(grid.horizontal_grid.local_geometry.coordinates.x)[1,1,1,:],
   parent(grid.horizontal_grid.local_geometry.coordinates.y)[1,1,1,:]
)

# Plot the ordering index vs coordinate index
fig = Figure(size = (800, 400))
ax = Axis(fig[1, 1]; title = "Element Traversal Order")
sc = scatterlines!(ax, getfield.(spacefilling, :I); markersize = (1:length(coords)) .* 2, label = "Order")

# Plot the physical coordinates and the path
ax2 = Axis(fig[1, 2]; title = "Physical Coordinates Path")
scatterlines!(ax2, coords; markersize = (1:length(coords)) .* 2)

save("grid_order.png", fig); nothing # hide
```

![Grid Order](grid_order.png)
