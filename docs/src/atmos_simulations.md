# Performing a `ClimaAtmos` simulation

`ClimaAtmos` supports a script-based interface. With this interface, simulations
are defined in a Julia script which uses `ClimaAtmos` as a library. Simulation
scripts can also contain post-processing and visualization.

## The computational grid

One of the first step in performing a simulation is defining the computational
grid. `ClimaAtmos` comes with a variety of pre-defined grids that are commonly
used in the field:

- `StretchedColumnGrid`, a single column (1D) domain with non-uniform resolution
  (higher resolution at the bottom of the column). The variant with constant
  resolution is called `UniformColumnGrid`. Columns can only be run on
  single-threaded simulations.
- `VerticallyStretchedBoxGrid`, a periodic box with columns (1D) domain with
  non-uniform resolution in the vertical direction (higher resolution at the
  bottom of the column). The variant with vertically uniform resolution is
  `VerticallyUniformBoxGrid`. Given that `VerticallyStretchedBoxGrid` is a
  commonly used domain, we also provide an alias `Box` for it.
- `VerticallyStretchedSphereGrid`, a equiangular cubed sphere with columns (1D)
  domain with non-uniform resolution in the vertical direction (higher
  resolution at the bottom of the column). The variant with vertically uniform
  resolution is `VerticallyUniformSphereGrid`. Given that
  `VerticallyStretchedSphereGrid` is a commonly used domain, we also provide an
  alias `Sphere` for it.

By convention, all the `AbstractAtmosGrid`s in `ClimaAtmos` have a name that
ends in `Grid`.

`AbstractAtmosGrid`s support a variety of features. For instance, to see an overview of
the computational grid:
```julia
println(summary(column)) # =>
# Grid: UniformColumnGrid
# Number of elements: 10
# Height: 30000.0 meters
# Grid stretching: Uniform
```

Users interested in adding new grids can do so by defining a new concrete
subtype `MyGrid` of the abstract `AbstractAtmosGrid` type. The only requirement
for `MyGrid` is that it has to have at least two fields: `center_space` and
`face_space`, which are `ClimaCore.Spaces` for the center and the face of the
cells respectively. We refer users to the `ClimaCore` documentation to learn
more about the notion of `Spaces`.

Developers are encouraged to also define a `Base.summary` method.
