"""
The single source of truth for a SOCRATES case's vertical grid.

**The default is the Atlas LES's own grid.** The model integrates on the LES levels
(`SSCF.default_new_z`: 320 for flights 1/9/10/11, 192 for 12/13), so comparing profiles needs no
vertical interpolation and the model resolves what the LES resolved.

A ClimaCore column is defined by its *faces*, so that is the coordinate everything here is built from.
The LES publishes centres, and [`faces_from_centers`](@ref) recovers its faces exactly — the Atlas
centres are true midpoints of a grid starting at the surface, and the reconstruction reproduces their
`zf` variable.

Two ways to depart from the native grid, both explicit:

  - `dz_min` merges adjacent LES cells until each is at least that thick. Coarsening drops *faces*, so
    every coarse cell is a whole number of LES cells and the faces stay increasing whatever the LES
    spacing does.
  - `faces` takes an arbitrary vector of cell faces.

For a fully different grid, build one with ClimaAtmos directly and hand it to
[`socrates_simulation`](@ref) as `grid` — e.g.
`CA.ColumnGrid(FT; z_elem = 60, z_max = 6000, z_stretch = true, dz_bottom = 30)`.
"""

using ClimaAtmos: ClimaAtmos as CA
using ClimaComms: ClimaComms
using SOCRATESSingleColumnForcings: SOCRATESSingleColumnForcings as SSCF

"""
    native_z(case)

The Atlas LES level centres [m] for `case`.
"""
native_z(case::SocratesCase) = collect(Float64, SSCF.default_new_z(case.flight_number))

"""
    native_faces(case)

The Atlas LES cell faces [m] for `case` — the model's default vertical grid.
"""
native_faces(case::SocratesCase) = faces_from_centers(native_z(case))

"""
    z_max_default(case)

Domain top [m] for `case`: the top face of the LES grid, so the model column spans exactly the LES
column and the forcing is never extrapolated above the data.
"""
z_max_default(case::SocratesCase) = last(native_faces(case))

"""
    faces_from_centers(centers; surface = 0.0)

Cell faces [m] for a grid whose centres are `centers`, from `f₁ = surface` and `fᵢ₊₁ = 2 cᵢ − fᵢ`.

The recursion gives the only candidate face set, and its centres are `centers` identically, so the
faces increasing is exactly the condition for `centers` to be a valid midpoint set — when they are not,
the faces fold back on themselves and this errors. Pass `faces` directly for such a grid.
"""
function faces_from_centers(centers::AbstractVector; surface::Real = 0.0)
    isempty(centers) && error("faces_from_centers needs at least one centre")
    faces = Vector{Float64}(undef, length(centers) + 1)
    faces[1] = Float64(surface)
    for (i, c) in enumerate(centers)
        faces[i + 1] = 2 * Float64(c) - faces[i]
    end
    issorted(faces) || error(
        "The given centres are not cell midpoints of a grid starting at $surface m: the implied faces \
         are not increasing. Pass `faces` directly for such a grid.",
    )
    return faces
end

"""
    centers_from_faces(faces)

Cell centres [m] of a grid with the given `faces`, which is what the model integrates on.
"""
centers_from_faces(faces::AbstractVector) =
    [(Float64(faces[i]) + Float64(faces[i + 1])) / 2 for i in 1:(length(faces) - 1)]

"""
    coarsen_faces_to_dz_min(faces, dz_min)

`faces` with interior faces dropped so every cell is at least `dz_min` thick, keeping the bottom and
top.

Each retained cell is a union of adjacent input cells, so the result is a subset of `faces` and is
therefore still increasing. `dz_min = nothing`, or a value no larger than the thinnest existing cell,
returns `faces` unchanged.
"""
function coarsen_faces_to_dz_min(faces::AbstractVector, dz_min)
    fs = collect(Float64, faces)
    length(fs) >= 2 || error("A grid needs at least two faces, got $(length(fs))")
    isnothing(dz_min) && return fs
    minimum(diff(fs)) >= dz_min && return fs
    kept = [first(fs)]
    for f in fs[2:(end - 1)]
        (f - last(kept)) >= dz_min && push!(kept, f)
    end
    # Keep the column depth: the top face always survives, absorbing a final thin cell if need be.
    (last(fs) - last(kept)) >= dz_min ? push!(kept, last(fs)) : (kept[end] = last(fs))
    length(kept) >= 2 || error(
        "dz_min = $dz_min m leaves no cells in a column of depth $(last(fs) - first(fs)) m.",
    )
    return kept
end

"""
    socrates_grid(FT, case; faces, dz_min, context)

The column grid for `case`, built from explicit cell faces.

`faces` defaults to the LES faces, coarsened by `dz_min` when given. The mesh is built from exactly
these faces, so the model's cells *are* the cells described here and [`socrates_z`](@ref) reports their
centres.
"""
function socrates_grid(
    ::Type{FT},
    case::SocratesCase;
    faces::AbstractVector = native_faces(case),
    dz_min = nothing,
    context = ClimaComms.context(),
) where {FT <: AbstractFloat}
    zf = coarsen_faces_to_dz_min(faces, dz_min)
    issorted(zf) ||
        error("Cell faces must be increasing; got $(length(zf)) faces spanning $(extrema(zf)) m.")
    domain = CA.CC.Domains.IntervalDomain(
        CA.CC.Geometry.ZPoint(FT(first(zf))),
        CA.CC.Geometry.ZPoint(FT(last(zf)));
        boundary_names = (:bottom, :top),
    )
    z_mesh = CA.CC.Meshes.IntervalMesh(domain, CA.CC.Geometry.ZPoint.(FT.(zf)))
    return CA.ColumnGrid(
        FT;
        context,
        z_elem = length(zf) - 1,
        z_max = FT(last(zf)),
        z_mesh,
    )
end

"""
    socrates_z(grid)

Centre-level heights [m] of `grid`, ascending. These are the levels the model's `output_at_levels`
diagnostics are written on, so they are also the levels observations live on.
"""
function socrates_z(grid)
    center_space = CA.get_spaces(grid).center_space
    z = CA.CC.Fields.coordinate_field(center_space).z
    return vec(Array(parent(z)))
end

"""
    socrates_z(FT, case; kwargs...)

Centre-level heights [m] for `case`, building the grid on the way. Accepts the same keyword
arguments as [`socrates_grid`](@ref).
"""
socrates_z(::Type{FT}, case::SocratesCase; kwargs...) where {FT <: AbstractFloat} =
    socrates_z(socrates_grid(FT, case; kwargs...))
