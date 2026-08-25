using NCDatasets
import Interpolations
using Statistics: mean
import ClimaUtilities.SpaceVaryingInputs: SpaceVaryingInput
import ClimaCore: Geometry, Fields, Spaces
import ClimaCore.Utilities: half

"""
    calc_orographic_tensor(elev, χ, lon, lat, earth_radius) -> (t11, t21, t12, t22)

Compute the orographic tensor `T = −∇χ (∇h)ᵀ` on the lat-lon grid.

Mutates `elev` in place, clipping ocean bathymetry to zero, then returns the four
components with the code's index convention `tᵢⱼ = −∂χ/∂xⱼ · ∂h/∂xᵢ`, so the stored `tᵢⱼ`
is the `(j,i)` entry of `−∇χ(∇h)ᵀ`. The leading minus reproduces Garner's sign convention,
so the resulting drag opposes the low-level flow.

Called from `compute_OGW_info` in the offline preprocessing pipeline.

# Arguments

  - `elev`: Surface elevation on the lon-lat grid [m]; clipped in place.
  - `χ`: Velocity potential from `calc_velocity_potential`.
  - `lon`, `lat`: Coordinate vectors [degrees].
  - `earth_radius`: Sphere radius [m].

Described in [garner2005](@cite) Eq. 6; see the *Orographic Gravity Waves* page.
"""
function calc_orographic_tensor(elev, χ, lon, lat, earth_radius)
    @info "Computing T tensor..."
    FT = eltype(elev)

    # compute ∇h
    @. elev = max(0, elev)
    dhdx, dhdy = calc_∇A(elev, lon, lat, earth_radius)
    # compute ∇χ
    # dχdx, dχdy = -bfscale .* calc_∇A(χ, lon, lat, earth_radius)
    # TODO: This needs to be double checked with Steve Garner.
    # It looks like bfscale is multiply when he compute the raw T tensor but
    # later divided again when creating the file that contains the actual T tensor
    # being used.
    dχdx, dχdy = .-calc_∇A(χ, lon, lat, earth_radius)


    t11 = dχdx .* dhdx
    t21 = dχdx .* dhdy
    t12 = dχdy .* dhdx
    t22 = dχdy .* dhdy

    return (t11, t21, t12, t22)
end

"""
    calc_∇A(A, lon, lat, earth_radius) -> (dAdx, dAdy)

Compute the horizontal gradient of a scalar on the spherical surface.

Returns the zonal and meridional derivatives per unit physical distance, both the same
shape as `A`, using centered differences in the interior and one-sided differences at the
boundaries of each dimension.

The zonal spacing shrinks as `cos(lat)`, so it is divided out; the factor is floored at
`cos(60°)` to keep the gradient finite poleward of 60°, where the grid converges. Longitude
is treated as non-periodic here, so the first and last columns use one-sided differences.

Called from `calc_orographic_tensor` in the offline preprocessing pipeline.

# Arguments

  - `A`: Scalar field on the lon-lat grid, indexed `[lon, lat]`.
  - `lon`, `lat`: Uniformly spaced coordinate vectors [degrees].
  - `earth_radius`: Sphere radius [m].
"""
function calc_∇A(A, lon, lat, earth_radius)
    FT = eltype(A)
    dlat = lat[2] - lat[1]
    dlon = lon[2] - lon[1]
    dAdx =
        vcat(
            (A[2, :] .- A[1, :])',
            (A[3:end, :] .- A[1:(end - 2), :]) / FT(2),
            (A[end, :] .- A[end - 1, :])',
        ) ./ (
            # Convert the longitudinal grid spacing (dlon) from degrees to
            # physical distance (meters) at each latitude. The zonal distance
            # per degree of longitude shrinks as cos(lat), so we multiply by
            # cos(lat). The max(..., cosd(60)) clamp prevents division by
            # near-zero values at high latitudes (|lat| > 60°). The result is
            # tiled across all longitudes to form a (lon × lat) matrix.
            deg2rad(dlon) * earth_radius .*
            reshape(repeat(max.(cosd.(lat), cosd(FT(60))), length(lon)), length(lat), :)'
        )
    dAdy =
        hcat(
            A[:, 2] .- A[:, 1],
            (A[:, 3:end] .- A[:, 1:(end - 2)]) / FT(2),
            A[:, end] .- A[:, end - 1],
        ) ./ (deg2rad(dlat) * earth_radius)
    return (dAdx, dAdy)
end

"""
    calc_velocity_potential(elev, lon, lat, earth_radius;
                            smoothing_length_scale = nothing,
                            n_smoothing_cells = nothing)

Compute the velocity potential `χ` as a windowed 2D Hilbert transform of the elevation.

`χ` captures the far-field influence of the surrounding topography on the low-level flow:
a `1/d` great-circle kernel weighted by `cos(lat)` and tapered by a Blackman window that
reaches zero at the smoothing scale, plus a Green's-function correction for the singular
self-cell. Mutates `elev` in place, clipping ocean bathymetry to zero, and returns `χ` with
the same `[nlon, nlat]` shape.

Following the GFDL `get_velpot.f90`, the kernel weights are precomputed once per latitude
row and reused across longitudes, and longitude wrapping is periodic. Great-circle
distances use the Haversine form, which stays accurate for the short arcs that dominate.
Rows with `|lat| > 89°` are skipped, matching the Fortran; this is a no-op in practice
given the Antarctic mask and zero Arctic elevation.

The smoothing scale is latitude-dependent, `sin(40°)/sin(max(20°, |lat|))` times the
requested distance, divided by the Earth radius to give an angular scale, and is floored at
one grid cell in each direction so the weights cannot blow up on a coarse grid.

Called from `compute_OGW_info` in the offline preprocessing pipeline.

# Arguments

  - `elev`: Surface elevation on the lon-lat grid [m]; clipped in place.
  - `lon`, `lat`: Uniformly spaced coordinate vectors [degrees].
  - `earth_radius`: Sphere radius [m].

# Keyword Arguments

  - `smoothing_length_scale = nothing`: Physical smoothing scale [m].
  - `n_smoothing_cells = nothing`: Smoothing scale in grid cells [-], measured against the
    diagonal grid spacing.

Exactly one of the two keywords must be given; supplying both or neither is an error.

Described in [garner2005](@cite); see the *Orographic Gravity Waves* page.
"""
function calc_velocity_potential(
    elev,
    lon,
    lat,
    earth_radius;
    smoothing_length_scale = nothing,
    n_smoothing_cells = nothing,
)
    @info "Computing velocity potential (Haversine arc, tiny=1e-12)..."
    FT = eltype(elev)

    # Ensure non-negative elevation
    @. elev = max(0, elev)

    nlon = length(lon)
    nlat = length(lat)

    dlat = lat[2] - lat[1]
    dlon = lon[2] - lon[1]

    # Convert to radians
    dlat_rad = deg2rad(dlat)
    dlon_rad = deg2rad(dlon)

    # Compute grid-aware scale
    # Physical grid spacing in meters
    Δx_lon = deg2rad(dlon) * earth_radius .* cosd.(lat)  # varies with latitude
    Δy_lat = deg2rad(dlat) * earth_radius                # constant
    Δh_grid = sqrt.(Δx_lon .^ 2 .+ Δy_lat^2)            # diagonal grid spacing

    # Determine scale_distance based on user input
    if n_smoothing_cells !== nothing && smoothing_length_scale !== nothing
        error(
            "Cannot specify both n_smoothing_cells and smoothing_length_scale. Choose one.",
        )
    elseif n_smoothing_cells !== nothing
        # User specified exact number of cells (on lat-lon grid)
        scale_distance = n_smoothing_cells .* Δh_grid
    elseif smoothing_length_scale !== nothing
        # User specified physical length scale in meters (from GCM grid)
        scale_distance = smoothing_length_scale
    else
        error(
            "Must specify either n_smoothing_cells or smoothing_length_scale for grid-aware preprocessing.",
        )
    end

    # Apply latitude-dependent scaling factor (matches Fortran)
    scale =
        sind(40) ./ earth_radius ./ sind.(max.(FT(20), abs.(lat))) .*
        scale_distance

    # Enforce minimum scale based on grid spacing to prevent weight blow-up
    # The scale must be at least 1 grid cell in each direction
    min_scale_lat = dlat_rad  # At least one latitude grid spacing
    min_scale_lon = dlon_rad .* cosd.(lat)  # At least one longitude grid spacing (latitude-dependent)
    # Use the larger of the two to ensure proper smoothing in both directions
    min_scale_combined = max.(min_scale_lat, min_scale_lon)
    scale = max.(scale, min_scale_combined)

    # Compute window sizes for spatial running mean using Blackman kernel
    ilat_range = Int.(round.(scale ./ dlat_rad))
    ilon_range =
        min.(
            Int.(trunc.(scale ./ (dlon_rad .* cosd.(lat)))),
            Int(round(nlon / 8)),
        )

    # Pre-compute trigonometric values
    cosphi = cosd.(lat)

    # Pre-compute sin²(Δlon/2) for Haversine formula (reused across all grid points)
    max_ilon_offset = maximum(ilon_range)
    sin_half_dlon_sq = [sin(deg2rad(i2 * dlon) / 2)^2 for i2 in 0:max_ilon_offset]

    # Initialize output
    χ = zeros(FT, nlon, nlat)

    # Main convolution loop: latitude (outer) then longitude (inner)
    # Note: Fortran skips |lat| > 89° (get_velpot.f90:81). This is a no-op
    # (Antarctic mask + zero Arctic elevation) but we match it for safety.
    for j in 1:nlat
        if abs(lat[j]) > FT(89)
            continue
        end
        # Determine latitude window for this j
        ja = max(1, j - ilat_range[j])
        jb = min(nlat, j + ilat_range[j])

        # Pre-compute weights for all latitude offsets and longitude offsets
        # wt[i_offset, j1] stores weight for longitude offset i_offset and latitude index j1
        # Note: Using ja+1:jb so array size is (jb - ja)
        max_i_offset = min(ilon_range[j], max_ilon_offset)
        wt = zeros(FT, max_i_offset + 1, jb - ja)  # +1 for zero offset in first dim

        cj = cosphi[j]

        # Compute weights once per latitude row (OPTIMIZATION: moved outside i-loop)
        # Note: Fortran uses ja+1:jb (skips first element), we match that here
        for (j1_idx, j1) in enumerate((ja + 1):jb)
            cj1 = cosphi[j1]

            # Haversine: sin²(Δlat/2) — hoisted out of i2 loop (depends only on j, j1)
            sin_half_dlat_sq = sin(deg2rad(lat[j1] - lat[j]) / 2)^2

            for i2 in 0:max_i_offset
                # Haversine formula for great-circle arc (numerically stable for small arcs)
                # h = sin²(Δlat/2) + cos(lat₁)·cos(lat₂)·sin²(Δlon/2)
                # arc = 2·asin(√h)
                h = sin_half_dlat_sq + cj * cj1 * sin_half_dlon_sq[i2 + 1]
                arc = FT(2) * asin(sqrt(clamp(h, FT(0), FT(1))))

                # Blackman window taper (matches Fortran exactly)
                arc1 = arc / max(arc, scale[j])
                blackman =
                    FT(0.42) +
                    FT(0.50) * cos(FT(π) * arc1) +
                    FT(0.08) * cos(FT(2π) * arc1)

                # Weight: cos(lat) / (arc + tiny) * Blackman_window
                # Matches Fortran get_velpot.f90:98-101 with tiny=1e-12
                wt[i2 + 1, j1_idx] = cj1 / (arc + FT(1e-12)) * blackman
            end

            # Zero out singularity (when j == j1 and i_offset == 0)
            if j == j1
                wt[1, j1_idx] = FT(0)
            end
        end

        # Now loop over all longitudes at this latitude
        for i in 1:nlon
            # Determine longitude window with PERIODIC boundaries
            ia = i - ilon_range[j]
            ib = i + ilon_range[j]

            sum_val = FT(0)

            # Accumulate weighted sum over window
            # Note: Fortran uses ja+1:jb and ia+1:ib (skips first elements)
            for (j1_idx, j1) in enumerate((ja + 1):jb)
                for i1 in (ia + 1):ib
                    # Handle periodic longitude boundaries
                    i1_wrapped = mod1(i1, nlon)  # Julia's mod1 handles periodic wrapping

                    # Compute offset distance
                    i2 = abs(i - i1_wrapped)
                    # Handle wrap-around: min(i2, nlon - i2) for shortest path
                    i2 = min(i2, nlon - i2)

                    # Only use pre-computed weights within range
                    if i2 <= max_i_offset
                        sum_val += wt[i2 + 1, j1_idx] * elev[i1_wrapped, j1]
                    end
                end
            end
            χ[i, j] = sum_val
        end
    end

    # Apply area element scaling
    # The weights already include cos(lat) normalization (cj1 term)
    χ .*= (dlon_rad * dlat_rad)

    # Add singularity correction term (Green's function correction for finite grid size)
    # Note: Fortran (get_velpot.f90:121-129) applies this at ALL latitudes.
    # The term is well-behaved near poles: as lat→90°, dlamj→0 but dlamj*log(...)→0.
    # Cell-centered grids never have exact polar points, so cos(lat) > 0 always.
    for j in 1:nlat
        jlat_deg = lat[j]

        cj = cosd(jlat_deg)
        dlamj = dlon_rad * cj
        rij = sqrt(dlamj^2 + dlat_rad^2)

        for i in 1:nlon
            χ[i, j] +=
                elev[i, j] * FT(2) *
                (
                    dlamj * log((rij + dlat_rad) / dlamj) +
                    dlat_rad * log((rij + dlamj) / dlat_rad)
                )
        end
    end

    # Final scaling
    χ .*= (earth_radius / (FT(2) * pi))

    return χ
end


"""
    smooth_field_latlon(field, lon, lat, earth_radius;
                        smoothing_length_scale = nothing,
                        n_smoothing_cells = nothing,
                        use_lat_factor = true)

Smooth a lat-lon field with the same Lorentzian kernel `calc_hpoz_latlon` uses.

Returns a new array of the same shape holding the weighted mean of `field` over a
latitude-dependent window, with weights `w = 1/(1 + arc²/scale²)` and `arc` the angular
separation. `field` is not modified. Unlike `calc_velocity_potential`, the longitude window
is clipped rather than wrapped.

Available for optionally pre-smoothing the elevation before the tensor gradients are taken;
`compute_OGW_info` leaves that off by default and passes the raw elevation instead.

# Arguments

  - `field`: Field on the lon-lat grid, indexed `[lon, lat]`.
  - `lon`, `lat`: Uniformly spaced coordinate vectors [degrees].
  - `earth_radius`: Sphere radius [m].

# Keyword Arguments

  - `smoothing_length_scale = nothing`: Physical smoothing scale [m].
  - `n_smoothing_cells = nothing`: Smoothing scale in grid cells [-]. Exactly one of this and
    `smoothing_length_scale` must be given.
  - `use_lat_factor = true`: Use the `sin(40°)/sin(max(20°, |lat|))` latitude weighting that
    matches the Fortran treatment of `χ` and `hmax`. When `false`, the scale is isotropic
    with a `1/max(0.3, cos(lat))` polar compensation instead.
"""
function smooth_field_latlon(
    field,
    lon,
    lat,
    earth_radius;
    smoothing_length_scale = nothing,
    n_smoothing_cells = nothing,
    use_lat_factor = true,
)
    @info "Smoothing field on lat-lon grid..."
    FT = eltype(field)

    dlat = lat[2] - lat[1]
    dlon = lon[2] - lon[1]

    # Compute grid-aware scale (same as calc_hpoz_latlon)
    Δx_lon = deg2rad(dlon) * earth_radius .* cosd.(lat)
    Δy_lat = deg2rad(dlat) * earth_radius
    Δh_grid = sqrt.(Δx_lon .^ 2 .+ Δy_lat^2)

    if n_smoothing_cells !== nothing && smoothing_length_scale !== nothing
        error(
            "Cannot specify both n_smoothing_cells and smoothing_length_scale. Choose one.",
        )
    elseif n_smoothing_cells !== nothing
        # User specified exact number of cells (on lat-lon grid)
        scale_distance = n_smoothing_cells .* Δh_grid
    elseif smoothing_length_scale !== nothing
        # User specified physical length scale in meters (from GCM grid)
        scale_distance = smoothing_length_scale
    else
        error(
            "Must specify either n_smoothing_cells or smoothing_length_scale for grid-aware preprocessing.",
        )
    end

    # Scale computation
    scale = if use_lat_factor
        # Latitude-dependent scaling (matches Fortran χ/hmax behavior)
        sind(40) ./ earth_radius ./ sind.(max.(FT(20), abs.(lat))) .*
        scale_distance
    else
        # Isotropic with 1/cos polar compensation
        scale_distance ./ earth_radius ./ max.(FT(0.3), cosd.(lat))
    end

    ilat_range = Int.(round.(scale ./ deg2rad(dlat)))
    ilon_range =
        min.(
            Int.(round.(scale ./ (deg2rad(dlon) .* cosd.(lat)))),
            Int(round(length(lon) / 8)),
        )

    result = zeros(FT, size(field))
    cosphi = cosd.(lat)

    nlon = length(lon)
    for j in eachindex(lat)
        jrange =
            max(j - ilat_range[j], 1):min(j + ilat_range[j], length(lat))
        max_ilon = ilon_range[j]
        s2 = scale[j]^2

        # Precompute full 2D weight table for this latitude band
        dlat_dists = deg2rad.(lat[jrange] .- lat[j])
        dlon_offsets = deg2rad.(((-max_ilon):max_ilon) .* dlon) .* cosphi[j]
        arc2 = dlon_offsets .^ 2 .+ dlat_dists' .^ 2
        wt_full = FT(1) ./ (FT(1) .+ arc2 ./ s2)

        for i in eachindex(lon)
            irange = max(i - max_ilon, 1):min(i + max_ilon, nlon)

            # Map irange to indices in the precomputed weight table
            wt_i =
                (first(irange) - i + max_ilon + 1):(last(irange) - i + max_ilon + 1)
            w = @view wt_full[wt_i, :]
            field_window = @view field[irange, jrange]

            result[i, j] = sum(w .* field_window) / sum(w)
        end
    end

    return result
end


"""
    calc_hpoz_latlon(elev, lon, lat, earth_radius;
                     smoothing_length_scale = nothing,
                     n_smoothing_cells = nothing)

Compute the raw subgrid height statistic `h₀` from a distance-weighted fourth moment.

Returns an array shaped like `elev` holding

    h₀ = (Σᵢ wᵢ (hᵢ − h̄)⁴ / Σᵢ wᵢ)^(1/4)    [m]

with `h̄` the distance-weighted local mean elevation and Lorentzian weights
`wᵢ = 1/(1 + arcᵢ²/s²)`, where `s` is the same latitude-dependent angular scale
`calc_velocity_potential` uses. The fourth moment emphasizes the tallest peaks, which
generate most of the wave drag, while still averaging over the subgrid terrain. Mutates
`elev` in place, clipping ocean bathymetry to zero.

This is only the raw statistic; the caller `compute_OGW_info` rescales it with the
mountain height-width exponent `γ` and the height fraction `h_frac` to obtain the `hmax`
and `hmin` the parameterization consumes. Ports the GFDL `get_hmax`.

# Arguments

  - `elev`: Surface elevation on the lon-lat grid [m]; clipped in place.
  - `lon`, `lat`: Uniformly spaced coordinate vectors [degrees].
  - `earth_radius`: Sphere radius [m].

# Keyword Arguments

  - `smoothing_length_scale = nothing`: Physical smoothing scale [m].
  - `n_smoothing_cells = nothing`: Smoothing scale in grid cells [-]. Exactly one of this and
    `smoothing_length_scale` must be given.

Described in [garner2005](@cite); see the *Orographic Gravity Waves* page.
"""
function calc_hpoz_latlon(
    elev,
    lon,
    lat,
    earth_radius;
    smoothing_length_scale = nothing,
    n_smoothing_cells = nothing,
)
    @info "Computing hmax..."
    FT = eltype(elev)

    # remove ocean topography
    elev[elev .< FT(0)] .= FT(0)

    dlat = lat[2] - lat[1]
    dlon = lon[2] - lon[1]

    # Compute grid-aware scale
    # Physical grid spacing in meters
    Δx_lon = deg2rad(dlon) * earth_radius .* cosd.(lat)  # varies with latitude
    Δy_lat = deg2rad(dlat) * earth_radius                # constant
    Δh_grid = sqrt.(Δx_lon .^ 2 .+ Δy_lat^2)            # diagonal grid spacing

    # Determine scale_distance based on user input (same logic as calc_velocity_potential)
    if n_smoothing_cells !== nothing && smoothing_length_scale !== nothing
        error(
            "Cannot specify both n_smoothing_cells and smoothing_length_scale. Choose one.",
        )
    elseif n_smoothing_cells !== nothing
        # User specified exact number of cells (on lat-lon grid)
        scale_distance = n_smoothing_cells .* Δh_grid
    elseif smoothing_length_scale !== nothing
        # User specified physical length scale in meters (from GCM grid)
        scale_distance = smoothing_length_scale
    else
        error(
            "Must specify either n_smoothing_cells or smoothing_length_scale for grid-aware preprocessing.",
        )
    end

    # Apply latitude-dependent scaling factor (matches Fortran)
    scale =
        sind(40) ./ earth_radius ./ sind.(max.(FT(20), abs.(lat))) .*
        scale_distance

    ilat_range = Int.(round.(scale ./ deg2rad(dlat)))
    # Changed from /16 to /8 to match Fortran
    ilon_range =
        min.(
            Int.(round.(scale ./ (deg2rad(dlon) .* cosd.(lat)))),
            Int(round(length(lon) / 8)),
        )

    hmax = zeros(size(elev))

    # Precompute trigonometric values for arc distance calculation
    cosphi = cosd.(lat)

    for i in eachindex(lon)
        for j in eachindex(lat)
            # TODO: irange may not need clipping at the boundaries since it is on the closed lat circle
            irange =
                max(i - ilon_range[j], 1):min(i + ilon_range[j], length(lon))
            jrange =
                max(j - ilat_range[j], 1):min(j + ilat_range[j], length(lat))

            # Distance-weighted 4th moment calculation (matches Fortran)
            # Pass 1: Compute weighted mean
            hmn = FT(0)
            wt = FT(0)

            for j1 in jrange
                for i1 in irange
                    # Compute angular arc distance on sphere (in radians)
                    # arc² is in radians² to match scale² units
                    dlon_dist = deg2rad(lon[i1] - lon[i]) * cosphi[j]
                    dlat_dist = deg2rad(lat[j1] - lat[j])
                    arc2 = dlon_dist^2 + dlat_dist^2

                    # Weight function: w = 1/(1 + arc²/scale²)
                    w = FT(1) / (FT(1) + arc2 / (scale[j]^2))

                    hmn += w * elev[i1, j1]
                    wt += w
                end
            end
            hmn = hmn / wt  # weighted mean elevation

            # Pass 2: Compute weighted 4th moment
            var = FT(0)
            for j1 in jrange
                for i1 in irange
                    dlon_dist = deg2rad(lon[i1] - lon[i]) * cosphi[j]
                    dlat_dist = deg2rad(lat[j1] - lat[j])
                    arc2 = dlon_dist^2 + dlat_dist^2

                    w = FT(1) / (FT(1) + arc2 / (scale[j]^2))

                    var += w * (elev[i1, j1] - hmn)^4
                end
            end

            # 4th root of weighted 4th moment
            hmax[i, j] = (var / wt)^FT(0.25)
        end
    end

    return hmax
end

"""
    compute_OGW_info(Y, elev_data, earth_radius, γ, h_frac; α_smoothing = 0.15)

Run the offline preprocessing pipeline that turns raw elevation into orographic drag input.

Returns `(; hmax, hmin, t11, t12, t21, t22)` on the surface cell-center space of `Y`. This
is the entry point of the offline path; the result is normally written to an HDF5 artifact
by `write_computed_drag!` and loaded at runtime rather than recomputed.

The smoothing scale is derived from the target grid: `α_smoothing` times the node
horizontal length scale of `Y`, and the raw elevation is subsampled by
`skip_pt = smoothing_length_scale/(4·Δ_raw)` so the source resolution matches. Less
smoothing gives stronger drag.

Steps, in order:

 1. `calc_hpoz_latlon` gives the raw fourth-moment statistic `h₀`, which is rescaled with
    `γ` and `h_frac` into `hmax` and `hmin = h_frac·hmax`.
 2. `calc_velocity_potential` gives `χ`, smoothed at the same scale for numerical
    stability.
 3. `calc_orographic_tensor` gives `t11`, `t21`, `t12`, `t22` from `χ` and the *unsmoothed*
    elevation, so the tensor gradients keep the full terrain detail.
 4. The six fields are written to a temporary NetCDF file and regridded onto the model grid
    by `regrid_OGW_info`, which reuses the GPU-capable `SpaceVaryingInput` path; the
    temporary file is then removed.

# Arguments

  - `Y`: Prognostic state, used only for its spaces.
  - `elev_data`: Path to a NetCDF file with `lon`, `lat`, and elevation `z`.
  - `earth_radius`: Sphere radius [m].
  - `γ`: Mountain height-width exponent [-].
  - `h_frac`: Ratio of the minimum to the maximum obstacle height [-].

# Keyword Arguments

  - `α_smoothing = 0.15`: Smoothing scale as a fraction of the target grid spacing [-].

See the *Orographic Gravity Waves* page for the pipeline diagram.
"""
function compute_OGW_info(
    Y,
    elev_data,
    earth_radius,
    γ,
    h_frac;
    α_smoothing = 0.15,  # tunable: smoothing scale as fraction of grid resolution
)
    # obtain lat, lon, elevation from the elev_data
    FT = Spaces.undertype(Spaces.axes(Y.c))

    # Extract GCM grid resolution for grid-aware smoothing
    hspace = Spaces.horizontal_space(Spaces.axes(Y.c))
    Δh_GCM = FT(Spaces.node_horizontal_length_scale(hspace))

    # Auto-compute smoothing parameters based on GCM grid resolution
    # The smoothing scale should be a fraction of the GCM grid scale
    smoothing_length_scale = α_smoothing * Δh_GCM

    # Determine skip_pt based on desired resolution relative to grid
    # Raw data is ~10km (21600×10800), we want to downsample appropriately
    # For a 400km grid with α=0.1, smoothing ~40km, so skip_pt ~4-6
    raw_data_resolution = FT(2 * π * earth_radius / 21600)  # ~10 km
    skip_pt = max(1, Int(round(smoothing_length_scale / (4 * raw_data_resolution))))

    @info "Grid-aware OGWD preprocessing" Δh_GCM smoothing_length_scale skip_pt α_smoothing

    # Apply smoothing for hmax and chi, but not for elevation used in drag tensors
    smoothing_length_scale_chi = smoothing_length_scale  # Smooth chi for numerical stability
    smoothing_length_scale_elev = nothing  # No smoothing for tensor gradients (keep raw elevation)
    # downsample to elev dims (3600×1800)
    nt = NCDataset(elev_data, "r") do ds
        lon = FT.(Array(ds["lon"]))[1:skip_pt:end]
        lat = FT.(Array(ds["lat"]))[1:skip_pt:end]
        elev = FT.(Array(ds["z"]))[1:skip_pt:end, 1:skip_pt:end]
        (; lon, lat, elev)
    end
    (; lon, lat, elev) = nt
    FT = eltype(elev)

    # compute hmax and hmin (with grid-aware smoothing)
    # The calc_hpoz_latlon function computes hmn (local mean) at smoothing_length_scale
    # and subtracts it when computing hmax, effectively filtering out features > smoothing_length_scale
    # less smoothing -> stronger drag
    @info "skip_pt = $skip_pt"
    @info "smoothing_length_scale for calc_hpoz_latlon = $smoothing_length_scale"
    hpoz = calc_hpoz_latlon(
        elev,
        lon,
        lat,
        earth_radius;
        smoothing_length_scale = smoothing_length_scale,
    )
    hpoz = @. max(FT(0), hpoz)^(FT(2) - γ)
    hmax = @. (
        abs(hpoz) * (γ + FT(2)) / (FT(2) * γ) * (FT(1) - h_frac^(FT(2) * γ)) /
        (FT(1) - h_frac^(γ + FT(2)))
    )^(FT(1) / (FT(2) - γ))
    hmin = hmax .* h_frac

    # compute χ (no smoothing if chi_length_scale is nothing)
    # less smoothing -> stronger drag
    chi_length_scale = smoothing_length_scale_chi
    # If no length scale, use n_smoothing_cells = 0 (no smoothing)
    chi_n_cells = chi_length_scale !== nothing ? nothing : 0.0
    @info "chi smoothing: length_scale=$chi_length_scale, n_cells=$chi_n_cells"
    χ = calc_velocity_potential(
        elev,
        lon,
        lat,
        earth_radius;
        smoothing_length_scale = chi_length_scale,
        n_smoothing_cells = chi_n_cells,
    )

    # Optionally smooth elevation for tensor gradient computation
    # (matches effective resolution of coarser Fortran elevation data)
    elev_for_tensor = if smoothing_length_scale_elev !== nothing
        @info "Smoothing elevation for tensor gradients (scale=$(smoothing_length_scale_elev)m)"
        smooth_field_latlon(
            copy(elev),
            lon,
            lat,
            earth_radius;
            smoothing_length_scale = smoothing_length_scale_elev,
        )
    else
        elev
    end

    # compute orographic tensor (t11, t21, t12, t22)
    t11, t21, t12, t22 =
        calc_orographic_tensor(elev_for_tensor, χ, lon, lat, earth_radius)

    # create ClimaCore.Fields
    topo_cg = fill(
        (;
            t11 = FT(0),
            t12 = FT(0),
            t21 = FT(0),
            t22 = FT(0),
            hmin = FT(0),
            hmax = FT(0),
        ),
        axes(Fields.level(Y.c.ρ, 1)),
    )

    # Save the computed lat-lon data to a temporary NetCDF file
    # This allows us to use the GPU-compatible SpaceVaryingInput infrastructure
    # Using the pattern from remap_helpers.jl for consistency
    temp_nc_file = tempname() * ".nc"
    nc = NCDataset(temp_nc_file, "c")

    # Define dimensions
    defDim(nc, "lon", length(lon))
    defDim(nc, "lat", length(lat))

    # Define coordinate variables (required for SpaceVaryingInput)
    nc_lon = defVar(nc, "lon", FT, ("lon",))
    nc_lat = defVar(nc, "lat", FT, ("lat",))

    # Define data variables with chunking and compression for large arrays
    # This prevents HDF5 errors when writing full-resolution data (skip_pt=1)
    chunk_size = (min(360, length(lon)), min(180, length(lat)))
    nc_hmax =
        defVar(nc, "hmax", FT, ("lon", "lat"); chunksizes = chunk_size, deflatelevel = 4)
    nc_hmin =
        defVar(nc, "hmin", FT, ("lon", "lat"); chunksizes = chunk_size, deflatelevel = 4)
    nc_t11 =
        defVar(nc, "t11", FT, ("lon", "lat"); chunksizes = chunk_size, deflatelevel = 4)
    nc_t12 =
        defVar(nc, "t12", FT, ("lon", "lat"); chunksizes = chunk_size, deflatelevel = 4)
    nc_t21 =
        defVar(nc, "t21", FT, ("lon", "lat"); chunksizes = chunk_size, deflatelevel = 4)
    nc_t22 =
        defVar(nc, "t22", FT, ("lon", "lat"); chunksizes = chunk_size, deflatelevel = 4)

    # Write coordinate data
    nc_lon[:] = lon
    nc_lat[:] = lat

    # Write field data
    nc_hmax[:, :] = hmax
    nc_hmin[:, :] = hmin
    nc_t11[:, :] = t11
    nc_t12[:, :] = t12
    nc_t21[:, :] = t21
    nc_t22[:, :] = t22

    close(nc)

    # Use the GPU-compatible regrid_OGW_info function to remap the data
    topo_cg = regrid_OGW_info(Y, temp_nc_file)

    # Clean up temporary file
    rm(temp_nc_file; force = true)

    return topo_cg
end

"""
    regrid_OGW_info(Y, orographic_info_rll)

Regrid orographic drag input from a lat-lon file onto the model's surface space.

Returns `(; t11, t12, t21, t22, hmin, hmax)` as `Field`s on the surface cell-center space
of `Y`. Called from `get_topo_info` for a GFDL restart, and from `compute_OGW_info` for
freshly preprocessed data.

The six variables are read by `get_topo_ll`, rewritten to a temporary two-dimensional
NetCDF file, and interpolated with `SpaceVaryingInput`, which is GPU-capable. The
round trip through a temporary file exists because GFDL restart files store the fields as
three-dimensional arrays with a singleton dimension, which the underlying interpolation
package rejects. Longitude is extrapolated periodically and latitude flat. The temporary
file is removed before returning.
"""
function regrid_OGW_info(Y, orographic_info_rll)
    FT = Spaces.undertype(axes(Y.c))

    # Read data from NetCDF - this handles both 2D and 3D arrays (slicing if needed)
    lon, lat, topo_ll = get_topo_ll(orographic_info_rll)

    # Create a temporary 2D NetCDF file for SpaceVaryingInput
    # This is necessary because GFDL restart files have 3D arrays with singleton dimensions
    # which are not supported by the Interpolations package used by SpaceVaryingInput
    temp_nc_file = tempname() * ".nc"
    nc = NCDataset(temp_nc_file, "c")

    # Define dimensions
    defDim(nc, "lon", length(lon))
    defDim(nc, "lat", length(lat))

    # Define coordinate variables (required for SpaceVaryingInput)
    nc_lon = defVar(nc, "lon", FT, ("lon",))
    nc_lat = defVar(nc, "lat", FT, ("lat",))

    # Define data variables with chunking and compression for large arrays
    # This prevents HDF5 errors when writing full-resolution data (skip_pt=1)
    chunk_size = (min(360, length(lon)), min(180, length(lat)))
    nc_hmax =
        defVar(nc, "hmax", FT, ("lon", "lat"); chunksizes = chunk_size, deflatelevel = 4)
    nc_hmin =
        defVar(nc, "hmin", FT, ("lon", "lat"); chunksizes = chunk_size, deflatelevel = 4)
    nc_t11 =
        defVar(nc, "t11", FT, ("lon", "lat"); chunksizes = chunk_size, deflatelevel = 4)
    nc_t12 =
        defVar(nc, "t12", FT, ("lon", "lat"); chunksizes = chunk_size, deflatelevel = 4)
    nc_t21 =
        defVar(nc, "t21", FT, ("lon", "lat"); chunksizes = chunk_size, deflatelevel = 4)
    nc_t22 =
        defVar(nc, "t22", FT, ("lon", "lat"); chunksizes = chunk_size, deflatelevel = 4)

    # Write coordinate data
    nc_lon[:] = lon
    nc_lat[:] = lat

    # Write field data (already sliced to 2D by get_topo_ll)
    nc_hmax[:, :] = topo_ll.hmax
    nc_hmin[:, :] = topo_ll.hmin
    nc_t11[:, :] = topo_ll.t11
    nc_t12[:, :] = topo_ll.t12
    nc_t21[:, :] = topo_ll.t21
    nc_t22[:, :] = topo_ll.t22

    close(nc)

    topo_cg = fill(
        (;
            t11 = FT(0),
            t12 = FT(0),
            t21 = FT(0),
            t22 = FT(0),
            hmin = FT(0),
            hmax = FT(0),
        ),
        axes(Fields.level(Y.c.ρ, 1)),
    )

    # Get the horizontal space (2D surface)
    hspace = axes(Fields.level(Y.c.ρ, 1))

    # Set up regridder with periodic longitude and flat latitude boundary conditions
    extrapolation_bc = (Interpolations.Periodic(), Interpolations.Flat())
    regridder_kwargs = (; extrapolation_bc)

    # Use SpaceVaryingInput to remap each variable from the temporary NetCDF file
    # This is GPU-compatible and handles the interpolation efficiently
    hmax_field = SpaceVaryingInput(
        temp_nc_file,
        "hmax",
        hspace;
        regridder_kwargs,
    )
    hmin_field = SpaceVaryingInput(
        temp_nc_file,
        "hmin",
        hspace;
        regridder_kwargs,
    )
    t11_field = SpaceVaryingInput(
        temp_nc_file,
        "t11",
        hspace;
        regridder_kwargs,
    )
    t12_field = SpaceVaryingInput(
        temp_nc_file,
        "t12",
        hspace;
        regridder_kwargs,
    )
    t21_field = SpaceVaryingInput(
        temp_nc_file,
        "t21",
        hspace;
        regridder_kwargs,
    )
    t22_field = SpaceVaryingInput(
        temp_nc_file,
        "t22",
        hspace;
        regridder_kwargs,
    )

    # Copy the remapped fields into the preallocated output. The broadcast
    # materializes the SpaceVaryingInput results into concrete Fields, which is
    # what can be serialized to HDF5.
    @. topo_cg.hmax = hmax_field
    @. topo_cg.hmin = hmin_field
    @. topo_cg.t11 = t11_field
    @. topo_cg.t12 = t12_field
    @. topo_cg.t21 = t21_field
    @. topo_cg.t22 = t22_field

    # Clean up temporary file
    rm(temp_nc_file; force = true)

    return topo_cg
end

"""
    get_topo_ll(orographic_info_rll) -> (lon, lat, topo_ll)

Read orographic drag input from a lat-lon NetCDF file.

Returns the coordinate vectors and a `NamedTuple` `(; hmax, hmin, t11, t12, t21, t22)` of
two-dimensional arrays. The variables are sliced at the first index of their trailing
dimension, which drops the singleton dimension GFDL restart files carry.

Called from `regrid_OGW_info`.
"""
function get_topo_ll(orographic_info_rll)
    nt = NCDataset(orographic_info_rll, "r") do ds
        lon = Array(ds["lon"])
        lat = Array(ds["lat"])
        hmax = ds["hmax"][:, :, 1]
        hmin = ds["hmin"][:, :, 1]
        t11 = ds["t11"][:, :, 1]
        t12 = ds["t12"][:, :, 1]
        t21 = ds["t21"][:, :, 1]
        t22 = ds["t22"][:, :, 1]
        (; lon, lat, hmax, hmin, t11, t12, t21, t22)
    end
    (; lon, lat, hmax, hmin, t11, t12, t21, t22) = nt

    return (lon, lat, (; hmax, hmin, t11, t12, t21, t22))
end


"""
    move_topo_info_to_gpu(topo_info, ᶜtarget_space)

Move the six orographic drag fields to the GPU and reattach them to a target space.

Returns a `NamedTuple` of `Field`s on `ᶜtarget_space` backed by CUDA arrays. Used when the
drag was preprocessed or read on the CPU but the run is on the GPU.
"""
function move_topo_info_to_gpu(topo_info, ᶜtarget_space)
    t11 = ClimaCore.to_device(ClimaComms.CUDADevice(), topo_info.t11)
    t12 = ClimaCore.to_device(ClimaComms.CUDADevice(), topo_info.t12)
    t21 = ClimaCore.to_device(ClimaComms.CUDADevice(), topo_info.t21)
    t22 = ClimaCore.to_device(ClimaComms.CUDADevice(), topo_info.t22)
    hmin = ClimaCore.to_device(ClimaComms.CUDADevice(), topo_info.hmin)
    hmax = ClimaCore.to_device(ClimaComms.CUDADevice(), topo_info.hmax)

    return set_topo_info_target_space(
        (; t11, t12, t21, t22, hmin, hmax),
        ᶜtarget_space,
    )
end

"""
    set_topo_info_target_space(topo_info, ᶜtarget_space)

Reattach the six orographic drag fields to a different, layout-compatible space.

Returns a `NamedTuple` of `Field`s on `ᶜtarget_space` sharing the input's data layout. This
is how a drag tensor loaded from an artifact, whose space object is not the one the current
run built, is made usable: only the space is swapped, the values are unchanged.

Called from `compute_ogw_drag` and `move_topo_info_to_gpu`.
"""
function set_topo_info_target_space(topo_info, ᶜtarget_space)
    (; t11, t12, t21, t22, hmin, hmax) = topo_info
    FT = eltype(t11)

    val = FT(1.0)
    t11 = Fields.Field(Fields.field_values(t11), ᶜtarget_space) .* val
    t12 = Fields.Field(Fields.field_values(t12), ᶜtarget_space) .* val
    t21 = Fields.Field(Fields.field_values(t21), ᶜtarget_space) .* val
    t22 = Fields.Field(Fields.field_values(t22), ᶜtarget_space) .* val
    hmin = Fields.Field(Fields.field_values(hmin), ᶜtarget_space) .* val
    hmax = Fields.Field(Fields.field_values(hmax), ᶜtarget_space) .* val

    topo_info = (;
        t11 = t11,
        t12 = t12,
        t21 = t21,
        t22 = t22,
        hmin = hmin,
        hmax = hmax,
    )

    return topo_info
end

"""
    generate_drag_filename(; topography, topo_smoothing, topography_damping_factor, h_elem)

Build the filename and metadata that identify a preprocessed-drag output.

Returns `(; output_filename, topography, topo_smoothing, topo_damping_factor, h_elem)`,
where `output_filename` is the extensionless
`computed_drag_<topography>_<topo_smoothing>_<damping factor>_<h_elem>`. Every field of the
topography configuration that changes the drag appears in the name, so two configurations
cannot collide.

Shared by `write_computed_drag!`, which writes the file and its attributes, and
`load_preprocessed_topography`, which reads it back.

# Keyword Arguments

  - `topography`: Topography name.
  - `topo_smoothing`: Whether topography smoothing was applied.
  - `topography_damping_factor`: Damping factor applied to the topography.
  - `h_elem`: Number of horizontal elements per panel direction [-].
"""
function generate_drag_filename(;
    topography,
    topo_smoothing,
    topography_damping_factor,
    h_elem,
)
    output_filename = "computed_drag_$(topography)_$(topo_smoothing)_$(topography_damping_factor)_$(h_elem)"
    return (;
        output_filename,
        topography,
        topo_smoothing,
        topo_damping_factor = topography_damping_factor,
        h_elem,
    )
end


"""
    load_preprocessed_topography(; topography, topo_smoothing,
                                 topography_damping_factor, h_elem)
    load_preprocessed_topography(filename::String, comms_ctx = ClimaComms.SingletonCommsContext())

Read a preprocessed orographic drag field from an HDF5 file in the package root.
The optional `comms_ctx` must be the run's MPI context on distributed runs so the
field is read as correctly.

Returns the `computed_drag` field written by `write_computed_drag!`. The keyword method
derives the filename from the topography configuration via `generate_drag_filename`; the
`String` method takes the extensionless filename directly. Both read on a single CPU
context, so a GPU run has to move the result with `move_topo_info_to_gpu`.
"""
function load_preprocessed_topography(;
    topography,
    topo_smoothing,
    topography_damping_factor,
    h_elem,
)
    (; output_filename) = generate_drag_filename(;
        topography, topo_smoothing, topography_damping_factor, h_elem,
    )
    @info "loading topography drag vector: $(output_filename)"

    reader = InputOutput.HDF5Reader(
        joinpath(@__DIR__, "../../../$(output_filename).hdf5"),
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
    )
    computed_drag = InputOutput.read_field(reader, "computed_drag")

    Base.close(reader)
    return computed_drag
end

function load_preprocessed_topography(
    filename::String,
    comms_ctx = ClimaComms.SingletonCommsContext(),
)
    @info "loading topography drag vector: $(filename)"
    reader = InputOutput.HDF5Reader(
        joinpath(@__DIR__, "../../../$(filename).hdf5"),
        comms_ctx,
    )
    computed_drag = InputOutput.read_field(reader, "computed_drag")

    Base.close(reader)
    return computed_drag
end
