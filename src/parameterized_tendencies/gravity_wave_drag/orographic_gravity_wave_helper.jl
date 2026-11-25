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

The velocity-potential gradient, and so the whole tensor, is zero poleward of 88°, where
the latitude circles are too short for the finite differences. `compute_OGW_info` applies
the same cut to the model grid points after regridding.

Called from `compute_OGW_latlon` in the offline preprocessing pipeline.

# Arguments

  - `elev`: Surface elevation on the lon-lat grid [m]; clipped in place.
  - `χ`: Velocity potential from `calc_velocity_potential`.
  - `lon`, `lat`: Coordinate vectors [degrees].
  - `earth_radius`: Sphere radius [m].

Described in [garner2005](@cite) Eq. 6; see the *Orographic Gravity Waves* page.
"""
function calc_orographic_tensor(elev, χ, lon, lat, earth_radius)
    @info "Computing T tensor..."

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
    polar_row = abs.(lat') .> 88
    @. dχdx = ifelse(polar_row, 0, dχdx)
    @. dχdy = ifelse(polar_row, 0, dχdy)

    t11 = dχdx .* dhdx
    t21 = dχdx .* dhdy
    t12 = dχdy .* dhdx
    t22 = dχdy .* dhdy

    return (t11, t21, t12, t22)
end

"""
    calc_∇A(A, lon, lat, earth_radius) -> (dAdx, dAdy)

Compute the horizontal gradient of a scalar on the spherical surface.

Returns the zonal and meridional derivatives per unit distance, each the same shape as `A`,
by centered differences. Longitude wraps around, so the first and last columns are
neighbors; the first and last latitude rows use one-sided differences.

The zonal step is one column, `R·Δλ·cos(lat)`, at every latitude, as in Garner.
`cos(lat)` is floored at `sin(Δlat/2)` so rows at or next to a pole get a finite slope. The
floor makes the east-west slope too small in the first and last rows of the ETOPO grid
(0.008° from the poles), but `calc_orographic_tensor` zeroes the tensor poleward of 88°, so
this error does not reach the drag.

Called from `calc_orographic_tensor` in the offline preprocessing pipeline.

# Arguments

  - `A`: Scalar field on the lon-lat grid, indexed `[lon, lat]`.
  - `lon`, `lat`: Uniformly spaced coordinate vectors [degrees]; `lon` covers the whole
    circle.
  - `earth_radius`: Sphere radius [m].
"""
function calc_∇A(A, lon, lat, earth_radius)
    FT = eltype(A)
    dlat = lat[2] - lat[1]
    dlon = lon[2] - lon[1]
    # Zonal grid spacing [m] of each latitude row, `R·Δλ·cos(lat)`. The floor
    # `sin(Δlat/2)` keeps rows on or next to a pole finite; those rows are poleward of
    # 88°, where `calc_orographic_tensor` zeroes the tensor.
    cos_floor = sind(abs(dlat) / 2)
    Δx_row = deg2rad(dlon) * earth_radius .* max.(cosd.(lat), cos_floor)
    # Centered difference with periodic longitude: `circshift(A, (-1, 0))[i, :]` is
    # `A[i + 1, :]`, wrapping from the last column to the first.
    dAdx = (circshift(A, (-1, 0)) .- circshift(A, (1, 0))) ./ (FT(2) .* Δx_row')
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

Called from `compute_OGW_latlon` in the offline preprocessing pipeline.

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

    # Main convolution loop: latitude (outer) then longitude (inner). Rows are
    # independent and run on separate threads.
    Threads.@threads for j in 1:nlat
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
separation. `field` is not modified. Unlike `calc_velocity_potential` and
`calc_hpoz_latlon`, the longitude window is clipped rather than wrapped.

Available for optionally pre-smoothing the elevation before the tensor gradients are taken;
`compute_OGW_latlon` does not use it and passes the raw elevation instead.

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
`elev` in place, clipping ocean bathymetry to zero. Longitude is periodic: `lon` is taken
to cover the whole circle in equal steps, and the window wraps across its ends.

This is only the raw statistic; the caller `compute_OGW_latlon` rescales it with the
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
    # The east-west half-window is capped at nlon/8 columns, the cap of Garner's
    # velocity-potential window (transform.f90, ngridx). His get_hmax.f90 caps this window
    # at lx/16 instead. The cap binds only at high latitudes, where a fixed distance spans
    # many columns: with lx/16 from roughly 77° at h_elem = 4 and 87° at h_elem = 16.
    ilon_range =
        min.(
            Int.(round.(scale ./ (deg2rad(dlon) .* cosd.(lat)))),
            Int(round(length(lon) / 8)),
        )

    hmax = zeros(FT, size(elev))

    # Precompute trigonometric values for arc distance calculation
    cosphi = cosd.(lat)

    # Each row's window spans ±ilon_range[j] columns and ±ilat_range[j] rows (fewer next
    # to a pole). Longitude wraps around (mod1), as in `calc_velocity_potential`; the
    # nlon/8 cap keeps the window shorter than the circle, so no column is counted twice.
    # The weights depend only on the offsets and the row, so each row builds them once.
    # Rows run on separate threads.
    nlon = length(lon)
    Threads.@threads for j in eachindex(lat)
        ioffsets = (-ilon_range[j]):ilon_range[j]
        jrange = max(j - ilat_range[j], 1):min(j + ilat_range[j], length(lat))
        # Lorentzian weight w = 1/(1 + arc²/scale²), arc² in radians².
        w = [
            FT(1) / (
                FT(1) +
                ((deg2rad(ioffset * dlon) * cosphi[j])^2 + deg2rad(lat[j1] - lat[j])^2) /
                (scale[j]^2)
            ) for ioffset in ioffsets, j1 in jrange
        ]
        wt = sum(w)
        for i in eachindex(lon)
            # hpos = (weighted mean of (h − mean h)⁴)^(1/4), as in Garner's get_hmax.f90.
            # Pass 1: weighted mean elevation
            hmn = FT(0)
            for (b, j1) in enumerate(jrange), (a, ioffset) in enumerate(ioffsets)
                hmn += w[a, b] * elev[mod1(i + ioffset, nlon), j1]
            end
            hmn = hmn / wt
            # Pass 2: weighted 4th moment about it
            var = FT(0)
            for (b, j1) in enumerate(jrange), (a, ioffset) in enumerate(ioffsets)
                var += w[a, b] * (elev[mod1(i + ioffset, nlon), j1] - hmn)^4
            end
            hmax[i, j] = (var / wt)^FT(0.25)
        end
    end

    return hmax
end

"""
    OGW_TOPOGRAPHY_SAMPLE_SPACING

Target spacing [m] at which `compute_OGW_info` samples the raw elevation: about 7.5 km,
every 4th point of the 1-arcminute (1.85 km) ETOPO grid. `ogw_topography_stride` samples
finer when the smoothing window is shorter than 30 km.
"""
const OGW_TOPOGRAPHY_SAMPLE_SPACING = 7.5e3

"""
    ogw_topography_stride(smoothing_length_scale, earth_radius; nlon_raw = 21600,
                          target_spacing = OGW_TOPOGRAPHY_SAMPLE_SPACING)

Return the stride (in raw grid points) at which `compute_OGW_info` subsamples the raw
elevation.

The drag tensor is built from local slopes, and short, steep features carry most of them,
so the sampling must stay fine whatever the smoothing window. The stride aims at
`target_spacing` (7.5 km), or at a quarter of the smoothing scale when that is smaller, so
a narrow window still holds a few points across. It is then rounded to the nearest divisor
of `nlon_raw`, so the subsampled longitudes close the circle in equal steps, which the
periodic gradients and windows assume. Garner's own preprocessing used the full
1-arcminute grid (`topo_drag.F90`, `resolution = 60` points per degree); 7.5 km keeps
the cost of the windowed sums affordable at coarse model resolutions, where the window
is several hundred kilometres wide.

# Arguments

  - `smoothing_length_scale`: Smoothing scale `α·Δx` [m].
  - `earth_radius`: Sphere radius [m].
"""
function ogw_topography_stride(
    smoothing_length_scale,
    earth_radius;
    nlon_raw = 21600,
    target_spacing = OGW_TOPOGRAPHY_SAMPLE_SPACING,
)
    raw_spacing = 2 * π * earth_radius / nlon_raw
    spacing = min(target_spacing, smoothing_length_scale / 4)
    stride_target = max(1, spacing / raw_spacing)
    divisors = filter(d -> nlon_raw % d == 0, 1:nlon_raw)
    return divisors[argmin(abs.(divisors .- stride_target))]
end

"""
    latlon_block_size(Δh, lat, nlon, nlat, earth_radius) -> n

Return the side `n` (in grid points) of the square blocks that `block_average_latlon` uses to
average a lat-lon field over about one model grid spacing, so each model grid point
draws on about one block. (A spectral element holds 4 × 4 grid points, so it spans about
3 × 3 blocks.)

`n` is the largest block whose north-south side `n·R·Δφ` is no larger than the model grid
spacing `Δh` and that divides both `nlon` and `nlat`, so the blocks tile the grid and
close the longitude circle. It is at least 1. The east-west side `n·R·Δλ·cos φ` shrinks
poleward, so away from the tropics a block is narrower east-west than the grid spacing.

For example, with the default α = 1.0, `compute_OGW_info` uses a 7.4 km grid
(5400 × 2700 points), and `n` is 25 (185 km) at `h_elem = 16`, 12 at 30 and 6 at 60.
A smaller α gives a finer grid and a larger `n`.

# Arguments

  - `Δh`: Model grid spacing [m], from `Spaces.node_horizontal_length_scale`.
  - `lat`: Latitude vector of the lat-lon grid [degrees], equally spaced.
  - `nlon`, `nlat`: Grid size.
  - `earth_radius`: Sphere radius [m].
"""
function latlon_block_size(Δh, lat, nlon, nlat, earth_radius)
    Δy = earth_radius * deg2rad(lat[2] - lat[1])
    n_target = max(1, floor(Int, Δh / Δy))
    return maximum(m -> (nlon % m == 0 && nlat % m == 0) ? m : 1, 1:n_target)
end

"""
    compute_OGW_latlon(elev, lon, lat, earth_radius, γ;
                       smoothing_length_scale, block_size = 1)

Compute the orographic drag input on the lat-lon grid of `elev` and average it over blocks
of `block_size × block_size` points.

Returns `(; lon, lat, hpos, t11, t12, t21, t22)` on the block-center grid, all `Float64`.
Inputs are copied to `Float64` first (the windowed sums and finite differences lose
accuracy in `Float32`), so `elev` is not modified. `compute_OGW_info` calls this
function, then `regrid_OGW_latlon` converts `hpos` into `hmax` and `hmin` on the model grid
points.

Steps, in order:

 1. `calc_hpoz_latlon` gives the fourth-moment statistic `h₀`; `hpos = max(0, h₀)^(2 − γ)`.
 2. `calc_velocity_potential` gives `χ`, smoothed at `smoothing_length_scale`.
 3. `calc_orographic_tensor` gives `t11`, `t21`, `t12`, `t22` from `χ` and the unsmoothed
    elevation (clipped at sea level), so the gradients keep the full terrain detail.
 4. `block_average_latlon` takes the area-weighted mean of `hpos` and of the four tensor
    entries over each block. A single point is a poor stand-in for the area around a grid
    point: the tensor `−∇χ (∇h)ᵀ` has rank one, so it is never negative definite and is
    zero where the elevation is flat. That would switch the drag off for some wind
    directions, and over the sea next to a coast for all of them. The block mean is
    negative definite on 99.9% of the land area at `h_elem = 16`.

With `block_size = 1` there is no averaging and the fields are the point values.

# Arguments

  - `elev`: Surface elevation on the lat-lon grid [m], indexed `[lon, lat]`.
  - `lon`, `lat`: Coordinate vectors [degrees].
  - `earth_radius`: Sphere radius [m].
  - `γ`: Mountain height-width exponent [-].

# Keyword Arguments

  - `smoothing_length_scale`: Window scale of the `hpos` and `χ` sums [m].
  - `block_size = 1`: Block side in grid points; must divide both grid dimensions
    (`latlon_block_size`).
"""
function compute_OGW_latlon(
    elev,
    lon,
    lat,
    earth_radius,
    γ;
    smoothing_length_scale,
    block_size = 1,
)
    # Float64 copies (see the docstring); `Array{Float64}` always copies, so the
    # sea-level clipping below never reaches the caller.
    elev = Array{Float64}(elev)
    lon, lat = Array{Float64}(lon), Array{Float64}(lat)
    earth_radius, γ = Float64(earth_radius), Float64(γ)
    smoothing_length_scale = Float64(smoothing_length_scale)
    h₀ = calc_hpoz_latlon(elev, lon, lat, earth_radius; smoothing_length_scale)
    hpos = @. max(0.0, h₀)^(2.0 - γ)
    χ = calc_velocity_potential(
        elev,
        lon,
        lat,
        earth_radius;
        smoothing_length_scale,
        n_smoothing_cells = nothing,
    )
    t11, t21, t12, t22 = calc_orographic_tensor(elev, χ, lon, lat, earth_radius)

    if block_size > 1
        lon_b, lat_b, hpos = block_average_latlon(hpos, lon, lat, block_size)
        t11, t21, t12, t22 = map((t11, t21, t12, t22)) do t
            block_average_latlon(t, lon, lat, block_size)[3]
        end
    else
        lon_b, lat_b = lon, lat
    end

    return (; lon = lon_b, lat = lat_b, hpos, t11, t12, t21, t22)
end

"""
    ogw_hmax_from_hpos(hpos, γ, h_frac)

Return the maximum obstacle height `hmax` [m] for `hpos = max(0, h₀)^(2 − γ)`, by
inverting the definition of `hpos`:
`hmax = (hpos·(γ + 2)/(2γ)·(1 − h_frac^(2γ))/(1 − h_frac^(γ + 2)))^(1/(2 − γ))`.
The minimum height is `hmin = h_frac·hmax`. A negative `hpos` (rounding in an
interpolation) gives zero.

For `0 < γ < 1` the conversion is concave in `hpos`, so averaging or interpolating `hpos`
and then converting gives a larger `hmax` than the reverse order. It is applied after the
regridding, as in GFDL.
"""
function ogw_hmax_from_hpos(hpos, γ, h_frac)
    FT = typeof(hpos)
    return (
        max(hpos, zero(FT)) * (γ + FT(2)) / (FT(2) * γ) *
        (FT(1) - h_frac^(FT(2) * γ)) / (FT(1) - h_frac^(γ + FT(2)))
    )^(FT(1) / (FT(2) - γ))
end

"""
    compute_OGW_info(Y, elev_data, earth_radius, γ, h_frac; α_smoothing = 1.0)

Run the offline preprocessing pipeline that turns raw elevation into orographic drag input.

Returns `(; hmax, hmin, t11, t12, t21, t22)` on the surface cell-center space of `Y`. This
is the entry point of the offline path; the result is normally written to an HDF5 artifact
by `write_computed_drag!` and loaded at runtime rather than recomputed.

The smoothing scale is derived from the target grid: `α_smoothing` times the grid
spacing of `Y` (`Spaces.node_horizontal_length_scale`). The raw elevation is subsampled at
about 7.5 km, or finer when the smoothing scale is shorter than 30 km
(`ogw_topography_stride`), so the slopes in the tensor keep the same detail at every model
resolution. The windowed sums run on several threads when Julia has them.

The elevation, its coordinates and every lat-lon step compute in `Float64`, whatever the
model float type. The results become the model float type only on the model grid points: the
regridding in step 2 interpolates in the model float type, and `hmax`, `hmin` and the
tensor are stored in it.

Steps, in order:

 1. `compute_OGW_latlon` computes `hpos` and the tensor on the subsampled lat-lon grid
    and averages them over blocks of about one model grid spacing (`latlon_block_size`),
    so each model grid point gets the mean over about one block, not a single point value.
 2. `regrid_OGW_latlon` interpolates `hpos` and the tensor onto the model grid points,
    converts `hpos` into `hmax` and `hmin` there, and sets the tensor to zero poleward of
    88°.

# Arguments

  - `Y`: Prognostic state, used only for its spaces.
  - `elev_data`: Path to a NetCDF file with `lon`, `lat`, and elevation `z`.
  - `earth_radius`: Sphere radius [m].
  - `γ`: Mountain height-width exponent [-].
  - `h_frac`: Ratio of the minimum to the maximum obstacle height [-].

# Keyword Arguments

  - `α_smoothing = 1.0`: Smoothing scale as a fraction of the target grid spacing [-]; the
    ClimaParams default `ogw_smoothing_scale_fraction`.

See the *Orographic Gravity Waves* page for the pipeline diagram.
"""
function compute_OGW_info(
    Y,
    elev_data,
    earth_radius,
    γ,
    h_frac;
    α_smoothing = 1.0,  # tunable: smoothing scale as fraction of grid resolution
)
    ctx = ClimaComms.context(Y.c)
    earth_radius = Float64(earth_radius)

    # The smoothing scale and the averaging blocks follow the model grid spacing.
    hspace = Spaces.horizontal_space(Spaces.axes(Y.c))
    Δh_GCM = Float64(Spaces.node_horizontal_length_scale(hspace))
    smoothing_length_scale = Float64(α_smoothing) * Δh_GCM

    nlon_raw = 21600
    skip_pt = ogw_topography_stride(smoothing_length_scale, earth_radius; nlon_raw)

    # The lat-lon steps (reading the elevation, velocity potential, tensor, block means)
    # give the same result on every rank, so root runs them once and broadcasts the
    # result. The regridding below (`regrid_OGW_latlon`) must run on all ranks.
    latlon_info = if ClimaComms.iamroot(ctx)
        # Strided read: slice the on-disk variables directly so only the subsampled
        # elements are loaded.
        (; lon, lat, elev) = NCDataset(elev_data, "r") do ds
            length(ds["lon"]) == nlon_raw || error(
                "Expected $nlon_raw longitudes in the elevation data, got $(length(ds["lon"])).",
            )
            lon = Float64.(ds["lon"][1:skip_pt:end])
            lat = Float64.(ds["lat"][1:skip_pt:end])
            elev = Float64.(ds["z"][1:skip_pt:end, 1:skip_pt:end])
            (; lon, lat, elev)
        end
        block_size = latlon_block_size(Δh_GCM, lat, size(elev)..., earth_radius)
        @info(
            "Grid-aware OGWD preprocessing",
            Δh_GCM,
            smoothing_length_scale,
            skip_pt,
            α_smoothing,
            block_size,
        )
        compute_OGW_latlon(
            elev,
            lon,
            lat,
            earth_radius,
            γ;
            smoothing_length_scale,
            block_size,
        )
    else
        nothing
    end

    # Broadcast the root-computed lat-lon fields to all ranks (no-op without MPI).
    latlon_info = ClimaComms.bcast(ctx, latlon_info)

    return regrid_OGW_latlon(
        axes(Fields.level(Y.c.ρ, 1)),
        latlon_info,
        γ,
        h_frac,
    )
end

"""
    regrid_OGW_latlon(hspace, latlon_info, γ, h_frac)

Regrid the lat-lon output of `compute_OGW_latlon` onto the model surface space `hspace`
and turn it into the drag input.

Returns `(; t11, t12, t21, t22, hmin, hmax)` as `Field`s on `hspace`. Steps, in order:

 1. `hpos` and the four tensor entries are written to a temporary NetCDF file in their own
    float type (`Float64` from `compute_OGW_latlon`) and interpolated linearly between
    block centers with the GPU-capable `SpaceVaryingInput` path (longitude periodic,
    latitude flat), which converts them to the float type of `hspace`; the temporary
    file is then removed.
 2. `hmax = ogw_hmax_from_hpos(hpos)` and `hmin = h_frac·hmax` are computed on the model
    grid points from the interpolated `hpos`. GFDL `topo_drag` does the same: it remaps
    `hpos` and converts it to `hmax` afterwards.
 3. After the interpolation, the tensor is set to zero at grid points poleward of 88°. The
    lat-lon tensor is already zero there (`calc_orographic_tensor`), but the blocks next
    to the pole straddle 88°, so the interpolation carries some of their value poleward.

# Arguments

  - `hspace`: Model surface space.
  - `latlon_info`: `(; lon, lat, hpos, t11, t12, t21, t22)` on a lat-lon grid.
  - `γ`: Mountain height-width exponent [-].
  - `h_frac`: Ratio of the minimum to the maximum obstacle height [-].
"""
function regrid_OGW_latlon(hspace, latlon_info, γ, h_frac)
    FT = Spaces.undertype(hspace)
    (; lon, lat) = latlon_info
    names = (:hpos, :t11, :t12, :t21, :t22)

    temp_nc_file = tempname() * ".nc"
    NCDataset(temp_nc_file, "c") do nc
        defDim(nc, "lon", length(lon))
        defDim(nc, "lat", length(lat))
        defVar(nc, "lon", eltype(lon), ("lon",))[:] = lon
        defVar(nc, "lat", eltype(lat), ("lat",))[:] = lat
        # Chunking and compression keep HDF5 happy with full-resolution arrays.
        chunk_size = (min(360, length(lon)), min(180, length(lat)))
        for name in names
            field = getproperty(latlon_info, name)
            defVar(
                nc,
                String(name),
                eltype(field),
                ("lon", "lat");
                chunksizes = chunk_size,
                deflatelevel = 4,
            )[:, :] = field
        end
    end

    # The field order matches `regrid_OGW_info`, so the written drag files have one layout.
    topo_cg = fill(
        (;
            t11 = FT(0),
            t12 = FT(0),
            t21 = FT(0),
            t22 = FT(0),
            hmin = FT(0),
            hmax = FT(0),
        ),
        hspace,
    )
    regridder_kwargs = (;
        extrapolation_bc = (Interpolations.Periodic(), Interpolations.Flat())
    )
    for name in (:t11, :t12, :t21, :t22)
        input = SpaceVaryingInput(temp_nc_file, String(name), hspace; regridder_kwargs)
        t = getproperty(topo_cg, name)
        @. t = input
    end
    hpos = SpaceVaryingInput(temp_nc_file, "hpos", hspace; regridder_kwargs)
    γ_FT, h_frac_FT = FT(γ), FT(h_frac)
    @. topo_cg.hmax = ogw_hmax_from_hpos(hpos, γ_FT, h_frac_FT)
    @. topo_cg.hmin = h_frac_FT * topo_cg.hmax
    rm(temp_nc_file; force = true)

    # The tensor is zero poleward of 88° on the model grid too.
    ᶜlat = Fields.coordinate_field(hspace).lat
    for name in (:t11, :t12, :t21, :t22)
        t = getproperty(topo_cg, name)
        @. t = ifelse(abs(ᶜlat) > 88, zero(t), t)
    end

    return topo_cg
end

"""
    regrid_OGW_info(Y, orographic_info_rll)

Regrid orographic drag input from a lat-lon file onto the model's surface space.

Returns `(; t11, t12, t21, t22, hmin, hmax)` as `Field`s on the surface cell-center space
of `Y`. Called from `get_topo_info` for a GFDL restart, whose file stores `hmax` and
`hmin`, so these are interpolated as they are. Freshly preprocessed data goes through
`regrid_OGW_latlon` instead, which interpolates `hpos`.

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
`computed_drag_<topography>_<topo_smoothing>_<damping factor>_<h_elem>`. The shape
parameters `γ`, `h_frac`, `α_smoothing` and the preprocessing version also change the drag
but are not in the name; they are stored in the file's build record and checked when the
file is read (`check_ogw_drag_stamp`).

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
    OGW_PREPROCESSING_VERSION

Version number of the orographic drag preprocessing, written into every drag file and
checked when a `raw_topo` run loads one.

Increase it by one whenever a change to `compute_OGW_info` (or anything it calls) changes
the drag. Then rebuild the `ogw_computed_drag_h*` artifacts and any local
`computed_drag_*.hdf5` files: loading refuses a file with a different version.
"""
const OGW_PREPROCESSING_VERSION = 1

"""
    write_computed_drag!(
        computed_drag,
        comms_ctx;
        topography,
        topo_smoothing,
        topography_damping_factor,
        h_elem,
        γ,
        h_frac,
        α_smoothing,
        elevation_source,
    )

Write the preprocessed drag `computed_drag` to `<output_filename>.hdf5` in the working
directory, with `output_filename` from `generate_drag_filename`, and return
`output_filename`. These files are what the `ogw_computed_drag_h*` artifacts ship.

Besides the field, the file stores two sets of attributes:

  - the topography configuration: `topography`, `topo_smoothing`, `topo_damping_factor`,
    `h_elem`;
  - the build record, checked on load by `check_ogw_drag_stamp`: the preprocessing
    version, `gamma`, `h_frac`, `alpha_smoothing`, the float type the fields are stored in
    (`field_float_type`) and computed in (`compute_float_type`, always `Float64`), and the
    elevation file name (`elevation_source`).

# Arguments

  - `computed_drag`: The result of `compute_OGW_info`, in the model float type.
  - `comms_ctx`: Communications context for the HDF5 writer.

# Keyword Arguments

  - `topography`, `topo_smoothing`, `topography_damping_factor`, `h_elem`: The topography
    configuration (see `generate_drag_filename`).
  - `γ`, `h_frac`, `α_smoothing`: The shape parameters `compute_OGW_info` was called with.
  - `elevation_source`: Name of the elevation file, e.g.
    `basename(AA.earth_orography_file_path())`.
"""
function write_computed_drag!(
    computed_drag,
    comms_ctx;
    topography,
    topo_smoothing,
    topography_damping_factor,
    h_elem,
    γ,
    h_frac,
    α_smoothing,
    elevation_source,
)
    (; output_filename, topo_damping_factor) = generate_drag_filename(;
        topography, topo_smoothing, topography_damping_factor, h_elem,
    )
    hdfwriter = InputOutput.HDF5Writer("$(output_filename).hdf5", comms_ctx)
    file = hdfwriter.file
    InputOutput.HDF5.write_attribute(file, "topography", topography)
    InputOutput.HDF5.write_attribute(file, "topo_smoothing", topo_smoothing)
    InputOutput.HDF5.write_attribute(file, "topo_damping_factor", topo_damping_factor)
    InputOutput.HDF5.write_attribute(file, "h_elem", h_elem)
    InputOutput.HDF5.write_attribute(
        file,
        "ogw_preprocessing_version",
        OGW_PREPROCESSING_VERSION,
    )
    InputOutput.HDF5.write_attribute(file, "gamma", Float64(γ))
    InputOutput.HDF5.write_attribute(file, "h_frac", Float64(h_frac))
    InputOutput.HDF5.write_attribute(file, "alpha_smoothing", Float64(α_smoothing))
    InputOutput.HDF5.write_attribute(
        file,
        "field_float_type",
        string(eltype(computed_drag.hmax)),
    )
    InputOutput.HDF5.write_attribute(file, "compute_float_type", string(Float64))
    InputOutput.HDF5.write_attribute(file, "elevation_source", string(elevation_source))

    @info "Writing computed drag data to $(output_filename).hdf5"
    InputOutput.write!(hdfwriter, computed_drag, "computed_drag")
    Base.close(hdfwriter)
    return output_filename
end

"""
    check_ogw_drag_stamp(stamp, path; h_elem, γ = nothing, h_frac = nothing,
                         α_smoothing = nothing)

Check the build record of a preprocessed drag file against the run that loads it.

`stamp` is the file's attributes as a `Dict` (see `read_ogw_drag_file`) and `path` the file
name used in the messages. Raises an error when the build record is missing (a file written
before files carried one), when its `ogw_preprocessing_version` differs from
`OGW_PREPROCESSING_VERSION`, or when its `h_elem` differs from the run's. Warns when the
run's `γ`, `h_frac`, or `α_smoothing` differ from the values the file was built with, since
the stored `hmax`, `hmin`, and tensor do not follow the run's values; a parameter passed as
`nothing` is not compared. A stored float type different from the run's is fine and only
logged at the debug level.
"""
function check_ogw_drag_stamp(
    stamp,
    path;
    h_elem,
    γ = nothing,
    h_frac = nothing,
    α_smoothing = nothing,
)
    how_to_fix = "Rebuild the drag file with the current preprocessing (for the shipped \
                  files, the ogw_computed_drag_h$(h_elem) artifact; a local \
                  computed_drag_*.hdf5 file is loaded before the artifact, so delete or \
                  rebuild it), or set \
                  orographic_gravity_wave: \"raw_topo_online\" to compute the drag at \
                  initialization."
    haskey(stamp, "ogw_preprocessing_version") || error(
        "Orographic drag file $(path) has no build record (no preprocessing version): it \
         was written before files carried one, so it may not match the current \
         preprocessing (version $(OGW_PREPROCESSING_VERSION)). $(how_to_fix)",
    )
    version = stamp["ogw_preprocessing_version"]
    version == OGW_PREPROCESSING_VERSION || error(
        "Orographic drag file $(path) was built with preprocessing version $(version), \
         but this code expects version $(OGW_PREPROCESSING_VERSION). $(how_to_fix)",
    )
    stored_h_elem = get(stamp, "h_elem", nothing)
    stored_h_elem == h_elem || error(
        "Orographic drag file $(path) was built for h_elem = $(stored_h_elem), but the \
         run has h_elem = $(h_elem). $(how_to_fix)",
    )

    # (name in messages, attribute name, run value)
    shape_parameters = (
        ("γ", "gamma", γ),
        ("h_frac", "h_frac", h_frac),
        ("α_smoothing", "alpha_smoothing", α_smoothing),
    )
    stale = String[]
    for (name, key, value) in shape_parameters
        isnothing(value) && continue
        stored = stamp[key]
        isapprox(Float64(value), stored; rtol = sqrt(eps(Float32))) ||
            push!(stale, "$(name) = $(value) (file $(stored))")
    end
    isempty(stale) || @warn "orographic_gravity_wave = \"raw_topo\" loads the drag file \
                             $(path), whose hmax, hmin and tensor were built with other \
                             shape parameters, so the drag input will not reflect the \
                             run's $(join(stale, ", ")). Use \"raw_topo_online\" to \
                             recompute the drag input from the current parameters."
    field_float_type = stamp["field_float_type"]
    compute_float_type = stamp["compute_float_type"]
    @debug "Orographic drag file float types" path field_float_type compute_float_type
    return nothing
end

"""
    read_ogw_drag_file(path, comms_ctx; h_elem, γ = nothing, h_frac = nothing,
                       α_smoothing = nothing)

Read the `computed_drag` field from the HDF5 drag file at `path`, after checking its
build record with `check_ogw_drag_stamp` (the keywords are passed on to it).

Used by every `raw_topo` load: `load_preprocessed_topography` and the artifact path of
`compute_ogw_drag`.
"""
function read_ogw_drag_file(
    path,
    comms_ctx;
    h_elem,
    γ = nothing,
    h_frac = nothing,
    α_smoothing = nothing,
)
    reader = InputOutput.HDF5Reader(path, comms_ctx)
    computed_drag = try
        attributes = InputOutput.HDF5.attrs(reader.file)
        stamp = Dict{String, Any}(key => attributes[key] for key in keys(attributes))
        check_ogw_drag_stamp(stamp, path; h_elem, γ, h_frac, α_smoothing)
        InputOutput.read_field(reader, "computed_drag")
    finally
        Base.close(reader)
    end
    return computed_drag
end

"""
    load_preprocessed_topography(; topography, topo_smoothing,
                                 topography_damping_factor, h_elem,
                                 γ = nothing, h_frac = nothing, α_smoothing = nothing)
    load_preprocessed_topography(filename::String,
                                 comms_ctx = ClimaComms.SingletonCommsContext();
                                 h_elem, γ = nothing, h_frac = nothing,
                                 α_smoothing = nothing)

Read a preprocessed orographic drag field from an HDF5 file in the package root.
The optional `comms_ctx` must be the run's MPI context on distributed runs so the
field is read correctly.

Returns the `computed_drag` field written by `write_computed_drag!`, read with
`read_ogw_drag_file`, so the file's build record is checked against `h_elem` and the
shape parameters given (see `check_ogw_drag_stamp`). The keyword method derives the
filename from the topography configuration via `generate_drag_filename` and reads on a
single-threaded CPU context; the `String` method takes the extensionless filename directly
and reads with the supplied `comms_ctx`. A GPU run has to move the result with
`move_topo_info_to_gpu`.
"""
function load_preprocessed_topography(;
    topography,
    topo_smoothing,
    topography_damping_factor,
    h_elem,
    γ = nothing,
    h_frac = nothing,
    α_smoothing = nothing,
)
    (; output_filename) = generate_drag_filename(;
        topography, topo_smoothing, topography_damping_factor, h_elem,
    )
    return load_preprocessed_topography(
        output_filename,
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded());
        h_elem, γ, h_frac, α_smoothing,
    )
end

function load_preprocessed_topography(
    filename::String,
    comms_ctx = ClimaComms.SingletonCommsContext();
    h_elem,
    γ = nothing,
    h_frac = nothing,
    α_smoothing = nothing,
)
    @info "loading topography drag vector: $(filename)"
    return read_ogw_drag_file(
        joinpath(@__DIR__, "../../../$(filename).hdf5"),
        comms_ctx;
        h_elem, γ, h_frac, α_smoothing,
    )
end

"""
    ifs_smoothing_weights(Δx, Δ, δ, J_max)

Return the normalized weights of the IFS orography smoothing operator sampled at spacing `Δx`.

The operator (IFS Documentation CY47R1, Part IV, Eq. 11.4) is a top hat of width `Δ` with
cosine edges of width `δ`:

    h(r) ∝ 1                                     for |r| < Δ/2 − δ,
    h(r) ∝ ½ + ½·cos[π(|r| − Δ/2 + δ)/(2δ)]      for Δ/2 − δ ≤ |r| < Δ/2 + δ,
    h(r) = 0                                     otherwise.

The weights cover the offsets `−J:J`, with `J = min(⌈(Δ/2 + δ)/Δx⌉, J_max)`, and sum to one.
When `Δx` exceeds the operator width only the center weight is nonzero (no smoothing).
Used by `smooth_ifs_latlon`.
"""
function ifs_smoothing_weights(Δx, Δ, δ, J_max)
    FT = typeof(Δx)
    J = min(ceil(Int, (Δ / 2 + δ) / Δx), J_max)
    # A plain loop rather than a closure over FT: a closure capturing FT is not
    # inferred, which makes the returned weights, and every multiply-add over them in
    # smooth_ifs_latlon, dynamically typed.
    w = Vector{FT}(undef, 2J + 1)
    for (n, j) in enumerate((-J):J)
        r = abs(j * Δx)
        w[n] =
            r < Δ / 2 - δ ? FT(1) :
            r < Δ / 2 + δ ? (1 + cos(FT(π) * (r - Δ / 2 + δ) / (2 * δ))) / 2 : FT(0)
    end
    return w ./ sum(w)
end

"""
    smooth_ifs_latlon(elev, lon, lat, earth_radius, Δ, δ)

Smooth a lat-lon field with the IFS operator of width `Δ` and edge width `δ` [m].

Returns a new array shaped like `elev` (indexed `[lon, lat]`). The operator
`ifs_smoothing_weights` is applied separably in physical distance: along longitude with the
spacing `R·Δλ·cos φ` of each latitude row (periodic), then along latitude with the spacing
`R·Δφ` (edge values repeated at the poles). The longitude half-width is capped at a quarter of
the row (`nlon ÷ 8` points each side) near the poles, where a fixed physical width spans a
large part of the latitude circle; those rows cover a negligible area.

The two separable passes give a square footprint, whereas the IFS operator and Beljaars et
al. (2004, Eq. A.1) are radially symmetric in `r`. The constants `I_H` and `k_flt` in
`tofd_drag_coefficient!` were computed for the radially symmetric filter, so the effective
band of this one can differ slightly.
"""
function smooth_ifs_latlon(elev, lon, lat, earth_radius, Δ, δ)
    FT = eltype(elev)
    nlon, nlat = size(elev)
    Δλ = deg2rad(FT(lon[2] - lon[1]))
    Δφ = deg2rad(FT(lat[2] - lat[1]))

    # Longitude pass, row by row, on a periodically padded copy of the row.
    smoothed_lon = similar(elev)
    J_max = nlon ÷ 8
    for j in 1:nlat
        Δx = max(FT(earth_radius) * Δλ * cosd(FT(lat[j])), eps(FT))
        w = ifs_smoothing_weights(Δx, FT(Δ), FT(δ), J_max)
        J = (length(w) - 1) ÷ 2
        row = [elev[mod1(i, nlon), j] for i in (1 - J):(nlon + J)]
        for i in 1:nlon
            acc = zero(FT)
            @inbounds for k in eachindex(w)
                acc += w[k] * row[i + k - 1]
            end
            smoothed_lon[i, j] = acc
        end
    end

    # Latitude pass with a constant kernel.
    w = ifs_smoothing_weights(FT(earth_radius) * Δφ, FT(Δ), FT(δ), nlat)
    J = (length(w) - 1) ÷ 2
    smoothed = similar(elev)
    for j in 1:nlat, i in 1:nlon
        acc = zero(FT)
        @inbounds for k in eachindex(w)
            acc += w[k] * smoothed_lon[i, clamp(j + k - 1 - J, 1, nlat)]
        end
        smoothed[i, j] = acc
    end
    return smoothed
end

"""
    calc_filtered_orography_variance_latlon(elev, lon, lat, earth_radius;
                                            Δ_small = 2000, Δ_large = 20000, δ = 1000)

Return the squared band-pass-filtered orography `(S_small h − S_large h)²` [m²] on the
lat-lon grid of `elev`.

This is step (i) of the IFS definition of the standard deviation of filtered orography
`σ_flt` used by the turbulent orographic form drag (IFS Documentation CY47R1, Part IV,
Section 11.3.3): the orography (ocean bathymetry clipped to zero) is smoothed with
`smooth_ifs_latlon` at `Δ_small = 2 km` and at `Δ_large = 20 km`, both with `δ = 1 km`, and
the difference is squared. The band pass keeps horizontal scales of roughly 3 to 22 km
(half wavelengths). Its integral `I_H` and representative wavenumber `k_flt` are the
constants used in `tofd_drag_coefficient!`. Averaging the result over a model grid box
gives `σ_flt²` there.

The IFS and Beljaars et al. (2004) filter 30 arc-second (about 1 km) data, on which the 2 km
pass removes the grid-scale aliasing tail. On 60 arc-second (about 1.85 km) input such as
ETOPO 2022 the 2 km pass removes little, and the input grid itself sets the small-scale
cutoff of the band.

`elev` is indexed `[lon, lat]` and is not modified. The filtering computes in `Float64`
whatever the float type of `elev`, and the result is `Float64`.
"""
function calc_filtered_orography_variance_latlon(
    elev,
    lon,
    lat,
    earth_radius;
    Δ_small = 2000,
    Δ_large = 20000,
    δ = 1000,
)
    # A sum of many weighted terms and a difference of two nearly equal smoothed fields:
    # Float64 keeps the small band-pass signal.
    land_elev = max.(Float64.(elev), 0.0)
    lon, lat, earth_radius = Float64.(lon), Float64.(lat), Float64(earth_radius)
    smoothed_small = smooth_ifs_latlon(land_elev, lon, lat, earth_radius, Δ_small, δ)
    smoothed_large = smooth_ifs_latlon(land_elev, lon, lat, earth_radius, Δ_large, δ)
    @. smoothed_small = (smoothed_small - smoothed_large)^2
    return smoothed_small
end

"""
    block_average_latlon(field, lon, lat, n) -> (lon_coarse, lat_coarse, field_coarse)

Average a lat-lon field over blocks of `n × n` points, weighting rows by `cos φ` (area).

`field` is indexed `[lon, lat]` and both dimensions must be divisible by `n`. Returns the
block-center coordinates and the block means.
"""
function block_average_latlon(field, lon, lat, n)
    FT = eltype(field)
    nlon, nlat = size(field)
    @assert nlon % n == 0 && nlat % n == 0 "block size $n must divide the grid ($nlon × $nlat)"
    mlon, mlat = nlon ÷ n, nlat ÷ n
    field_coarse = zeros(FT, mlon, mlat)
    for jb in 1:mlat
        rows = ((jb - 1) * n + 1):(jb * n)
        row_weights = [max(cosd(FT(lat[j])), zero(FT)) for j in rows]
        weight_sum = sum(row_weights) * n
        for ib in 1:mlon
            cols = ((ib - 1) * n + 1):(ib * n)
            acc = zero(FT)
            for (jw, j) in enumerate(rows), i in cols
                acc += row_weights[jw] * field[i, j]
            end
            field_coarse[ib, jb] = weight_sum > 0 ? acc / weight_sum : zero(FT)
        end
    end
    lon_coarse = [mean(lon[((ib - 1) * n + 1):(ib * n)]) for ib in 1:mlon]
    lat_coarse = [mean(lat[((jb - 1) * n + 1):(jb * n)]) for jb in 1:mlat]
    return FT.(lon_coarse), FT.(lat_coarse), field_coarse
end

"""
    compute_filtered_orography_variance(Y, elev_data, earth_radius)

Return `σ_flt²` [m²], the variance of the filtered orography used by the turbulent
orographic form drag, as a `Field` on the surface cell-center space of `Y`.

Follows the IFS definition (IFS Documentation CY47R1, Part IV, Section 11.3.3):

 1. `calc_filtered_orography_variance_latlon` computes the squared band-pass-filtered
    orography on the lat-lon grid of the elevation file `elev_data` (NetCDF with `lon`, `lat`
    and elevation `z`).
 2. The result is averaged with `block_average_latlon` over blocks approximately the size of
    the model's horizontal grid spacing (step ii of the IFS definition, averaging over the
    target grid box). Since the drag is linear in `σ_flt²`, the variance rather than its
    square root is averaged.
 3. The block means are regridded onto the model surface with `SpaceVaryingInput`.

Steps 1 and 2 run on the root process, in `Float64` whatever the model float type, and are
broadcast; step 3 runs on every process and gives the model float type.
"""
function compute_filtered_orography_variance(Y, elev_data, earth_radius)
    FT = Spaces.undertype(axes(Y.c))
    ctx = ClimaComms.context(Y.c)
    hspace = axes(Fields.level(Y.c.ρ, 1))
    earth_radius = Float64(earth_radius)
    Δh_GCM = Float64(
        Spaces.node_horizontal_length_scale(Spaces.horizontal_space(axes(Y.c))),
    )

    latlon_variance = if ClimaComms.iamroot(ctx)
        (lon, lat, elev) = NCDataset(elev_data, "r") do ds
            (Float64.(ds["lon"][:]), Float64.(ds["lat"][:]), Float64.(ds["z"][:, :]))
        end
        @info "Computing the filtered-orography variance for TOFD" size(elev) Δh_GCM
        variance = calc_filtered_orography_variance_latlon(
            elev,
            lon,
            lat,
            earth_radius,
        )
        n = latlon_block_size(Δh_GCM, lat, size(elev)..., earth_radius)
        block_average_latlon(variance, lon, lat, n)
    else
        nothing
    end
    (lon, lat, variance) = ClimaComms.bcast(ctx, latlon_variance)

    temp_nc_file = tempname() * ".nc"
    NCDataset(temp_nc_file, "c") do nc
        defDim(nc, "lon", length(lon))
        defDim(nc, "lat", length(lat))
        defVar(nc, "lon", Float64, ("lon",))[:] = lon
        defVar(nc, "lat", Float64, ("lat",))[:] = lat
        defVar(nc, "sigma2_flt", Float64, ("lon", "lat"))[:, :] = variance
    end
    regridder_kwargs =
        (; extrapolation_bc = (Interpolations.Periodic(), Interpolations.Flat()))
    variance_input =
        SpaceVaryingInput(temp_nc_file, "sigma2_flt", hspace; regridder_kwargs)
    σ²_flt = Fields.zeros(FT, hspace)
    @. σ²_flt = max(variance_input, zero(FT))
    rm(temp_nc_file; force = true)
    return σ²_flt
end
