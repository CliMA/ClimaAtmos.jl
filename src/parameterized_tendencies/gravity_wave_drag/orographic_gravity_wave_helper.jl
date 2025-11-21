using NCDatasets
import Interpolations
using Statistics: mean

"""
    calc_orographic_tensor(elev, χ, lon, lat, earth_radius)

    Calculate orographic tensor (T) from
    - elev: surface elevation
    - χ: velocity potential
    - lon: longitude
    - lat: latitude
    - earth_radius: radius of the Earth
"""
function calc_orographic_tensor(elev, χ, lon, lat, earth_radius)
    @info "Computing T tensor..."
    FT = eltype(elev)
    bfscale = FT(1e-2)

    # @Main.infiltrate
    # compute ∇h
    @. elev = max(0, elev)
    dhdx, dhdy = calc_∇A(elev, lon, lat, earth_radius)
    # @Main.infiltrate
    # compute ∇χ
    # dχdx, dχdy = -bfscale .* calc_∇A(χ, lon, lat, earth_radius)
    # TODO: This needs to be double checked with Steve Garner.
    # It looks like bfscale is multiply when he compute the raw T tensor but
    # later divided again when creating the file that contains the actual T tensor
    # being used.
    dχdx, dχdy = .-calc_∇A(χ, lon, lat, earth_radius)
    # @Main.infiltrate
    # for antarctic
    dχdx[:, lat .< FT(-88)] .= FT(0)
    dχdy[:, lat .< FT(-88)] .= FT(0)
    # @Main.infiltrate
    t11 = dχdx .* dhdx
    t21 = dχdx .* dhdy
    t12 = dχdy .* dhdx
    t22 = dχdy .* dhdy

    # Zero out tensor components at polar regions to prevent blow-up from finite differences
    # This matches the polar masking in calc_velocity_potential
    t11[:, abs.(lat) .> FT(89)] .= FT(0)
    t12[:, abs.(lat) .> FT(89)] .= FT(0)
    t21[:, abs.(lat) .> FT(89)] .= FT(0)
    t22[:, abs.(lat) .> FT(89)] .= FT(0)

    return (t11, t21, t12, t22)
end

"""
    calc_∇A(A, lon, lat, earth_radius)

    Calculate the horizontal gradient of scalar A on the Earth surface
    - A: the scalar
    - lon: longitude
    - lat: latitude
    - earth_radius: radius of the Earth
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
            deg2rad(dlon) * earth_radius .*
            reshape(repeat(cosd.(lat), length(lon)), length(lat), :)'
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
                           smoothing_length_scale=nothing,
                           n_smoothing_cells=nothing,
                           min_smoothing_cells=1.0)

    Calculate velocity potential via optimized 2D Hilbert transform.
    This implementation follows the Fortran strategy of pre-computing weights
    once per latitude using offset-based indexing for significant performance gains.

    Arguments:
    - elev: surface elevation [nlon × nlat]
    - lon: longitude array (degrees)
    - lat: latitude array (degrees)
    - earth_radius: radius of the Earth (meters)
    - smoothing_length_scale: smoothing scale in meters (optional)
                             Default: 100e3 (100 km), but enforces min_smoothing_cells
                             Note: Some Fortran codes use 200e3 (200 km)
    - n_smoothing_cells: number of grid cells for smoothing (optional, overrides smoothing_length_scale)
    - min_smoothing_cells: minimum number of grid cells to smooth over (default: 1.0)
                          Prevents physically meaningless smoothing for coarse grids

    Returns:
    - χ: velocity potential [nlon × nlat]

    Note: Either smoothing_length_scale OR n_smoothing_cells should be specified, not both.
          If neither is specified, uses default 100 km with min_smoothing_cells=1.0 constraint.
"""
function calc_velocity_potential(
    elev,
    lon,
    lat,
    earth_radius;
    smoothing_length_scale = nothing,
    n_smoothing_cells = nothing,
    min_smoothing_cells = 1.0,
)
    @info "Computing velocity potential (optimized)..."
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
        # User specified exact number of cells
        scale_distance = n_smoothing_cells .* Δh_grid
    elseif smoothing_length_scale !== nothing
        # User specified physical length scale in meters
        # Compute equivalent number of cells and enforce minimum
        n_cells = smoothing_length_scale ./ Δh_grid
        n_cells_clamped = max.(n_cells, min_smoothing_cells)
        scale_distance = n_cells_clamped .* Δh_grid
    else
        # Default: 100 km scale with minimum cell constraint
        # Note: FT(200e3) was used in some codes sent by Steve Garner
        default_scale = FT(100e3)
        n_cells = default_scale ./ Δh_grid
        n_cells_clamped = max.(n_cells, FT(min_smoothing_cells))
        scale_distance = n_cells_clamped .* Δh_grid
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
    sinphi = sind.(lat)

    # Pre-compute longitude offset cosines (reused across all grid points)
    max_ilon_offset = maximum(ilon_range)
    cosdlam = [cosd(i2 * dlon) for i2 in 0:max_ilon_offset]

    # Initialize output
    χ = zeros(FT, nlon, nlat)

    # Main convolution loop: latitude (outer) then longitude (inner)
    # This matches Fortran structure and allows weight pre-computation
    for j in 1:nlat
        jlat_deg = lat[j]

        # Skip polar regions (matches Fortran threshold)
        if abs(jlat_deg) > 89
            continue
        end

        # Determine latitude window for this j
        ja = max(1, j - ilat_range[j])
        jb = min(nlat, j + ilat_range[j])

        # Pre-compute weights for all latitude offsets and longitude offsets
        # wt[i_offset, j1] stores weight for longitude offset i_offset and latitude index j1
        max_i_offset = min(ilon_range[j], max_ilon_offset)
        wt = zeros(FT, max_i_offset + 1, jb - ja + 1)  # +1 for zero offset

        cj = cosphi[j]
        sj = sinphi[j]

        # Compute weights once per latitude row (OPTIMIZATION: moved outside i-loop)
        for (j1_idx, j1) in enumerate(ja:jb)
            cj1 = cosphi[j1]
            sj1 = sinphi[j1]
            ccj = cj * cj1  # cos(lat_j) * cos(lat_j1)
            ssj = sj * sj1  # sin(lat_j) * sin(lat_j1)

            for i2 in 0:max_i_offset
                # Compute great circle arc distance using offset
                # arc = acos(cos(lat1)*cos(lat2)*cos(dlon) + sin(lat1)*sin(lat2))
                arc = acos(min(FT(1), ccj * cosdlam[i2 + 1] + ssj))

                # Blackman window taper
                arc1 = arc / max(arc, scale[j])
                blackman =
                    FT(0.42) + FT(0.50) * cos(pi * arc1) + FT(0.08) * cos(FT(2) * pi * arc1)

                # Weight: cos(lat) / arc * Blackman_window
                # Latitude-dependent regularization: more at high latitudes, less at low
                # At high latitudes (|lat| > 70°), use larger min_arc to prevent blow-up
                # At low latitudes, use smaller min_arc to preserve detail
                lat_factor = max(FT(0.01), abs(sind(lat[j])) / FT(10))  # Ranges from 1% to 10%
                min_arc = max(scale[j] * lat_factor, dlat_rad * lat_factor)
                wt[i2 + 1, j1_idx] = cj1 / (arc + min_arc) * blackman
            end
        end

        # Zero out singularity (when j == j1 and i_offset == 0)
        if j >= ja && j <= jb
            wt[1, j - ja + 1] = FT(0)
        end

        # Now loop over all longitudes at this latitude
        for i in 1:nlon
            # Determine longitude window with PERIODIC boundaries
            ia = i - ilon_range[j]
            ib = i + ilon_range[j]

            sum_val = FT(0)

            # Accumulate weighted sum over window
            for j1 in ja:jb
                j1_idx = j1 - ja + 1
                for i1 in ia:ib
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
            if j == 71 && i == 2487  # The problematic location
                @info "Debug at blow-up location (i=$i, j=$j, lat=$(lat[j])):"
                @info "  sum_val = $sum_val"
                @info "  max weight = $(maximum(abs.(wt)))"
                @info "  ilon_range[j] = $(ilon_range[j]), ilat_range[j] = $(ilat_range[j])"
                @info "  scale[j] = $(scale[j])"
            end
            χ[i, j] = sum_val
        end
    end

    # Apply area element scaling
    χ .*= (dlon_rad * dlat_rad)
    # Debug: find where the huge values are
    max_val, max_idx = findmax(abs.(χ))
    max_i, max_j = Tuple(max_idx)
    @info "Max |χ| location: i=$max_i (lon=$(lon[max_i])), j=$max_j (lat=$(lat[max_j])), value=$(χ[max_i, max_j])"
    @info "χ range before final scaling: min=$(minimum(χ)), max=$(maximum(χ))"
    # Add singularity correction term (Green's function correction for finite grid size)
    for j in 1:nlat
        jlat_deg = lat[j]

        # Skip polar regions where dlamj → 0 causes log singularity (matches Fortran)
        if abs(jlat_deg) > 89
            continue
        end

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
    calc_hpoz_latlon(elev, lon, lat, earth_radius;
                    smoothing_length_scale=nothing,
                    n_smoothing_cells=nothing,
                    min_smoothing_cells=1.0)

    Calculated hmax used in orographic gravity wave parameterization.
    Implements distance-weighted 4th moment calculation similar to Fortran get_hmax.

    Arguments:
    - elev: surface elevation
    - lon: longitude array (degrees)
    - lat: latitude array (degrees)
    - earth_radius: radius of the Earth (meters)
    - smoothing_length_scale: smoothing scale in meters (optional)
                             Default: 100e3 (100 km), but enforces min_smoothing_cells
    - n_smoothing_cells: number of grid cells for smoothing (optional, overrides smoothing_length_scale)
    - min_smoothing_cells: minimum number of grid cells to smooth over (default: 1.0)
"""
function calc_hpoz_latlon(
    elev,
    lon,
    lat,
    earth_radius;
    smoothing_length_scale = nothing,
    n_smoothing_cells = nothing,
    min_smoothing_cells = 1.0,
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
        # User specified exact number of cells
        scale_distance = n_smoothing_cells .* Δh_grid
    elseif smoothing_length_scale !== nothing
        # User specified physical length scale in meters
        n_cells = smoothing_length_scale ./ Δh_grid
        n_cells_clamped = max.(n_cells, min_smoothing_cells)
        scale_distance = n_cells_clamped .* Δh_grid
    else
        # Default: 100 km scale with minimum cell constraint
        default_scale = FT(100e3)
        n_cells = default_scale ./ Δh_grid
        n_cells_clamped = max.(n_cells, FT(min_smoothing_cells))
        scale_distance = n_cells_clamped .* Δh_grid
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

    hmax[:, abs.(lat) .> FT(89)] .= FT(0)
    return hmax
end

function compute_OGW_info(
    Y,
    elev_data,
    earth_radius,
    γ,
    h_frac;
    smoothing_length_scale = nothing,
    n_smoothing_cells = nothing,
    min_smoothing_cells = 1.0,
)
    # obtain lat, lon, elevation from the elev_data
    FT = Spaces.undertype(Spaces.axes(Y.c))
    # downsample to elev dims (3600×1800)
    skip_pt = 6
    nt = NCDataset(elev_data, "r") do ds
        lon = FT.(Array(ds["lon"]))[1:skip_pt:end]
        lat = FT.(Array(ds["lat"]))[1:skip_pt:end]
        elev = FT.(Array(ds["z"]))[1:skip_pt:end, 1:skip_pt:end]
        (; lon, lat, elev)
    end
    (; lon, lat, elev) = nt
    FT = eltype(elev)

    # compute hmax and hmin (with grid-aware smoothing)
    hpoz = calc_hpoz_latlon(
        elev,
        lon,
        lat,
        earth_radius;
        smoothing_length_scale,
        n_smoothing_cells,
        min_smoothing_cells,
    )
    hpoz = @. max(FT(0), hpoz)^(FT(2) - γ)
    hmax = @. (
        abs(hpoz) * (γ + FT(2)) / (FT(2) * γ) * (FT(1) - h_frac^(FT(2) * γ)) /
        (FT(1) - h_frac^(γ + FT(2)))
    )^(FT(1) / (FT(2) - γ))
    hmin = hmax .* h_frac

    # compute χ (with grid-aware smoothing)
    χ = calc_velocity_potential(
        elev,
        lon,
        lat,
        earth_radius;
        smoothing_length_scale,
        n_smoothing_cells,
        min_smoothing_cells,
    )

    # compute orographic tensor (t11, t21, t12, t22)
    # Debug: find where the huge values are
    # max_val, max_idx = findmax(abs.(χ))
    # max_i, max_j = Tuple(max_idx)
    # @info "Max |χ| location: i=$max_i (lon=$(lon[max_i])), j=$max_j (lat=$(lat[max_j])), value=$(χ[max_i, max_j])"
    # @info "χ range before final scaling: min=$(minimum(χ)), max=$(maximum(χ))"
    t11, t21, t12, t22 = calc_orographic_tensor(elev, χ, lon, lat, earth_radius)

    topo_ll = (; hmax, hmin, t11, t12, t21, t22)

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

    cg_lat = Fields.level(Fields.coordinate_field(Y.c).lat, 1)
    cg_lon = Fields.level(Fields.coordinate_field(Y.c).long, 1)

    # NOTE: GFDL may incorporate some smoothing when inerpolate it to model grid
    for varname in (:hmax, :hmin, :t11, :t12, :t21, :t22)
        li_obj = Interpolations.linear_interpolation(
            (lon, lat),
            getproperty(topo_ll, varname),
            extrapolation_bc = (
                Interpolations.Periodic(),
                Interpolations.Flat(),
            ),
        )
        Fields.bycolumn(axes(Y.c.ρ)) do colidx
            parent(getproperty(topo_cg, varname)[colidx]) .=
                FT.(li_obj(parent(cg_lon[colidx]), parent(cg_lat[colidx])))
        end
    end
    return topo_cg
end

function regrid_OGW_info(Y, orographic_info_rll)
    FT = Spaces.undertype(axes(Y.c))

    lon, lat, topo_ll = get_topo_ll(orographic_info_rll)

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

    cg_lat = Fields.level(Fields.coordinate_field(Y.c).lat, 1)
    cg_lon = Fields.level(Fields.coordinate_field(Y.c).long, 1)

    for varname in (:hmax, :hmin, :t11, :t12, :t21, :t22)
        li_obj = Interpolations.linear_interpolation(
            (lon, lat),
            getproperty(topo_ll, varname),
            extrapolation_bc = (
                Interpolations.Periodic(),
                Interpolations.Flat(),
            ),
        )
        Fields.bycolumn(axes(Y.c.ρ)) do colidx
            parent(getproperty(topo_cg, varname)[colidx]) .=
                FT.(li_obj(parent(cg_lon[colidx]), parent(cg_lat[colidx])))
        end
    end
    return topo_cg
end

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

function gen_fn(parsed_args)
    ### generate output filename
    # get grid info necessary to specify unique output file
    topography = parsed_args["topography"]
    topo_smoothing = parsed_args["topo_smoothing"]
    topo_damping_factor = parsed_args["topography_damping_factor"]
    h_elem = parsed_args["h_elem"]

    # construct output filename
    output_filename = "computed_drag_$(topography)_$(topo_smoothing)_$(topo_damping_factor)_$(h_elem)"
    ###
    return (; output_filename, topography, topo_smoothing, topo_damping_factor, h_elem)
end


function load_preprocessed_topography(parsed_args::Dict{String, Any})
    (; output_filename,) = gen_fn(parsed_args)
    @info "loading topography drag vector: $(output_filename)"

    reader = InputOutput.HDF5Reader(
        joinpath(@__DIR__, "../../../$(output_filename).hdf5"),
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
    )
    computed_drag = InputOutput.read_field(reader, "computed_drag")

    Base.close(reader)
    return computed_drag
end

# For direct filename
function load_preprocessed_topography(filename::String)
    @info "loading topography drag vector: $(filename)"
    reader = InputOutput.HDF5Reader(
        joinpath(@__DIR__, "../../../$(filename).hdf5"),
        ClimaComms.SingletonCommsContext(),
    )
    computed_drag = InputOutput.read_field(reader, "computed_drag")

    Base.close(reader)
    return computed_drag
end
