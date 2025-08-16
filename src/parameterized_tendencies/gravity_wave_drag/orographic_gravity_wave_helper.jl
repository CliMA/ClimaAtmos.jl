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
    calc_velocity_potential(elev, lon, lat, earth_radius)

    Calculate velocity potential
    - elev: surface elevation
    - lon: longitude
    - lat: latitude
    - earth_radius: radius of the Earth
"""
function calc_velocity_potential(elev, lon, lat, earth_radius)
    @info "Computing velocity potential..."
    FT = eltype(elev)
    
    # Ensure non-negative elevation
    @. elev = max(0, elev)
    
    nx, ny = size(elev)
    @assert nx == length(lon)
    @assert ny == length(lat)
    
    # Grid spacing in radians
    dlam = deg2rad(lon[2] - lon[1])  # longitude spacing in radians
    dphi = deg2rad(lat[2] - lat[1])  # latitude spacing in radians
    
    # Pre-compute trigonometric values for latitude
    phi = deg2rad.(lat)
    cosphi = cos.(phi)
    sinphi = sin.(phi)
    
    # Pre-compute cosine values for longitude differences
    # Fortran: do i=0,lx; cosdlam(i) = cos(i*dlam); enddo
    cosdlam = [cos(i * dlam) for i in 0:nx]
    
    # Scale parameter (matching Fortran's scale array)
    scale = sind(40) ./ earth_radius ./ sind.(max.(FT(20), abs.(lat))) .* FT(100e3)
    
    # Initialize output
    χ = zeros(FT, nx, ny)
    
    # Main loop over output grid points
    for j in 1:ny
        # Skip poles (matching Fortran: if abs(lat(j)) > 89 cycle)
        if abs(lat[j]) > 89
            continue
        end
        
        cj = cosphi[j]
        sj = sinphi[j]
        dlamj = dlam * cj
        
        # Compute search radii in grid points
        # Fortran: idiff = int( scale(j)/dlamj ); idiff = min(idiff,lx/8)
        # Handle potential division by very small dlamj at high latitudes
        if dlamj > FT(1e-10)
            idiff = Int(trunc(scale[j] / dlamj))  # Fortran int() is truncation, not rounding
        else
            idiff = div(nx, 8)  # Use maximum allowed value
        end
        idiff = min(idiff, div(nx, 8))
        jdiff = Int(trunc(scale[j] / dphi))  # Fortran int() is truncation
        
        # Latitude bounds for integration
        # Fortran: ja = max( ly1, j - jdiff ); jb = min( ly2, j + jdiff )
        ja = max(1, j - jdiff)
        jb = min(ny, j + jdiff)
        
        # Additional safety check to prevent excessive ranges
        if (jb - ja) > div(ny, 2)
            jdiff = min(jdiff, div(ny, 4))
            ja = max(1, j - jdiff)
            jb = min(ny, j + jdiff)
        end
        
        # Pre-compute weights for this latitude
        # Fortran: real, dimension(0:lx/8+1,ly1:ly2) :: wt
        wt = zeros(FT, idiff + 1, ny)
        
        # Fortran: do j1=ja+1,jb
        for j1 in (ja+1):jb
            cj1 = cosphi[j1]
            sj1 = sinphi[j1]
            ccj = cj * cj1
            ssj = sj * sj1
            
            # Fortran: do i2=0,idiff
            for i2 in 0:idiff
                # Great circle distance
                arc = acos(min(FT(1), ccj * cosdlam[i2+1] + ssj))
                arc1 = arc / max(arc, scale[j])
                
                # Blackman window weight
                wt[i2+1, j1] = cj1 / (arc + FT(1e-12)) * 
                    (FT(0.42) + FT(0.50) * cos(π * arc1) + FT(0.08) * cos(2π * arc1))
            end
            
            # Zero weight at same point (matching Fortran: if (j == j1) wt(0,j1) = 0.)
            if j == j1
                wt[1, j1] = FT(0)
            end
        end
        
        # Now compute the convolution for each longitude
        # Fortran: do i=ix1,ix2
        for i in 1:nx
            # Longitude bounds for integration
            # Fortran: ia = max( lx1, i - idiff ); ib = min( lx2, i + idiff )
            ia = max(1, i - idiff)
            ib = min(nx, i + idiff)
            
            # Accumulate weighted sum
            # Fortran: do j1=ja+1,jb; do i1=ia+1,ib
            for j1 in (ja+1):jb
                for i1 in (ia+1):ib
                    # Fortran: i2 = iabs(i-i1)
                    i2 = abs(i - i1)
                    χ[i, j] += wt[i2+1, j1] * elev[i1, j1]
                end
            end
        end
    end
    
    # Apply area element scaling
    χ .*= (dlam * dphi)
    
    # Add the singular part of the integral
    for j in 1:ny
        cj = cosphi[j]
        dlamj = dlam * cj
        rij = sqrt(dlamj^2 + dphi^2)
        
        for i in 1:nx
            χ[i, j] += elev[i, j] * FT(2) * (
                dlamj * log((rij + dphi) / dlamj) + 
                dphi * log((rij + dlamj) / dphi)
            )
        end
    end
    
    # Final scaling
    χ .*= (earth_radius / (FT(2) * π))
    
    # Set polar values to zero (matching calc_hpoz_latlon approach)
    χ[:, abs.(lat) .> FT(89)] .= FT(0)
    
    return χ
end

"""
    calc_hmax_latlon(elev, lon, lat, earth_radius)

    Calculated hmax used in orographic gravity wave
    - elev: surface elevation
    - lon: longitude
    - lat: latitude
    - earth_radius: radius of the Earth
"""
function calc_hpoz_latlon(elev, lon, lat, earth_radius)
    @info "Computing hmax..."
    FT = eltype(elev)

    # remove ocean topography
    elev[elev .< FT(0)] .= FT(0)

    dlat = lat[2] - lat[1]
    dlon = lon[2] - lon[1]

    scale =
        sind(40) ./ earth_radius ./ sind.(max.(FT(20), abs.(lat))) .* FT(100e3) # FT(200e3) is used in the codes sent by Steve Garner

    ilat_range = Int.(round.(scale ./ deg2rad(dlat)))
    ilon_range =
        min.(
            Int.(round.(scale ./ (deg2rad(dlon) .* cosd.(lat)))),
            Int(round(length(lon) / 16)),
        )

    hmax = zeros(size(elev))
    for i in 1:length(lon)
        for j in 1:length(lat)
            # TODO: irange may not need clipping at the boundaries since it is on the closed lat circle
            irange =
                max(i - ilon_range[j], 1):min(i + ilon_range[j], length(lon))
            jrange =
                max(j - ilat_range[j], 1):min(j + ilat_range[j], length(lat))

            # 4th moments of elevation
            hmax[i, j] =
                (mean(
                    (elev[irange, jrange] .- mean(elev[irange, jrange])) .^
                    FT(4),
                ))^FT(0.25)
        end
    end
    hmax[:, abs.(lat) .> FT(89)] .= FT(0)
    return hmax
end

function compute_OGW_info(Y, elev_data, earth_radius, γ, h_frac)
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

    # compute hmax and hmin
    hpoz = calc_hpoz_latlon(elev, lon, lat, earth_radius)
    hpoz = @. max(FT(0), hpoz)^(FT(2) - γ)
    hmax = @. (
        abs(hpoz) * (γ + FT(2)) / (FT(2) * γ) * (FT(1) - h_frac^(FT(2) * γ)) /
        (FT(1) - h_frac^(γ + FT(2)))
    )^(FT(1) / (FT(2) - γ))
    hmin = hmax .* h_frac

    # compute χ
    χ = calc_velocity_potential(elev, lon, lat, earth_radius)

    # compute orographic tensor (t11, t21, t12, t22)
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

    t11 = Fields.Field(Fields.field_values(t11), ᶜtarget_space)
    t12 = Fields.Field(Fields.field_values(t12), ᶜtarget_space)
    t21 = Fields.Field(Fields.field_values(t21), ᶜtarget_space)
    t22 = Fields.Field(Fields.field_values(t22), ᶜtarget_space)
    hmin = Fields.Field(Fields.field_values(hmin), ᶜtarget_space)
    hmax = Fields.Field(Fields.field_values(hmax), ᶜtarget_space)

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


function load_preprocessed_topography(parsed_args)
    (; output_filename,) = gen_fn(parsed_args)

    reader = InputOutput.HDF5Reader(
        joinpath(@__DIR__, "../../../$(output_filename).hdf5"),
        ClimaComms.SingletonCommsContext(ClimaComms.CPUSingleThreaded()),
    )
    computed_drag = InputOutput.read_field(reader, "computed_drag")

    Base.close(reader)
    return computed_drag
end