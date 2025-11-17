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

    dlat = lat[2] - lat[1]
    dlon = lon[2] - lon[1]

    scale =
        sind(40) ./ earth_radius ./ sind.(max.(FT(20), abs.(lat))) .* FT(100e3) # FT(200e3) is used in the codes sent by Steve Garner

    # compute weights for the spatial running mean using the Blackman kernel
    ilat_range = Int.(round.(scale ./ deg2rad(dlat)))
    ilon_range =
        min.(
            Int.(trunc.(scale ./ (deg2rad(dlon) .* cosd.(lat)))),
            Int(round(length(lon) / 8)),
        )

    # @Main.infiltrate

    χ = zeros(size(elev))
    for (i, ilon) in enumerate(lon)
        for (j, jlat) in enumerate(lat)
            if abs(jlat) > 89
                continue
            end
            # irange may not need clipping at the boundaries since it is on the closed lat circle
            irange =
                max(i - ilon_range[j], 1):min(i + ilon_range[j], length(lon))
            jrange =
                max(j - ilat_range[j], 1):min(j + ilat_range[j], length(lat))

            # 1. Define the longitude and latitude coordinates in the window
            lons_in_window = lon[irange]
            lats_in_window = lat[jrange]

            # 2. Calculate the components, being explicit about dimensions
            # Note the transpose ' on the LATITUDE vectors this time
            sin_lats_term = sind.(lats_in_window') .* sind(jlat) # This is now a ROW vector (1 x M)
            cos_lats_term = cosd.(lats_in_window') .* cosd(jlat) # Also a ROW vector (1 x M)

            cos_lon_diff = cosd.(lons_in_window .- ilon)      # This is now a COLUMN vector (N x 1)

            # 3. Broadcasting combines the column (lon) and row (lat) vectors into a matrix
            # The result will be N rows (longitude) x M columns (latitude)
            inner_product = sin_lats_term .+ cos_lon_diff .* cos_lats_term

            # 4. Final arc calculation
            arc = acos.(min.(FT(1), inner_product))

            arc1 = arc ./ max.(arc, scale[j])
            # @Main.infiltrate
            # Compute weights: cos(latitude) / arc * Blackman_window(arc1)
            # Note: cosd.(lat[jrange])' creates a row vector that broadcasts across longitude dimension
            wts =
                cosd.(lat[jrange]') ./ (arc .+ eps(FT)) .* (
                    FT(0.42) .+ FT(0.50) .* cos.(pi * arc1) .+
                    FT(0.08) .* cos.(FT(2) * pi * arc1)
                )
            @. wts[arc < 1e-9] = 0.0
            # calculate the spatial integration
            χ[i, j] = sum(wts .* elev[irange, jrange])
            # @Main.infiltrate
        end
    end
    # @Main.infiltrate
    @. χ = χ * deg2rad(dlon) * deg2rad(dlat)
    for (j, jlat) in enumerate(lat)
        rij = sqrt((deg2rad(dlon) * cosd(jlat))^2 + deg2rad(dlat)^2)
        for (i, ilon) in enumerate(lon)
            χ[i, j] =
                χ[i, j] +
                elev[i, j] *
                FT(2) *
                (
                    deg2rad(dlon) *
                    cosd(jlat) *
                    log((rij + deg2rad(dlat)) / (deg2rad(dlon) * cosd(jlat))) +
                    deg2rad(dlat) *
                    log((rij + deg2rad(dlon) * cosd(jlat)) / deg2rad(dlat))
                )
        end
    end

    @. χ = χ * (earth_radius / FT(2) / pi)

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