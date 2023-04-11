using NCDatasets
using Interpolations

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
    FT = eltype(elev)
    bfscale = FT(1e-2)

    # compute ∇h
    @. elev = max(0, elev)
    dhdx, dhdy = calc_∇A(elev, lon, lat, earth_radius)

    # compute ∇χ
    dχdx, dχdy = -bfscale .* calc_∇A(χ, lon, lat, earth_radius)

    # for antarctic
    dχdx[:, lat .< FT(-88)] .= FT(0)
    dχdy[:, lat .< FT(-88)] .= FT(0)

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
    dAdx =
        vcat(
            (A[2, :] .- A[1, :])',
            (A[3:end, :] .- A[1:(end - 2), :]) / FT(2),
            (A[end, :] .- A[end - 1, :])',
        ) ./ (
            deg2rad(maximum(lon) - minimum(lon)) * earth_radius .*
            reshape(repeat(cosd.(lat), length(lon)), length(lat), :)'
        )
    dAdy =
        hcat(
            A[:, 2] .- A[:, 1],
            (A[:, 3:end] .- A[:, 1:(end - 2)]) / FT(2),
            A[:, end] .- A[:, end - 1],
        ) ./ (deg2rad(maximum(lat) - minimum(lat)) * earth_radius)
    return (dAdx, dAdy)
end

"""
    calc_hmax_latlon(elev, lon, lat)

Calculated hmax used in orographic gravity wave
    - elev: surface elevation
    - lon: longitude
    - lat: latitude
"""
function calc_hmax_latlon(elev, lon, lat)
    FT = eltype(elev)

    dlat = lat[2] - lat[1]
    dlon = lon[2] - lon[1]
    scale =
        [lat[1], lat..., lat[end]][3:end] .-
        [lat[1], lat..., lat[end]][1:(end - 2)]

    # compute weights for the spatial running mean using the Blackman kernel
    ilat_range = Int.(floor.(scale ./ dlat))
    ilon_range =
        min.(
            Int.(floor.(scale ./ (dlon .* cosd.(lat)))),
            Int(floor(length(lon) / 16)),
        )

    hmax = zeros(size(elev))
    for (i, ilon) in enumerate(lon)
        for (j, jlat) in enumerate(lat)
            # irange may not need clipping at the boundaries since it is on the closed lat circle
            irange =
                max(i - ilon_range[j], 1):min(i + ilon_range[j], length(lon))
            jrange =
                max(j - ilat_range[j], 1):min(j + ilat_range[j], length(lat))

            # compute weights for spatial running-mean filter using the Blackman kernel
            arc =
                acos.(
                    min.(
                        FT(1),
                        reshape(
                            repeat(
                                cosd(jlat) .* cosd.(lat[jrange]),
                                length(irange),
                            ),
                            length(jrange),
                            :,
                        )' .* reshape(
                            repeat(cosd.(ilon .- lon[irange]), length(jrange)),
                            length(irange),
                            :,
                        ) .+
                        reshape(
                            repeat(
                                sind(jlat) .* sind.(lat[jrange]),
                                length(irange),
                            ),
                            length(jrange),
                            :,
                        )',
                    )
                )
            hp_wts =
                FT(0.42) .+ FT(0.50) .* cos.(pi * arc) .+
                FT(0.08) .* cos.(FT(2) * pi * arc)

            # high pass elevation and use its 4th moment as hmax
            elev_highpass = elev[irange, jrange] .* hp_wts
            hmax[i, j] =
                (mean((elev_highpass .- mean(elev_highpass)) .^ FT(4)))^FT(0.25)
        end
    end
    hmax[:, abs.(lat) .> FT(89)] .= FT(0)
    return hmax
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
    FT = eltype(elev)

    dlat = lat[2] - lat[1]
    dlon = lon[2] - lon[1]
    scale =
        [lat[1], lat..., lat[end]][3:end] .-
        [lat[1], lat..., lat[end]][1:(end - 2)]

    # compute weights for the spatial running mean using the Blackman kernel
    ilat_range = Int.(floor.(scale ./ dlat))
    ilon_range =
        min.(
            Int.(floor.(scale ./ (dlon .* cosd.(lat)))),
            Int(floor(length(lon) / 8)),
        )

    χ = zeros(size(elev))
    for (i, ilon) in enumerate(lon)
        for (j, jlat) in enumerate(lat)
            # irange may not need clipping at the boundaries since it is on the closed lat circle
            irange =
                max(i - ilon_range[j], 1):min(i + ilon_range[j], length(lon))
            jrange =
                max(j - ilat_range[j], 1):min(j + ilat_range[j], length(lat))

            # compute weights for spatial (area) integration
            arc =
                acos.(
                    min.(
                        FT(1),
                        reshape(
                            repeat(
                                cosd(jlat) .* cosd.(lat[jrange]),
                                length(irange),
                            ),
                            length(jrange),
                            :,
                        )' .* reshape(
                            repeat(cosd.(ilon .- lon[irange]), length(jrange)),
                            length(irange),
                            :,
                        ) .+
                        reshape(
                            repeat(
                                sind(jlat) .* sind.(lat[jrange]),
                                length(irange),
                            ),
                            length(jrange),
                            :,
                        )',
                    )
                )
            arc1 = arc ./ max.(arc, deg2rad(scale[j]))
            wts =
                reshape(
                    repeat(cosd.(lat[jrange]), length(irange)),
                    length(jrange),
                    :,
                )' ./ (arc .+ eps(FT)) .* (
                    FT(0.42) .+ FT(0.50) .* cos.(pi * arc1) .+
                    FT(0.08) .* cos.(FT(2) * pi * arc1)
                )
            if i != irange[1] && j != jrange[1]
                wts[i - irange[1], j - jrange[1]] = FT(0)
            end

            # calculate the spatial integration
            χ[i, j] = sum(wts .* elev[irange, jrange])
        end
    end
    @. χ = χ * dlon * dlat
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
                    log((rij + deg2rad(dlon) * cosd(jlat)) / deg2rad(dlon))
                )
        end
    end

    @. χ = χ * (earth_radius / FT(2) / pi)

    return χ
end

function regrid_OGW_info(Y, orographic_info_rll)
    FT = Spaces.undertype(axes(Y.c))

    lon, lat, topo_ll = get_topo_ll(orographic_info_rll)

    topo_cg = FieldFromNamedTuple(
        axes(Fields.level(Y.c.ρ, 1)),
        (;
            t11 = FT(0),
            t12 = FT(0),
            t21 = FT(0),
            t22 = FT(0),
            hmin = FT(0),
            hmax = FT(0),
        ),
    )

    cg_lat = Fields.level(Fields.coordinate_field(Y.c).lat, 1)
    cg_lon = Fields.level(Fields.coordinate_field(Y.c).long, 1)

    for varname in (:hmax, :hmin, :t11, :t12, :t21, :t22)
        li_obj = linear_interpolation(
            (lon, lat),
            getproperty(topo_ll, varname),
            extrapolation_bc = (Periodic(), Flat()),
        )
        Fields.bycolumn(axes(Y.c.ρ)) do colidx
            parent(getproperty(topo_cg, varname)[colidx]) .=
                FT.(li_obj(parent(cg_lon[colidx]), parent(cg_lat[colidx])))
        end
    end
    return topo_cg
end

function compute_OGW_info(Y, elev_data, earth_radius)
    # obtain lat, lon, elevation from the elev_data
    nt = NCDataset(elev_data, "r") do ds
        lon = ds["longitude"][:]
        lat = ds["latitude"][:]
        elev = ds["elevation"][:]
        (; lon, lat, elev)
    end
    (; lon, lat, elev) = nt
    FT = eltype(elev)

    # compute hmax and hmin
    hmax = calc_hmax_latlon(elev, lon, lat)

    # compute χ
    χ = calc_velocity_potential(elev, lon, lat, earth_radius)

    # compute orographic tensor (t11, t21, t12, t22)
    t11, t21, t12, t22 = calc_orographic_tensor(elev, χ, lon, lat, earth_radius)

    topo_ll = (; hmax = hmax, hmin = FT(0.1) .* hmax, t11, t12, t21, t22)

    # create ClimaCore.Fields
    topo_cg = FieldFromNamedTuple(
        axes(Fields.level(Y.c.ρ, 1)),
        (;
            t11 = FT(0),
            t12 = FT(0),
            t21 = FT(0),
            t22 = FT(0),
            hmin = FT(0),
            hmax = FT(0),
        ),
    )

    cg_lat = Fields.level(Fields.coordinate_field(Y.c).lat, 1)
    cg_lon = Fields.level(Fields.coordinate_field(Y.c).long, 1)

    for varname in (:hmax, :hmin, :t11, :t12, :t21, :t22)
        li_obj = linear_interpolation(
            (lon, lat),
            getproperty(topo_ll, varname),
            extrapolation_bc = (Periodic(), Flat()),
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
        lon = ds["lon"][:]
        lat = ds["lat"][:]
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

function FieldFromNamedTuple(space, nt::NamedTuple)
    cmv(z) = nt
    return cmv.(Fields.coordinate_field(space))
end
