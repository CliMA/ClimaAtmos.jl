# Turn an ERA5 model-level file into the altitude-level file `WeatherModel`
# reads. Ported from `to_z_levels_3d_model` in WeatherQuest
# `processing/interpolate.jl`.

import Thermodynamics as TD
import ClimaInterpolations.Interpolation1D: interpolate1d!, Linear, Flat

# ERA5 / IFS L137 hybrid interface `a` coefficients [Pa], index 1 = model top.
# https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
const L137_A_HALF = (
    0.000000, 2.000365, 3.102241, 4.666084,
    6.827977, 9.746966, 13.605424, 18.608931,
    24.985718, 32.985710, 42.879242, 54.955463,
    69.520576, 86.895882, 107.415741, 131.425507,
    159.279404, 191.338562, 227.968948, 269.539581,
    316.420746, 368.982361, 427.592499, 492.616028,
    564.413452, 643.339905, 729.744141, 823.967834,
    926.344910, 1037.201172, 1156.853638, 1285.610352,
    1423.770142, 1571.622925, 1729.448975, 1897.519287,
    2076.095947, 2265.431641, 2465.770508, 2677.348145,
    2900.391357, 3135.119385, 3381.743652, 3640.468262,
    3911.490479, 4194.930664, 4490.817383, 4799.149414,
    5119.895020, 5452.990723, 5798.344727, 6156.074219,
    6526.946777, 6911.870605, 7311.869141, 7727.412109,
    8159.354004, 8608.525391, 9076.400391, 9562.682617,
    10065.978516, 10584.631836, 11116.662109, 11660.067383,
    12211.547852, 12766.873047, 13324.668945, 13881.331055,
    14432.139648, 14975.615234, 15508.256836, 16026.115234,
    16527.322266, 17008.789063, 17467.613281, 17901.621094,
    18308.433594, 18685.718750, 19031.289063, 19343.511719,
    19620.042969, 19859.390625, 20059.931641, 20219.664063,
    20337.863281, 20412.308594, 20442.078125, 20425.718750,
    20361.816406, 20249.511719, 20087.085938, 19874.025391,
    19608.572266, 19290.226563, 18917.460938, 18489.707031,
    18006.925781, 17471.839844, 16888.687500, 16262.046875,
    15596.695313, 14898.453125, 14173.324219, 13427.769531,
    12668.257813, 11901.339844, 11133.304688, 10370.175781,
    9617.515625, 8880.453125, 8163.375000, 7470.343750,
    6804.421875, 6168.531250, 5564.382813, 4993.796875,
    4457.375000, 3955.960938, 3489.234375, 3057.265625,
    2659.140625, 2294.242188, 1961.500000, 1659.476563,
    1387.546875, 1143.250000, 926.507813, 734.992188,
    568.062500, 424.414063, 302.476563, 202.484375,
    122.101563, 62.781250, 22.835938, 3.757813,
    0.000000, 0.000000,
)

# ERA5 / IFS L137 hybrid interface `b` coefficients, same source as L137_A_HALF.
const L137_B_HALF = (
    0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000007,
    0.000024, 0.000059, 0.000112, 0.000199,
    0.000340, 0.000562, 0.000890, 0.001353,
    0.001992, 0.002857, 0.003971, 0.005378,
    0.007133, 0.009261, 0.011806, 0.014816,
    0.018318, 0.022355, 0.026964, 0.032176,
    0.038026, 0.044548, 0.051773, 0.059728,
    0.068448, 0.077958, 0.088286, 0.099462,
    0.111505, 0.124448, 0.138313, 0.153125,
    0.168910, 0.185689, 0.203491, 0.222333,
    0.242244, 0.263242, 0.285354, 0.308598,
    0.332939, 0.358254, 0.384363, 0.411125,
    0.438391, 0.466003, 0.493800, 0.521619,
    0.549301, 0.576692, 0.603648, 0.630036,
    0.655736, 0.680643, 0.704669, 0.727739,
    0.749797, 0.770798, 0.790717, 0.809536,
    0.827256, 0.843881, 0.859432, 0.873929,
    0.887408, 0.899900, 0.911448, 0.922096,
    0.931881, 0.940860, 0.949064, 0.956550,
    0.963352, 0.969513, 0.975078, 0.980072,
    0.984542, 0.988500, 0.991984, 0.995003,
    0.997630, 1.000000,
)

"""
    era5_hybrid_coeffs(ds, nlev, FT)

Hybrid coefficients `(a, b)` at the `nlev + 1` layer interfaces, surface
first, giving interface pressures `a[k] + b[k] * sp`. Read from `ds` when it
carries them, otherwise from the compiled-in L137 table.
"""
function era5_hybrid_coeffs(ds, nlev::Int, ::Type{FT}) where {FT}
    a_half, b_half = if haskey(ds, "a") && haskey(ds, "b")
        FT.(vec(Array(ds["a"]))), FT.(vec(Array(ds["b"])))
    elseif nlev == 137
        FT.(collect(L137_A_HALF)), FT.(collect(L137_B_HALF))
    else
        error(
            "The input carries no hybrid coefficients `a` and `b`, and the " *
            "compiled-in table only covers the 137 ERA5 model levels, not $(nlev).",
        )
    end
    if length(a_half) != nlev + 1 || length(b_half) != nlev + 1
        error(
            "Hybrid coefficients must be half-level arrays of length " *
            "nlev + 1 = $(nlev + 1); got a = $(length(a_half)), b = $(length(b_half)).",
        )
    end
    # b runs 0 at the model top to 1 at the surface, so it tells us the order
    if b_half[1] < b_half[end]
        a_half, b_half = reverse(a_half), reverse(b_half)
    end
    return a_half, b_half
end

"""
    era5_model_level_name(ds)

Name of the model-level dimension of `ds`. Errors on a pressure-level file,
which [`to_z_levels_1d`](@ref) handles.
"""
function era5_model_level_name(ds)
    haskey(ds, "pressure_level") && error(
        "to_z_levels_3d_model takes ERA5 model levels, but this file has a " *
        "`pressure_level` dimension. Use to_z_levels_1d instead.",
    )
    for name in ("model_level", "level", "hybrid")
        haskey(ds, name) && return name
    end
    return error("No model-level dimension found in the input.")
end

"""
    read_reordered(ds, name, FT; lev_len = nothing)

Variable `name` as `(lon, lat[, lev][, time])` with `missing` mapped to `NaN`.
Singleton dimensions are dropped, then the rest are identified by length.
"""
function read_reordered(ds, name, ::Type{FT}; lev_len = nothing) where {FT}
    arr = Array(ds[name])
    arr = dropdims(arr; dims = Tuple(findall(size(arr) .== 1)))
    lengths = (
        length(ds["longitude"]),
        length(ds["latitude"]),
        lev_len,
        haskey(ds, "valid_time") ? length(ds["valid_time"]) : 1,
    )
    sz = size(arr)
    perm = Int[]
    for len in lengths
        isnothing(len) && continue
        idx = findfirst(==(len), sz)
        (isnothing(idx) || idx in perm) && continue
        push!(perm, idx)
    end
    append!(perm, setdiff(1:length(sz), perm))
    return FT.(coalesce.(permutedims(arr, Tuple(perm)), NaN))
end

read_surface(ds, name, ::Type{FT}) where {FT} =
    let a = read_reordered(ds, name, FT)
        ndims(a) == 3 ? a[:, :, 1] : a
    end

"""
    read_surface_pressure(ds, FT)

Surface pressure [Pa] as `(lon, lat)`, from `sp` or from `exp(lnsp)`.
"""
function read_surface_pressure(ds, ::Type{FT}) where {FT}
    haskey(ds, "sp") && return read_surface(ds, "sp", FT)
    haskey(ds, "lnsp") && return exp.(read_surface(ds, "lnsp", FT))
    return error("No surface pressure: expected `sp` or `lnsp` in the input.")
end

"""
    hydrostatic_heights(t, q, sp, phi_sfc, a_half, b_half, grav)

Geopotential height [m] at every model level, on a `(lon, lat, lev)` grid
ordered surface first. Integrates upward from the surface over the interface
pressures `a_half[k] + b_half[k] * sp`, using moist virtual temperature and
the ECMWF alpha weighting.
"""
function hydrostatic_heights(t, q, sp, phi_sfc, a_half, b_half, grav::FT) where {FT}
    R_d = FT(287.06)
    q_factor = FT(0.609133)
    # Stands in for p = 0 in the layer next to the model top
    p_top = FT(0.1)

    (nx, ny, nlev) = size(t)
    z = Array{FT}(undef, nx, ny, nlev)
    p_half = Vector{FT}(undef, nlev + 1)
    @inbounds for j in 1:ny, i in 1:nx
        for k in 1:(nlev + 1)
            p_half[k] = a_half[k] + b_half[k] * sp[i, j]
        end
        phi = phi_sfc[i, j]
        for k in 1:nlev
            p_below, p_above = p_half[k], p_half[k + 1]
            rt = R_d * t[i, j, k] * (FT(1) + q_factor * q[i, j, k])
            if k == nlev
                dlog_p, alpha = log(p_below / p_top), log(FT(2))
            else
                dlog_p = log(p_below / p_above)
                alpha = FT(1) - (p_above / (p_below - p_above)) * dlog_p
            end
            z[i, j, k] = (phi + rt * alpha) / grav
            phi += rt * dlog_p
        end
    end
    return z
end

"""
    interpz_3d_to_3d(z_target, z_source, f_source)

`f_source` interpolated column by column from `z_source` onto `z_target`, all
`(lon, lat, level)`. Linear in z, flat outside the source column. Unlike
[`interpz_3d`](@ref), the source heights vary per column.
"""
function interpz_3d_to_3d(z_target, z_source, f_source)
    (nx, ny, _) = size(z_source)
    @assert size(f_source) == size(z_source)
    @assert size(z_target)[1:2] == (nx, ny)
    nz = size(z_target, 3)
    out = Array{eltype(f_source)}(undef, nx, ny, nz)
    column = Vector{eltype(f_source)}(undef, nz)
    @inbounds for j in 1:ny, i in 1:nx
        z_col, f_col = view(z_source, i, j, :), view(f_source, i, j, :)
        z_out = view(z_target, i, j, :)
        if z_col[1] > z_col[end]
            interpolate1d!(column, reverse(z_col), z_out, reverse(f_col), Linear(), Flat())
        else
            interpolate1d!(column, z_col, z_out, f_col, Linear(), Flat())
        end
        out[i, j, :] .= column
    end
    return out
end

"""
    clean_era5_attrib(var)

Attributes of `var` without the ones describing how the source was encoded.
"""
function clean_era5_attrib(var)
    dropped =
        ("_FillValue", "missing_value", "scale_factor", "add_offset", "coordinates")
    return Dict(String(k) => v for (k, v) in var.attrib if !(String(k) in dropped))
end

"""
    to_z_levels_3d_model(source_file, target_file, target_levels, FT; interp_w = false)

Interpolate an ERA5 model-level file onto `target_levels` and write the result,
the file `WeatherModel` reads.

The vertical coordinate is geopotential height from
[`hydrostatic_heights`](@ref). Pressure comes from the same hybrid
coefficients and is written as `p_3d`, interpolated in `log(p)`.

The ERA5 horizontal grid is kept, with latitude sorted increasing.
`SpaceVaryingInput` regrids horizontally when the file is read, so pick
`target_levels` to match the model grid.

Writes `u`, `v`, `t`, `q`, `w`, `p_3d`, the surface fields `skt`, `p`, and
`z_sfc` broadcast over z, and any of `crwc`, `cswc`, `clwc`, `ciwc` the source
carries.

# Arguments

  - `source_file`: an `era5_raw_*.nc` on ERA5 model levels, holding `u`, `v`,
    `t`, `q`, `skt`, `sp`, and `surface_geopotential`.
  - `target_file`: the file to create, overwriting any existing one.
  - `target_levels`: target altitudes [m].
  - `FT`: floating point type of the output.

# Keyword Arguments

  - `interp_w = false`: write `w = 0` when `false`. When `true`, convert the
    ERA5 pressure velocity with `w = -omega * R_d * T / (p * g)`, tapered to 0
    between 100 and 10 hPa. Files from `ClimaInitialConditions.ERA5` carry no `w`,
    giving `w = 0` either way.
"""
function to_z_levels_3d_model(
    source_file,
    target_file,
    target_levels,
    ::Type{FT};
    interp_w::Bool = false,
) where {FT}
    grav = FT(TD.Parameters.grav(TD.Parameters.ThermodynamicsParameters(FT)))
    z_t = FT.(collect(target_levels))
    nz = length(z_t)

    NCDataset(source_file) do ncin
        lev_name = era5_model_level_name(ncin)
        for name in ("u", "v", "t", "q", "skt", "sp", "surface_geopotential")
            haskey(ncin, name) || error(
                "Missing required variable $(name) in $(source_file). A " *
                "model-level download needs the single-level fields merged in " *
                "so that sp, skt, and surface_geopotential are present.",
            )
        end

        lon = FT.(Array(ncin["longitude"]))
        lat = FT.(Array(ncin["latitude"]))
        lev = FT.(Array(ncin[lev_name]))
        (nx, ny, nlev) = (length(lon), length(lat), length(lev))

        read3d = name -> read_reordered(ncin, name, FT; lev_len = nlev)[:, :, :, 1]
        (t_src, q_src, u_src, v_src) = map(read3d, ("t", "q", "u", "v"))
        sp_src = read_surface_pressure(ncin, FT)
        phi_sfc = read_surface(ncin, "surface_geopotential", FT)

        # ERA5 puts level 1 at the model top; the integration wants the surface first
        if nlev >= 2 && lev[1] < lev[end]
            (t_src, q_src, u_src, v_src) =
                map(a -> reverse(a; dims = 3), (t_src, q_src, u_src, v_src))
        end

        a_half, b_half = era5_hybrid_coeffs(ncin, nlev, FT)
        z_src = hydrostatic_heights(t_src, q_src, sp_src, phi_sfc, a_half, b_half, grav)

        p_src = Array{FT}(undef, nx, ny, nlev)
        @inbounds for j in 1:ny, i in 1:nx
            p_prev = a_half[1] + b_half[1] * sp_src[i, j]
            for k in 1:nlev
                p_next = a_half[k + 1] + b_half[k + 1] * sp_src[i, j]
                p_src[i, j, k] = (p_prev + p_next) / FT(2)
                p_prev = p_next
            end
        end

        z_target = FT[z_t[k] for _ in 1:nx, _ in 1:ny, k in 1:nz]
        to_levels = field -> interpz_3d_to_3d(z_target, z_src, field)

        # SpaceVaryingInput wants increasing coordinates; ERA5 latitude decreases
        lat_order = sortperm(lat)
        sorted = a -> ndims(a) == 3 ? a[:, lat_order, :] : a[:, lat_order]

        NCDataset(target_file, "c", attrib = copy(ncin.attrib)) do ncout
            defDim(ncout, "lon", nx)
            defDim(ncout, "lat", ny)
            defDim(ncout, "z", nz)
            defVar(
                ncout,
                "lon",
                lon,
                ("lon",),
                attrib = Dict(
                    "standard_name" => "longitude", "units" => "degrees_east"),
            )
            defVar(
                ncout,
                "lat",
                lat[lat_order],
                ("lat",),
                attrib = Dict(
                    "standard_name" => "latitude", "units" => "degrees_north"),
            )
            defVar(
                ncout,
                "z",
                z_t,
                ("z",),
                attrib = Dict(
                    "standard_name" => "altitude", "long_name" => "altitude",
                    "units" => "m"),
            )

            t_out = to_levels(t_src)
            for (name, field) in
                (("u", u_src), ("v", v_src), ("t", t_src), ("q", q_src))
                out = name == "t" ? t_out : to_levels(field)
                # Interpolation in z can undershoot into negative humidity
                name == "q" && (out = max.(out, FT(0)))
                defVar(ncout, name, sorted(out), ("lon", "lat", "z"),
                    attrib = clean_era5_attrib(ncin[name]))
            end

            # log(p) because pressure is close to exponential in z
            p_out = exp.(to_levels(log.(p_src)))
            defVar(
                ncout,
                "p_3d",
                sorted(p_out),
                ("lon", "lat", "z"),
                attrib = Dict(
                    "standard_name" => "air_pressure",
                    "long_name" => "air pressure on the target levels",
                    "units" => "Pa",
                    "source" =>
                        "ERA5 model levels via hybrid coefficients, " *
                        "interpolated in log(p) against z",
                ),
            )

            w_out = if interp_w && haskey(ncin, "w")
                omega = to_levels(read3d("w"))
                w = @. -omega * FT(287.06) * t_out / (p_out * grav)
                # Hydrostatic omega stops being useful near the model top
                @. w * clamp((p_out - FT(1000)) / FT(9000), FT(0), FT(1))
            else
                zeros(FT, nx, ny, nz)
            end
            defVar(
                ncout,
                "w",
                sorted(w_out),
                ("lon", "lat", "z"),
                attrib = Dict(
                    "standard_name" => "upward_air_velocity",
                    "long_name" => "geometric vertical velocity",
                    "units" => "m s-1",
                ),
            )

            # The reader does not take 2D fields yet, so broadcast over z
            for (src_name, dst_name) in
                (("skt", "skt"), ("sp", "p"), ("surface_geopotential", "z_sfc"))
                field = src_name == "sp" ? sp_src : read_surface(ncin, src_name, FT)
                dst_name == "z_sfc" && (field = field ./ grav)
                attrib =
                    dst_name == "z_sfc" ?
                    Dict("standard_name" => "surface_altitude",
                        "long_name" => "surface altitude from ERA5",
                        "units" => "m", "source_variable" => src_name) :
                    clean_era5_attrib(ncin[src_name])
                broadcast_z = FT[field[i, j] for i in 1:nx, j in 1:ny, _ in 1:nz]
                defVar(ncout, dst_name, sorted(broadcast_z), ("lon", "lat", "z"),
                    attrib = attrib)
            end

            for name in ("crwc", "cswc", "clwc", "ciwc")
                haskey(ncin, name) || continue
                field = read3d(name)
                nlev >= 2 && lev[1] < lev[end] && (field = reverse(field; dims = 3))
                defVar(ncout, name, sorted(to_levels(field)), ("lon", "lat", "z"),
                    attrib = clean_era5_attrib(ncin[name]))
            end
        end
    end
    return target_file
end
