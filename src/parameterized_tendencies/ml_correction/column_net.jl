#####
##### Column tendency-corrector network: weights, featurizer, and forward pass.
#####
##### Everything here works on plain arrays laid out `(lon, lat, level, ...)` on a
##### regular latitude-longitude grid, so the same code runs on CPU `Array`s and
##### on `CuArray`s (no scalar indexing): horizontal neighbor operations are
##### written as shifted views, and the 1D convolutions as dense matrix products.
#####

import NCDatasets
import LinearAlgebra: mul!

"""
    ColumnNetMember

Weights and normalization of one ensemble member of the column corrector,
in the layout used by `column_net_forward!`.

Matrices are stored as `(in, out)` so that a layer is `mul!(y, x, W)` with
activations laid out `(columns × levels, channels)`.

# Fields

  - `inp_w`, `inp_b`: Input 1×1 convolution, `(fin, width)` and `(1, 1, width)`.
  - `conv_w`, `conv_b`: Dilated kernel-3 convolutions, `(width, width, 3, block)` and
    `(1, 1, width, block)`.
  - `mix_w`, `mix_b`: 1×1 mixing convolutions, `(width, width, block)` and
    `(1, 1, width, block)`.
  - `ctx_w`, `ctx_b`: Column-context linear layers, `(width, width, block)` and
    `(1, width, block)`.
  - `out_w`, `out_b`: Output 1×1 convolution, `(width, target)` and `(1, 1, target)`.
  - `feat_mu`, `feat_sd`: Feature normalization, `(1, 1, feature)`.
  - `ysd`: Output scale per target [target units per hour], `(1, 1, target)`.
  - `q_ref`: Reference specific humidity of the humidity output scale [kg/kg].
"""
struct ColumnNetMember{FT, A2, A3, A4}
    inp_w::A2
    inp_b::A3
    conv_w::A4
    conv_b::A4
    mix_w::A3
    mix_b::A4
    ctx_w::A3
    ctx_b::A3
    out_w::A2
    out_b::A3
    feat_mu::A3
    feat_sd::A3
    ysd::A3
    q_ref::FT
end

"""
    ColumnNet

An ensemble of column correctors, with the grid, level, and static fields it was
trained on. Loaded by `load_column_net`; evaluated by `column_net_predict`.

# Fields

  - `members`: Vector of `ColumnNetMember`.
  - `dilations`: Dilation of each convolution block [-].
  - `targets`: Predicted variables, e.g. `("t", "q")`.
  - `pressure`: Pressure levels, surface to top [Pa] (host `Vector`).
  - `lat`, `lon`: Cell-centered grid coordinates [degrees] (host `Vector`s).
  - `orography`, `land_fraction`: Static inputs, `(lon, lat)` [m], [-].
  - `q_taper`: Humidity below which the humidity output tapers to zero [kg/kg].
  - `edge`: Number of cells by which the above-ground mask is eroded [-].
  - `stencil`: Radius of the semi-local anomaly box, in cells [-].
  - `window`: Length of the training windows [s]; sets the cos-zenith phases.
"""
struct ColumnNet{FT, M, A2, V}
    members::M
    dilations::Vector{Int}
    targets::Vector{String}
    pressure::V
    lat::V
    lon::V
    orography::A2
    land_fraction::A2
    q_taper::FT
    edge::Int
    stencil::Int
    window::FT
end

"""
    column_net_grid(path, ::Type{FT})

Read the grid and level definition of an exported column net without loading the
weights: a `NamedTuple` with `pressure` (surface to top) [Pa], `lat`, `lon`
[degrees], `targets`, and `window` [s]. Cheap enough to call on every process.
"""
function column_net_grid(path, ::Type{FT}) where {FT}
    NCDatasets.NCDataset(path) do ds
        (;
            pressure = FT.(Array(ds["pressure"])),
            lat = FT.(Array(ds["lat"])),
            lon = FT.(Array(ds["lon"])),
            targets = String.(split(ds.attrib["targets"], ",")),
            window = FT(ds.attrib["window_hours"]) * FT(3600),
        )
    end
end

"""
    load_column_net(path, ::Type{FT}, ArrayType = Array)

Load a column-corrector ensemble exported by the training repository
(`tendency_corr/export_julia.py`) and move it to `ArrayType`.

NCDatasets reverses the dimension order of the file, so torch's `(out, in, k)`
convolution weights arrive as `(k, in, out)`; they are permuted here into the
`(in, out)` matrices that `column_net_forward!` multiplies by.
"""
function load_column_net(path, ::Type{FT}, ArrayType = Array) where {FT}
    NCDatasets.NCDataset(path) do ds
        to(a) = ArrayType(FT.(a))
        rd(name) = Array(ds[name])
        inp_w, inp_b = rd("inp_w"), rd("inp_b")            # (fin, width, m), (width, m)
        conv_w, conv_b = rd("conv_w"), rd("conv_b")        # (k, in, out, b, m), (width, b, m)
        mix_w, mix_b = rd("mix_w"), rd("mix_b")            # (in, out, b, m), (width, b, m)
        ctx_w, ctx_b = rd("ctx_w"), rd("ctx_b")
        out_w, out_b = rd("out_w"), rd("out_b")            # (width, target, m), (target, m)
        mu, sd, ysd, q_ref = rd("feat_mu"), rd("feat_sd"), rd("ysd"), rd("q_ref")
        width = size(inp_w, 2)
        nblock = size(conv_w, 4)
        ntarget = size(out_w, 2)
        nfeature = size(mu, 1)
        members = map(1:size(inp_w, 3)) do m
            ColumnNetMember(
                to(inp_w[:, :, m]),
                to(reshape(inp_b[:, m], 1, 1, width)),
                to(permutedims(conv_w[:, :, :, :, m], (2, 3, 1, 4))),
                to(reshape(conv_b[:, :, m], 1, 1, width, nblock)),
                to(mix_w[:, :, :, m]),
                to(reshape(mix_b[:, :, m], 1, 1, width, nblock)),
                to(ctx_w[:, :, :, m]),
                to(reshape(ctx_b[:, :, m], 1, width, nblock)),
                to(out_w[:, :, m]),
                to(reshape(out_b[:, m], 1, 1, ntarget)),
                to(reshape(mu[:, m], 1, 1, nfeature)),
                to(reshape(sd[:, m], 1, 1, nfeature)),
                to(reshape(ysd[:, m], 1, 1, ntarget)),
                FT(q_ref[m]),
            )
        end
        attr = ds.attrib
        ColumnNet(
            members,
            Int.(vec(collect(attr["dilations"]))),
            String.(split(attr["targets"], ",")),
            FT.(rd("pressure")),
            FT.(rd("lat")),
            FT.(rd("lon")),
            to(rd("orography")),
            to(rd("land_fraction")),
            FT(attr["q_taper"]),
            Int(attr["edge"]),
            Int(attr["stencil"]),
            FT(attr["window_hours"]) * FT(3600),
        )
    end
end

#####
##### Horizontal neighbor operations on (lon, lat, ...) arrays
#####

_trailing(a) = ntuple(_ -> Colon(), ndims(a) - 2)

"""
    roll_lon(a, s)

Return `a` rolled by `s` cells along the periodic longitude (first) dimension, so
that `roll_lon(a, s)[i, ...] == a[mod1(i - s, n), ...]` (`numpy.roll` semantics).
"""
function roll_lon(a, s)
    n = size(a, 1)
    s = mod(s, n)
    s == 0 && return copy(a)
    out = similar(a)
    rest = (Colon(), _trailing(a)...)
    view(out, (s + 1):n, rest...) .= view(a, 1:(n - s), rest...)
    view(out, 1:s, rest...) .= view(a, (n - s + 1):n, rest...)
    return out
end

"""
    shift_lat_nearest(a, s)

Return `a` sampled `s` cells along latitude (second dimension) with edge values
repeated, `shift_lat_nearest(a, s)[:, j, ...] == a[:, clamp(j + s, 1, n), ...]`
(`scipy.ndimage` `mode = "nearest"`).
"""
function shift_lat_nearest(a, s)
    n = size(a, 2)
    s = clamp(s, -n, n)
    s == 0 && return copy(a)
    out = similar(a)
    rest = _trailing(a)
    if s > 0
        view(out, :, 1:(n - s), rest...) .= view(a, :, (1 + s):n, rest...)
        view(out, :, (n - s + 1):n, rest...) .= view(a, :, n:n, rest...)
    else
        view(out, :, (1 - s):n, rest...) .= view(a, :, 1:(n + s), rest...)
        view(out, :, 1:(-s), rest...) .= view(a, :, 1:1, rest...)
    end
    return out
end

"""
    erode_mask(m, n)

Shrink the `true` region of the `(lon, lat, ...)` mask `m` by `n` cells: a cell
stays valid only if its four horizontal neighbors are valid (longitude periodic,
latitude edges checked against their single interior neighbor).
"""
function erode_mask(m, n)
    nlat = size(m, 2)
    rest = _trailing(m)
    for _ in 1:n
        e = m .& roll_lon(m, 1) .& roll_lon(m, -1)
        view(e, :, 2:nlat, rest...) .&= view(m, :, 1:(nlat - 1), rest...)
        view(e, :, 1:(nlat - 1), rest...) .&= view(m, :, 2:nlat, rest...)
        m = e
    end
    return m
end

"""
    box_sum(a, r)

Sum of `a` over the `(2r + 1) × (2r + 1)` horizontal box around each cell, with a
periodic longitude and repeated latitude edges.
"""
function box_sum(a, r)
    lon_sum = copy(a)
    for s in 1:r
        lon_sum .+= roll_lon(a, s) .+ roll_lon(a, -s)
    end
    out = copy(lon_sum)
    for s in 1:r
        out .+= shift_lat_nearest(lon_sum, s) .+ shift_lat_nearest(lon_sum, -s)
    end
    return out
end

"""
    neighbor_anomaly(x, m, r)

Semi-local horizontal anomaly: `x` minus its mask-weighted `(2r + 1)²`
neighborhood mean, zeroed where `m` is invalid. `x` is `(lon, lat, level, var)`
and `m` is `(lon, lat, level)`.
"""
function neighbor_anomaly(x, m, r)
    FT = eltype(x)
    r <= 0 && return zero(x)
    mf = FT.(m)
    den = box_sum(mf, r)
    num = box_sum(x .* mf, r)
    return (x .- num ./ max.(den, FT(1e-6))) .* mf
end

"""
    gaussian_smooth(x, m, σ)

Mask-weighted Gaussian smoothing of `x` (`(lon, lat, level, ...)`) over the
horizontal, with standard deviation `σ` in grid cells, a periodic longitude, and
repeated latitude edges; zero where `m` (`(lon, lat, level)`) is invalid. Uses
the `scipy.ndimage.gaussian_filter` kernel (truncated at 4σ).
"""
function gaussian_smooth(x, m, σ)
    FT = eltype(x)
    σ <= 0 && return x
    r = floor(Int, 4 * σ + FT(0.5))
    w = [exp(-FT(0.5) * (s / FT(σ))^2) for s in (-r):r]
    w ./= sum(w)
    filt(a) = begin
        lon_f = w[r + 1] .* a
        for s in 1:r
            lon_f .+= w[r + 1 + s] .* (roll_lon(a, s) .+ roll_lon(a, -s))
        end
        out = w[r + 1] .* lon_f
        for s in 1:r
            out .+=
                w[r + 1 + s] .*
                (shift_lat_nearest(lon_f, s) .+ shift_lat_nearest(lon_f, -s))
        end
        out
    end
    mf = FT.(m) .* one.(x)
    num = filt(x .* mf)
    den = filt(mf)
    return ifelse.(mf .> 0, num ./ max.(den, FT(1e-6)), zero(FT))
end

#####
##### Features and forward pass
#####

"""
    approx_erf(x)

Error function, Abramowitz & Stegun 7.1.26 (absolute error below 1.5e-7). Used by
the exact (erf-based) GELU of the network without a SpecialFunctions dependency.
"""
@inline function approx_erf(x)
    FT = typeof(x)
    t = 1 / (1 + FT(0.3275911) * abs(x))
    y =
        1 -
        t *
        (
            FT(0.254829592) +
            t * (
                FT(-0.284496736) +
                t * (FT(1.421413741) + t * (FT(-1.453152027) + t * FT(1.061405429)))
            )
        ) * exp(-x * x)
    return copysign(y, x)
end

@inline gelu_erf(x) = x * (1 + approx_erf(x / sqrt(typeof(x)(2)))) / 2

"""
    column_net_raw_features(net, x, ps, cz, m)

Return the unnormalized level-shared features `(lon, lat, level, feature)` of the
column state `x` (`(lon, lat, level, var)` with vars `t, u, v, q`) [K, m/s, kg/kg],
the reference surface pressure `ps` (`(lon, lat)`) [Pa], the cos-zenith phases
`cz` (`(lon, lat, 3)`), and the valid mask `m` (`(lon, lat, level)`).

Feature order matches the training `Featurizer` (`use_p = false`, `use_ctx = true`):
`t, 1e3 q, log q, RH, u, v`, the semi-local anomaly of `t, u, v, 1e3 q`, then the
column scalars `ps / 1e5`, land fraction, orography [km], and the three cos-zenith
phases.
"""
function column_net_raw_features(net::ColumnNet, x, ps, cz, m)
    FT = eltype(x)
    nlon, nlat, L, _ = size(x)
    p = level_pressure(net, x)
    t = view(x, :, :, :, 1)
    u = view(x, :, :, :, 2)
    v = view(x, :, :, :, 3)
    q = max.(view(x, :, :, :, 4), zero(FT))
    e = @. q * p / (FT(0.622) + FT(0.378) * q)
    tc = @. t - FT(273.15)
    es = @. ifelse(
        t >= FT(273.15),
        FT(611.2) * exp(FT(17.67) * tc / (t - FT(29.65))),
        FT(611.2) * exp(FT(22.46) * tc / (t - FT(0.53))),
    )
    rh = @. clamp(e / es, zero(FT), FT(1.5))
    xnb = neighbor_anomaly(x, m, net.stencil)

    f = similar(x, nlon, nlat, L, 16)
    level_features = (
        t,
        q .* FT(1e3),
        log.(max.(q, FT(1e-7))),
        rh,
        u,
        v,
        view(xnb, :, :, :, 1),
        view(xnb, :, :, :, 2),
        view(xnb, :, :, :, 3),
        view(xnb, :, :, :, 4) .* FT(1e3),
    )
    for (i, a) in enumerate(level_features)
        view(f, :, :, :, i) .= a
    end
    column_features = (
        ps ./ FT(1e5),
        net.land_fraction,
        net.orography ./ FT(1e3),
        view(cz, :, :, 1),
        view(cz, :, :, 2),
        view(cz, :, :, 3),
    )
    for (i, a) in enumerate(column_features)
        view(f, :, :, :, length(level_features) + i) .= a
    end
    return f
end

"""
    column_net_forward!(y, member, f, mm, dilations, buffers)

Evaluate one ensemble member on a batch of columns.

`f` holds the normalized, masked features with the mask appended as the last
channel, `(column, level, fin)`; `mm` is the mask `(column, level)`; `y` receives the
normalized outputs `(column, level, target)`. Mirrors the training `ColNet`: a 1×1
input convolution, residual blocks of `h ← (h + mix(gelu(conv_d(h) + ctx(pool(h))))) · mask` with a dilated kernel-3 convolution and a mask-weighted column mean, and a 1×1
output convolution, all masked. `buffers` is a `NamedTuple` of `(column, level, width)` arrays `h`, `z`, `s` and `(column, 1, width)` array `pool`.
"""
function column_net_forward!(y, member, f, mm, dilations, buffers)
    FT = eltype(f)
    (; h, z, s, pool) = buffers
    N, L, W = size(h)
    flat(a) = reshape(a, N * L, size(a, 3))
    mm3 = reshape(mm, N, L, 1)
    den = max.(sum(mm3; dims = 2), one(FT))                 # (N, 1, 1)

    mul!(flat(h), flat(f), member.inp_w)
    @. h = (h + member.inp_b) * mm3
    for (b, d) in enumerate(dilations)
        @. s = h * mm3
        sum!(pool, s)
        pool ./= den
        ctx = reshape(pool, N, W) * view(member.ctx_w, :, :, b)  # (N, W)
        ctx .+= view(member.ctx_b, :, :, b)
        conv_b = view(member.conv_b, :, :, :, b)
        z .= conv_b .+ reshape(ctx, N, 1, W)
        for k in 1:3
            shift = (k - 2) * d
            fill!(s, zero(FT))
            if shift >= 0
                view(s, :, 1:(L - shift), :) .= view(h, :, (1 + shift):L, :)
            else
                view(s, :, (1 - shift):L, :) .= view(h, :, 1:(L + shift), :)
            end
            mul!(flat(z), flat(s), view(member.conv_w, :, :, k, b), true, true)
        end
        @. z = gelu_erf(z)
        mul!(flat(s), flat(z), view(member.mix_w, :, :, b))
        mix_b = view(member.mix_b, :, :, :, b)
        @. h = (h + s + mix_b) * mm3
    end
    mul!(flat(y), flat(h), member.out_w)
    @. y = (y + member.out_b) * mm3
    return y
end

"""
    column_net_mask(net, ps)

Valid-level mask `(lon, lat, level)`: levels at or above the reference surface
pressure `ps` [Pa], eroded by `net.edge` cells to drop terrain-edge columns whose
pressure-level values mix in below-ground extrapolation.
"""
function column_net_mask(net::ColumnNet, ps)
    return erode_mask(level_pressure(net, ps) .<= ps, net.edge)
end

"""
    level_pressure(net, like)

Return the pressure levels of `net` [Pa] as a `(1, 1, level)` array of the same
array type as `like`, for broadcasting against `(lon, lat, level)` arrays.
"""
function level_pressure(net::ColumnNet, like)
    L = length(net.pressure)
    p = copyto!(similar(like, L), net.pressure)
    return reshape(p, 1, 1, L)
end

"""
    column_net_predict(net, x, ps, cz; members = eachindex(net.members), chunk = 8192)

Ensemble-mean drift-rate prediction of `net` for a global state, and the valid
mask it was evaluated with.

`x` is `(lon, lat, level, var)` with vars `t, u, v, q` [K, m/s, kg/kg] on
`net.pressure`, `ps` the reference surface pressure `(lon, lat)` [Pa], and `cz`
the cos-zenith phases `(lon, lat, 3)`. Returns `(pred, m)`: `pred` is
`(lon, lat, level, target)` in target units per hour (zero where invalid), `m`
the `Bool` mask `(lon, lat, level)`. Columns are processed in batches of `chunk`
to bound the activation memory.
"""
function column_net_predict(
    net::ColumnNet,
    x,
    ps,
    cz;
    members = eachindex(net.members),
    chunk = 8192,
)
    FT = eltype(x)
    nlon, nlat, L, _ = size(x)
    N = nlon * nlat
    T = length(net.targets)
    W = size(first(net.members).inp_w, 2)
    m = column_net_mask(net, ps)
    raw = reshape(column_net_raw_features(net, x, ps, cz, m), N, L, :)
    mm = reshape(FT.(m), N, L)
    q = reshape(max.(view(x, :, :, :, 4), zero(FT)), N, L)
    iq = findfirst(==("q"), net.targets)

    pred = fill!(similar(x, N, L, T), zero(FT))
    nb = min(chunk, N)
    f = similar(x, nb, L, size(raw, 3) + 1)
    y = similar(x, nb, L, T)
    buffers = (;
        h = similar(x, nb, L, W),
        z = similar(x, nb, L, W),
        s = similar(x, nb, L, W),
        pool = similar(x, nb, 1, W),
    )
    for member in view(net.members, members)
        for start in 1:nb:N
            cols = start:min(start + nb - 1, N)
            n = length(cols)
            if n < nb
                fb = similar(x, n, L, size(f, 3))
                yb = similar(x, n, L, T)
                bb = map(a -> similar(a, n, size(a, 2), size(a, 3)), buffers)
            else
                fb, yb, bb = f, y, buffers
            end
            mmb = view(mm, cols, :)
            rawb = view(raw, cols, :, :)
            fv = view(fb, :, :, 1:size(raw, 3))
            @. fv = (rawb - member.feat_mu) / member.feat_sd * mmb
            view(fb, :, :, size(fb, 3)) .= mmb
            column_net_forward!(yb, member, fb, mmb, net.dilations, bb)
            yb .*= member.ysd
            if !isnothing(iq)
                qb = view(q, cols, :)
                scale = @. min(qb / member.q_ref, FT(5)) * qb^2 /
                           (qb^2 + net.q_taper^2)
                view(yb, :, :, iq) .*= scale
            end
            view(pred, cols, :, :) .+= yb
        end
    end
    pred ./= length(members)
    return reshape(pred, nlon, nlat, L, T), m
end

"""
    cos_zenith_noaa(date, lat, lon)

Cosine of the solar zenith angle `(lon, lat)` on the grid `lat`, `lon` [degrees] at
`date` (UTC), from the NOAA approximation used to build the training inputs.
"""
function cos_zenith_noaa(date::Dates.DateTime, lat, lon)
    hour =
        Dates.hour(date) + Dates.minute(date) / 60 + Dates.second(date) / 3600
    γ = 2π / 365 * (Dates.dayofyear(date) - 1 + (hour - 12) / 24)
    decl =
        0.006918 - 0.399912 * cos(γ) + 0.070257 * sin(γ) - 0.006758 * cos(2γ) +
        0.000907 * sin(2γ) - 0.002697 * cos(3γ) + 0.00148 * sin(3γ)
    eqtime =
        229.18 * (
            0.000075 + 0.001868 * cos(γ) - 0.032077 * sin(γ) -
            0.014615 * cos(2γ) - 0.040849 * sin(2γ)
        )
    FT = eltype(lat)
    return [
        FT(
            sind(φ) * sin(decl) +
            cosd(φ) * cos(decl) * cosd((hour * 60 + eqtime + 4λ) / 4 - 180),
        ) for λ in lon, φ in lat
    ]
end
