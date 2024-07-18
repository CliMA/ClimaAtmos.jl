using ClimaComms
@static pkgversion(ClimaComms) >= v"0.6" && ClimaComms.@import_required_backends
using NCDatasets
using Dates
using Interpolations
using Statistics
import ClimaAtmos
import ClimaAtmos as CA
const FT = Float64

include("../gw_plotutils.jl")

# compute the source parameters
function non_orographic_gravity_wave(
    lat,
    ::Type{FT};
    gw_source_pressure = FT(31500),
    gw_damp_pressure = FT(85),
    Bw = FT(0.4),
    Bn = FT(0),
    Bt_0 = FT(0.0043),
    Bt_n = FT(0.001),
    Bt_s = FT(0.001),
    Bt_eq = FT(0.0043),
    ϕ0_n = FT(15),
    ϕ0_s = FT(-15),
    dϕ_n = FT(10),
    dϕ_s = FT(-10),
    dc = FT(0.6),
    cmax = FT(99.6),
    c0 = FT(0),
    nk = Int(1),
    cw = FT(35.0),
    cw_tropics = FT(35.0),
    cn = FT(2.0),
) where {FT}

    nc = Int(floor(FT(2 * cmax / dc + 1)))
    c = [FT((n - 1) * dc - cmax) for n in 1:nc]

    # source amplitude following GFDL: smooth transition from SP to NP
    # source_ampl = @. Bt_0 +
    #    Bt_n * FT(0.5) * (FT(1) + tanh((lat - ϕ0_n) / dϕ_n)) +
    #    Bt_s * FT(0.5) * (FT(1) + tanh((lat - ϕ0_s) / dϕ_s))

    gw_Bn = @. ifelse(dϕ_s <= lat <= dϕ_n, FT(0), Bn)
    gw_cw = @. ifelse(dϕ_s <= lat <= dϕ_n, cw_tropics, cw)
    gw_flag = zeros(axes(lat))

    # source amplitude following MiMA: radical change between subtropics and the tropic
    one_half = FT(0.5)
    source_ampl = @. ifelse(
        (lat > ϕ0_n) | (lat < ϕ0_s),
        Bt_0 +
        Bt_n * one_half * (1 + tanh((lat - ϕ0_n) / dϕ_n)) +
        Bt_s * one_half * (1 + tanh((lat - ϕ0_s) / dϕ_s)),
        ifelse(
            dϕ_s <= lat <= dϕ_n,
            Bt_eq,
            ifelse(
                dϕ_n <= lat <= ϕ0_n,
                Bt_0 + (Bt_eq - Bt_0) / (ϕ0_n - dϕ_n) * (ϕ0_n - lat),
                Bt_0 + (Bt_eq - Bt_0) / (ϕ0_s - dϕ_s) * (ϕ0_s - lat),
            ),
        ),
    )

    return (;
        gw_source_pressure = gw_source_pressure,
        gw_damp_pressure = gw_damp_pressure,
        gw_source_ampl = source_ampl,
        gw_Bw = Bw,
        gw_Bn = gw_Bn,
        gw_c = c,
        gw_cw = gw_cw,
        gw_cn = cn,
        gw_c0 = c0,
        gw_flag = gw_flag,
        gw_nk = nk,
    )
end


# MiMA data
include(joinpath(pkgdir(ClimaAtmos), "artifacts", "artifact_funcs.jl"))
mima_data = joinpath(mima_gwf_path(), "mima_gwf.nc")

nt = NCDataset(mima_data) do ds
    lon = Array(ds["lon"])
    lat = Array(ds["lat"])
    pfull = Array(ds["pfull"])
    phalf = Array(ds["phalf"])
    time = ds["time"][1:13]
    lev = ds["level"][:, :, :, 1:13]
    z = ds["hght"][:, :, :, 1:13]
    T = ds["temp"][:, :, :, 1:13]
    u = ds["ucomp"][:, :, :, 1:13]
    v = ds["vcomp"][:, :, :, 1:13]
    q = ds["sphum"][:, :, :, 1:13]
    gwfu_cgwd = ds["gwfu_cgwd"][:, :, :, 1:13]
    gwfv_cgwd = ds["gwfv_cgwd"][:, :, :, 1:13]
    (; lon, lat, pfull, phalf, lev, time, z, T, u, v, q, gwfu_cgwd, gwfv_cgwd)
end
(; lon, lat, lev, time, pfull, phalf, z, T, u, v, q, gwfu_cgwd, gwfv_cgwd) = nt

# compute density and buoyancy frequency
R_d = 287.0
grav = 9.8
cp_d = 1004.0
eps = 0.622

ρ = lev ./ T / R_d ./ (1 .- q .+ q ./ eps)

dTdz = zeros(size(T))
@. dTdz[:, :, 1, :] =
    (T[:, :, 2, :] - T[:, :, 1, :]) / (z[:, :, 2, :] - z[:, :, 1, :])
@. dTdz[:, :, end, :] =
    (T[:, :, end, :] - T[:, :, end - 1, :]) /
    (z[:, :, end, :] - z[:, :, end - 1, :])
@. dTdz[:, :, 2:(end - 1), :] =
    (T[:, :, 3:end, :] - T[:, :, 1:(end - 2), :]) /
    (z[:, :, 3:end, :] - z[:, :, 1:(end - 2), :])
bf = @. (grav / T) * (dTdz + grav / cp_d)
bf = @. ifelse(bf < 2.5e-5, sqrt(2.5e-5), sqrt(abs(bf)))

# compute u/v forcings from convective gravity waves
params = non_orographic_gravity_wave(lat, FT)
B0 = similar(params.gw_c)
# nogw forcing

kmax = length(pfull) - 1
k_source = findfirst(pfull * 100 .> params.gw_source_pressure)
k_damp = findlast(pfull * 100 .< params.gw_damp_pressure)

uforcing = zeros(size(lev))
vforcing = zeros(size(lev))
for i in 1:length(lon)
    for j in 1:length(lat)
        for it in 1:length(time)
            source_level =
                kmax + 2 - Int(
                    floor(
                        (kmax + 1) -
                        ((kmax + 1 - k_source) * cosd(lat[j]) + 0.5),
                    ),
                )
            damp_level = length(pfull) + 1 - k_damp

            uforcing[i, j, :, it] = CA.non_orographic_gravity_wave_forcing(
                u[i, j, end:-1:1, it],
                bf[i, j, end:-1:1, it],
                ρ[i, j, end:-1:1, it],
                z[i, j, end:-1:1, it],
                source_level,
                damp_level,
                params.gw_source_ampl[j],
                params.gw_Bw,
                params.gw_Bn[j],
                B0,
                params.gw_cw[j],
                params.gw_cn,
                params.gw_flag[j],
                params.gw_c,
                params.gw_c0,
                params.gw_nk,
            )
        end
    end
end
uforcing_zonalave = dropdims(mean(uforcing, dims = 1), dims = 1)

for i in 1:length(lon)
    for j in 1:length(lat)
        for it in 1:length(time)
            source_level =
                kmax + 2 - Int(
                    floor(
                        (kmax + 1) -
                        ((kmax + 1 - k_source) * cosd(lat[j]) + 0.5),
                    ),
                )
            damp_level = length(pfull) + 1 - k_damp

            vforcing[i, j, :, it] = CA.non_orographic_gravity_wave_forcing(
                v[i, j, end:-1:1, it],
                bf[i, j, end:-1:1, it],
                ρ[i, j, end:-1:1, it],
                z[i, j, end:-1:1, it],
                source_level,
                damp_level,
                params.gw_source_ampl[j],
                params.gw_Bw,
                params.gw_Bn[j],
                B0,
                params.gw_cw[j],
                params.gw_cn,
                params.gw_flag[j],
                params.gw_c,
                params.gw_c0,
                params.gw_nk,
            )
        end
    end
end
vforcing_zonalave = dropdims(mean(vforcing, dims = 1), dims = 1)

# plots
gwfu_zonalave = dropdims(mean(gwfu_cgwd, dims = 1), dims = 1)
gwfv_zonalave = dropdims(mean(gwfv_cgwd, dims = 1), dims = 1)



ENV["GKSwstype"] = "nul"
output_dir = "nonorographic_gravity_wave_test_mima"
mkpath(output_dir)

for it in 1:length(time)
    # Generate empty figure
    fig = generate_empty_figure()

    # Generic axis properties
    yreversed = true
    yscale = log10

    # Generic plot properties
    colormap = :balance
    extendlow = :cyan
    extendhigh = :magenta

    # Populate figure grid
    title = "gwfu clima (m/s/day)"
    create_plot!(
        fig;
        X = lat,
        Y = pfull,
        Z = uforcing_zonalave[:, end:-1:1, it] * 86400,
        title,
        p_loc = (1, 1),
        levels = range(-3, 3; length = 20),
    )

    title = "gwfu mima (m/s/day)"
    create_plot!(
        fig;
        X = lat,
        Y = pfull,
        Z = gwfu_zonalave[:, :, it] * 86400,
        levels = range(-3, 3; length = 20),
        title,
        p_loc = (1, 2),
        colormap,
        extendhigh,
        extendlow,
    )

    title = "gwfu (clima - mima) (%)"
    create_plot!(
        fig;
        X = lat,
        Y = pfull,
        Z = (uforcing_zonalave[:, end:-1:1, it] .- gwfu_zonalave[:, :, it]) ./
            gwfu_zonalave[:, :, it] .* 100,
        levels = range(-100, 100; length = 20),
        title,
        p_loc = (1, 3),
    )

    title = "gwfv clima (m/s/day)"
    create_plot!(
        fig;
        X = lat,
        Y = pfull,
        Z = vforcing_zonalave[:, end:-1:1, it] * 86400,
        levels = range(-3, 3; length = 20),
        title,
        p_loc = (2, 1),
    )

    title = "gwfv mima (m/s/day)"
    create_plot!(
        fig;
        X = lat,
        Y = pfull,
        Z = gwfv_zonalave[:, :, it] * 86400,
        levels = range(-3, 3; length = 20),
        title,
        p_loc = (2, 2),
    )

    title = "gwfv (clima-mima) (%)"
    create_plot!(
        fig;
        X = lat,
        Y = pfull,
        Z = (vforcing_zonalave[:, end:-1:1, it] .- gwfv_zonalave[:, :, it]) ./
            gwfv_zonalave[:, :, it] .* 100,
        levels = range(-100, 100; length = 20),
        title,
        p_loc = (2, 3),
    )

    # Save to disk
    CairoMakie.save(joinpath(output_dir, "$it.png"), fig)
end
