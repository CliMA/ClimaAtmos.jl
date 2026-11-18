import ClimaComms
ClimaComms.@import_required_backends

using NCDatasets
using Dates
using Statistics
using Test

import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore: to_cpu

include("../gw_plotutils.jl")

const FT = Float64

# Limits of the numeric gate at the end of this file (see `mima_agreement`). The test
# feeds the kernel MiMA's own column heights (`mima_column_space`) and MiMA's uniform
# source amplitude. Over the 13 months, this test gives zonal
# correlations 0.981-0.988, meridional correlations 0.64-0.76, and zonal RMS ratios
# (ClimaAtmos over MiMA) 0.64-0.68. The ratio is below one because of the model top
# alone. ClimaAtmos spreads the flux that escapes the top as one momentum-conserving
# acceleration over the mass of the damping layer (p < gw_damp_pressure), while MiMA
# adds the drag of the top layer, divided by the number of damping levels, to each of
# them; with MiMA's rule for the top, the same deposit gives ratios 1.02-1.07 (an
# offline NumPy port of both kernels). Split by height, the RMS ratio is 0.99 between
# 100 and 10 hPa, 1.00 between 10 and 1 hPa, 1.12 below 100 hPa, and 0.66 in the three
# damping-layer cells, whose large drag dominates the RMS over the whole domain. The
# limits leave a margin of 0.03 on the zonal correlation, 0.09 on the meridional one,
# and about 15% on the ratio; a change to the AD99 kernel that moves the drag shows up
# as a drop in correlation or a shift of the ratio.
const MIMA_MIN_CORR_U = 0.95
const MIMA_MIN_CORR_V = 0.55
const MIMA_RATIO_RANGE_U = (0.55, 0.8)

# Set up ClimaAtmos infrastructure
comms_ctx = ClimaComms.SingletonCommsContext()
@show ClimaComms.device(comms_ctx)

config_file = joinpath(
    @__DIR__,
    "../../../../config/model_configs/sphere_nogw_mima_test.yml",
)
config = CA.AtmosConfig(config_file; job_id = "nogw_mima_test", comms_ctx)

# Build simulation
simulation = CA.get_simulation(config)
p = simulation.integrator.p
Y = simulation.integrator.u

# Extract physical constants from parameters
(; params) = p
grav = CAP.grav(params)
R_d = CAP.R_d(params)
cp_d = CAP.cp_d(params)

# Extract NOGW parameters from simulation
nogw_params = CAP.non_orographic_gravity_wave_params(params)
gw_source_pressure = nogw_params.source_pressure
gw_damp_pressure = nogw_params.damp_pressure

# MiMA data
include(joinpath(@__DIR__, "../../../artifact_funcs.jl"))
mima_data = mima_gwf_path()

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

# Compute density and buoyancy frequency
eps = FT(0.622)

ρ = lev ./ T / R_d ./ (FT(1) .- q .+ q ./ eps)

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
bf = @. ifelse(bf < FT(2.5e-5), sqrt(FT(2.5e-5)), sqrt(abs(bf)))

# Compute the source parameters using latitude-dependent functions (matching MiMA)
function non_orographic_gravity_wave_param(
    lat,
    ::Type{FT};
    gw_source_pressure = FT(31500),
    gw_damp_pressure = FT(85),
    Bw = FT(0.4),
    Bn = FT(0),
    Bt_0 = FT(0.0043),
    # No extra amplitude poleward of ϕ0: the drag in the MiMA file matches a uniform
    # source amplitude of Bt_0 = Bt_eq = 0.0043 at all latitudes.
    Bt_n = FT(0),
    Bt_s = FT(0),
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

    gw_Bn = @. ifelse(dϕ_s <= lat <= dϕ_n, FT(0), Bn)
    gw_cw = @. ifelse(dϕ_s <= lat <= dϕ_n, cw_tropics, cw)

    # Source amplitude following MiMA: radical change between subtropics and the tropic
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
        gw_flag = 0,
        gw_nk = nk,
    )
end

param = non_orographic_gravity_wave_param(lat, FT)

# nogw forcing setup
kmax = length(pfull) - 1
k_source = findfirst(pfull * 100 .> param.gw_source_pressure)
k_damp = findlast(pfull * 100 .< param.gw_damp_pressure)

uforcing = zeros(size(lev))
vforcing = zeros(size(lev))

# One single-column space per MiMA column, built from that column's own full-level
# heights: the cell faces lie midway between neighbouring full levels, and the two outer
# faces mirror the nearest interior face about the outermost full level. The kernel
# deposits the drag into the cell masses ρ·Δz of this space, so they must be the
# column's own; a mesh that does not follow the file's heights changes the deposited
# drag (see the gate constants above).
function mima_column_space(z_full)   # heights bottom to top
    n = length(z_full)
    z_faces = similar(z_full, n + 1)
    z_faces[2:n] .= (z_full[1:(n - 1)] .+ z_full[2:n]) ./ 2
    z_faces[1] = 2 * z_full[1] - z_faces[2]
    z_faces[n + 1] = 2 * z_full[n] - z_faces[n]
    domain = ClimaCore.Domains.IntervalDomain(
        ClimaCore.Geometry.ZPoint(z_faces[1]) ..
        ClimaCore.Geometry.ZPoint(z_faces[end]),
        boundary_names = (:bottom, :top),
    )
    mesh = ClimaCore.Meshes.IntervalMesh(domain, ClimaCore.Geometry.ZPoint.(z_faces))
    face_space = Spaces.FaceFiniteDifferenceSpace(ClimaComms.device(), mesh)
    return Spaces.CenterFiniteDifferenceSpace(face_space)
end

gw_ncval = Val(333)

# The AD99 drag of one MiMA column (inputs bottom to top), with the source amplitude
# and spectrum parameters of its latitude.
function mima_column_forcing(
    z_full,
    ρ_full,
    u_full,
    v_full,
    bf_full,
    lat_param,
    source_level,
    damp_level,
)
    space = mima_column_space(z_full)
    ᶜz = copy(Fields.coordinate_field(space).z)
    parent(ᶜz) .= z_full
    ᶜρ = copy(ᶜz)
    ᶜu = copy(ᶜz)
    ᶜv = copy(ᶜz)
    ᶜbf = copy(ᶜz)
    parent(ᶜρ) .= ρ_full
    parent(ᶜu) .= u_full
    parent(ᶜv) .= v_full
    parent(ᶜbf) .= bf_full
    ᶜlevel = similar(ᶜρ, FT)
    for k in 1:Spaces.nlevels(space)
        fill!(Fields.level(ᶜlevel, k), k)
    end
    ᶜuforcing = fill!(similar(ᶜz, FT), FT(0))
    ᶜvforcing = fill!(similar(ᶜz, FT), FT(0))
    level_field() = fill!(similar(Fields.level(ᶜz, 1), FT), FT(0))
    scratch = (;
        ᶜtemp_scalar = similar(ᶜz, FT),
        ᶜtemp_scalar_2 = similar(ᶜz, FT),
        ᶜtemp_scalar_3 = similar(ᶜz, FT),
        ᶜtemp_scalar_4 = similar(ᶜz, FT),
        ᶜtemp_scalar_5 = similar(ᶜz, FT),
        ᶜtemp_scalar_6 = similar(ᶜz, FT),
        temp_field_level = similar(Fields.level(ᶜz, 1), FT),
    )
    non_orographic_gravity_wave = (;
        lat_param...,
        gw_temp_surface = similar(Fields.level(ᶜz, 1), FT),
        gw_beres_active = level_field(),
        gw_Q0 = level_field(),
        gw_h_heat = level_field(),
        gw_u_heat = level_field(),
        gw_v_heat = level_field(),
        gw_N_source = level_field(),
        gw_a_cover = level_field(),
        gw_beres_source = nothing,
        beres_source_ρ_z_u_v_level = similar(
            Fields.level(ᶜz, 1),
            Tuple{FT, FT, FT, FT, FT},
        ),
    )
    CA.non_orographic_gravity_wave_forcing(
        ᶜu,
        ᶜv,
        ᶜbf,
        ᶜρ,
        ᶜz,
        ᶜlevel,
        source_level,
        damp_level,
        Fields.level(ᶜρ, source_level),
        Fields.level(ᶜu, source_level),
        Fields.level(ᶜv, source_level),
        ᶜuforcing,
        ᶜvforcing,
        gw_ncval,
        similar(ᶜu),
        similar(ᶜu),
        (; non_orographic_gravity_wave, scratch),
    )
    return vec(parent(ᶜuforcing)), vec(parent(ᶜvforcing))
end

for j in eachindex(lat)
    local lat_param = non_orographic_gravity_wave_param(lat[j], FT)
    local source_level =
        kmax + 2 -
        Int(floor((kmax + 1) - ((kmax + 1 - k_source) * cosd(lat[j]) + 0.5)))
    local damp_level = length(pfull) + 1 - k_damp
    for i in 1:length(lon)
        for it in 1:length(time)
            column(a) = FT.(a[i, j, end:-1:1, it])
            local (uf, vf) = mima_column_forcing(
                column(z),
                column(ρ),
                column(u),
                column(v),
                column(bf),
                lat_param,
                source_level,
                damp_level,
            )
            uforcing[i, j, :, it] = uf
            vforcing[i, j, :, it] = vf
        end
    end
end
uforcing_zonalave = dropdims(mean(uforcing, dims = 1), dims = 1)
vforcing_zonalave = dropdims(mean(vforcing, dims = 1), dims = 1)

# Plots
gwfu_zonalave = dropdims(mean(gwfu_cgwd, dims = 1), dims = 1)
gwfv_zonalave = dropdims(mean(gwfv_cgwd, dims = 1), dims = 1)

# Numeric gate against MiMA: for each month, the pattern correlation over latitude and
# pressure of the zonal-mean drag, and the ratio of its root-mean-square values
# (ClimaAtmos over MiMA). ClimaAtmos levels run bottom to top, MiMA's top to bottom.
# The limits and the reference run they come from are given at the top of this file.
function mima_agreement(clima_zonalave, mima_zonalave)
    clima = clima_zonalave[:, end:-1:1, :]
    mima = convert(Array{FT}, mima_zonalave)
    corr = [cor(vec(clima[:, :, it]), vec(mima[:, :, it])) for it in axes(mima, 3)]
    rms(x) = sqrt(mean(abs2, x))
    ratio = [rms(clima[:, :, it]) / rms(mima[:, :, it]) for it in axes(mima, 3)]
    return (; corr, ratio)
end
agreement_u = mima_agreement(uforcing_zonalave, gwfu_zonalave)
agreement_v = mima_agreement(vforcing_zonalave, gwfv_zonalave)
@info "Agreement with MiMA (minimum and maximum over the months)" extrema(
    agreement_u.corr,
) extrema(agreement_u.ratio) extrema(agreement_v.corr) extrema(agreement_v.ratio)

ENV["GKSwstype"] = "nul"
output_dir = "nonorographic_gravity_wave_test_mima"
mkpath(output_dir)

function percent_difference(clima, mima; floor_fraction = FT(1e-3))
    # NCDatasets hands back `Union{Missing, Float32}` arrays. Converting to `FT`
    # first keeps the arithmetic below in one type, and it throws loudly if the
    # MiMA file ever does carry masked points, rather than quietly turning them
    # into more non-finite plot input.
    reference = convert(Array{FT}, mima)
    # `floatmin` only matters for an all-zero MiMA slice, which would otherwise
    # put a zero floor back in place.
    denominator_floor =
        max(floor_fraction * maximum(abs, reference), floatmin(FT))
    return (clima .- reference) ./ max.(abs.(reference), denominator_floor) .*
           100
end

for it in 1:length(time)
    # Generate empty figure
    local fig = generate_empty_figure()

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
        Z = percent_difference(
            uforcing_zonalave[:, end:-1:1, it],
            gwfu_zonalave[:, :, it],
        ),
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
        Z = percent_difference(
            vforcing_zonalave[:, end:-1:1, it],
            gwfv_zonalave[:, :, it],
        ),
        levels = range(-100, 100; length = 20),
        title,
        p_loc = (2, 3),
    )

    # Save to disk
    CairoMakie.save(joinpath(output_dir, "$it.png"), fig)
end

# The gate runs after the plots, so a failure still leaves them for inspection.
@testset "AD99 drag agrees with MiMA" begin
    @test all(>(MIMA_MIN_CORR_U), agreement_u.corr)
    @test all(>(MIMA_MIN_CORR_V), agreement_v.corr)
    @test all(r -> MIMA_RATIO_RANGE_U[1] < r < MIMA_RATIO_RANGE_U[2], agreement_u.ratio)
end
