import ClimaComms
ClimaComms.@import_required_backends

using NCDatasets
using Dates
using Statistics

import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Geometry as Geometry
import ClimaCore: to_cpu

include("../gw_plotutils.jl")

const FT = Float64

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

    gw_Bn = @. ifelse(dϕ_s <= lat <= dϕ_n, FT(0), Bn)
    gw_cw = @. ifelse(dϕ_s <= lat <= dϕ_n, cw_tropics, cw)
    gw_flag = zeros(axes(lat))

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

# Generate domain, space and field for single column computation
column_domain = ClimaCore.Domains.IntervalDomain(
    ClimaCore.Geometry.ZPoint(FT(z[1, 1, end, 1])) ..
    ClimaCore.Geometry.ZPoint(FT(z[1, 1, 1, 1])),
    boundary_names = (:bottom, :top),
)

column_mesh = ClimaCore.Meshes.IntervalMesh(column_domain, nelems = 40)

# Construct the face space from the center one
column_face_space = ClimaCore.Spaces.FaceFiniteDifferenceSpace(column_mesh)
column_center_space =
    Spaces.CenterFiniteDifferenceSpace(column_face_space)

coordinate = Fields.coordinate_field(column_center_space)

gw_ncval = Val(333)
ᶜz = coordinate.z
ᶜρ = copy(ᶜz)
ᶜu = copy(ᶜz)
ᶜv = copy(ᶜz)
ᶜbf = copy(ᶜz)
ᶜlevel = similar(ᶜρ, FT)
u_waveforcing = similar(ᶜu)
v_waveforcing = similar(ᶜu)
for i in 1:Spaces.nlevels(axes(ᶜρ))
    fill!(Fields.level(ᶜlevel, i), i)
end
ᶜuforcing = similar(ᶜρ, FT)
ᶜvforcing = similar(ᶜρ, FT)

scratch = (;
    ᶜtemp_scalar = similar(ᶜz, FT),
    ᶜtemp_scalar_2 = similar(ᶜz, FT),
    ᶜtemp_scalar_3 = similar(ᶜz, FT),
    ᶜtemp_scalar_4 = similar(ᶜz, FT),
    ᶜtemp_scalar_5 = similar(ᶜz, FT),
    temp_field_level = similar(Fields.level(ᶜz, 1), FT),
)

for j in 1:length(lat)
    local non_orographic_gravity_wave = non_orographic_gravity_wave_param(lat[j], FT)
    # Create input parameters at each level
    local params_nogw = (; non_orographic_gravity_wave, scratch)
    for i in 1:length(lon)
        for it in 1:length(time)
            local source_level =
                kmax + 2 - Int(
                    floor(
                        (kmax + 1) -
                        ((kmax + 1 - k_source) * cosd(lat[j]) + 0.5),
                    ),
                )
            local damp_level = length(pfull) + 1 - k_damp
            parent(ᶜz) .= z[i, j, end:-1:1, it]
            parent(ᶜρ) .= ρ[i, j, end:-1:1, it]
            parent(ᶜu) .= u[i, j, end:-1:1, it]
            parent(ᶜv) .= v[i, j, end:-1:1, it]
            parent(ᶜbf) .= bf[i, j, end:-1:1, it]
            ᶜρ_source = Fields.level(ᶜρ, source_level)
            ᶜu_source = Fields.level(ᶜu, source_level)
            ᶜv_source = Fields.level(ᶜv, source_level)
            ᶜuforcing .= 0
            ᶜvforcing .= 0

            CA.non_orographic_gravity_wave_forcing(
                ᶜu,
                ᶜv,
                ᶜbf,
                ᶜρ,
                ᶜz,
                ᶜlevel,
                source_level,
                damp_level,
                ᶜρ_source,
                ᶜu_source,
                ᶜv_source,
                ᶜuforcing,
                ᶜvforcing,
                gw_ncval,
                u_waveforcing,
                v_waveforcing,
                params_nogw,
            )

            uforcing[i, j, :, it] = parent(ᶜuforcing)
            vforcing[i, j, :, it] = parent(ᶜvforcing)
        end
    end
end
uforcing_zonalave = dropdims(mean(uforcing, dims = 1), dims = 1)
vforcing_zonalave = dropdims(mean(vforcing, dims = 1), dims = 1)

# Plots
gwfu_zonalave = dropdims(mean(gwfu_cgwd, dims = 1), dims = 1)
gwfv_zonalave = dropdims(mean(gwfv_cgwd, dims = 1), dims = 1)

ENV["GKSwstype"] = "nul"
output_dir = "nonorographic_gravity_wave_test_mima"
mkpath(output_dir)

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
