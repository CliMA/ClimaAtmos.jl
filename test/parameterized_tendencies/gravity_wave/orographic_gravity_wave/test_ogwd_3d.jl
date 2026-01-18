using ClimaCore:
    Fields, Geometry, Domains, Meshes, Topologies, Spaces, Operators, DataLayouts, Utilities
using ClimaCore
using ClimaCore.CommonSpaces
using NCDatasets
import ClimaAtmos
import ClimaAtmos as CA
import ClimaComms
import ClimaAtmos: AtmosArtifacts as AA
import ClimaUtilities: SpaceVaryingInputs.SpaceVaryingInput
using ClimaCoreTempestRemap

import Interpolations

using CUDA
using Dates

const FT = Float64
include(
    joinpath(pkgdir(ClimaAtmos), "post_processing/remap", "remap_helpers.jl"),
)
include("../gw_remap_plot_utils.jl")

comms_ctx = ClimaComms.SingletonCommsContext()
@show CUDA.functional()
@show ClimaComms.device(comms_ctx)

(; config_file, job_id) = CA.commandline_kwargs()
config = CA.AtmosConfig(config_file; job_id, comms_ctx)

config.parsed_args["topography"] = "Earth";
config.parsed_args["topo_smoothing"] = false;
config.parsed_args["mesh_warp_type"] = "Linear";
(; parsed_args) = config

# load gfdl data
include(joinpath(@__DIR__, "../../../artifact_funcs.jl"))
ncfile = joinpath(gfdl_ogw_data_path(), "gfdl_ogw.nc")
nt = NCDataset(ncfile) do ds
    lon = Array(ds["lon"])
    lat = Array(ds["lat"])
    z_full = Array(ds["z_full"])
    temp = Array(ds["temp"])
    ucomp = Array(ds["ucomp"])
    vcomp = Array(ds["vcomp"])
    udt_topo = Array(ds["udt_topo"])
    vdt_topo = Array(ds["vdt_topo"])
    sphum = Array(ds["sphum"])
    ps = Array(ds["ps"])
    pk = Array(ds["pk"])
    bk = Array(ds["bk"])
    (;
        lon,
        lat,
        z_full,
        temp,
        ucomp,
        vcomp,
        udt_topo,
        vdt_topo,
        sphum,
        ps,
        pk,
        bk,
    )
end
(;
    lon,
    lat,
    z_full,
    temp,
    ucomp,
    vcomp,
    udt_topo,
    vdt_topo,
    sphum,
    ps,
    pk,
    bk,
) = nt

# compute pressure
p_half = zeros((size(ps)[1], size(ps)[2], length(bk)))
for k in 1:size(bk)[1]
    p_half[:, :, k, :] = bk[k] * ps .+ pk[k]
end
p_center = 0.5 * (p_half[:, :, 1:(end - 1), :] .+ p_half[:, :, 2:end, :])

# Create meshes and spaces
h_elem = 16
nh_poly = 3
z_max = 42e3
z_elem = 33
dz_bottom = 300.0
radius = 6.371229e6

quad = Quadratures.GLL{nh_poly + 1}()
horizontal_mesh = CA.cubed_sphere_mesh(; radius, h_elem)
h_space = CA.make_horizontal_space(horizontal_mesh, quad, comms_ctx, false)
z_stretch = Meshes.HyperbolicTangentStretching(dz_bottom)
ᶜspace, ᶠspace =
    CA.make_hybrid_spaces(h_space, z_max, z_elem, z_stretch; parsed_args)

ᶜlocal_geometry = Fields.local_geometry_field(ᶜspace)
ᶠlocal_geometry = Fields.local_geometry_field(ᶠspace)

# interpolation functions
function ᶜinterp_latlon2cg(lon, lat, datain, ᶜlocal_geometry)
    # create interpolcation obj
    li_obj = []
    for k in 1:size(datain)[3]
        push!(
            li_obj,
            Interpolations.linear_interpolation(
                (lon, lat),
                datain[:, :, k],
                extrapolation_bc = (
                    Interpolations.Periodic(),
                    Interpolations.Flat(),
                ),
            ),
        )
    end

    # interpolate onto cg grid and return Fields.Field
    gfdl_data = Fields.Field(FT, axes(ᶜlocal_geometry))
    Fields.bycolumn(axes(ᶜlocal_geometry)) do colidx
        lg = ᶜlocal_geometry[colidx]
        lat = lg.coordinates.lat
        lon = lg.coordinates.long
        data_center = zeros(size(datain)[3])
        for k in 1:size(z_full)[3]
            data_center[k] = FT(li_obj[k](parent(lon)[k], parent(lat)[k]))
        end
        parent(gfdl_data[colidx]) .= data_center[end:-1:1]
    end
    return gfdl_data
end

function ᶜinterp2CAlevels(gfdl_z_full, gfdl_data, ᶜlocal_geometry)
    gfdl_ca_data = Fields.Field(FT, axes(ᶜlocal_geometry))
    Fields.bycolumn(axes(ᶜlocal_geometry)) do colidx
        li = Interpolations.linear_interpolation(
            parent(gfdl_z_full[colidx])[:],
            parent(gfdl_data[colidx])[:],
            extrapolation_bc = Interpolations.Line(),
        )
        parent(gfdl_ca_data[colidx]) .=
            li.(parent(ᶜlocal_geometry.coordinates.z[colidx]))
    end
    return gfdl_ca_data
end

# interpolate to cg grid and assign to Fields
gfdl_z_full = ᶜinterp_latlon2cg(lon, lat, z_full, ᶜlocal_geometry)
gfdl_temp = ᶜinterp_latlon2cg(lon, lat, temp, ᶜlocal_geometry)
gfdl_ucomp = ᶜinterp_latlon2cg(lon, lat, ucomp, ᶜlocal_geometry)
gfdl_vcomp = ᶜinterp_latlon2cg(lon, lat, vcomp, ᶜlocal_geometry)
gfdl_udt_topo = ᶜinterp_latlon2cg(lon, lat, udt_topo, ᶜlocal_geometry)
gfdl_vdt_topo = ᶜinterp_latlon2cg(lon, lat, vdt_topo, ᶜlocal_geometry)
gfdl_sphum = ᶜinterp_latlon2cg(lon, lat, sphum, ᶜlocal_geometry)
gfdl_p = ᶜinterp_latlon2cg(lon, lat, p_center, ᶜlocal_geometry)

# interpolate gfdl data (cg) onto our model levels
gfdl_ca_temp = ᶜinterp2CAlevels(gfdl_z_full, gfdl_temp, ᶜlocal_geometry)
gfdl_ca_ucomp = ᶜinterp2CAlevels(gfdl_z_full, gfdl_ucomp, ᶜlocal_geometry)
gfdl_ca_vcomp = ᶜinterp2CAlevels(gfdl_z_full, gfdl_vcomp, ᶜlocal_geometry)
gfdl_ca_udt_topo = ᶜinterp2CAlevels(gfdl_z_full, gfdl_udt_topo, ᶜlocal_geometry)
gfdl_ca_vdt_topo = ᶜinterp2CAlevels(gfdl_z_full, gfdl_vdt_topo, ᶜlocal_geometry)
gfdl_ca_sphum = ᶜinterp2CAlevels(gfdl_z_full, gfdl_sphum, ᶜlocal_geometry)
gfdl_ca_p = ᶜinterp2CAlevels(gfdl_z_full, gfdl_p, ᶜlocal_geometry)

# create Y
Yc = map(ᶜlocal_geometry) do lg
    return (;
        ρ = FT(1.0),
        uₕ = Geometry.Covariant12Vector(Geometry.UVVector(FT(0), FT(0)), lg),
        T = FT(0),
        qt = FT(0),
    )
end
Yc.uₕ .= Geometry.Covariant12Vector.(Geometry.UVVector.(gfdl_ca_ucomp, gfdl_ca_vcomp))
# Yc.uₕ.components.data.:2 .= Geometry.Covariant12Vector.(gfdl_ca_vcomp)
Yc.T .= gfdl_ca_temp
Yc.qt .= gfdl_ca_sphum
Yf = map(ᶠlocal_geometry) do lg
    return (; u₃ = Geometry.Covariant3Vector(FT(0), lg))
end
Y = Fields.FieldVector(c = Yc, f = Yf)

# compute density from temperature, humidiry, pressure
R_d = 287.0
grav = 9.8
cp_d = 1004.0
epsilon = 0.622
@. Y.c.ρ = gfdl_ca_p / Y.c.T / R_d / (1 - Y.c.qt + Y.c.qt / epsilon)

γ = 0.4
ϵ = 0.0
β = 0.5
h_frac = 0.1
ρscale = 1.2
L0 = 80000.0
a0 = 0.9
a1 = 3.0
Fr_crit = 0.7
topo_info = Val(:raw_topo)
topography = Val(:Earth)

ogw = CA.FullOrographicGravityWave{FT, typeof(topo_info), typeof(topography)}(;
    γ,
    ϵ,
    β,
    h_frac,
    ρscale,
    L0,
    a0,
    a1,
    Fr_crit,
    topo_info,
    topography,
)

topo_info = CA.get_topo_info(Y, ogw)

Y = ClimaCore.to_device(ClimaComms.CUDADevice(), copy(Y))

# pre-compute thermal vars
thermo_params = CA.TD.Parameters.ThermodynamicsParameters(FT)
# thermo_params = ClimaCore.to_device(ClimaComms.CUDADevice(), thermo_params)

ᶜT_cpu = gfdl_ca_temp
ᶜp_cpu = gfdl_ca_p

ᶜz = Fields.coordinate_field(Y.c).z

ᶜp = ClimaCore.to_device(ClimaComms.CUDADevice(), ᶜp_cpu)
ᶜT = ClimaCore.to_device(ClimaComms.CUDADevice(), ᶜT_cpu)

ᶜtarget_space = Spaces.axes(Y.c)
ᶜp = Fields.Field(Fields.field_values(ᶜp), ᶜtarget_space)
ᶜT = Fields.Field(Fields.field_values(ᶜT), ᶜtarget_space)

topo_info = CA.move_topo_info_to_gpu(topo_info, ᶜtarget_space)

ᶜts = similar(Y.c, CA.TD.PhaseEquil{FT})
ᶜts = @. CA.TD.PhaseEquil_ρpq(thermo_params, Y.c.ρ, ᶜp, Y.c.qt)
ᶜts = Fields.Field(Fields.field_values(ᶜts), ᶜtarget_space)

# emulate `p` from a ClimaAtmos run
atmos = (; turbconv_model = nothing)
atmos = cu(atmos)
# atmos = cu(ClimaComms.CUDADevice(), atmos)
p = (;
    orographic_gravity_wave = CA.orographic_gravity_wave_cache(Y, ogw, topo_info),
    scratch = CA.temporary_quantities(Y, atmos),
    precomputed = (; ᶜts = ᶜts, ᶜp = ᶜp, thermo_params = thermo_params),
    params = CA.ClimaAtmosParameters(config),
)

(; topo_ᶜz_pbl, topo_ᶠz_pbl, topo_τ_x, topo_τ_y, topo_τ_l, topo_τ_p, topo_τ_np) =
    p.orographic_gravity_wave
(; topo_ᶜτ_sat, topo_ᶠτ_sat) = p.orographic_gravity_wave
(; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) =
    p.orographic_gravity_wave
(; topo_ᶠVτ, values_at_z_pbl) =
    p.orographic_gravity_wave
(; ᶜdTdz) = p.orographic_gravity_wave
(; ᶜuforcing, ᶜvforcing) = p.orographic_gravity_wave

# Extract parameters
ogw_params = p.orographic_gravity_wave.ogw_params

CA.orographic_gravity_wave_compute_tendency!(Y, p, ogw)

# (; ᶜuforcing, ᶜvforcing) = p.orographic_gravity_wave

# Move GPU arrays back to CPU for plotting
uforcing_cpu = ClimaCore.to_cpu(ᶜuforcing)
vforcing_cpu = ClimaCore.to_cpu(ᶜvforcing)
gfdl_ca_udt_topo_cpu = ClimaCore.to_cpu(gfdl_ca_udt_topo)
gfdl_ca_vdt_topo_cpu = ClimaCore.to_cpu(gfdl_ca_vdt_topo)
ᶜz_cpu = ClimaCore.to_cpu(ᶜz)
Y_cpu = ClimaCore.to_cpu(Y)

##################
# plotting!!!!
##################
ENV["GKSwstype"] = "nul"
output_dir = "orographic_gravity_wave_test_3d"
mkpath(output_dir)


# Prepare field data dictionary
field_data = Dict(
    "ogwd_u" => uforcing_cpu,
    "ogwd_v" => vforcing_cpu,
    "gfdl_udt_topo" => gfdl_ca_udt_topo_cpu,
    "gfdl_vdt_topo" => gfdl_ca_vdt_topo_cpu,
    "z_3d" => ᶜz_cpu,
)

# Define all panel configurations
u_panels = [
    PlotPanel("ogwd_u", "climaatmos at z = {z}", (1, 1); scale_factor = 86400),
    PlotPanel("gfdl_udt_topo", "gfdl at z = {z}", (2, 1); scale_factor = 86400),
]

v_panels = [
    PlotPanel("ogwd_v", "climaatmos at z = {z}", (1, 1); scale_factor = 86400),
    PlotPanel("gfdl_vdt_topo", "gfdl at z = {z}", (2, 1); scale_factor = 86400),
]

# Configure plots
config = PlotConfig(
    vertical_levels = [21, 31],
    contour_levels = range(-10, 10; length = 20),
    nlat = 90,
    nlon = 180,
    yreversed = false,
)

# Generate all figures efficiently (remaps only once!)
figure_specs = Dict(
    "uforcing" => u_panels,
    "vforcing" => v_panels,
)

create_figure_set(
    output_dir,
    collect(keys(field_data)),
    field_data,
    Y_cpu,
    ᶜspace,
    figure_specs,
    config;
    remap_dir = joinpath(@__DIR__, "ogwd_3d", "remap_data/"),
    FT = FT,
)

# That's it! Lines reduced from ~125 to ~40, and much more flexible!

##############################################################################
# BONUS: Additional plots you can easily add now
##############################################################################

# Example: Add a 4-panel comparison
uv_comparison_panels = [
    PlotPanel("ogwd_u", "CA u-forcing", (1, 1); scale_factor = 86400),
    PlotPanel("gfdl_udt_topo", "GFDL u-forcing", (1, 2); scale_factor = 86400),
    PlotPanel("ogwd_v", "CA v-forcing", (2, 1); scale_factor = 86400),
    PlotPanel("gfdl_vdt_topo", "GFDL v-forcing", (2, 2); scale_factor = 86400),
]

# Just add to figure_specs to include in the batch!
# figure_specs["uv_comparison"] = uv_comparison_panels

# Example: Add difference plots
# field_data["u_diff"] = uforcing_cpu .- gfdl_ca_udt_topo_cpu
# field_data["v_diff"] = vforcing_cpu .- gfdl_ca_vdt_topo_cpu
#
# diff_panels = [
#     PlotPanel("u_diff", "u difference at z = {z}", (1, 1);
#               scale_factor = 86400, colormap = :balance),
#     PlotPanel("v_diff", "v difference at z = {z}", (2, 1);
#               scale_factor = 86400, colormap = :balance),
# ]
# figure_specs["differences"] = diff_panels

# end
