using ClimaCore:
    Fields, Geometry, Domains, Meshes, Topologies, Spaces, Operators, DataLayouts,
    Utilities, Grids, Quadratures
using ClimaCore
using ClimaCore.CommonSpaces
using NCDatasets
import ClimaAtmos
import ClimaAtmos as CA
import ClimaComms
import ClimaAtmos: AtmosArtifacts as AA
import ClimaUtilities: SpaceVaryingInputs.SpaceVaryingInput
using ClimaCoreTempestRemap

import Interpolations as Intp
import ClimaInterpolations.Interpolation1D: interpolate1d!, Linear, Flat

using CUDA
using Dates
const FT = Float32

# Create output directory early for debug plots
const output_dir = "orographic_gravity_wave_test_3d"
mkpath(output_dir)

# Column-wise vertical interpolation (following pattern from weather_model.jl)
function interpz_3d(ztarget, zsource, fsource)
    nx, ny, nz = size(zsource)
    zsourcep = permutedims(zsource, (3, 1, 2))
    fsourcep = permutedims(fsource, (3, 1, 2))
    ftargetp = similar(fsourcep, length(ztarget), nx, ny)
    interpolate1d!(ftargetp, zsourcep, ztarget, fsourcep, Linear(), Flat())
    return permutedims(ftargetp, (2, 3, 1))
end

# Preprocess GFDL data to ClimaAtmos z-levels
function preprocess_gfdl_to_ca_levels(source_file, target_file, target_levels, FT)
    NCDataset(source_file) do ncin
        lon = FT.(ncin["lon"][:])
        lat = FT.(ncin["lat"][:])
        source_z_raw = FT.(ncin["z_full"][:, :, :])
        # GFDL z is descending (top-of-atmosphere first), reverse to ascending for interpolation
        source_z = reverse(source_z_raw, dims = 3)

        # Read pressure components for p_center calculation
        ps = FT.(ncin["ps"][:, :])
        pk = FT.(ncin["pk"][:])
        bk = FT.(ncin["bk"][:])

        # Compute p_center on GFDL levels
        p_half = zeros(FT, size(ps, 1), size(ps, 2), length(bk))
        for k in 1:length(bk)
            p_half[:, :, k] .= bk[k] .* ps .+ pk[k]
        end
        p_center_gfdl = FT(0.5) .* (p_half[:, :, 1:(end - 1)] .+ p_half[:, :, 2:end])

        NCDataset(target_file, "c") do ncout
            # Define dimensions
            defDim(ncout, "lon", length(lon))
            defDim(ncout, "lat", length(lat))
            defDim(ncout, "z", length(target_levels))

            # Write coordinate variables
            defVar(
                ncout,
                "lon",
                lon,
                ("lon",),
                attrib = Dict(
                    "units" => "degrees_east",
                    "standard_name" => "longitude",
                ),
            )
            defVar(
                ncout,
                "lat",
                lat,
                ("lat",),
                attrib = Dict(
                    "units" => "degrees_north",
                    "standard_name" => "latitude",
                ),
            )
            defVar(
                ncout,
                "z",
                FT.(target_levels),
                ("z",),
                attrib = Dict(
                    "units" => "m",
                    "standard_name" => "altitude",
                ),
            )

            # Interpolate each 3D variable to target z-levels
            # Must also reverse data along z to match reversed source_z
            var_names = ["temp", "ucomp", "vcomp", "udt_topo", "vdt_topo", "sphum"]
            for var_name in var_names
                data_in_raw = FT.(ncin[var_name][:, :, :])
                data_in = reverse(data_in_raw, dims = 3)  # Reverse to match source_z ordering
                data_out = interpz_3d(FT.(target_levels), source_z, data_in)
                defVar(ncout, var_name, data_out, ("lon", "lat", "z"))
            end

            # Handle pressure - interpolate p_center (also reversed)
            p_center_gfdl_rev = reverse(p_center_gfdl, dims = 3)
            p_center_out = interpz_3d(FT.(target_levels), source_z, p_center_gfdl_rev)
            defVar(ncout, "p_center", p_center_out, ("lon", "lat", "z"))
        end
    end
    return target_file
end
include(
    joinpath(pkgdir(ClimaAtmos), "post_processing/remap", "remap_helpers.jl"),
)
include("../gw_remap_plot_utils.jl")

# Include helper functions from test directory
include(joinpath(@__DIR__, "..", "..", "..", "test_helpers.jl"))

comms_ctx = ClimaComms.SingletonCommsContext()
@show CUDA.functional()
@show ClimaComms.device(comms_ctx)

# GFDL data file path
include(joinpath(@__DIR__, "../../../artifact_funcs.jl"))
ncfile = joinpath(gfdl_ogw_data_path(), "gfdl_ogw.nc")

# Initialize p
(; config_file, job_id) = CA.commandline_kwargs()
config = CA.AtmosConfig(config_file; job_id, comms_ctx)
config.parsed_args["h_elem"] = 8
config.parsed_args["nh_poly"] = 3
config.parsed_args["z_max"] = 42e3
config.parsed_args["z_elem"] = 33
config.parsed_args["dz_bottom"] = 300.0
config.parsed_args["orographic_gravity_wave"] = "gfdl_restart"
config.parsed_args["topography"] = "Earth";
(; parsed_args) = config

simulation = CA.get_simulation(config)
p = simulation.integrator.p
Y = simulation.integrator.u

ᶜspace = axes(Y.c)
ᶜlocal_geometry = Fields.local_geometry_field(axes(Y.c))
ᶠlocal_geometry = Fields.local_geometry_field(axes(Y.f))

# Get target z-levels from the ClimaAtmos grid
z_arr = Array(Fields.field2array(Fields.coordinate_field(Y.c).z))
target_levels = FT.(unique(z_arr[:, 1]))
sort!(target_levels)  # Ensure ascending order

# Preprocess GFDL data to ClimaAtmos z-levels (creates file if needed)
preprocessed_file = joinpath(gfdl_ogw_data_path(), "gfdl_ogw_ca_levels.nc")
if !isfile(preprocessed_file)
    @info "Preprocessing GFDL data to ClimaAtmos z-levels..."
    preprocess_gfdl_to_ca_levels(ncfile, preprocessed_file, target_levels, FT)
    @info "Preprocessed file created: $preprocessed_file"
end

# Set up SpaceVaryingInput regridder
regridder_type = :InterpolationsRegridder
extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())
interpolation_method = Intp.Linear()
regridder_kwargs = (; extrapolation_bc, interpolation_method)

# Load GFDL data onto ClimaAtmos grid using SpaceVaryingInput
@info "Loading GFDL data with SpaceVaryingInput..."
gfdl_ca_temp =
    SpaceVaryingInput(preprocessed_file, "temp", ᶜspace; regridder_type, regridder_kwargs)
gfdl_ca_ucomp =
    SpaceVaryingInput(preprocessed_file, "ucomp", ᶜspace; regridder_type, regridder_kwargs)
gfdl_ca_vcomp =
    SpaceVaryingInput(preprocessed_file, "vcomp", ᶜspace; regridder_type, regridder_kwargs)
gfdl_ca_udt_topo = SpaceVaryingInput(
    preprocessed_file,
    "udt_topo",
    ᶜspace;
    regridder_type,
    regridder_kwargs,
)
gfdl_ca_vdt_topo = SpaceVaryingInput(
    preprocessed_file,
    "vdt_topo",
    ᶜspace;
    regridder_type,
    regridder_kwargs,
)
gfdl_ca_sphum =
    SpaceVaryingInput(preprocessed_file, "sphum", ᶜspace; regridder_type, regridder_kwargs)
gfdl_ca_p = SpaceVaryingInput(
    preprocessed_file,
    "p_center",
    ᶜspace;
    regridder_type,
    regridder_kwargs,
)

@info "Regridded field ranges:" temp = extrema(parent(gfdl_ca_temp)) udt_topo =
    extrema(parent(gfdl_ca_udt_topo)) vdt_topo = extrema(parent(gfdl_ca_vdt_topo))

# create Y
Yc = map(ᶜlocal_geometry) do lg
    return (;
        ρ = FT(1.0),
        uₕ = Geometry.Covariant12Vector(Geometry.UVVector(FT(0), FT(0)), lg),
        T = FT(0),
        qt = FT(0),
    )
end
Yc.uₕ .=
    Geometry.Covariant12Vector.(
        Geometry.UVVector.(gfdl_ca_ucomp, gfdl_ca_vcomp),
        ᶜlocal_geometry,
    )
Yc.T .= gfdl_ca_temp
Yc.qt .= gfdl_ca_sphum
Yf = map(ᶠlocal_geometry) do lg
    return (; u₃ = Geometry.Covariant3Vector(FT(0), lg))
end
Y = Fields.FieldVector(c = Yc, f = Yf)

# compute density from temperature, humidiry, pressure
R_d = 287.0
epsilon = 0.622
@. Y.c.ρ = gfdl_ca_p / Y.c.T / R_d / (1 - Y.c.qt + Y.c.qt / epsilon)

ᶜT = gfdl_ca_temp
ᶜp = gfdl_ca_p

ᶜz = Fields.coordinate_field(Y.c).z

ᶜtarget_space = Spaces.axes(Y.c)
ᶜp = Fields.Field(Fields.field_values(ᶜp), ᶜtarget_space)
ᶜT = Fields.Field(Fields.field_values(ᶜT), ᶜtarget_space)

@. p.precomputed.ᶜT = ᶜT
@. p.precomputed.ᶜp = ᶜp

# Pre-compute OGWD column quantities for diagnostic plotting
# Get orographic gravity wave parameters from cache
ogw_cache = p.orographic_gravity_wave

# Get the topography info - topo_info has fields: t11, t12, t21, t22, hmin, hmax
(; topo_info) = ogw_cache
(; hmax, hmin) = topo_info

# Get z fields
ᶜz_field = Fields.coordinate_field(Y.c).z
ᶠz_field = Fields.coordinate_field(Y.f).z

# Get pressure fields from precomputed
ᶜp_pre = p.precomputed.ᶜp

# Convert fields to CPU for column diagnostics (no-op on CPU, needed for GPU)
hmax_cpu = ClimaCore.to_cpu(hmax)
hmin_cpu = ClimaCore.to_cpu(hmin)
ᶜz_field_cpu = ClimaCore.to_cpu(ᶜz_field)
ᶠz_field_cpu = ClimaCore.to_cpu(ᶠz_field)
ᶜp_pre_cpu = ClimaCore.to_cpu(ᶜp_pre)
ᶜT_pre_cpu = ClimaCore.to_cpu(p.precomputed.ᶜT)

# Find columns with largest hmax (where OGWD is most active)
hmax_arr = parent(hmax_cpu)

max_hmax_idx = argmax(hmax_arr)
max_hmax_cartesian = CartesianIndices(hmax_arr)[max_hmax_idx]
@info "Max hmax:" idx = max_hmax_cartesian value = hmax_arr[max_hmax_idx]

# Extract column indices (surface fields have IJFH layout, no vertical dim)
i_col, j_col, f_col, h_col = Tuple(max_hmax_cartesian)

# Get column data (3D fields have VIJFH layout)
z_col = parent(ᶜz_field_cpu)[:, i_col, j_col, f_col, h_col]
p_col = parent(ᶜp_pre_cpu)[:, i_col, j_col, f_col, h_col]
ᶠz_col = parent(ᶠz_field_cpu)[:, i_col, j_col, f_col, h_col]

# Estimate z_pbl using the same criteria as get_pbl_z
# Look for where p > 0.8 * p_surface AND dT/dz > -g/cp_d
p_surface = p_col[1]
cp_d = 1004.0f0
grav_val = 9.81f0
dTdz_crit = -grav_val / cp_d

T_col = parent(ᶜT_pre_cpu)[:, i_col, j_col, f_col, h_col]
dTdz = diff(T_col) ./ diff(z_col)

z_pbl_est = z_col[1]
for k in 1:(length(z_col) - 1)
    if p_col[k] > 0.8f0 * p_surface && dTdz[k] > dTdz_crit
        global z_pbl_est = z_col[k]
    end
end

# z_ref = z_pbl + hmax
hmax_val = hmax_arr[i_col, j_col, f_col, h_col]
z_ref_est = z_pbl_est + hmax_val

# Find p_ref by interpolating pressure at z_ref
p_ref_est = p_col[end]  # default to top pressure
for k in 1:(length(z_col) - 1)
    if z_col[k] <= z_ref_est <= z_col[k + 1]
        frac = (z_ref_est - z_col[k]) / (z_col[k + 1] - z_col[k])
        global p_ref_est = p_col[k] + frac * (p_col[k + 1] - p_col[k])
        break
    end
end

# Compute weights = p - p_ref for each cell
weights_col = p_col .- p_ref_est

# Find cells in mask (between z_pbl and z_ref)
mask_col = (z_col .> z_pbl_est) .& (z_col .< z_ref_est)

@info "Column at max hmax:" hmax = hmax_val hmin =
    parent(hmin_cpu)[i_col, j_col, f_col, h_col] z_pbl = z_pbl_est z_ref = z_ref_est p_ref =
    p_ref_est
@info "  Mask: $(sum(mask_col)) cells, $(sum(mask_col .& (abs.(weights_col) .< 100))) with near-zero weight"

# Plot the column profile using gw_plotutils.jl
fig = generate_empty_figure(size = (1800, 600), fontsize = 20)

create_plot!(fig;
    X = Float64.(p_col), Y = Float64.(z_col),
    p_loc = (1, 1), title = "Pressure Profile",
    xlabel = "Pressure (Pa)", ylabel = "Altitude (m)",
    label = ("p(z)",), yreversed = false,
)

create_plot!(fig;
    X = Float64.(weights_col), Y = Float64.(z_col),
    p_loc = (1, 2), title = "Weights (p - p_ref)",
    xlabel = "Weight (Pa)", ylabel = "Altitude (m)",
    label = ("weight",), yreversed = false,
)

create_plot!(fig;
    X = Float64.(T_col), Y = Float64.(z_col),
    p_loc = (1, 3), title = "Temperature Profile",
    xlabel = "Temperature (K)", ylabel = "Altitude (m)",
    label = ("T(z)",), yreversed = false,
)

save(joinpath(output_dir, "column_profile.png"), fig)

CA.orographic_gravity_wave_compute_tendency!(Y, p, p.atmos.orographic_gravity_wave)

(; ᶜuforcing, ᶜvforcing) = p.orographic_gravity_wave

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
# output_dir already created at top of file

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

ᶜspace_cpu = axes(Y_cpu.c)
create_figure_set(
    output_dir,
    collect(keys(field_data)),
    field_data,
    Y_cpu,
    ᶜspace_cpu,
    figure_specs,
    config;
    remap_dir = joinpath(@__DIR__, "ogwd_3d", "remap_data/"),
    FT = FT,
)

# Example: Add a 4-panel comparison
uv_comparison_panels = [
    PlotPanel("ogwd_u", "CA u-forcing", (1, 1); scale_factor = 86400),
    PlotPanel("gfdl_udt_topo", "GFDL u-forcing", (1, 2); scale_factor = 86400),
    PlotPanel("ogwd_v", "CA v-forcing", (2, 1); scale_factor = 86400),
    PlotPanel("gfdl_vdt_topo", "GFDL v-forcing", (2, 2); scale_factor = 86400),
]
