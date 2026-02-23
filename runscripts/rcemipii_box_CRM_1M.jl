"""
Runscript for cloud-resolving box with RCEMIP-II setup
"""

# When Julia 1.10+ is used interactively, stacktraces contain reduced type information to make them shorter.
# On the other hand, the full type information is printed when julia is not run interactively.
# Given that ClimaCore objects are heavily parametrized, non-abbreviated stacktraces are hard to read,
# so we force abbreviated stacktraces even in non-interactive runs.
# (See also Base.type_limited_string_from_context())
redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))

# Set up environment
import ClimaAtmos as CA
import ClimaCore as CC
import Random
Random.seed!(1234)
import ClimaComms
ClimaComms.@import_required_backends
context = CA.get_comms_context(Dict("device" => "auto"))

### -------------------- ###
### Variable model setup ###
### -------------------- ###
job_id = "rcemipii_box_CRM_1M"
reference_job_id = Val(:les_box)  # for plotting
output_dir = nothing  # customize if desired
FT = Float32

t_end = "4hours"

##### RCE-small #####
x_max = y_max = 96_000

# spatio-temporal discretization

#~8km grid
x_elem = y_elem = 4
dt = "10secs"
# ~4km grid
# x_elem = y_elem = 8
# dt = "5secs"
# ~2km grid
# x_elem = y_elem = 16
# dt = "2secs"
# ~1km grid
# x_elem = y_elem = 32
# dt = "1secs"

### -------------------- ###


## Load model parameters
params = CA.ClimaAtmosParameters(
    CA.CP.create_toml_dict(FT; override_file = "toml/rcemipii_box.toml"),
)

## RCEMIP-II model prescriptions
insolation = CA.RCEMIPIIInsolation()
sfc_temperature = CA.RCEMIPIISST()
initial_condition = CA.InitialConditions.RCEMIPIIProfile_300()


## Construct the model
model = CA.AtmosModel(;
    # AtmosWater - Moisture, Precipitation & Clouds
    moisture_model = CA.NonEquilMoistModel(),
    microphysics_model = CA.Microphysics1Moment(),
    cloud_model = CA.GridScaleCloud(),
    microphysics_tendency_timestepping = CA.Explicit(),
    tracer_nonnegativity_method = CA.TracerNonnegativityMethod("elementwise_constraint"),

    # AtmosRadiation
    radiation_mode = CA.RRTMGPInterface.AllSkyRadiationWithClearSkyDiagnostics(),
    insolation,

    # TODO: See if you need to set: `edmfx_model`
    smagorinsky_lilly = CA.SmagorinskyLilly(; axes = :UV_W),
    rayleigh_sponge = CA.RayleighSponge{FT}(; zd = 30_000),

    # AtmosSurface
    sfc_temperature,
    surface_model = CA.PrescribedSST(),
    surface_albedo = CA.ConstantAlbedo{FT}(; α = 0.07),

    # numerics
    numerics = CA.AtmosNumerics(; hyperdiff = nothing),
)
# @info "AtmosModel: \n$(summary(atmos))"

## Grid creation
function rcemipii_z_mesh(::Type{FT}) where {FT}
    z_max = 33_000
    z_elem = 74
    boundary_layer =
        FT[0, 37, 112, 194, 288, 395, 520, 667, 843, 1062, 1331, 1664, 2055, 2505]
    n_bl = length(boundary_layer) - 1
    free_atmosphere = range(3000, FT(z_max), length = z_elem - n_bl)  # z_elem=74, z_max=33_000m --> 500m spacing
    CT = CC.Geometry.ZPoint{FT}
    faces = CT.([boundary_layer; free_atmosphere])
    z_domain = CC.Domains.IntervalDomain(CT(0), CT(z_max); boundary_names = (:bottom, :top))
    z_mesh = CC.Meshes.IntervalMesh(z_domain, faces)
    return z_mesh
end
z_mesh = rcemipii_z_mesh(FT)
grid = CA.BoxGrid(FT; context, x_elem, x_max, y_elem, y_max, z_mesh)

## Discretization
import ClimaTimeSteppers as CTS
newtons_method =
    CTS.NewtonsMethod(; max_iters = 1, update_j = CTS.UpdateEvery(CTS.NewNewtonIteration))
ode_config = CTS.IMEXAlgorithm(CTS.ARS343(), newtons_method)

## Output diagnostics
diagnostics = [
    Dict(
        "short_name" => [
            "wa", "ua", "va", "ta", "thetaa", "ha",  # dynamics & thermodynamics
            "hus", "hur", "cl", "clw", "cli",  # liquid
            "pr",  # precipitation
            "ke",  # kinetic energy for spectrum
            # Smagorinsky diagnostics
            "Dh_smag", "strainh_smag",  # horizontal
            "Dv_smag", "strainv_smag",  # vertical
        ],
        "period" => "1hours",
    ),
]
### 1M microphysics
if model.microphysics_model ∈ (CA.Microphysics1Moment(), CA.Microphysics2Moment())
    push!(diagnostics, Dict("short_name" => ["husra", "hussn"], "period" => "1hours"))
end
### 2M microphysics
if model.microphysics_model == CA.Microphysics2Moment()
    push!(diagnostics, Dict("short_name" => ["cdnc", "ncra"], "period" => "1hours"))
end

## Assemble simulation
simulation = CA.AtmosSimulation{FT}(; job_id,
    model, params, context, grid,
    initial_condition,
    surface_setup = CA.SurfaceConditions.DefaultMoninObukhov(),
    dt, t_end,
    ode_config,
    output_dir,
    # Callbacks
    callback_kwargs = (; dt_cloud_fraction = "3hours", dt_rad = "1hours"),
    # Diagnostics
    default_diagnostics = false,
    diagnostics,
    # Numerics
    approximate_linear_solve_iters = 2,  # TODO: Fix implicit diffusion for LES
    # Misc
    checkpoint_frequency = "1days",
    log_to_file = true,
)

## Solve
sol_res = CA.solve_atmos!(simulation)
(; sol) = sol_res

# Post-solve checks
CA.error_if_crashed(sol_res.ret_code)
CA.verify_callbacks(sol.t)


## Postprocessing

# Check if selected output has changed from the previous recorded output (bit-wise comparison)
if haskey(ENV, "CI")
    include(joinpath(@__DIR__, "..", "reproducibility_tests", "reproducibility_tools.jl"))
    export_reproducibility_results(sol.u[end], context;
        simulation.job_id, computed_dir = simulation.output_dir,
    )
end
# --> Make ci plots
if ClimaComms.iamroot(context)
    include(joinpath(pkgdir(CA), "post_processing", "ci_plots.jl"))
    make_plots(reference_job_id, simulation.output_dir)
    # plot_profiles_kw = (; first_time_relative = Dates.Hour(42))
    # les_debug_plots([simulation.output_dir]; save_jpeg_copy=true, plot_profiles_kw)
end
# <--
