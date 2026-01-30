redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore as CC
import Random
Random.seed!(1234)

### -------------------- ###
### Variable model setup ###
### -------------------- ###
output_dir = nothing  # customize if desired
FT = Float32

t_end = "150days"

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
    noneq_cloud_formation_mode = CA.Explicit(),
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
grid = CA.BoxGrid(FT; x_elem, x_max, y_elem, y_max, z_mesh)

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
simulation = CA.AtmosSimulation{FT}(; job_id = "rcemipii_box",
    model, params, grid,
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


## Postprocessing
# include(joinpath(@__DIR__, "..", "reproducibility_tests", "reproducibility_tools.jl"))
# export_reproducibility_results(sol.u[end], ClimaComms.context(); simulation.job_id, computed_dir = simulation.output_dir)
