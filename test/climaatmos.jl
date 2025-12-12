using Dates

import ClimaParams as CP
import ClimaTimeSteppers as CTS
import ClimaAtmos as CA
import ClimaComms
FT = Float32
days = 86400
start_date = DateTime(2010, 1, 1)

param_dict =
    CP.create_toml_dict(FT; override_file = "toml/longrun_aquaplanet_diagedmf.toml")
params = CA.ClimaAtmosParameters(param_dict)

deep_atmosphere = true
ozone = CA.PrescribedOzone()
cloud = CA.InteractiveCloudInRadiation()
co2 = CA.MaunaLoaCO2()
rayleigh_sponge = CA.RayleighSponge{FT}(;
    zd = params.zd_rayleigh,
    α_uₕ = params.alpha_rayleigh_uh,
    α_w = params.alpha_rayleigh_w,
    α_sgs_tracer = params.alpha_rayleigh_sgs_tracer,
)
viscous_sponge = CA.ViscousSponge{FT}(; zd = params.zd_viscous, κ₂ = params.kappa_2_sponge)

diff_mode = CA.Implicit()
hyperdiff = CA.ClimaHyperdiffusion{FT}(;
    ν₄_vorticity_coeff = 0.150 * 1.238,
    ν₄_scalar_coeff = 0.751 * 1.238,
    divergence_damping_factor = 5,
)

tracers = (
    "CB1", "CB2",
    "DST01", "DST02", "DST03", "DST04", "DST05",
    "OC1", "OC2",
    "SO4",
    "SSLT01", "SSLT02", "SSLT03", "SSLT04", "SSLT05",
)
cloud_model = CA.SGSQuadratureCloud(CA.SGSQuadrature(FT))
microphysics_model = CA.Microphysics0Moment()
moisture_model = CA.EquilMoistModel()

# Radiation mode
idealized_h2o = false
idealized_clouds = false
add_isothermal_boundary_layer = true
aerosol_radiation = true
reset_rng_seed = false
radiation_mode = CA.RRTMGPInterface.AllSkyRadiationWithClearSkyDiagnostics(
    idealized_h2o,
    idealized_clouds,
    cloud,
    add_isothermal_boundary_layer,
    aerosol_radiation,
    reset_rng_seed,
    deep_atmosphere,
)

insolation = CA.TimeVaryingInsolation(start_date)

surface_setup = CA.SurfaceConditions.DefaultMoninObukhov()

n_updrafts = 1
prognostic_tke = true
turbconv_model = CA.DiagnosticEDMFX{n_updrafts, prognostic_tke}(1e-5)

edmfx_model = CA.EDMFXModel(;
    entr_model = CA.InvZEntrainment(),
    detr_model = CA.BuoyancyVelocityDetrainment(),
    sgs_mass_flux = Val(true),
    sgs_diffusive_flux = Val(true),
    nh_pressure = Val(true),
    vertical_diffusion = Val(false),
    filter = Val(false),
    scale_blending_method = CA.SmoothMinimumBlending(),
)
topography = CA.EarthTopography()
h_elem = 16
z_elem = 63
z_max = 60000.0
dz_bottom = 30.0

model = CA.AtmosModel(;
    moisture_model,
    microphysics_model,
    turbconv_model,
    edmfx_model,
    radiation_mode,
    insolation,
    co2,
    rayleigh_sponge,
    viscous_sponge,
    hyperdiff,
    diff_mode
)

grid = CA.SphereGrid(FT; topography, h_elem, z_elem, z_max, dz_bottom)

# TODO: Use jacobian flags
approximate_linear_solve_iters = 2
max_newton_iters_ode = 1

newtons_method = CTS.NewtonsMethod(;
    max_iters = max_newton_iters_ode,
    update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
)

ode_config = CTS.IMEXAlgorithm(
    CTS.ARS343(),
    newtons_method,
)

callback_kwargs = (;
    dt_rad = "1hours",
    dt_cloud_fraction = "1hours",
)

simulation = CA.AtmosSimulation{FT}(;
    model,
    grid,
    job_id = "test_climaatmos",
    context = ClimaComms.SingletonCommsContext(),
    tracers,
    t_end = 120days,
    checkpoint_frequency = 30days,
    approximate_linear_solve_iters,
    callback_kwargs,
)

CA.solve_atmos!(simulation)
