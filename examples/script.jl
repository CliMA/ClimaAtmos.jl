import ClimaCore
import ClimaTimeSteppers as CTS
import SciMLBase
import ClimaAtmos as CA

import ClimaParams
import Thermodynamics as TD
import SurfaceFluxes: Parameters.SurfaceFluxesParameters, UniversalFunctions.BusingerParams
import CloudMicrophysics as CM
import Insolation.Parameters.InsolationParameters
import RRTMGP.Parameters.RRTMGPParameters

# Define the floating-point type
FT = Float32

meter = meters = seconds = second = one(FT)
kilometers = 1000meters
day = days = 86400seconds
earth_radius = 6700kilometers

# Parameters of the grid
# Using a vertical grid stretched with an hyperbolic tangent
number_of_vertical_points = 10
minimum_elevation = 0meters
maximum_elevation = 30kilometers
height_of_bottom_cell = 500meters
vertical_stretching =
    ClimaCore.Meshes.HyperbolicTangentStretching(height_of_bottom_cell)

# Horizontal spectral grid
number_of_horizontal_elements_per_panel = 6
polynomial_order = 3

# Time
t_start = 0seconds
t_end = 10days
dt = 400seconds

# Create a 3D CubedSphere defined on the cell/face centers
center_space = ClimaCore.CommonSpaces.ExtrudedCubedSphereSpace(
    FT;
    radius = earth_radius,
    h_elem = number_of_horizontal_elements_per_panel,
    z_elem = number_of_vertical_points,
    z_min = minimum_elevation,
    z_max = maximum_elevation,
    n_quad_points = polynomial_order + 1,
    stretch = vertical_stretching,
    staggering = ClimaCore.Grids.CellCenter(),
)
face_space = ClimaCore.CommonSpaces.ExtrudedCubedSphereSpace(
    FT;
    radius = earth_radius,
    h_elem = number_of_horizontal_elements_per_panel,
    z_elem = number_of_vertical_points,
    z_min = minimum_elevation,
    z_max = maximum_elevation,
    n_quad_points = polynomial_order + 1,
    stretch = vertical_stretching,
    staggering = ClimaCore.Grids.CellFace(),
)

model = CA.AtmosModel(;
    moisture_model = CA.DryModel(),
    surface_model = CA.PrescribedSurfaceTemperature(),
    precip_model = CA.NoPrecipitation(),
)

# Parameters
grav = 20
tot_solar_irrad = 4_000
Prandtl_number_0 = 0.04
planet_radius = 6_371_000 / 10

thermodynamics_params = TD.Parameters.ThermodynamicsParameters(FT)
surface_fluxes_params = SurfaceFluxesParameters(FT, BusingerParams)
microphysics_cloud_params =
    (; liquid = CM.Parameters.CloudLiquid(FT), ice = CM.Parameters.CloudIce(FT))
water_params = CM.Parameters.WaterProperties(FT)
microphysics_precipitation_params = CM.Parameters.Parameters0M(FT)
insolation_params = InsolationParameters(FT, (; tot_solar_irrad))
rrtmgp_params = RRTMGPParameters(FT, (; grav))
surface_temp_params = CA.SurfaceTemperatureParameters(FT)
turbconv_params = CA.TurbulenceConvectionParameters(FT, (; Prandtl_number_0))

parameters = CA.ClimaAtmosParameters(
    FT;
    turbconv_params,
    thermodynamics_params,
    microphysics_cloud_params,
    insolation_params,
    rrtmgp_params,
    surface_temp_params,
    planet_radius,
    # more params....
)

initial_condition = CA.InitialConditions.DryBaroclinicWave()

Y = CA.InitialConditions.atmos_state(
    initial_condition(parameters),
    model,
    center_space,
    face_space,
)
precomputed = CA.precomputed_quantities(Y, model)

surface_setup = CA.SurfaceConditions.DefaultMoninObukhov()

prescribed_aerosols = []
# AtmosModel object is a lot of work to build
cache = CA.build_cache(
    Y,
    types,
    params,
    surface_setup,
    dt,
    t_start,
    prescribed_aerosols,
)
