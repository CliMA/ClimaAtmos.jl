"""
    InitialCondition

A mechanism for specifying the `LocalState` of an `AtmosModel` at every point in
the domain. Given some `initial_condition`, calling `initial_condition(params)`
returns a function of the form `local_state(local_geometry)::LocalState`.
"""
abstract type InitialCondition end

# TODO: remove once we fully migrate to Setups (duplicated in setups/common.jl)
# Perturbation coefficient for the initial conditions
# It would be better to be able to specify the wavenumbers
# but we don't have access to the domain size here

perturb_coeff(p::Geometry.AbstractPoint{FT}) where {FT} = FT(0)
perturb_coeff(p::Geometry.LatLongZPoint{FT}) where {FT} = sind(p.long)
perturb_coeff(p::Geometry.XZPoint{FT}) where {FT} = sin(p.x)
perturb_coeff(p::Geometry.XYZPoint{FT}) where {FT} = sin(p.x)

# TODO: remove once we fully migrate to Setups (duplicated in setups/hydrostatic.jl)
"""
    ColumnInterpolatableField(::Fields.ColumnField)

A column field object that can be interpolated
in the z-coordinate. For example:

```julia
cif = ColumnInterpolatableField(column_field)
z = 1.0
column_field_at_z = cif(z)
```

!!! warn
    This function allocates and is not GPU-compatible
    so please avoid using this inside `step!` only use
    this for initialization.
"""
struct ColumnInterpolatableField{F, D}
    f::F
    data::D
    function ColumnInterpolatableField(f::Fields.ColumnField)
        zdata = vec(parent(Fields.Fields.coordinate_field(f).z))
        fdata = vec(parent(f))
        data = Intp.extrapolate(
            Intp.interpolate((zdata,), fdata, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
        return new{typeof(f), typeof(data)}(f, data)
    end
end
(f::ColumnInterpolatableField)(z) = Spaces.undertype(axes(f.f))(f.data(z))

import ClimaComms
import ClimaCore.Domains as Domains
import ClimaCore.Meshes as Meshes
import ClimaCore.Geometry as Geometry
import ClimaCore.Operators as Operators
import ClimaCore.Topologies as Topologies
import ClimaCore.Spaces as Spaces

# TODO: remove once we fully migrate to Setups (duplicated in setups/hydrostatic.jl)
"""
    column_indefinite_integral(f, ϕ₀, zspan; nelems = 100)

The column integral, returned as
an interpolate-able field.
"""
function column_indefinite_integral(
    f::Function,
    ϕ₀::FT,
    zspan::Tuple{FT, FT};
    nelems = 100, # sets resolution for integration
) where {FT <: Real}
    # --- Make a space for integration:
    z_domain = Domains.IntervalDomain(
        Geometry.ZPoint(first(zspan)),
        Geometry.ZPoint(last(zspan));
        boundary_names = (:bottom, :top),
    )
    z_mesh = Meshes.IntervalMesh(z_domain; nelems)
    context = ClimaComms.SingletonCommsContext()
    z_topology = Topologies.IntervalTopology(context, z_mesh)
    cspace = Spaces.CenterFiniteDifferenceSpace(z_topology)
    fspace = Spaces.FaceFiniteDifferenceSpace(z_topology)
    # ---
    zc = Fields.coordinate_field(cspace)
    ᶠintegral = Fields.Field(FT, fspace)
    Operators.column_integral_indefinite!(f, ᶠintegral, ϕ₀)
    return ColumnInterpolatableField(ᶠintegral)
end

##
## Simple Profiles
##

"""
    ConstantBuoyancyFrequencyProfile()

An `InitialCondition` with a constant Brunt-Vaisala frequency and constant wind
velocity, where the pressure profile is hydrostatically balanced. This is
currently the only `InitialCondition` that supports the approximation of a
steady-state solution.
"""
struct ConstantBuoyancyFrequencyProfile <: InitialCondition end
function (::ConstantBuoyancyFrequencyProfile)(params)
    function local_state(local_geometry)
        FT = eltype(params)
        coord = local_geometry.coordinates
        return LocalState(;
            params,
            geometry = local_geometry,
            constant_buoyancy_frequency_initial_state(params, coord)...,
        )
    end
    return local_state
end

"""
    IsothermalProfile(; temperature = 300)

An `InitialCondition` with a uniform temperature profile.
"""
Base.@kwdef struct IsothermalProfile{T} <: InitialCondition
    temperature::T = 300
end

function (initial_condition::IsothermalProfile)(params)
    (; temperature) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        R_d = CAP.R_d(params)
        MSLP = CAP.MSLP(params)
        grav = CAP.grav(params)
        T = FT(temperature)

        (; z) = local_geometry.coordinates
        p = MSLP * exp(-z * grav / (R_d * T))

        return LocalState(;
            params,
            geometry = local_geometry,
            T = T,
            p = p,
        )
    end
    return local_state
end

"""
    DecayingProfile(; perturb = true)

An `InitialCondition` with a decaying temperature profile, and with an optional
perturbation to the temperature.
"""
Base.@kwdef struct DecayingProfile <: InitialCondition
    perturb::Bool = true
end

function (initial_condition::DecayingProfile)(params)
    (; perturb) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        grav = CAP.grav(params)
        thermo_params = CAP.thermodynamics_params(params)
        temp_profile = DecayingTemperatureProfile{FT}(
            thermo_params,
            FT(290),
            FT(220),
            FT(8e3),
        )

        (; z) = local_geometry.coordinates
        coeff = perturb_coeff(local_geometry.coordinates)
        T, p = temp_profile(thermo_params, z)
        if perturb
            T += coeff * FT(0.1) * (z < 5000)
        end

        return LocalState(;
            params,
            geometry = local_geometry,
            T = T,
            p = p,
        )
    end
    return local_state
end

"""
    MoistFromFile(file_path)

This function assigns an empty initial condition for , populating the `LocalState` with
`NaN`, and later overwriting it with the content of the given file
"""
struct MoistFromFile <: InitialCondition
    file_path::String
end

"""
    WeatherModel(start_date)

An `InitialCondition` that initializes the model with an empty state, and then overwrites
it with the content of a NetCDF file that contains the initial conditions, stored in the 
artifact `weather_model_ic`/raw/era5_raw_YYYYMMDD_HHMM.nc. We interpolate the initial 
conditions from ERA5 pressure level grid to a z grid, saving to the artifact 
weather_model_ic/init/era5_init_YYYYMMDD_HHMM.nc. It is then interpolated to the model
grid in `_overwrite_initial_conditions_from_file!`, which documents the required variables.
Recall running `ClimaUtilities.ClimaArtiffacts.@clima_artifact("weather_model_ic")` gets 
the artifact path.
"""
struct WeatherModel <: InitialCondition
    start_date::String
    era5_initial_condition_dir::Union{Nothing, String}
end

"""
    AMIPFromERA5(start_date)

An `InitialCondition` for AMIP simulations using ERA5 monthly reanalysis data.
Uses the `amip_era5_ic` artifact containing pre-processed ERA5 data interpolated
to z-levels, matching the format expected by the WeatherModel machinery.

Expected artifact structure: `amip_era5_ic/era5_init_YYYYMMDD_0000.nc`
"""
struct AMIPFromERA5 <: InitialCondition
    start_date::String
end

function (initial_condition::Union{MoistFromFile, WeatherModel, AMIPFromERA5})(params)
    function local_state(local_geometry)
        FT = eltype(params)

        T, p = FT(NaN), FT(NaN) # placeholder values

        return LocalState(;
            params,
            geometry = local_geometry,
            T = T,
            p = p,
        )
    end
    return local_state
end

"""
    DryDensityCurrentProfile(; perturb = false)

An `InitialCondition` with an isothermal background profile, with a negatively
buoyant bubble, and with an optional
perturbation to the temperature.
"""
Base.@kwdef struct DryDensityCurrentProfile <: InitialCondition
    perturb::Bool = false
end

function (initial_condition::DryDensityCurrentProfile)(params)
    (; perturb) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        grav = CAP.grav(params)
        ndims = length(propertynames(local_geometry.coordinates))
        (; x, z) = local_geometry.coordinates
        x_c = FT(25600)
        x_r = FT(4000)
        z_c = FT(2000)
        z_r = FT(2000)
        r_c = FT(1)
        θ_b = FT(300)
        θ_c = FT(-15)
        cp_d = CAP.cp_d(params)
        cv_d = CAP.cv_d(params)
        p_0 = CAP.p_ref_theta(params)
        R_d = CAP.R_d(params)
        T_0 = CAP.T_0(params)

        # auxiliary quantities
        r² = FT(0)
        r² += ((x - x_c) / x_r)^2 + ((z - z_c) / z_r)^2
        if ndims == 3
            (; y) = local_geometry.coordinates
            y_r = FT(2000)
            y_c = FT(3200)
            r² += ((y - y_c) / y_r)^2
        end
        θ_p =
            sqrt(r²) < r_c ? FT(1 / 2) * θ_c * (FT(1) + cospi(sqrt(r²) / r_c)) :
            FT(0) # potential temperature perturbation
        θ = θ_b + θ_p # potential temperature
        π_exn = FT(1) - grav * z / cp_d / θ # exner function
        T = π_exn * θ # temperature
        p = p_0 * π_exn^(cp_d / R_d) # pressure
        ρ = p / R_d / T # density

        return LocalState(;
            params,
            geometry = local_geometry,
            T = T,
            p = p,
        )
    end
    return local_state
end

"""
    RisingThermalBubbleProfile(; perturb = false)

An `InitialCondition` with an isothermal background profile, with a positively
buoyant bubble, and with an optional perturbation to the temperature.
"""
Base.@kwdef struct RisingThermalBubbleProfile <: InitialCondition
    perturb::Bool = false
end

function (initial_condition::RisingThermalBubbleProfile)(params)
    (; perturb) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        grav = CAP.grav(params)
        ndims = length(propertynames(local_geometry.coordinates))
        (; x, z) = local_geometry.coordinates
        x_c = FT(500)
        x_r = FT(250)
        z_c = FT(350)
        z_r = FT(250)
        r_c = FT(1)
        θ_b = FT(300)
        θ_c = FT(0.5)
        cp_d = CAP.cp_d(params)
        cv_d = CAP.cv_d(params)
        p_0 = CAP.p_ref_theta(params)
        R_d = CAP.R_d(params)
        T_0 = CAP.T_0(params)

        # auxiliary quantities
        r² = FT(0)
        r² += ((x - x_c) / x_r)^2 + ((z - z_c) / z_r)^2
        if ndims == 3
            (; y) = local_geometry.coordinates
            y_c = FT(500)
            y_r = FT(250)
            r² += ((y - y_c) / y_r)^2
        end
        θ_p =
            sqrt(r²) < r_c ? FT(1 / 2) * θ_c * (FT(1) + cospi(sqrt(r²) / r_c)) :
            FT(0) # potential temperature perturbation
        θ = θ_b + θ_p # potential temperature
        π_exn = FT(1) - grav * z / cp_d / θ # exner function
        T = π_exn * θ # temperature
        p = p_0 * π_exn^(cp_d / R_d) # pressure
        ρ = p / R_d / T # density

        return LocalState(;
            params,
            geometry = local_geometry,
            T = T,
            p = p,
        )
    end
    return local_state
end

"""
    RCEMIPIIProfile(temperature, humidity)

An `InitialCondition` following the sounding to initialize simulations for
RCEMIPII as described by Wing et. al. (2018)
(https://doi.org/10.5194/gmd-11-793-2018). There are three input profiles:
RCEMIPIIProfile_295, RCEMIPIIProfile_300, and RCEMIPIIProfile_305, that specify
three different SST temperatures and different initial specific humidity
profiles. Note: this should be used for RCE_small and NOT
RCE_large - RCE_large must be initialized with the final state of RCE_small.
"""
@kwdef struct RCEMIPIIProfile{FT} <: InitialCondition
    temperature::FT
    humidity::FT
end

RCEMIPIIProfile_295() = RCEMIPIIProfile(295.0, 12e-3)
RCEMIPIIProfile_300() = RCEMIPIIProfile(300.0, 18.65e-3)
RCEMIPIIProfile_305() = RCEMIPIIProfile(305.0, 24e-3)

function (initial_condition::RCEMIPIIProfile)(params)
    (; temperature, humidity) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        R_d = CAP.R_d(params)
        grav = CAP.grav(params)

        T_0 = FT(temperature)
        q_0 = FT(humidity)

        q_t = FT(10^(-14)) # kg kg -1
        z_q1 = FT(4000) # m
        z_q2 = FT(7500) # m
        z_t = FT(15000) # m
        Γ = FT(0.0067) # K m-1
        p_0 = FT(101480) # Pa

        T_v0 = T_0 * (1 + FT(0.608) * q_0)
        T_vt = T_v0 - Γ * z_t

        p_t = p_0 * (T_vt / T_v0)^(grav / (R_d * Γ))

        (; z) = local_geometry.coordinates

        q = z ≤ z_t ? q_0 * exp(-z / z_q1) * exp(-(z / z_q2)^2) : q_t
        T_v = z ≤ z_t ? T_v0 - Γ * z : T_vt
        T = T_v / (1 + FT(0.608) * q)
        p =
            z ≤ z_t ? p_0 * ((T_v0 - Γ * z) / T_v0)^(grav / (R_d * Γ)) :
            p_t * exp(-grav * (z - z_t) / (R_d * T_vt))

        return LocalState(;
            params,
            geometry = local_geometry,
            T = T,
            p = p,
            q_tot = q,
        )
    end
    return local_state
end

"""
    overwrite_initial_conditions!(initial_condition, args...)

Do-nothing fallback method for the operation overwriting initial conditions
(this functionality required in instances where we interpolate initial conditions from NetCDF files).
Future work may revisit this design choice.
"""
function overwrite_initial_conditions!(
    initial_condition::InitialCondition,
    args...,
)
    return nothing
end

# Restored original MoistFromFile function behavior
function overwrite_initial_conditions!(
    initial_condition::MoistFromFile,
    Y,
    thermo_params,
)
    return _overwrite_initial_conditions_from_file!(
        initial_condition.file_path,
        nothing, # use default extrapolation bc
        Y,
        thermo_params,
    )
end

# TODO: remove once we fully migrate to Setups (duplicated in setups/overwrite_from_file.jl)
"""
    correct_surface_pressure_for_topography!(
        p_sfc,
        file_path,
        face_space,
        Y,
        ᶜT,
        ᶜq_tot,
        thermo_params,
        regridder_kwargs;
        surface_altitude_var = "z_sfc",
    )

Adjusts the surface pressure field `p_sfc` to account for mismatches between
ERA5 (file) surface altitude and the model orography when specifying pressure.

    Δz = z_model_surface - z_sfc

and applies a hydrostatic correction at the surface using the local moist gas
constant and temperature at the surface:

    p_sfc .= p_sfc .* exp.(-Δz * g ./ (R_m_sfc .* T_sfc))

where:
- `g` is gravitational acceleration from `thermo_params`
- `R_m_sfc` is the moist-air gas constant evaluated from `ᶜq_tot` at the surface
- `T_sfc` is the air temperature from `ᶜT` at the surface

Returns `true` if the correction is applied; returns `false` if the surface
altitude field cannot be loaded.

Arguments
- `p_sfc`: face field of surface pressure to be corrected (modified in-place)
- `file_path`: path to the ERA5-derived initialization NetCDF file
- `face_space`: face space of the model grid (for reading/regridding)
- `Y`: prognostic state, used to obtain model surface height
- `ᶜT`: center field of temperature
- `ᶜq_tot`: center field of total specific humidity
- `thermo_params`: thermodynamics parameter set
- `regridder_kwargs`: keyword arguments forwarded to the regridder
- `surface_altitude_var`: variable name for surface altitude (default `"z_sfc"`)
"""
function correct_surface_pressure_for_topography!(
    p_sfc,
    file_path,
    face_space,
    Y,
    ᶜT,
    ᶜq_tot,
    thermo_params,
    regridder_kwargs;
    surface_altitude_var = "z_sfc",
)
    regridder_type = :InterpolationsRegridder
    ᶠz_surface = Fields.level(
        SpaceVaryingInputs.SpaceVaryingInput(
            file_path,
            surface_altitude_var,
            face_space;
            regridder_type,
            regridder_kwargs = regridder_kwargs,
        ),
        Fields.half,
    )

    if ᶠz_surface === nothing
        return false
    end

    grav = thermo_params.grav

    ᶠz_model_surface = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶠΔz = @. ᶠz_model_surface - ᶠz_surface

    ᶠR_m = ᶠinterp.(TD.gas_constant_air.(thermo_params, ᶜq_tot))
    ᶠR_m_sfc = Fields.level(ᶠR_m, Fields.half)

    ᶠT = ᶠinterp.(ᶜT)
    ᶠT_sfc = Fields.level(ᶠT, Fields.half)

    @. p_sfc = p_sfc * exp(-(ᶠΔz) * grav / (ᶠR_m_sfc * ᶠT_sfc))

    @info "Adjusted surface pressure to account for ERA5/model surface-height differences."
    return true
end

# WeatherModel function using the shared implementation
"""
    overwrite_initial_conditions!(initial_condition::WeatherModel, Y, thermo_params; use_full_pressure=false)

Overwrite the prognostic state `Y` with ERA5-derived initial conditions on the model grid.


- Derives the model's target vertical levels from `Y.c` (on CPU).
- Obtains the ERA5-derived IC NetCDF path via `weather_model_data_path` (with
  any caller-provided kwargs forwarded there), then constructs `SpaceVaryingInput`
  fields to regrid ERA5 variables onto the model's center and face spaces.
- Populates the thermodynamic state, density, velocity components, total energy,
  and moisture. If EDMF subdomains exist, initializes those as well.

Pressure initialization (`use_full_pressure`):
- If `use_full_pressure == true` and the IC file contains a 3D pressure field
  `p_3d(lon,lat,z)`, then pressure is taken directly from `p_3d` and regridded.
- Otherwise, pressure is obtained by hydrostatic integration starting from the
  surface pressure `p(lon,lat)` (broadcast in `z` in the IC file), using the
  regridded temperature `t` and specific humidity `q`. If the dataset provides
  surface altitude `z_sfc` (derived from ERA5 surface geopotential), the surface
  pressure is first corrected for model-versus-ERA5 topographic differences
  in order to adjust the reported ERA5 surface pressure.

Expected variables in the IC file:
- 3D: `u`, `v`, `w`, `t`, `q` (and optionally `p_3d`, cloud water variables)
- 2D broadcast in `z`: `p` (surface pressure), `skt` (skin temperature), and
  optionally `z_sfc` (surface altitude)

Notes:
- When generating 3D ICs (via `to_z_levels(...; interp3d=true)` in
  `weather_model_data_path`), the file can include `p_3d` and a `z_physical`
  field on the target grid, enabling the full-pressure path described above.
"""
function overwrite_initial_conditions!(
    initial_condition::WeatherModel,
    Y,
    thermo_params;
    use_full_pressure::Bool = false,
)
    regridder_type = :InterpolationsRegridder
    interpolation_method = Intp.Linear()
    extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())

    # target_levels defines vertical z for 1D interp,
    # if upstream processed ERA5 init file is missing, it is generated with to_z_levels_1d.
    z_arr = Array(Fields.field2array(Fields.coordinate_field(Y.c).z))
    z_top = round(maximum(z_arr))
    target_levels = collect(0.0:300.0:z_top)

    @info "Calling weather_model_data_path" (
        start_date = initial_condition.start_date,
        era5_dir = initial_condition.era5_initial_condition_dir,
    )

    file_path = weather_model_data_path(
        initial_condition.start_date,
        target_levels,
        initial_condition.era5_initial_condition_dir;
    )

    regridder_kwargs = (; extrapolation_bc, interpolation_method)

    isfile(file_path) || error("$(file_path) is not a file")
    @info "Overwriting initial conditions with data from file $(file_path)"

    center_space = Fields.axes(Y.c)
    face_space = Fields.axes(Y.f)

    ᶜT = SpaceVaryingInputs.SpaceVaryingInput(
        file_path,
        "t",
        center_space;
        regridder_type,
        regridder_kwargs = regridder_kwargs,
    )
    ᶜq_tot = SpaceVaryingInputs.SpaceVaryingInput(
        file_path,
        "q",
        center_space;
        regridder_type,
        regridder_kwargs = regridder_kwargs,
    )

    use_p3d = use_full_pressure && NC.NCDataset(file_path) do ds
        haskey(ds, "p_3d")
    end
    ᶠp = if use_p3d
        SpaceVaryingInputs.SpaceVaryingInput(
            file_path,
            "p_3d",
            face_space;
            regridder_type,
            regridder_kwargs = regridder_kwargs,
        )
    else
        if use_full_pressure
            @warn "Requested full pressure initialization, but variable `p_3d` is missing in $(file_path). Falling back to hydrostatic integration from surface pressure."
        end
        # Using surface pressure, air temperature and specific humidity
        # from the dataset, compute air pressure by hydrostatic integration.
        p_sfc = Fields.level(
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path,
                "p",
                face_space;
                regridder_type,
                regridder_kwargs = regridder_kwargs,
            ),
            Fields.half,
        )
        # Apply hydrostatic surface-pressure correction only if surface altitude is available
        surface_altitude_var = "z_sfc"
        has_surface_altitude = NC.NCDataset(file_path) do ds
            haskey(ds, surface_altitude_var)
        end
        if has_surface_altitude
            correct_surface_pressure_for_topography!(
                p_sfc,
                file_path,
                face_space,
                Y,
                ᶜT,
                ᶜq_tot,
                thermo_params,
                regridder_kwargs;
                surface_altitude_var = surface_altitude_var,
            )
        else
            @warn "Skipping topographic correction because variable `$surface_altitude_var` is missing from $(file_path)."
        end
        # With the known temperature (ᶜT) and moisture (ᶜq_tot) profile,
        # recompute the pressure levels assuming hydrostatic balance is maintained.
        ᶜ∂lnp∂z = @. -thermo_params.grav /
           (TD.gas_constant_air(thermo_params, ᶜq_tot) * ᶜT)
        ᶠlnp_over_psfc = zeros(face_space)
        Operators.column_integral_indefinite!(ᶠlnp_over_psfc, ᶜ∂lnp∂z)
        p_sfc .* exp.(ᶠlnp_over_psfc)
    end
    ᶜρ = TD.air_density.(thermo_params, ᶜT, ᶜinterp.(ᶠp), ᶜq_tot)

    # Assign prognostic variables from equilibrium moisture models
    Y.c.ρ .= TD.air_density.(thermo_params, ᶜT, ᶜinterp.(ᶠp), ᶜq_tot)
    # Velocity is first assigned on cell-centers and then interpolated onto
    # cell faces.
    vel =
        Geometry.UVWVector.(
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path,
                "u",
                center_space;
                regridder_type,
                regridder_kwargs = regridder_kwargs,
            ),
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path,
                "v",
                center_space;
                regridder_type,
                regridder_kwargs = regridder_kwargs,
            ),
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path,
                "w",
                center_space;
                regridder_type,
                regridder_kwargs = regridder_kwargs,
            ),
        )
    Y.c.uₕ .= C12.(Geometry.UVVector.(vel))
    Y.f.u₃ .= ᶠinterp.(C3.(Geometry.WVector.(vel)))
    e_kin = similar(ᶜT)
    e_kin .= compute_kinetic(Y.c.uₕ, Y.f.u₃)
    e_pot = geopotential.(thermo_params.grav, Fields.coordinate_field(Y.c).z)
    Y.c.ρe_tot .= TD.total_energy.(thermo_params, e_kin, e_pot, ᶜT, ᶜq_tot) .* Y.c.ρ
    # Initialize prognostic EDMF 0M subdomains if present
    if hasproperty(Y.c, :sgsʲs)
        ᶜmse = TD.enthalpy.(thermo_params, ᶜT, ᶜq_tot) .+ e_pot
        for name in propertynames(Y.c.sgsʲs)
            s = getproperty(Y.c.sgsʲs, name)
            hasproperty(s, :ρa) && fill!(s.ρa, 0)
            hasproperty(s, :mse) && (s.mse .= ᶜmse)
            hasproperty(s, :q_tot) && (s.q_tot .= ᶜq_tot)
        end
    end
    if hasproperty(Y.c, :ρq_tot)
        Y.c.ρq_tot .= ᶜq_tot .* Y.c.ρ
    else
        error(
            "`dry` configurations are incompatible with the interpolated initial conditions.",
        )
    end

    if hasproperty(Y.c, :ρq_liq)
        fill!(Y.c.ρq_liq, 0)
    end
    if hasproperty(Y.c, :ρq_ice)
        fill!(Y.c.ρq_ice, 0)
    end

    if hasproperty(Y.c, :ρq_sno) && hasproperty(Y.c, :ρq_rai)
        Y.c.ρq_sno .=
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path,
                "cswc",
                center_space;
                regridder_type,
                regridder_kwargs = regridder_kwargs,
            ) .* Y.c.ρ
        Y.c.ρq_rai .=
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path,
                "crwc",
                center_space;
                regridder_type,
                regridder_kwargs = regridder_kwargs,
            ) .* Y.c.ρ
    end

    if hasproperty(Y.c, :ρtke)
        # NOTE: This is not the most consistent, but it is better than NaNs
        fill!(Y.c.ρtke, 0)
    end

    return nothing
end

"""
    overwrite_initial_conditions!(initial_condition::AMIPFromERA5, Y, thermo_params)

Overwrite the prognostic state `Y` with ERA5-derived initial conditions for AMIP simulations.
Initial condition corresponds to Jan 1, 2010 at 00Z.
Uses hydrostatic integration from surface pressure to compute the pressure profile.

Expected variables in the IC file:
- 3D: `u`, `v`, `w`, `t`, `q`
- 2D broadcast in `z`: `p` (surface pressure)
"""
function overwrite_initial_conditions!(
    initial_condition::AMIPFromERA5,
    Y,
    thermo_params,
)
    # Get file path from AMIP artifact
    dt = parse_date(initial_condition.start_date)
    start_date_str = Dates.format(dt, "yyyymmdd")

    file_path = joinpath(
        @clima_artifact("era5_inst_model_levels"),
        "era5_init_processed_internal_$(start_date_str)_0000.nc",
    )

    extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())

    return _overwrite_initial_conditions_from_file!(
        file_path,
        extrapolation_bc,
        Y,
        thermo_params;
        regridder_type = :InterpolationsRegridder,
        interpolation_method = Intp.Linear(),
    )
end

# TODO: remove once we fully migrate to Setups (duplicated in setups/overwrite_from_file.jl)
"""
    _overwrite_initial_conditions_from_file!(file_path, extrapolation_bc, Y, thermo_params;
                                              regridder_type=nothing, interpolation_method=nothing)

Given a prognostic state `Y`, an `initial condition` (specifically, where initial values are
assigned from interpolations of existing datasets), and `thermo_params`, this function
overwrites the default initial condition and populates prognostic variables with
interpolated values using the `SpaceVaryingInputs` tool. To mitigate issues related to
unbalanced states following the interpolation operation, we recompute vertical pressure
levels assuming hydrostatic balance, given the surface pressure.

We expect the file to contain the following variables:
- `p`, for pressure,
- `t`, for temperature,
- `q`, for humidity,
- `u, v, w`, for velocity,
- `cswc, crwc` for snow and rain water content (for 1 moment microphysics).

If the file contains `z_sfc` (surface altitude), a hydrostatic correction is applied
to account for differences between the file's orography and the model's topography.

Optional keyword arguments:
- `regridder_type`: The regridder type to use (e.g., `:InterpolationsRegridder`)
- `interpolation_method`: The interpolation method (e.g., `Intp.Linear()`)
"""
function _overwrite_initial_conditions_from_file!(
    file_path::String,
    extrapolation_bc,
    Y,
    thermo_params;
    regridder_type = nothing,
    interpolation_method = nothing,
)
    regridder_kwargs = if isnothing(extrapolation_bc) && isnothing(interpolation_method)
        ()
    elseif isnothing(interpolation_method)
        (; extrapolation_bc)
    elseif isnothing(extrapolation_bc)
        (; interpolation_method)
    else
        (; extrapolation_bc, interpolation_method)
    end
    svi_kwargs =
        isnothing(regridder_type) ? (; regridder_kwargs) :
        (; regridder_type, regridder_kwargs)
    isfile(file_path) || error("$(file_path) is not a file")
    @info "Overwriting initial conditions with data from file $(file_path)"
    center_space = Fields.axes(Y.c)
    face_space = Fields.axes(Y.f)
    # Using surface pressure, air temperature and specific humidity
    # from the dataset, compute air pressure.
    p_sfc = Fields.level(
        SpaceVaryingInputs.SpaceVaryingInput(
            file_path,
            "p",
            face_space;
            svi_kwargs...,
        ),
        Fields.half,
    )
    ᶜT = SpaceVaryingInputs.SpaceVaryingInput(
        file_path,
        "t",
        center_space;
        svi_kwargs...,
    )
    ᶜq_tot = SpaceVaryingInputs.SpaceVaryingInput(
        file_path,
        "q",
        center_space;
        svi_kwargs...,
    )

    # Apply hydrostatic surface-pressure correction only if surface altitude is available
    surface_altitude_var = "z_sfc"
    has_surface_altitude = NC.NCDataset(file_path) do ds
        haskey(ds, surface_altitude_var)
    end
    if has_surface_altitude
        correct_surface_pressure_for_topography!(
            p_sfc,
            file_path,
            face_space,
            Y,
            ᶜT,
            ᶜq_tot,
            thermo_params,
            regridder_kwargs;
            surface_altitude_var = surface_altitude_var,
        )
    end

    # With the known temperature (ᶜT) and moisture (ᶜq_tot) profile,
    # recompute the pressure levels assuming hydrostatic balance is maintained.
    # Uses the ClimaCore `column_integral_indefinite!` function to solve
    # ∂(ln𝑝)/∂z = -g/(Rₘ(q)T), where
    # p is the local pressure
    # g is the gravitational constant
    # q is the specific humidity
    # Rₘ is the gas constant for moist air
    # T is the air temperature
    # p is then updated with the integral result, given p_sfc,
    # following which the thermodynamic state is constructed.
    ᶜ∂lnp∂z = @. -thermo_params.grav /
                 (TD.gas_constant_air(thermo_params, ᶜq_tot) * ᶜT)
    ᶠlnp_over_psfc = zeros(face_space)
    Operators.column_integral_indefinite!(ᶠlnp_over_psfc, ᶜ∂lnp∂z)
    ᶠp = p_sfc .* exp.(ᶠlnp_over_psfc)
    # Assign prognostic variables from equilibrium moisture models
    Y.c.ρ .= TD.air_density.(thermo_params, ᶜT, ᶜinterp.(ᶠp), ᶜq_tot)
    # Velocity is first assigned on cell-centers and then interpolated onto
    # cell faces.
    vel =
        Geometry.UVWVector.(
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path,
                "u",
                center_space;
                svi_kwargs...,
            ),
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path,
                "v",
                center_space;
                svi_kwargs...,
            ),
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path,
                "w",
                center_space;
                svi_kwargs...,
            ),
        )
    Y.c.uₕ .= C12.(Geometry.UVVector.(vel))
    Y.f.u₃ .= ᶠinterp.(C3.(Geometry.WVector.(vel)))
    e_kin = similar(ᶜT)
    e_kin .= compute_kinetic(Y.c.uₕ, Y.f.u₃)
    e_pot = geopotential.(thermo_params.grav, Fields.coordinate_field(Y.c).z)
    Y.c.ρe_tot .= TD.total_energy.(thermo_params, e_kin, e_pot, ᶜT, ᶜq_tot) .* Y.c.ρ
    if hasproperty(Y.c, :ρq_tot)
        Y.c.ρq_tot .= ᶜq_tot .* Y.c.ρ
    else
        error(
            "`dry` configurations are incompatible with the interpolated initial conditions.",
        )
    end

    if hasproperty(Y.c, :ρq_liq)
        fill!(Y.c.ρq_liq, 0)
    end
    if hasproperty(Y.c, :ρq_ice)
        fill!(Y.c.ρq_ice, 0)
    end
    if hasproperty(Y.c, :ρq_sno) && hasproperty(Y.c, :ρq_rai)
        has_microphysics_vars = NC.NCDataset(file_path) do ds
            haskey(ds, "cswc") && haskey(ds, "crwc")
        end
        if has_microphysics_vars
            Y.c.ρq_sno .=
                SpaceVaryingInputs.SpaceVaryingInput(
                    file_path,
                    "cswc",
                    center_space;
                    svi_kwargs...,
                ) .* Y.c.ρ
            Y.c.ρq_rai .=
                SpaceVaryingInputs.SpaceVaryingInput(
                    file_path,
                    "crwc",
                    center_space;
                    svi_kwargs...,
                ) .* Y.c.ρ
        else
            fill!(Y.c.ρq_sno, 0)
            fill!(Y.c.ρq_rai, 0)
        end
    end
    # Initialize prognostic EDMF 0M subdomains if present
    if hasproperty(Y.c, :sgsʲs)
        ᶜmse = TD.enthalpy.(thermo_params, ᶜT, ᶜq_tot) .+ e_pot
        for name in propertynames(Y.c.sgsʲs)
            s = getproperty(Y.c.sgsʲs, name)
            hasproperty(s, :ρa) && fill!(s.ρa, 0)
            hasproperty(s, :mse) && (s.mse .= ᶜmse)
            hasproperty(s, :q_tot) && (s.q_tot .= ᶜq_tot)
        end
    end

    if hasproperty(Y.c, :ρtke)
        # NOTE: This is not the most consistent, but it is better than NaNs
        fill!(Y.c.ρtke, 0)
    end

    return nothing
end

##
## Baroclinic Wave
##
function shallow_atmos_baroclinic_wave_values(z, ϕ, λ, params, perturb)
    FT = eltype(params)
    R_d = CAP.R_d(params)
    MSLP = CAP.MSLP(params)
    grav = CAP.grav(params)
    Ω = CAP.Omega(params)
    R = CAP.planet_radius(params)

    # Constants from paper
    k = 3
    T_e = FT(310) # temperature at the equator
    T_p = FT(240) # temperature at the pole
    T_0 = FT(0.5) * (T_e + T_p)
    Γ = FT(0.005)
    A = 1 / Γ
    B = (T_0 - T_p) / T_0 / T_p
    C = FT(0.5) * (k + 2) * (T_e - T_p) / T_e / T_p
    b = 2
    H = R_d * T_0 / grav
    z_t = FT(15e3)
    λ_c = FT(20)
    ϕ_c = FT(40)
    d_0 = R / 6
    V_p = FT(1)

    # Virtual temperature and pressure
    τ_z_1 = exp(Γ * z / T_0)
    τ_z_2 = 1 - 2 * (z / b / H)^2
    τ_z_3 = exp(-(z / b / H)^2)
    τ_1 = 1 / T_0 * τ_z_1 + B * τ_z_2 * τ_z_3
    τ_2 = C * τ_z_2 * τ_z_3
    τ_int_1 = A * (τ_z_1 - 1) + B * z * τ_z_3
    τ_int_2 = C * z * τ_z_3
    I_T = cosd(ϕ)^k - k * (cosd(ϕ))^(k + 2) / (k + 2)
    T_v = (τ_1 - τ_2 * I_T)^(-1)
    p = MSLP * exp(-grav / R_d * (τ_int_1 - τ_int_2 * I_T))

    # Horizontal velocity
    U = grav * k / R * τ_int_2 * T_v * (cosd(ϕ)^(k - 1) - cosd(ϕ)^(k + 1))
    u = -Ω * R * cosd(ϕ) + sqrt((Ω * R * cosd(ϕ))^2 + R * cosd(ϕ) * U)
    v = FT(0)
    if perturb
        F_z = (1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3) * (z ≤ z_t)
        r = R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c))
        c3 = cos(π * r / 2 / d_0)^3
        s1 = sin(π * r / 2 / d_0)
        cond = (0 < r < d_0) * (r != R * pi)
        u +=
            -16 * V_p / 3 / sqrt(FT(3)) *
            F_z *
            c3 *
            s1 *
            (-sind(ϕ_c) * cosd(ϕ) + cosd(ϕ_c) * sind(ϕ) * cosd(λ - λ_c)) /
            sin(r / R) * cond
        v +=
            16 * V_p / 3 / sqrt(FT(3)) *
            F_z *
            c3 *
            s1 *
            cosd(ϕ_c) *
            sind(λ - λ_c) / sin(r / R) * cond
    end

    return (; T_v, p, u, v)
end

function deep_atmos_baroclinic_wave_values(z, ϕ, λ, params, perturb)
    FT = eltype(params)
    R_d = CAP.R_d(params)
    MSLP = CAP.MSLP(params)
    grav = CAP.grav(params)
    Ω = CAP.Omega(params)
    R = CAP.planet_radius(params)

    # Constants from paper (See Table 1. in Ullrich et al (2014))
    k = 3         # Power for temperature field
    T_e = FT(310) # Surface temperature at the equator
    T_p = FT(240) # Surface temperature at the pole
    T_0 = FT(0.5) * (T_e + T_p)
    Γ = FT(0.005) # Lapse rate
    A = 1 / Γ  # (Eq 16)
    B = (T_0 - T_p) / T_0 / T_p # (Eq 17)
    C = FT(0.5) * (k + 2) * (T_e - T_p) / T_e / T_p # (Eq 17)
    b = 2 # half-width parameter
    H = R_d * T_0 / grav
    z_t = FT(15e3) # Top of perturbation domain
    λ_c = FT(20) # Geographical location (λ dim) of perturbation center
    ϕ_c = FT(40) # Geographical location (ϕ dim) of perturbation center
    d_0 = R / 6
    V_p = FT(1)

    # Virtual temperature and pressure
    τ̃₁ =
        A * Γ / T_0 * exp(Γ * z / T_0) +
        B * (1 - 2 * (z / b / H)^2) * exp(-(z / b / H)^2)# (Eq 14)
    τ̃₂ = C * (1 - 2 * (z / b / H)^2) * exp(-(z / b / H)^2) # (Eq 15)
    ∫τ̃₁ = (A * (exp(Γ * z / T_0) - 1)) + B * z * exp(-(z / b / H)^2) # (Eq A1)
    ∫τ̃₂ = C * z * exp(-(z / b / H)^2) # (Eq A2)
    I_T =
        ((z + R) / R * cosd(ϕ))^k -
        (k / (k + 2)) * ((z + R) / R * cosd(ϕ))^(k + 2)
    T_v = FT((R / (z + R))^2 * (τ̃₁ - τ̃₂ * I_T)^(-1)) # (Eq A3)
    p = FT(MSLP * exp(-grav / R_d * (∫τ̃₁ - ∫τ̃₂ * I_T))) # (Eq A6)
    # Horizontal velocity
    U =
        grav / R *
        k *
        T_v *
        ∫τ̃₂ *
        (((z + R) * cosd(ϕ) / R)^(k - 1) - ((R + z) * cosd(ϕ) / R)^(k + 1)) # wind-proxy (Eq A4)
    u = FT(
        -Ω * (R + z) * cosd(ϕ) +
        sqrt((Ω * (R + z) * cosd(ϕ))^2 + (R + z) * cosd(ϕ) * U),
    )
    v = FT(0)
    if perturb
        F_z = (1 - 3 * (z / z_t)^2 + 2 * (z / z_t)^3) * (z ≤ z_t)
        r = R * acos(sind(ϕ_c) * sind(ϕ) + cosd(ϕ_c) * cosd(ϕ) * cosd(λ - λ_c))
        c3 = cos(π * r / 2 / d_0)^3
        s1 = sin(π * r / 2 / d_0)
        cond = (0 < r < d_0) * (r != R * pi)
        u +=
            -16 * V_p / 3 / sqrt(FT(3)) *
            F_z *
            c3 *
            s1 *
            (-sind(ϕ_c) * cosd(ϕ) + cosd(ϕ_c) * sind(ϕ) * cosd(λ - λ_c)) /
            sin(r / R) * cond
        v +=
            16 * V_p / 3 / sqrt(FT(3)) *
            F_z *
            c3 *
            s1 *
            cosd(ϕ_c) *
            sind(λ - λ_c) / sin(r / R) * cond
    end
    return (; T_v, p, u, v)
end

function moist_baroclinic_wave_values(z, ϕ, λ, params, perturb, deep_atmosphere)
    FT = eltype(params)
    MSLP = CAP.MSLP(params)

    # Constants from paper
    p_w = FT(3.4e4)
    p_t = FT(1e4)
    q_t = FT(1e-12)
    q_0 = FT(0.018)
    ϕ_w = FT(40)
    ε = FT(0.608)

    if deep_atmosphere
        (; p, T_v, u, v) =
            deep_atmos_baroclinic_wave_values(z, ϕ, λ, params, perturb)
    else
        (; p, T_v, u, v) =
            shallow_atmos_baroclinic_wave_values(z, ϕ, λ, params, perturb)
    end

    q_tot =
        (p <= p_t) ? q_t : q_0 * exp(-(ϕ / ϕ_w)^4) * exp(-((p - MSLP) / p_w)^2)
    T = T_v / (1 + ε * q_tot) # This is the formula used in the paper.

    # This is the actual formula, which would be consistent with TD:
    # T = T_v * (1 + q_tot) / (1 + q_tot * CAP.Rv_over_Rd(params))

    return (; T, p, q_tot, u, v)
end

"""
    DryBaroclinicWave(; perturb = true, deep_atmosphere = false)

An `InitialCondition` with a dry baroclinic wave, and with an optional
perturbation to the horizontal velocity.
"""
Base.@kwdef struct DryBaroclinicWave <: InitialCondition
    perturb::Bool = true
    deep_atmosphere::Bool = false
end

function (initial_condition::DryBaroclinicWave)(params)
    (; perturb, deep_atmosphere) = initial_condition
    function local_state(local_geometry)
        (; z, lat, long) = local_geometry.coordinates
        if deep_atmosphere
            (; p, T_v, u, v) =
                deep_atmos_baroclinic_wave_values(z, lat, long, params, perturb)
        else
            (; p, T_v, u, v) = shallow_atmos_baroclinic_wave_values(
                z,
                lat,
                long,
                params,
                perturb,
            )
        end

        return LocalState(;
            params,
            geometry = local_geometry,
            T = T_v,
            p = p,
            velocity = Geometry.UVVector(u, v),
        )
    end
    return local_state
end

"""
    MoistBaroclinicWave(; perturb = true, deep_atmosphere = false)

An `InitialCondition` with a moist baroclinic wave, and with an optional
perturbation to the horizontal velocity.
"""
Base.@kwdef struct MoistBaroclinicWave <: InitialCondition
    perturb::Bool = true
    deep_atmosphere::Bool = false
end

function (initial_condition::MoistBaroclinicWave)(params)
    (; perturb, deep_atmosphere) = initial_condition
    function local_state(local_geometry)
        (; z, lat, long) = local_geometry.coordinates
        (; p, T, q_tot, u, v) = moist_baroclinic_wave_values(
            z,
            lat,
            long,
            params,
            perturb,
            deep_atmosphere,
        )
        return LocalState(;
            params,
            geometry = local_geometry,
            T = T,
            p = p,
            q_tot = q_tot,
            velocity = Geometry.UVVector(u, v),
        )
    end
    return local_state
end

"""
    MoistBaroclinicWaveWithEDMF(; perturb = true, deep_atmosphere = false)

The same `InitialCondition` as `MoistBaroclinicWave`, except with an initial TKE
of 0 and an initial draft area fraction of 0.2.
"""
Base.@kwdef struct MoistBaroclinicWaveWithEDMF <: InitialCondition
    perturb::Bool = true
    deep_atmosphere::Bool = false
end

function (initial_condition::MoistBaroclinicWaveWithEDMF)(params)
    (; perturb, deep_atmosphere) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        (; z, lat, long) = local_geometry.coordinates
        (; p, T, q_tot, u, v) = moist_baroclinic_wave_values(
            z,
            lat,
            long,
            params,
            perturb,
            deep_atmosphere,
        )
        return LocalState(;
            params,
            geometry = local_geometry,
            T = T,
            p = p,
            q_tot = q_tot,
            velocity = Geometry.UVVector(u, v),
            turbconv_state = EDMFState(; tke = FT(0), draft_area = FT(0.2)),
        )
    end
    return local_state
end

##
## EDMFX Test
##
"""
    MoistAdiabaticProfileEDMFX(; perturb = true)

An `InitialCondition` with a moist adiabatic temperature profile, and with an optional
perturbation to the temperature.
"""
Base.@kwdef struct MoistAdiabaticProfileEDMFX <: InitialCondition
    perturb::Bool = false
end

draft_area(::Type{FT}) where {FT} =
    z -> z < 0.7e4 ? FT(0.5) * exp(-(z - FT(4e3))^2 / 2 / FT(1e3)^2) : FT(0)

edmfx_q_tot(::Type{FT}) where {FT} =
    z -> z < 0.7e4 ? FT(1e-3) * exp(-(z - FT(4e3))^2 / 2 / FT(1e3)^2) : FT(0)

function (initial_condition::MoistAdiabaticProfileEDMFX)(params)
    (; perturb) = initial_condition
    function local_state(local_geometry)
        FT = eltype(params)
        thermo_params = CAP.thermodynamics_params(params)
        temp_profile = DryAdiabaticProfile{FT}(thermo_params, FT(330), FT(200))

        (; z) = local_geometry.coordinates
        coeff = perturb_coeff(local_geometry.coordinates)
        T, p = temp_profile(thermo_params, z)
        if perturb
            T += coeff * FT(0.1) * (z < 5000)
        end
        q_tot = edmfx_q_tot(FT)(z)

        return LocalState(;
            params,
            geometry = local_geometry,
            T = T,
            p = p,
            q_tot = q_tot,
            turbconv_state = EDMFState(;
                tke = FT(0),
                draft_area = draft_area(FT)(z),
                velocity = Geometry.WVector(FT(1.0)),
            ),
        )
    end
    return local_state
end

"""
    SimplePlume(; perturb = true)

An `InitialCondition` with a moist adiabatic temperature profile
"""
Base.@kwdef struct SimplePlume <: InitialCondition
    prognostic_tke::Bool = false
end

function (initial_condition::SimplePlume)(params)
    function local_state(local_geometry)
        FT = eltype(params)
        thermo_params = CAP.thermodynamics_params(params)
        temp_profile = DryAdiabaticProfile{FT}(thermo_params, FT(310), FT(290))

        (; z) = local_geometry.coordinates
        T, p = temp_profile(thermo_params, z)
        q_tot = FT(0)

        return LocalState(;
            params,
            geometry = local_geometry,
            T = T,
            p = p,
            q_tot = q_tot,
            turbconv_state = EDMFState(; tke = FT(0)),
        )
    end
    return local_state
end
##
## EDMF Test Cases
##
# TODO: Get rid of this
import AtmosphericProfilesLibrary as APL

# TODO: remove once we fully migrate to Setups (duplicated in setups/hydrostatic.jl)
const FunctionOrSpline =
    Union{Function, APL.AbstractProfile, Intp.Extrapolation}

"""
    hydrostatic_pressure_profile(; thermo_params, p_0, [T, θ, q_tot, z_max])

Solves the initial value problem `p'(z) = -g * ρ(z)` for all `z ∈ [0, z_max]`,
given `p(0)`, either `T(z)` or `θ(z)`, and optionally also `q_tot(z)`. If
`q_tot(z)` is not given, it is assumed to be 0. If `z_max` is not given, it is
assumed to be 30 km. Note that `z_max` should be the maximum elevation to which
the specified profiles T(z), θ(z), and/or q_tot(z) are valid.
"""
function hydrostatic_pressure_profile(;
    thermo_params,
    p_0,
    T = nothing,
    θ = nothing,
    q_tot = nothing,
    z_max = 30000,
)
    FT = eltype(thermo_params)
    grav = TD.Parameters.grav(thermo_params)

    # Compute air density from (p, z) using either T(z) or θ(z), with optional q_tot(z)
    function ρ_from_profile(p, z, ::Nothing, ::Nothing, _)
        error("Either T or θ must be specified")
    end
    function ρ_from_profile(p, z, T::FunctionOrSpline, θ::FunctionOrSpline, _)
        error("Only one of T and θ can be specified")
    end
    function ρ_from_profile(p, z, T::FunctionOrSpline, ::Nothing, ::Nothing)
        TD.air_density(thermo_params, oftype(p, T(z)), p)
    end
    function ρ_from_profile(p, z, ::Nothing, θ::FunctionOrSpline, ::Nothing)
        T_val = TD.air_temperature(thermo_params, TD.pθ_li(), p, oftype(p, θ(z)))
        TD.air_density(thermo_params, T_val, p)
    end
    function ρ_from_profile(p, z, T::FunctionOrSpline, ::Nothing, q_tot::FunctionOrSpline)
        TD.air_density(thermo_params, oftype(p, T(z)), p, oftype(p, q_tot(z)), FT(0), FT(0))
    end
    function ρ_from_profile(p, z, ::Nothing, θ::FunctionOrSpline, q_tot::FunctionOrSpline)
        q = oftype(p, q_tot(z))
        T_val = TD.air_temperature(thermo_params, TD.pθ_li(), p, oftype(p, θ(z)), q)
        TD.air_density(thermo_params, T_val, p, q, FT(0), FT(0))
    end
    dp_dz(p, z) = -grav * ρ_from_profile(p, z, T, θ, q_tot)

    return column_indefinite_integral(dp_dz, p_0, (FT(0), FT(z_max)))
end

"""
    GABLS

The `InitialCondition` described in [Kosovic2000](@cite), but with a hydrostatically
balanced pressure profile.
"""
Base.@kwdef struct GABLS <: InitialCondition
    prognostic_tke::Bool = false
end

for IC in (:GABLS,)
    θ_func_name = Symbol(IC, :_θ_liq_ice)
    u_func_name = Symbol(IC, :_u)
    tke_func_name = Symbol(IC, :_tke_prescribed)
    @eval function (initial_condition::$IC)(params)
        (; prognostic_tke) = initial_condition
        FT = eltype(params)
        thermo_params = CAP.thermodynamics_params(params)
        p_0 = FT(100000.0)
        θ = APL.$θ_func_name(FT)
        p = hydrostatic_pressure_profile(; thermo_params, p_0, θ)
        u = APL.$u_func_name(FT)
        tke = APL.$tke_func_name(FT)
        function local_state(local_geometry)
            (; z) = local_geometry.coordinates
            T = TD.air_temperature(thermo_params, TD.pθ_li(), p(z), θ(z))
            return LocalState(;
                params,
                geometry = local_geometry,
                T = T,
                p = p(z),
                velocity = Geometry.UVector(u(z)),
                turbconv_state = EDMFState(;
                    tke = prognostic_tke ? FT(0) : tke(z),
                ),
            )
        end
        return local_state
    end
end

"""
    GATE_III

The `InitialCondition` described in [Khairoutdinov2009](@cite), but with a
hydrostatically balanced pressure profile.
"""
Base.@kwdef struct GATE_III <: InitialCondition
    prognostic_tke::Bool = false
end

function (initial_condition::GATE_III)(params)
    (; prognostic_tke) = initial_condition
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    p_0 = FT(101500.0)
    T_profile = APL.GATE_III_T(FT)
    q_tot_profile = APL.GATE_III_q_tot(FT)
    p = hydrostatic_pressure_profile(;
        thermo_params,
        p_0,
        T = T_profile,
        q_tot = q_tot_profile,
    )
    u = APL.GATE_III_u(FT)
    tke = APL.GATE_III_tke(FT)
    function local_state(local_geometry)
        (; z) = local_geometry.coordinates
        return LocalState(;
            params,
            geometry = local_geometry,
            T = T_profile(z),
            p = p(z),
            q_tot = q_tot_profile(z),
            velocity = Geometry.UVector(u(z)),
            turbconv_state = EDMFState(; tke = prognostic_tke ? FT(0) : tke(z)),
        )
    end
    return local_state
end

"""
    Soares

The `InitialCondition` described in [Soares2004](@cite), but with a
hydrostatically balanced pressure profile.
"""
Base.@kwdef struct Soares <: InitialCondition
    prognostic_tke::Bool = false
end

function (initial_condition::Soares)(params)
    (; prognostic_tke) = initial_condition
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    p_0 = FT(100000.0)
    θ = APL.Soares_θ_liq_ice(FT)
    q_tot = APL.Soares_q_tot(FT)
    p = hydrostatic_pressure_profile(; thermo_params, p_0, θ, q_tot)
    u = APL.Soares_u(FT)
    tke = APL.Soares_tke_prescribed(FT)
    function local_state(local_geometry)
        (; z) = local_geometry.coordinates
        T = TD.air_temperature(thermo_params, TD.pθ_li(), p(z), θ(z), q_tot(z))
        return LocalState(;
            params,
            geometry = local_geometry,
            T = T,
            p = p(z),
            q_tot = q_tot(z),
            velocity = Geometry.UVector(u(z)),
            turbconv_state = EDMFState(;
                tke = prognostic_tke ? FT(0) : tke(z),
            ),
        )
    end
    return local_state
end

"""
    DYCOMS_RF01

The `InitialCondition` described in [Stevens2005](@cite), but with a
hydrostatically balanced pressure profile.
"""
Base.@kwdef struct DYCOMS_RF01 <: InitialCondition
    prognostic_tke::Bool = false
end

"""
    DYCOMS_RF02

The `InitialCondition` described in [Ackerman2009](@cite), but with a
hydrostatically balanced pressure profile.
"""
Base.@kwdef struct DYCOMS_RF02 <: InitialCondition
    prognostic_tke::Bool = false
end

for IC in (:Dycoms_RF01, :Dycoms_RF02)
    IC_Type = Symbol(uppercase(string(IC)))
    θ_func_name = Symbol(IC, :_θ_liq_ice)
    q_tot_func_name = Symbol(IC, :_q_tot)
    u_func_name = Symbol(IC, IC == :Dycoms_RF01 ? :_u0 : :_u)
    v_func_name = Symbol(IC, IC == :Dycoms_RF01 ? :_v0 : :_v)
    tke_func_name = Symbol(IC, :_tke_prescribed)
    @eval function (initial_condition::$IC_Type)(params)
        (; prognostic_tke) = initial_condition
        FT = eltype(params)
        thermo_params = CAP.thermodynamics_params(params)
        p_0 = FT(101780.0)
        θ = APL.$θ_func_name(FT)
        q_tot = APL.$q_tot_func_name(FT)
        p = hydrostatic_pressure_profile(; thermo_params, p_0, θ, q_tot)
        u = APL.$u_func_name(FT)
        v = APL.$v_func_name(FT)
        tke = APL.Dycoms_RF01_tke_prescribed(FT)
        function local_state(local_geometry)
            (; z) = local_geometry.coordinates
            T = TD.air_temperature(thermo_params, TD.pθ_li(), p(z), θ(z), q_tot(z))
            return LocalState(;
                params,
                geometry = local_geometry,
                T = T,
                p = p(z),
                q_tot = q_tot(z),
                velocity = Geometry.UVVector(u(z), v(z)),
                turbconv_state = EDMFState(;
                    tke = prognostic_tke ? FT(0) : tke(z),
                ),
            )
        end
        return local_state
    end
end


"""
    TRMM_LBA

The `InitialCondition` described in [Grabowski2006](@cite), but with a
hydrostatically balanced pressure profile.
"""
Base.@kwdef struct TRMM_LBA <: InitialCondition
    prognostic_tke::Bool = false
end

function (initial_condition::TRMM_LBA)(params)
    (; prognostic_tke) = initial_condition
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    p_0 = FT(99130.0)
    T_profile = APL.TRMM_LBA_T(FT)

    # Set q_tot to the value implied by the measured pressure and relative
    # humidity profiles (see the definition of relative humidity and equation 37
    # in Pressel et al., 2015). Note that the measured profiles are different from the
    # ones required for hydrostatic balance.
    # TODO: Move this to APL.
    Rv_over_Rd = TD.Parameters.Rv_over_Rd(thermo_params)
    measured_p = APL.TRMM_LBA_p(FT)
    measured_RH = APL.TRMM_LBA_RH(FT)
    measured_z_values = APL.TRMM_LBA_z(FT)
    measured_q_tot_values = map(measured_z_values) do z
        p_v_sat = TD.saturation_vapor_pressure(thermo_params, T_profile(z), TD.Liquid())
        denominator =
            measured_p(z) - p_v_sat +
            (1 / Rv_over_Rd) * p_v_sat * measured_RH(z) / 100
        q_v_sat = p_v_sat * (1 / Rv_over_Rd) / denominator
        return q_v_sat * measured_RH(z) / 100
    end
    q_tot = Intp.extrapolate(
        Intp.interpolate(
            (measured_z_values,),
            measured_q_tot_values,
            Intp.Gridded(Intp.Linear()),
        ),
        Intp.Flat(),
    )

    p = hydrostatic_pressure_profile(; thermo_params, p_0, T = T_profile, q_tot)
    u = APL.TRMM_LBA_u(FT)
    v = APL.TRMM_LBA_v(FT)
    tke = APL.TRMM_LBA_tke_prescribed(FT)
    function local_state(local_geometry)
        (; z) = local_geometry.coordinates
        return LocalState(;
            params,
            geometry = local_geometry,
            T = T_profile(z),
            p = p(z),
            q_tot = q_tot(z),
            velocity = Geometry.UVVector(u(z), v(z)),
            turbconv_state = EDMFState(; tke = prognostic_tke ? FT(0) : tke(z)),
        )
    end
    return local_state
end

"""
    PrecipitatingColumn

A 1-dimensional precipitating column test
"""
struct PrecipitatingColumn <: InitialCondition end

prescribed_prof(::Type{FT}, z_mid, z_max, val) where {FT} =
    z -> z < z_max ? FT(val) * exp(-(z - FT(z_mid))^2 / 2 / FT(1e3)^2) : FT(0)

function (initial_condition::PrecipitatingColumn)(params)
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    p_0 = FT(101300.0)
    qᵣ = prescribed_prof(FT, 2000, 5000, 1e-6)
    qₛ = prescribed_prof(FT, 5000, 8000, 2e-6)
    qₗ = prescribed_prof(FT, 4000, 5500, 2e-5)
    qᵢ = prescribed_prof(FT, 6000, 9000, 1e-5)
    nₗ = prescribed_prof(FT, 4000, 5500, 1e7)
    nᵣ = prescribed_prof(FT, 2000, 5000, 1e3)
    θ = APL.Rico_θ_liq_ice(FT)
    q_tot_profile = APL.Rico_q_tot(FT)
    u = prescribed_prof(FT, 0, Inf, 0)
    v = prescribed_prof(FT, 0, Inf, 0)
    p = hydrostatic_pressure_profile(; thermo_params, p_0, θ, q_tot = q_tot_profile)
    function local_state(local_geometry)
        (; z) = local_geometry.coordinates
        q_liq_z = qₗ(z) + qᵣ(z)
        q_ice_z = qᵢ(z) + qₛ(z)
        q_tot_z = q_tot_profile(z)
        T = TD.air_temperature(
            thermo_params,
            TD.pθ_li(),
            p(z),
            θ(z),
            q_tot_z,
            q_liq_z,
            q_ice_z,
        )
        return LocalState(;
            params,
            geometry = local_geometry,
            T = T,
            p = p(z),
            q_tot = q_tot_z,
            q_liq = q_liq_z,
            q_ice = q_ice_z,
            velocity = Geometry.UVVector(u(z), v(z)),
            turbconv_state = nothing,
            precip_state = PrecipStateMassNum(;
                n_liq = nₗ(z),
                n_rai = nᵣ(z),
                q_rai = qᵣ(z),
                q_sno = qₛ(z),
            ),
        )
    end
    return local_state
end

"""
    InterpolatedColumnProfile <: InitialCondition

Initial data condition for a column model. Stored as a tuple of Interpolation
objects. Temperature, zonal wind velocity, meridional wind velocity,
total specific humidity, and density are all needed to construct the initial
condition. Type `F` must be callable, i.e., F(z) where z is a number. This
could be an Interpolations.Extrapolation object or a function.
"""
struct InterpolatedColumnProfile{F} <: InitialCondition
    "temperature"
    T::F
    "zonal wind velocity"
    u::F
    "meridional wind velocity"
    v::F
    "total specific humidity"
    q_tot::F
    "air density"
    ρ₀::F
end

function (initial_condition::InterpolatedColumnProfile)(params)
    (; T, u, v, q_tot, ρ₀) = initial_condition
    thermo_params = CAP.thermodynamics_params(params)
    function local_state(local_geometry)
        (; z) = local_geometry.coordinates
        FT = typeof(z)
        T_val = FT(T(z))
        q_tot_val = FT(q_tot(z))
        ρ_val = FT(ρ₀(z))
        p = TD.air_pressure(thermo_params, T_val, ρ_val, q_tot_val, FT(0), FT(0))
        return LocalState(;
            params,
            geometry = local_geometry,
            T = T_val,
            p = p,
            q_tot = q_tot_val,
            velocity = Geometry.UVVector(FT(u(z)), FT(v(z))),
            turbconv_state = EDMFState(; tke = FT(0)),
        )
    end
    return local_state
end

"""
    external_tv_initial_condition(external_forcing_file, start_date)

Returns an `InterpolatedColumnProfile` object with the initial conditions
from the external forcing file for time varying data. The
`external_forcing_file` is a NetCDF file containing the external forcing
data, and `start_date` is a string in the format "yyyymmdd" that specifies
the date to use for the initial conditions.
"""
function external_tv_initial_condition(external_forcing_file, start_date)
    start_date = Dates.DateTime(start_date, "yyyymmdd")
    z, T, u, v, q_tot, ρ₀ = NC.NCDataset(external_forcing_file) do ds
        time_index = argmin(abs.(ds["time"][:] .- start_date))
        (
            z = ds["z"][:],
            T = ds["ta"][1, 1, :, time_index],
            u = ds["ua"][1, 1, :, time_index],
            v = ds["va"][1, 1, :, time_index],
            q_tot = ds["hus"][1, 1, :, time_index],
            ρ₀ = ds["rho"][1, 1, :, time_index],
        )
    end
    T, u, v, q_tot, ρ₀ = map((T, u, v, q_tot, ρ₀)) do value
        Intp.extrapolate(
            Intp.interpolate((z,), value, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
    end
    return InterpolatedColumnProfile(T, u, v, q_tot, ρ₀)
end

Base.@kwdef struct ISDAC <: InitialCondition
    prognostic_tke::Bool = false
    perturb::Bool = false
end

function (initial_condition::ISDAC)(params)
    (; prognostic_tke, perturb) = initial_condition
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    p_0 = FT(102000)  # 1020 hPa
    θ = APL.ISDAC_θ_liq_ice(FT) # K
    q_tot = APL.ISDAC_q_tot(FT)  # kg/kg
    # Note: ISDAC top-of-domain is ~1.5km, but we don't have access to that information here, so we use 5km to be safe
    p = hydrostatic_pressure_profile(;
        thermo_params,
        p_0,
        θ,
        q_tot,
        z_max = 5000,
    )  # Pa

    u = APL.ISDAC_u(FT)  # m/s
    v = APL.ISDAC_v(FT)  # m/s
    tke = APL.ISDAC_tke(FT)  # m²/s²

    # pseudorandom fluctuations with amplitude 0.1 K
    θ_pert(z::FT) where {FT} =
        perturb && (z < 825) ? FT(0.1) * randn(FT) : FT(0)

    function local_state(local_geometry)
        (; z) = local_geometry.coordinates
        T = TD.air_temperature(thermo_params, TD.pθ_li(), p(z), θ(z) + θ_pert(z), q_tot(z))
        return LocalState(;
            params,
            geometry = local_geometry,
            T = T,
            p = p(z),
            q_tot = q_tot(z),
            velocity = Geometry.UVVector(u(z), v(z)),
            turbconv_state = EDMFState(; tke = prognostic_tke ? tke(z) : FT(0)),
        )
    end
end

"""
    ShipwayHill2012

The `InitialCondition` described in [ShipwayHill2012](@cite), but with a hydrostatically
balanced pressure profile.

B. J. Shipway and A. A. Hill. 
Diagnosis of systematic differences between multiple parametrizations of warm rain microphysics using a kinematic framework. 
Quarterly Journal of the Royal Meteorological Society 138, 2196-2211 (2012).
"""
struct ShipwayHill2012 <: InitialCondition end
function (initial_condition::ShipwayHill2012)(params)
    FT = eltype(params)

    ## Initialize the profile
    z_values = FT[0, 740, 3260]
    rv_values = FT[0.015, 0.0138, 0.0024]  # water vapor mixing ratio
    θ_values = FT[297.9, 297.9, 312.66]    # potential temperature
    linear_profile(zs, vals) = Intp.extrapolate(
        Intp.interpolate((zs,), vals, Intp.Gridded(Intp.Linear())), Intp.Linear(),
    )
    # profile of water vapour mixing ratio
    rv(z) = linear_profile(z_values, rv_values)(z)
    q_tot(z) = rv(z) / (1 + rv(z))
    # profile of potential temperature
    θ(z) = linear_profile(z_values, θ_values)(z)
    ## Hydrostatically balanced pressure profile
    thermo_params = CAP.thermodynamics_params(params)
    p_0 = FT(100_700) # Pa
    p = hydrostatic_pressure_profile(; thermo_params, p_0, θ, q_tot)
    function local_state(local_geometry)
        (; z) = local_geometry.coordinates
        T = TD.air_temperature(thermo_params, TD.pθ_li(), p(z), θ(z), q_tot(z))
        return LocalState(; params, geometry = local_geometry,
            T = T,
            p = p(z),
            q_tot = q_tot(z),
            precip_state = NoPrecipState{typeof(z)}(),
        )
    end
    return local_state
end
