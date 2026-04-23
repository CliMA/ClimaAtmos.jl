"""
    ARMVARANAL — single column IC and forcing from ARM VARANAL format files.

Reads initial condition profiles and time-varying forcing from ARM Variational
Analysis (VARANAL) format NetCDF files. The VARANAL product provides semi-continuous
atmospheric forcing at ARM sites like SGP (Southern Great Plains).

File format characteristics:
- Dimensions: time (UNLIMITED), lev (pressure levels in hPa)
- Time: seconds since base_time (epoch format)
- Pressure levels: 1000 to 100 hPa (surface to TOA)
- Variables: T, q, u, v, omega, T_adv_h, T_adv_v, q_adv_h, q_adv_v, LH, SH, etc.
- Units: T in K, q in g/kg, tendencies in K/hr and g/kg/hr, omega in hPa/hr
"""

"""
    ARMVARANAL{P <: ColumnProfiles, FT, LHI, SHI}

Pointwise column IC driven by ARM VARANAL format NetCDF data.

Surface fluxes (LH, SH) are prescribed from the file as time-varying values.

## Fields
- `external_forcing_file`: Path to the ARM VARANAL NetCDF file.
- `case_name`: Name of the case (e.g., "SGP_VARANAL").
- `profiles`: `ColumnProfiles` of 1D interpolators `(T, u, v, q_tot, ρ)`.
- `T_sfc`: Initial surface temperature from the forcing file (K).
- `p_sfc`: Initial surface pressure from the forcing file (Pa).
- `lat`: Latitude of the site (degrees).
- `lon`: Longitude of the site (degrees).
- `alt`: Altitude above mean sea level (m).
- `start_date`: Start date/time from the file (for insolation calculations).
- `LH_init`: Initial latent heat flux (W/m²), for diagnostics.
- `SH_init`: Initial sensible heat flux (W/m²), for diagnostics.
- `LH_interp`: Time interpolator for latent heat flux (W/m²), or nothing.
- `SH_interp`: Time interpolator for sensible heat flux (W/m²), or nothing.
"""
struct ARMVARANAL{P <: ColumnProfiles, FT, LHI, SHI}
    external_forcing_file::String
    case_name::String
    profiles::P
    T_sfc::FT
    p_sfc::FT
    lat::FT
    lon::FT
    alt::FT
    start_date::Dates.DateTime
    LH_init::FT
    SH_init::FT
    LH_interp::LHI
    SH_interp::SHI
end

"""
    ARMVARANAL(external_forcing_file, case_name; thermo_params)

Construct an ARMVARANAL setup by reading initial condition profiles from an
ARM VARANAL format forcing file and building 1D vertical interpolators.

Surface fluxes (LH, SH) are prescribed from the file as time-varying values.

## Arguments
- `external_forcing_file`: Path to the ARM VARANAL NetCDF file.
- `case_name`: Case identifier (default: "SGP_VARANAL").
- `thermo_params`: Thermodynamics parameters (for physical constants).
"""
function ARMVARANAL(
    external_forcing_file::String,
    case_name::String = "SGP_VARANAL";
    thermo_params,
)
    # Local helpers
    function first_timestep(ds, varname, nlev)
        raw = Array(ds[varname])
        any(ismissing, raw) && error("$varname contains missing data; choose a different period")
        data = Float64.(raw)
        ndims(data) == 1 ? vec(data) :
        size(data, 1) == nlev ? vec(data[:, 1]) : vec(data[1, :])
    end

    function read_surface_scalar(ds, keys_convert, default)
        for (key, convert_fn) in keys_convert
            key in Base.keys(ds) || continue
            data = Float64.(Array(ds[key]))
            return convert_fn(length(data) > 1 ? data[1] : first(data))
        end
        default
    end

    function read_flux_init(ds, varname)
        varname in Base.keys(ds) || return 0.0
        raw = Array(ds[varname])
        val = length(raw) > 1 ? raw[1] : first(raw)
        ismissing(val) ? 0.0 : Float64(val)
    end

    function extract_start_date(ds)
        time_raw = vec(ds["time"][:])
        eltype(time_raw) <: Dates.DateTime && return time_raw[1]
        base_time = Dates.unix2datetime(ds.attrib["base_time"])
        base_time + Dates.Second(round(Int, time_raw[1]))
    end

    function create_flux_interpolator(ds, varname)
        time_raw = vec(ds["time"][:])
        time_sec = if eltype(time_raw) <: Dates.DateTime
            Float64.([Dates.value(t - time_raw[1]) / 1000.0 for t in time_raw])
        else
            Float64.(time_raw)
        end
        data = replace(Float64.(vec(ds[varname][:])), -9999.0 => NaN)
        Intp.extrapolate(
            Intp.interpolate((time_sec,), data, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
    end

    NC.NCDataset(external_forcing_file) do ds
        lev_pa = Float64.(vec(ds["lev"][:])) .* 100.0  # hPa -> Pa
        nlev = length(lev_pa)

        # Read IC profiles (first time step), convert q from g/kg
        T_raw = first_timestep(ds, "T", nlev)
        q_raw = first_timestep(ds, "q", nlev) ./ 1000.0
        u_raw = first_timestep(ds, "u", nlev)
        v_raw = first_timestep(ds, "v", nlev)

        # Ensure surface-first ordering (decreasing pressure with increasing z)
        if lev_pa[1] < lev_pa[end]
            lev_pa, T_raw, q_raw, u_raw, v_raw =
                reverse.((lev_pa, T_raw, q_raw, u_raw, v_raw))
        end

        z_lev = pressure_to_height(lev_pa, T_raw, q_raw, thermo_params)
        R_d, R_v = TD.Parameters.R_d(thermo_params), TD.Parameters.R_v(thermo_params)
        ρ_raw = lev_pa ./ (R_d .* T_raw .* (1 .+ (R_v / R_d - 1) .* q_raw))

        # Surface conditions with fallback chains (degC → K, hPa → Pa)
        sfc_T = read_surface_scalar(ds, [("T_skin", x -> x + 273.15), ("T_srf", x -> x + 273.15)], T_raw[1])
        sfc_p = read_surface_scalar(ds, [("p_srf_aver", x -> x * 100.0), ("p_srf_center", x -> x * 100.0)], lev_pa[1])

        # Sort by height (required for interpolation)
        idx = sortperm(z_lev)
        z, T, u, v, q, ρ = (arr[idx] for arr in (z_lev, T_raw, u_raw, v_raw, q_raw, ρ_raw))
        @info "ARMVARANAL data range" z_min=minimum(z) z_max=maximum(z)

        ARMVARANAL(
            external_forcing_file,
            case_name,
            ColumnProfiles(z, T, u, v, q, ρ),
            sfc_T,
            sfc_p,
            Float64(first(ds["lat"][:])),
            Float64(first(ds["lon"][:])),
            Float64(first(ds["alt"][:])),
            extract_start_date(ds),
            read_flux_init(ds, "LH"),
            read_flux_init(ds, "SH"),
            "LH" in Base.keys(ds) ? create_flux_interpolator(ds, "LH") : nothing,
            "SH" in Base.keys(ds) ? create_flux_interpolator(ds, "SH") : nothing,
        )
    end
end

center_initial_condition(setup::ARMVARANAL, local_geometry, params) =
    column_profiles_ic(setup.profiles, local_geometry)

function surface_condition(setup::ARMVARANAL, params)
    FT = eltype(params)
    z0 = FT(0.05)  # grassland/cropland at SGP
    T_sfc = FT(setup.T_sfc)
    LH_interp = setup.LH_interp
    SH_interp = setup.SH_interp

    function surface_state(surface_coordinates, interior_z, t)
        _FT = eltype(surface_coordinates)
        shf = _FT(SH_interp(t))
        lhf = _FT(LH_interp(t))
        shf = isnan(shf) ? _FT(0) : shf
        lhf = isnan(lhf) ? _FT(0) : lhf
        parameterization = MoninObukhov(; z0, shf, lhf)
        return SurfaceState(; parameterization, T = T_sfc)
    end
    return surface_state
end

external_forcing(setup::ARMVARANAL, ::Type{FT}) where {FT} =
    ARMVARANALForcing{FT}(setup.external_forcing_file)

surface_temperature_model(::ARMVARANAL) = ARMVARANALTimeVaryingSST()

insolation_model(setup::ARMVARANAL) = 
    ColumnTimeVaryingInsolation(setup.start_date, setup.lat, setup.lon)
