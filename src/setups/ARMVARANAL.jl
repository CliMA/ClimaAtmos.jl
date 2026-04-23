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

import NCDatasets as NC
import Interpolations as Intp
import Dates

"""
    ARMVARANAL{P <: ColumnProfiles, FT, LHI, SHI}

Pointwise column IC driven by ARM VARANAL format NetCDF data.

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
- `prescribe_surface_fluxes`: If true, prescribe LH/SH from file (default: true).
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
    prescribe_surface_fluxes::Bool
    LH_init::FT
    SH_init::FT
    LH_interp::LHI
    SH_interp::SHI
end

"""
    varanal_pressure_to_height_profile(p_pa, T, q)

Convert pressure levels to approximate geometric height using hypsometric equation.
p_pa: pressure levels in Pa
T: temperature profile in K
q: specific humidity profile in kg/kg
Returns heights in meters above the surface.
"""
function varanal_pressure_to_height_profile(p_pa, T, q)
    g = 9.80665
    R_d = 287.0
    
    # Sort by pressure (descending = surface to TOA)
    sort_idx = sortperm(p_pa, rev = true)
    p_sorted = p_pa[sort_idx]
    T_sorted = T[sort_idx]
    q_sorted = q[sort_idx]
    
    # Virtual temperature
    Tv = T_sorted .* (1.0 .+ 0.61 .* q_sorted)
    
    # Integrate hypsometric equation from surface
    n = length(p_sorted)
    z = zeros(n)
    z[1] = 0.0  # Surface
    
    for i in 2:n
        Tv_mean = 0.5 * (Tv[i-1] + Tv[i])
        dz = R_d * Tv_mean / g * log(p_sorted[i-1] / p_sorted[i])
        z[i] = z[i-1] + dz
    end
    
    # Return in original order
    inv_sort_idx = invperm(sort_idx)
    return z[inv_sort_idx]
end

"""
    ARMVARANAL(external_forcing_file::String, case_name::String="SGP_VARANAL"; 
               prescribe_surface_fluxes::Bool=true)

Construct an ARMVARANAL setup by reading initial condition profiles from an
ARM VARANAL format forcing file and building 1D vertical interpolators.

## Arguments
- `external_forcing_file`: Path to the ARM VARANAL NetCDF file.
- `case_name`: Case identifier (default: "SGP_VARANAL").
- `prescribe_surface_fluxes`: If true (default), prescribe surface LH/SH from file.
  If false, compute fluxes interactively via Monin-Obukhov.
"""
function ARMVARANAL(
    external_forcing_file::String,
    case_name::String = "SGP_VARANAL";
    prescribe_surface_fluxes::Bool = true,
)
    z_ref, T, u, v, q_tot, ρ, T_sfc, p_sfc, lat, lon, alt, start_date, LH_init, SH_init, LH_interp, SH_interp = NC.NCDataset(external_forcing_file) do ds
        # Pressure levels (hPa in file)
        lev_hpa = Float64.(vec(ds["lev"][:]))
        lev_pa = lev_hpa .* 100.0  # Convert to Pa
        
        # Read initial condition profiles (first time step)
        # NCDatasets gives Julia arrays with reversed dimension order from NetCDF
        # NetCDF: (time, lev) -> Julia: [lev, time] or [time, lev] depending on version
        T_data = Float64.(Array(ds["T"]))
        q_data = Float64.(Array(ds["q"]))  # g/kg in file
        u_data = Float64.(Array(ds["u"]))
        v_data = Float64.(Array(ds["v"]))
        
        # Get first time step for IC
        if ndims(T_data) == 2
            # Check dimension order by comparing sizes
            if size(T_data, 1) == length(lev_hpa)
                # [lev, time]
                T_raw = vec(T_data[:, 1])
                q_raw = vec(q_data[:, 1])
                u_raw = vec(u_data[:, 1])
                v_raw = vec(v_data[:, 1])
            else
                # [time, lev]
                T_raw = vec(T_data[1, :])
                q_raw = vec(q_data[1, :])
                u_raw = vec(u_data[1, :])
                v_raw = vec(v_data[1, :])
            end
        else
            T_raw = vec(T_data)
            q_raw = vec(q_data)
            u_raw = vec(u_data)
            v_raw = vec(v_data)
        end
        
        # Convert q from g/kg to kg/kg
        q_raw = q_raw ./ 1000.0
        
        # Handle missing values (-9999)
        T_raw = replace(T_raw, -9999.0 => NaN)
        q_raw = replace(q_raw, -9999.0 => NaN)
        u_raw = replace(u_raw, -9999.0 => NaN)
        v_raw = replace(v_raw, -9999.0 => NaN)
        
        # Handle pressure level ordering
        # We want surface (high p) to TOA (low p), i.e., decreasing pressure for increasing z
        if lev_pa[1] < lev_pa[end]
            # Already TOA to surface, need to reverse for surface-first
            lev_pa = reverse(lev_pa)
            T_raw = reverse(T_raw)
            q_raw = reverse(q_raw)
            u_raw = reverse(u_raw)
            v_raw = reverse(v_raw)
        end
        
        # Convert pressure to height
        z_lev = varanal_pressure_to_height_profile(lev_pa, T_raw, q_raw)
        
        # Compute density from ideal gas law
        R_d = 287.0
        R_v = 461.5
        ρ_raw = lev_pa ./ (R_d .* T_raw .* (1 .+ (R_v / R_d - 1) .* q_raw))
        
        # Surface conditions
        # Use T_skin (surface skin temperature) for Monin-Obukhov, in degC -> K
        # T_skin is more appropriate than T_srf (2m air temp) for surface flux calculations
        sfc_T = if "T_skin" in keys(ds)
            T_skin_data = Float64.(Array(ds["T_skin"]))
            T_val = length(T_skin_data) > 1 ? T_skin_data[1] : first(T_skin_data)
            T_val + 273.15  # Convert degC to K
        elseif "T_srf" in keys(ds)
            T_srf_data = Float64.(Array(ds["T_srf"]))
            T_val = length(T_srf_data) > 1 ? T_srf_data[1] : first(T_srf_data)
            T_val + 273.15  # Convert degC to K
        else
            T_raw[1]  # Use lowest level temperature
        end
        
        # Surface pressure (hPa in file)
        sfc_p = if "p_srf_aver" in keys(ds)
            p_data = Float64.(Array(ds["p_srf_aver"]))
            p_val = length(p_data) > 1 ? p_data[1] : first(p_data)
            p_val * 100.0  # Convert hPa to Pa
        elseif "p_srf_center" in keys(ds)
            p_data = Float64.(Array(ds["p_srf_center"]))
            p_val = length(p_data) > 1 ? p_data[1] : first(p_data)
            p_val * 100.0
        else
            lev_pa[1]  # Use lowest level pressure
        end
        
        # Site coordinates
        lat_val = Float64(first(ds["lat"][:]))
        lon_val = Float64(first(ds["lon"][:]))
        alt_val = Float64(first(ds["alt"][:]))
        
        # Surface fluxes (W/m²) - for prescribed flux option
        # Note: ARM convention is upward positive
        LH_val = if "LH" in keys(ds)
            LH_data = Float64.(Array(ds["LH"]))
            val = length(LH_data) > 1 ? LH_data[1] : first(LH_data)
            val == -9999.0 ? 0.0 : val
        else
            0.0
        end
        
        SH_val = if "SH" in keys(ds)
            SH_data = Float64.(Array(ds["SH"]))
            val = length(SH_data) > 1 ? SH_data[1] : first(SH_data)
            val == -9999.0 ? 0.0 : val
        else
            0.0
        end
        
        # Create time interpolators for surface fluxes (for time-varying prescribed fluxes)
        LH_time_interp = if "LH" in keys(ds)
            _create_flux_time_interpolator(ds, "LH")
        else
            nothing
        end
        
        SH_time_interp = if "SH" in keys(ds)
            _create_flux_time_interpolator(ds, "SH")
        else
            nothing
        end
        
        # Extract start date from the time coordinate
        time_raw = vec(ds["time"][:])
        file_start_date = if eltype(time_raw) <: Dates.DateTime
            time_raw[1]
        else
            # VARANAL files use base_time attribute + time offset in seconds
            base_time = if haskey(ds.attrib, "base_time")
                # base_time is Unix epoch seconds
                Dates.unix2datetime(ds.attrib["base_time"])
            else
                # Fallback: try to parse from filename (sgp60varanarucC1.c1.YYYYMMDD.HHMMSS.cdf)
                @warn "No base_time attribute, using Unix epoch"
                Dates.DateTime(1970, 1, 1)
            end
            base_time + Dates.Second(round(Int, time_raw[1]))
        end
        
        # Ensure heights are sorted increasing (required for interpolation)
        sort_idx = sortperm(z_lev)
        z_sorted = z_lev[sort_idx]
        T_sorted = T_raw[sort_idx]
        u_sorted = u_raw[sort_idx]
        v_sorted = v_raw[sort_idx]
        q_sorted = q_raw[sort_idx]
        ρ_sorted = ρ_raw[sort_idx]
        
        # Log the data range for diagnostic purposes
        @info "ARMVARANAL data range" z_min=minimum(z_sorted) z_max_data=maximum(z_sorted) 
        
        (z_sorted, T_sorted, u_sorted, v_sorted, q_sorted, ρ_sorted, 
         sfc_T, sfc_p, lat_val, lon_val, alt_val, file_start_date, LH_val, SH_val, LH_time_interp, SH_time_interp)
    end

    return ARMVARANAL(
        external_forcing_file,
        case_name,
        ColumnProfiles(z_ref, T, u, v, q_tot, ρ),
        T_sfc,
        p_sfc,
        lat,
        lon,
        alt,
        start_date,
        prescribe_surface_fluxes,
        LH_init,
        SH_init,
        LH_interp,
        SH_interp,
    )
end

"""
    _create_flux_time_interpolator(ds, varname)

Create a 1D time interpolator for a surface flux variable from ARM VARANAL file.
Returns an interpolator that takes time in seconds and returns flux in W/m².
"""
function _create_flux_time_interpolator(ds, varname)
    # Read time coordinate
    time_raw = vec(ds["time"][:])
    
    # Convert to seconds from start
    if eltype(time_raw) <: Dates.DateTime
        base_time = time_raw[1]
        time_sec = Float64.([Float64(Dates.value(t - base_time)) / 1000.0 for t in time_raw])
    else
        time_sec = Float64.(time_raw)
    end
    
    data_raw = Float64.(vec(ds[varname][:]))
    
    # Replace missing values with NaN
    data_clean = replace(data_raw, -9999.0 => NaN)
    
    Intp.extrapolate(
        Intp.interpolate((time_sec,), data_clean, Intp.Gridded(Intp.Linear())),
        Intp.Flat(),
    )
end

center_initial_condition(setup::ARMVARANAL, local_geometry, params) =
    column_profiles_ic(setup.profiles, local_geometry)

function surface_condition(setup::ARMVARANAL, params)
    FT = eltype(params)
    # z0 = 0.05 m is appropriate for grassland/cropland at SGP
    # (z0 = 1e-4 is for open ocean, way too smooth for land)
    z0 = FT(0.05)
    T_sfc_init = FT(setup.T_sfc)
    
    if setup.prescribe_surface_fluxes
        # Time-varying prescribed surface fluxes from observations
        # Return a closure that evaluates fluxes at current time t
        LH_interp = setup.LH_interp
        SH_interp = setup.SH_interp
        
        if isnothing(LH_interp) || isnothing(SH_interp)
            # Fall back to initial values if interpolators not available
            @warn "LH/SH interpolators not available, using initial values"
            fluxes = HeatFluxes(; 
                shf = FT(setup.SH_init), 
                lhf = FT(setup.LH_init),
            )
            parameterization = MoninObukhov(; z0, fluxes)
            return SurfaceState(; parameterization, T = T_sfc_init)
        else
            # Return time-varying surface state function (like TRMM_LBA)
            function surface_state(surface_coordinates, interior_z, t)
                _FT = eltype(surface_coordinates)
                # Evaluate time-varying fluxes from interpolators
                shf = _FT(SH_interp(t))
                lhf = _FT(LH_interp(t))
                # Handle NaN (missing data) - use zero as fallback
                shf = isnan(shf) ? _FT(0) : shf
                lhf = isnan(lhf) ? _FT(0) : lhf
                parameterization = MoninObukhov(; z0 = _FT(z0), shf, lhf)
                # Note: T is overridden by ARMVARANALTimeVaryingSST in update_surface_conditions!
                return SurfaceState(; parameterization, T = _FT(T_sfc_init))
            end
            return surface_state
        end
    else
        # Interactive fluxes from Monin-Obukhov
        parameterization = MoninObukhov(; z0)
        return SurfaceState(; parameterization, T = T_sfc_init)
    end
end

external_forcing(setup::ARMVARANAL, ::Type{FT}) where {FT} =
    ARMVARANALForcing{FT}(setup.external_forcing_file)

surface_temperature_model(::ARMVARANAL) = ARMVARANALTimeVaryingSST()

insolation_model(setup::ARMVARANAL) = 
    ColumnTimeVaryingInsolation(setup.start_date, setup.lat, setup.lon)
