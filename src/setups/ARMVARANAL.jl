"Single-column IC and forcing from ARM VARANAL format NetCDF files."

"""
    ARMVARANAL{P <: ColumnProfiles, FT, LHI, SHI}

Pointwise column IC driven by ARM VARANAL format NetCDF data.

Surface fluxes (LH, SH) are prescribed from the file as time-varying values.

## Fields

  - `external_forcing_file`: Path to the ARM VARANAL NetCDF file.
  - `profiles`: `ColumnProfiles` of 1D interpolators `(T, u, v, q_tot, ρ)`.
  - `T_sfc`: Initial surface temperature from the forcing file (K).
  - `p_sfc`: Initial surface pressure from the forcing file (Pa).
  - `lat`: Latitude of the site (degrees).
  - `lon`: Longitude of the site (degrees).
  - `alt`: Altitude above mean sea level (m).
  - `start_date`: Start date/time from the file (for insolation calculations).
  - `LH_interp`: Time interpolator for latent heat flux (W/m²), or nothing.
  - `SH_interp`: Time interpolator for sensible heat flux (W/m²), or nothing.
"""
struct ARMVARANAL{P <: ColumnProfiles, FT, LHI, SHI}
    external_forcing_file::String
    profiles::P
    T_sfc::FT
    p_sfc::FT
    lat::FT
    lon::FT
    alt::FT
    start_date::Dates.DateTime
    LH_interp::LHI
    SH_interp::SHI
end

"""
    ARMVARANAL(external_forcing_file; thermo_params, start_date)

Construct an ARMVARANAL setup by reading initial condition profiles from an
ARM VARANAL format forcing file and building 1D vertical interpolators.

Surface fluxes (LH, SH) are prescribed from the file as time-varying values.

## Arguments

  - `external_forcing_file`: Path to the ARM VARANAL NetCDF file.
  - `thermo_params`: Thermodynamics parameters (for physical constants).
  - `start_date`: Simulation start date string (e.g., "20100918"). When
    provided and the file spans multiple days, IC profiles are read from
    the time step closest to this date rather than the first time step.
"""
function ARMVARANAL(
    external_forcing_file::String;
    thermo_params,
    start_date::Union{String, Nothing} = nothing,
)
    # Local helpers

    """Find the time index in the file closest to `target_date`."""
    function find_time_index(ds, target_date)
        time_raw = vec(ds["time"][:])
        if eltype(time_raw) <: Dates.DateTime
            _, idx = findmin(t -> abs(Dates.value(t - target_date)), time_raw)
            return idx
        elseif haskey(ds.attrib, "base_time")
            base_time = Dates.unix2datetime(ds.attrib["base_time"])
            target_sec = Float64(Dates.value(target_date - base_time) / 1000)
            _, idx = findmin(t -> abs(Float64(t) - target_sec), time_raw)
            return idx
        else
            return 1
        end
    end

    function timestep_at(ds, varname, nlev, tidx)
        raw = Array(ds[varname])
        any(ismissing, raw) &&
            error("$varname contains missing data; choose a different period")
        data = Float64.(raw)
        ndims(data) == 1 ? vec(data) :
        size(data, 1) == nlev ? vec(data[:, tidx]) : vec(data[tidx, :])
    end

    function read_surface_scalar(ds, keys_convert, default; tidx = 1)
        for (key, convert_fn) in keys_convert
            key in Base.keys(ds) || continue
            data = Float64.(Array(ds[key]))
            return convert_fn(length(data) > 1 ? data[tidx] : first(data))
        end
        default
    end

    function extract_start_date(ds)
        time_raw = vec(ds["time"][:])
        eltype(time_raw) <: Dates.DateTime && return time_raw[1]
        base_time = Dates.unix2datetime(ds.attrib["base_time"])
        base_time + Dates.Second(round(Int, time_raw[1]))
    end

    function create_flux_interpolator(ds, varname; time_offset = 0.0)
        time_raw = vec(ds["time"][:])
        time_sec = if eltype(time_raw) <: Dates.DateTime
            Float64.([Dates.value(t - time_raw[1]) / 1000.0 for t in time_raw])
        else
            Float64.(time_raw)
        end
        # Shift to simulation time: sim_time = file_time - offset
        time_sim = time_sec .- time_offset
        data = replace(Float64.(vec(ds[varname][:])), -9999.0 => NaN)
        Intp.extrapolate(
            Intp.interpolate((time_sim,), data, Intp.Gridded(Intp.Linear())),
            Intp.Flat(),
        )
    end

    # Parse start_date string → DateTime for IC time selection
    sim_start_dt = isnothing(start_date) ? nothing : parse_date(start_date)

    NC.NCDataset(external_forcing_file) do ds
        lev_pa = Float64.(vec(ds["lev"][:])) .* 100.0  # hPa -> Pa
        nlev = length(lev_pa)

        # Select IC time: use start_date if provided, else first time step
        tidx = if !isnothing(sim_start_dt)
            idx = find_time_index(ds, sim_start_dt)
            @info "ARMVARANAL IC" time_index = idx start_date = sim_start_dt
            idx
        else
            1
        end

        # Read IC profiles at the selected time, convert q from g/kg
        T_raw = timestep_at(ds, "T", nlev, tidx)
        q_raw = timestep_at(ds, "q", nlev, tidx) ./ 1000.0
        u_raw = timestep_at(ds, "u", nlev, tidx)
        v_raw = timestep_at(ds, "v", nlev, tidx)

        # Ensure surface-first ordering (decreasing pressure with increasing z)
        if lev_pa[1] < lev_pa[end]
            lev_pa, T_raw, q_raw, u_raw, v_raw =
                reverse.((lev_pa, T_raw, q_raw, u_raw, v_raw))
        end

        z_lev = pressure_to_height(lev_pa, T_raw, q_raw, thermo_params)
        R_d, R_v = TD.Parameters.R_d(thermo_params), TD.Parameters.R_v(thermo_params)
        ρ_raw = lev_pa ./ (R_d .* T_raw .* (1 .+ (R_v / R_d - 1) .* q_raw))

        # Surface conditions at the IC time
        sfc_T = read_surface_scalar(
            ds,
            [("T_skin", x -> x + 273.15), ("T_srf", x -> x + 273.15)],
            T_raw[1];
            tidx,
        )
        sfc_p = read_surface_scalar(
            ds,
            [("p_srf_aver", x -> x * 100.0), ("p_srf_center", x -> x * 100.0)],
            lev_pa[1];
            tidx,
        )

        # Sort by height (required for interpolation)
        idx = sortperm(z_lev)
        z, T, u, v, q, ρ = (arr[idx] for arr in (z_lev, T_raw, u_raw, v_raw, q_raw, ρ_raw))
        @info "ARMVARANAL data range" z_min = minimum(z) z_max = maximum(z)

        # Time offset so flux interpolators use simulation time (t=0 at start_date)
        time_raw_all = vec(ds["time"][:])
        flux_time_offset =
            if !isnothing(sim_start_dt) && eltype(time_raw_all) <: Dates.DateTime
                Float64(Dates.value(sim_start_dt - time_raw_all[1]) / 1000)
            elseif !isnothing(sim_start_dt)
                base_time = Dates.unix2datetime(ds.attrib["base_time"])
                Float64(Dates.value(sim_start_dt - base_time) / 1000)
            else
                0.0
            end

        ARMVARANAL(
            external_forcing_file,
            ColumnProfiles(z, T, u, v, q, ρ),
            sfc_T,
            sfc_p,
            Float64(first(ds["lat"][:])),
            Float64(first(ds["lon"][:])),
            Float64(first(ds["alt"][:])),
            isnothing(sim_start_dt) ? extract_start_date(ds) : sim_start_dt,
            "LH" in Base.keys(ds) ?
            create_flux_interpolator(ds, "LH"; time_offset = flux_time_offset) : nothing,
            "SH" in Base.keys(ds) ?
            create_flux_interpolator(ds, "SH"; time_offset = flux_time_offset) : nothing,
        )
    end
end

center_initial_condition(setup::ARMVARANAL, local_geometry, params) =
    column_profiles_ic(setup.profiles, local_geometry)

function surface_condition(setup::ARMVARANAL, params)
    FT = eltype(params)
    lh_interp = setup.LH_interp
    sh_interp = setup.SH_interp
    if !isnothing(lh_interp) && !isnothing(sh_interp)
        prescribed_fluxes = function (t, ::Type{_FT}) where {_FT}
            t_sec = Float64(t isa Number ? t : float(t))
            shf = sh_interp(t_sec)
            lhf = lh_interp(t_sec)
            shf = isnan(shf) ? _FT(0) : _FT(shf)
            lhf = isnan(lhf) ? _FT(0) : _FT(lhf)
            HeatFluxes(; shf, lhf)
        end
        return (;
            flux_scheme = MoninObukhov(; z0 = FT(0.05), fluxes = prescribed_fluxes),
            temperature = nothing,
            overrides = nothing,
        )
    else
        return (;
            flux_scheme = MoninObukhov(; z0 = FT(0.05)),
            temperature = nothing,
            overrides = nothing,
        )
    end
end

external_forcing(setup::ARMVARANAL, ::Type{FT}) where {FT} =
    ARMVARANALForcing{FT}(setup.external_forcing_file)

surface_temperature_model(::ARMVARANAL) = ARMVARANALTimeVaryingSST()

insolation_model(setup::ARMVARANAL) =
    ColumnTimeVaryingInsolation(setup.start_date, setup.lat, setup.lon)
