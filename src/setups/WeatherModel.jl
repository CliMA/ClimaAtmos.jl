"""
    WeatherModel

ERA5-derived initial condition for weather/forecast simulations.

Assigns NaN placeholders during pointwise construction, then overwrites the
full prognostic state with ERA5 data obtained via `weather_model_data_path`.

## Fields
- `start_date`: Date string in format "yyyymmdd" or "yyyymmdd-HHMM".
- `era5_initial_condition_dir`: Optional directory with pre-processed ERA5 files.
  When `nothing`, uses the `weather_model_ic` ClimaArtifact.
- `use_full_pressure`: If `true`, attempt to read 3D pressure from the file
  rather than computing it hydrostatically. Defaults to `false`.
"""
struct WeatherModel
    start_date::String
    era5_initial_condition_dir::Union{Nothing, String}
    use_full_pressure::Bool
end

function WeatherModel(
    start_date::String,
    era5_initial_condition_dir = nothing;
    use_full_pressure::Bool = false,
)
    return WeatherModel(start_date, era5_initial_condition_dir, use_full_pressure)
end

function center_initial_condition(setup::WeatherModel, local_geometry, params)
    FT = eltype(params)
    return physical_state(; T = FT(NaN), p = FT(NaN))
end

function overwrite_initial_state!(setup::WeatherModel, Y, thermo_params)
    regridder_type = :InterpolationsRegridder
    interpolation_method = Intp.Linear()
    extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())

    # Compute target vertical levels for 1D interpolation
    z_arr = Array(Fields.field2array(Fields.coordinate_field(Y.c).z))
    z_top = round(maximum(z_arr))
    target_levels = collect(0.0:300.0:z_top)

    @info "Calling weather_model_data_path" (
        start_date = setup.start_date,
        era5_dir = setup.era5_initial_condition_dir,
    )

    file_path = weather_model_data_path(
        setup.start_date,
        target_levels,
        setup.era5_initial_condition_dir,
    )

    if !setup.use_full_pressure
        # Standard path: delegate to shared overwrite infrastructure
        return overwrite_from_file!(
            file_path, extrapolation_bc, Y, thermo_params;
            regridder_type, interpolation_method,
        )
    end

    # Full-pressure path: read p_3d if available, otherwise fall back to
    # hydrostatic integration (same as overwrite_from_file!, but with p_3d support)
    regridder_kwargs = (; extrapolation_bc, interpolation_method)
    svi_kwargs = (; regridder_type, regridder_kwargs)

    isfile(file_path) || error("$(file_path) is not a file")
    @info "Overwriting initial conditions with data from file $(file_path)"

    center_space = Fields.axes(Y.c)
    face_space = Fields.axes(Y.f)

    ᶜT = SpaceVaryingInputs.SpaceVaryingInput(
        file_path, "t", center_space; svi_kwargs...,
    )
    ᶜq_tot = SpaceVaryingInputs.SpaceVaryingInput(
        file_path, "q", center_space; svi_kwargs...,
    )

    use_p3d = NC.NCDataset(file_path) do ds
        haskey(ds, "p_3d")
    end
    ᶠp = if use_p3d
        SpaceVaryingInputs.SpaceVaryingInput(
            file_path, "p_3d", face_space; svi_kwargs...,
        )
    else
        @warn "Requested full pressure initialization, but variable `p_3d` " *
              "is missing in $(file_path). Falling back to hydrostatic " *
              "integration from surface pressure."
        p_sfc = Fields.level(
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path, "p", face_space; svi_kwargs...,
            ),
            Fields.half,
        )
        surface_altitude_var = "z_sfc"
        has_surface_altitude = NC.NCDataset(file_path) do ds
            haskey(ds, surface_altitude_var)
        end
        if has_surface_altitude
            correct_surface_pressure_for_topography!(
                p_sfc, file_path, face_space, Y, ᶜT, ᶜq_tot,
                thermo_params, regridder_kwargs;
                surface_altitude_var,
            )
        else
            @warn "Skipping topographic correction because variable " *
                  "`$surface_altitude_var` is missing from $(file_path)."
        end
        _hydrostatic_pressure(p_sfc, ᶜT, ᶜq_tot, face_space, thermo_params)
    end

    # Density
    Y.c.ρ .= TD.air_density.(thermo_params, ᶜT, ᶜinterp.(ᶠp), ᶜq_tot)

    # Velocity and energy
    e_pot = _assign_velocity_energy!(
        Y, ᶜT, ᶜq_tot, ᶠp, thermo_params, file_path, svi_kwargs,
    )

    # Moisture and EDMF
    _assign_moisture_edmf!(
        Y, ᶜT, ᶜq_tot, e_pot, thermo_params, file_path, svi_kwargs,
    )

    return nothing
end
