"""
    WeatherModel(start_date, era5_initial_condition_dir = nothing;
                 use_full_pressure = false)

ERA5-derived initial condition for weather and forecast simulations.

The pointwise construction assigns NaN placeholders; `overwrite_initial_state!`
then overwrites the whole prognostic state with ERA5 data located by
`weather_model_data_path`.

# Arguments

  - `start_date`: Date string in the format `"yyyymmdd"` or `"yyyymmdd-HHMM"`.
  - `era5_initial_condition_dir = nothing`: Directory of pre-processed ERA5
    files. When `nothing`, the `wxquest_initial_conditions` ClimaArtifact is
    used. It is stashed in the module-level `_ERA5_IC_DIR` rather than stored on
    the struct, because a captured string cannot be adapted to the GPU.

# Keyword Arguments

  - `use_full_pressure = false`: Whether to read the 3D pressure from the file
    instead of integrating it hydrostatically.
  - `hydrostatic_rebalance = false`: Whether to rebalance the column pressure
    against the model's discrete Exner-split vertical pressure gradient,
    anchored at the first level of the integrated or file pressure (see
    `rebalance_hydrostatic_pressure`).

# Fields

  - `start_date`: The parsed `DateTime`.
  - `use_full_pressure`: As above [-].
  - `hydrostatic_rebalance`: As above [-].
"""
struct WeatherModel
    start_date::Dates.DateTime
    use_full_pressure::Bool
    hydrostatic_rebalance::Bool
end

const _ERA5_IC_DIR = Ref{Any}(nothing)

function WeatherModel(
    start_date::String,
    era5_initial_condition_dir = nothing;
    use_full_pressure::Bool = false,
    hydrostatic_rebalance::Bool = false,
)
    _ERA5_IC_DIR[] = era5_initial_condition_dir
    return WeatherModel(
        parse_date(start_date),
        use_full_pressure,
        hydrostatic_rebalance,
    )
end

function center_initial_condition(setup::WeatherModel, local_geometry, params)
    FT = eltype(params)
    return physical_state(; T = FT(NaN), p = FT(NaN))
end

"""
    temperature_below_surface(T₁, z, z_sfc)

Temperature [K] at height `z` [m]: `T₁` at or above the source surface height
`z_sfc` [m], and `T₁ + Γ (z_sfc - z)` below it, with the standard lapse rate
`Γ = 0.0065` K/m.
"""
function temperature_below_surface(T₁, z, z_sfc)
    Γ = oftype(T₁, 0.0065)
    return T₁ + Γ * max(z_sfc - z, zero(z))
end

"""
    pressure_below_surface(thermo_params, T₁, p₁, q_vap, z, z_sfc)

Pressure [Pa] at height `z` [m]: `p₁` at or above the source surface height
`z_sfc` [m], and below it the hydrostatic profile `p₁ (T / T₁)^(g / (R_m Γ))`
of the temperature from [`temperature_below_surface`](@ref), with `R_m` the gas
constant of air with vapor specific humidity `q_vap`.
"""
function pressure_below_surface(thermo_params, T₁, p₁, q_vap, z, z_sfc)
    Γ = oftype(T₁, 0.0065)
    T = temperature_below_surface(T₁, z, z_sfc)
    R_m = TD.gas_constant_air(thermo_params, q_vap, zero(q_vap), zero(q_vap))
    return p₁ * (T / T₁)^(thermo_params.grav / (R_m * Γ))
end

function overwrite_initial_state!(setup::WeatherModel, Y, thermo_params)
    regridder_type = :InterpolationsRegridder
    interpolation_method = Intp.Linear()
    extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())

    # Compute target vertical levels for 1D interpolation
    z_arr = Array(Fields.field2array(Fields.coordinate_field(Y.c).z))
    z_top = round(maximum(z_arr))
    target_levels = collect(0.0:300.0:z_top)

    era5_dir = _ERA5_IC_DIR[]
    @info "Calling weather_model_data_path" (
        start_date = setup.start_date,
        era5_dir = era5_dir,
    )

    file_path = weather_model_data_path(
        setup.start_date,
        target_levels,
        era5_dir;
        comms_ctx = ClimaComms.context(Fields.axes(Y.c)),
    )

    if !setup.use_full_pressure
        # Standard path: delegate to shared overwrite infrastructure
        return overwrite_from_file!(
            file_path, extrapolation_bc, Y, thermo_params;
            regridder_type, interpolation_method,
            hydrostatic_rebalance = setup.hydrostatic_rebalance,
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
    ᶜq_vap = SpaceVaryingInputs.SpaceVaryingInput(
        file_path, "q", center_space; svi_kwargs...,
    )

    # Optional condensate species and the thermodynamic moisture partition, read
    # before ρ / p / ρe_tot so the initial state is consistent with ρq_tot.
    (; ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno) =
        read_microphysics_from_file(file_path, center_space, svi_kwargs)
    (; ᶜq_tot, ᶜq_liq, ᶜq_ice) =
        thermodynamic_partition(Y, ᶜq_vap, ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno)

    use_p3d, has_z_sfc = NC.NCDataset(file_path) do ds
        haskey(ds, "p_3d"), haskey(ds, "z_sfc")
    end
    use_p3d && !has_z_sfc &&
        error(
            "Full-pressure initialization needs `z_sfc` in $(file_path) to extend the " *
            "column below the ERA5 surface.",
        )
    ᶠp = if use_p3d
        @info "Using full 3D pressure from file variable `p_3d` (use_full_pressure=true)"
        ᶠp_3d = SpaceVaryingInputs.SpaceVaryingInput(
            file_path, "p_3d", face_space; svi_kwargs...,
        )
        # Below the ERA5 surface the file repeats the lowest-level values, which
        # leaves no vertical pressure gradient to hold the layer up.
        read_face =
            name -> SpaceVaryingInputs.SpaceVaryingInput(
                file_path, name, face_space; svi_kwargs...,
            )
        ᶜz_sfc = SpaceVaryingInputs.SpaceVaryingInput(
            file_path, "z_sfc", center_space; svi_kwargs...,
        )
        ᶠz_sfc, ᶠT_file, ᶠq_vap = read_face("z_sfc"), read_face("t"), read_face("q")
        ᶠz = Fields.coordinate_field(face_space).z
        ᶜz = Fields.coordinate_field(center_space).z
        @. ᶠp_3d = pressure_below_surface(
            thermo_params, ᶠT_file, ᶠp_3d, ᶠq_vap, ᶠz, ᶠz_sfc,
        )
        @. ᶜT = temperature_below_surface(ᶜT, ᶜz, ᶜz_sfc)
        ᶠp_3d
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
                p_sfc, file_path, face_space, Y, ᶜT,
                ᶜq_tot, ᶜq_liq, ᶜq_ice,
                thermo_params, regridder_kwargs;
                surface_altitude_var,
            )
        else
            @warn "Skipping topographic correction because variable " *
                  "`$surface_altitude_var` is missing from $(file_path)."
        end
        hydrostatic_pressure(
            p_sfc, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice, face_space,
            thermo_params,
        )
    end

    ᶜp = initial_center_pressure(
        ᶠp, Y, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice, thermo_params, setup.hydrostatic_rebalance,
    )
    Y.c.ρ .= TD.air_density.(thermo_params, ᶜT, ᶜp, ᶜq_tot, ᶜq_liq, ᶜq_ice)

    # Velocity and energy
    e_pot = assign_velocity_energy!(
        Y, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice, ᶠp, thermo_params,
        file_path, svi_kwargs,
    )

    # Moisture and EDMF
    assign_moisture_edmf!(
        Y, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice, e_pot, thermo_params,
        ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno,
    )

    return nothing
end
