"""
    AMIPFromERA5(start_date)

AMIP initial condition using ERA5 monthly reanalysis data.

Assigns NaN placeholders during pointwise construction, then overwrites
the full prognostic state with ERA5 data from the `era5_inst_model_levels`
ClimaArtifact, delegating to `overwrite_from_file!`.

## Fields
- `start_date`: DateTime parsed from a date string in format "yyyymmdd" or "yyyymmdd-HHMM".

## Expected artifact structure
`era5_inst_model_levels/era5_init_processed_internal_YYYYMMDD_0000.nc`
"""
struct AMIPFromERA5
    start_date::Dates.DateTime
end

AMIPFromERA5(start_date::String) = AMIPFromERA5(parse_date(start_date))

function center_initial_condition(::AMIPFromERA5, local_geometry, params)
    FT = eltype(params)
    return physical_state(; T = FT(NaN), p = FT(NaN))
end

function overwrite_initial_state!(setup::AMIPFromERA5, Y, thermo_params)
    start_date_str = Dates.format(setup.start_date, "yyyymmdd")

    file_path = joinpath(
        @clima_artifact("era5_inst_model_levels"),
        "era5_init_processed_internal_$(start_date_str)_0000.nc",
    )

    extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())

    return overwrite_from_file!(
        file_path, extrapolation_bc, Y, thermo_params;
        regridder_type = :InterpolationsRegridder,
        interpolation_method = Intp.Linear(),
    )
end
