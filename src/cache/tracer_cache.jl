using ClimaUtilities.ClimaArtifacts
import Dates: Year
import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput, LinearPeriodFillingInterpolation

function tracer_cache(
    Y,
    atmos,
    prescribe_ozone,
    prescribed_aerosol_names,
    start_date,
)
    if isempty(prescribed_aerosol_names) && !prescribe_ozone
        return (;)
    end

    target_space = axes(Y.c)
    if prescribe_ozone
        o3 = similar(Y.c.ρ)
        prescribed_o3_timevaryinginput = TimeVaryingInput(
            joinpath(
                @clima_artifact(
                    "ozone_concentrations",
                    ClimaComms.context(Y.c)
                ),
                "ozone_concentrations.nc",
            ),
            "vmro3",
            target_space;
            reference_date = start_date,
            regridder_type = :InterpolationsRegridder,
            method = LinearPeriodFillingInterpolation(Year(1)),
        )
        o3_cache = (; o3, prescribed_o3_timevaryinginput)
    else
        o3_cache = (;)
    end

    if !isempty(prescribed_aerosol_names)
        # Take the aerosol concentration file, read the keys with names matching
        # the ones passed in the prescribed_aerosol_names option, and create a
        # NamedTuple that uses the same keys and has as values the TimeVaryingInputs
        # for those variables.
        #
        # The keys in the aerosol_concentrations.nc file have to match the ones passed with the
        # configuration. The file also has to be defined on the globe and provide
        # time series of lon-lat-z data.
        prescribed_aerosol_names_as_symbols = Symbol.(prescribed_aerosol_names)
        target_space = axes(Y.c)
        timevaryinginputs = [
            TimeVaryingInput(
                joinpath(
                    @clima_artifact(
                        "aerosol_concentrations",
                        ClimaComms.context(Y.c)
                    ),
                    "aerosol_concentrations.nc",
                ),
                name,
                target_space;
                reference_date = start_date,
                regridder_type = :InterpolationsRegridder,
                method = LinearPeriodFillingInterpolation(Year(1)),
            ) for name in prescribed_aerosol_names
        ]

        # Field is updated in the radiation callback
        prescribed_aerosols_field = similar(
            Y.c,
            NamedTuple{
                prescribed_aerosol_names_as_symbols,
                NTuple{
                    length(prescribed_aerosol_names_as_symbols),
                    eltype(Y.c.ρ),
                },
            },
        )
        prescribed_aerosol_timevaryinginputs =
            (; zip(prescribed_aerosol_names_as_symbols, timevaryinginputs)...)
        aerosol_cache =
            (; prescribed_aerosols_field, prescribed_aerosol_timevaryinginputs)
    else
        aerosol_cache = (;)
    end
    return (; o3_cache..., aerosol_cache...)
end
