using ClimaUtilities.ClimaArtifacts
import ClimaUtilities.TimeVaryingInputs: TimeVaryingInput

function tracer_cache(Y, atmos, prescribed_aerosol_names, start_date)
    prescribed_aerosol_timevaryinginputs = (;)
    prescribed_aerosol_fields = (;)

    # Take the aerosol2005/aero_2005.nc file, read the keys with names matching
    # the ones passed in the prescribed_aerosol_names option, and create a
    # NamedTuple that uses the same keys and has as values the TimeVaryingInputs
    # for those variables.
    #
    # The keys in the aero_2005.nc file have to match the ones passed with the
    # configuration. The file also has to be defined on the globe and provide
    # time series of lon-lat-z data.
    if !isempty(prescribed_aerosol_names)
        prescribed_aerosol_names_as_symbols = Symbol.(prescribed_aerosol_names)
        target_space = axes(Y.c)
        timevaryinginputs = [
            TimeVaryingInput(
                joinpath(
                    @clima_artifact("aerosol2005", ClimaComms.context(Y.c)),
                    "aero_2005.nc",
                ),
                name,
                target_space;
                reference_date = start_date,
                regridder_type = :InterpolationsRegridder,
            ) for name in prescribed_aerosol_names
        ]
        empty_fields =
            [zero(Y.c.ρ) for _ in prescribed_aerosol_names_as_symbols]

        prescribed_aerosol_timevaryinginputs =
            (; zip(prescribed_aerosol_names_as_symbols, timevaryinginputs)...)

        # We add empty Fields here. Fields are updated in the radiation callback
        prescribed_aerosol_fields =
            (; zip(prescribed_aerosol_names_as_symbols, empty_fields)...)
    end
    return (; prescribed_aerosol_fields, prescribed_aerosol_timevaryinginputs)
end
