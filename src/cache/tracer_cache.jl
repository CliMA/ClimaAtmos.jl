using ClimaUtilities.ClimaArtifacts
import ClimaUtilities.TimeVaryingInputs: TimeVaryingInput

function tracer_cache(Y, atmos, prescribed_aerosol_names, start_date)
    if isempty(prescribed_aerosol_names)
        return (;)
    end

    # Take the aerosol concentration file, read the keys with names matching
    # the ones passed in the prescribed_aerosol_names option, and create a
    # NamedTuple that uses the same keys and has as values the TimeVaryingInputs
    # for those variables.
    #
    # The keys in the aero_2005.nc file have to match the ones passed with the
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
        ) for name in prescribed_aerosol_names
    ]

    # Field is updated in the radiation callback
    prescribed_aerosols_field = similar(
        Y.c,
        NamedTuple{
            prescribed_aerosol_names_as_symbols,
            NTuple{length(prescribed_aerosol_names_as_symbols), eltype(Y.c.ρ)},
        },
    )
    prescribed_aerosol_timevaryinginputs =
        (; zip(prescribed_aerosol_names_as_symbols, timevaryinginputs)...)
    return (; prescribed_aerosols_field, prescribed_aerosol_timevaryinginputs)
end
