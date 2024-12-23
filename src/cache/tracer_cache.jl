import Dates: Year
import ClimaUtilities.TimeVaryingInputs
import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput, LinearPeriodFillingInterpolation
import Interpolations as Intp

ozone_cache(_, _, _) = (;)
function ozone_cache(::PrescribedOzone, Y, start_date)
    o3 = similar(Y.c.ρ)
    extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())
    prescribed_o3_timevaryinginput = TimeVaryingInput(
        AA.ozone_concentration_file_path(; context = ClimaComms.context(Y.c)),
        "vmro3",
        axes(o3);
        reference_date = start_date,
        regridder_type = :InterpolationsRegridder,
        regridder_kwargs = (; extrapolation_bc),
        method = LinearPeriodFillingInterpolation(
            Year(1),
            TimeVaryingInputs.Flat(),
        ),
    )
    return (; o3, prescribed_o3_timevaryinginput)
end

aerosols_cache(_, _, _) = (;)
function aerosols_cache(aerosols::PrescribedCMIP5Aerosols, Y, start_date)
    aerosol_names = aerosols.aerosol_names
    target_space = axes(Y.c)

    # Take the aerosol concentration file, read the keys with names matching
    # the ones passed in the prescribed_aerosol_names option, and create a
    # NamedTuple that uses the same keys and has as values the TimeVaryingInputs
    # for those variables.
    #
    # The keys in the aerosol_concentrations.nc file have to match the ones passed with the
    # configuration. The file also has to be defined on the globe and provide
    # time series of lon-lat-z data.
    target_space = axes(Y.c)
    extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())
    timevaryinginputs = [
        TimeVaryingInput(
            AA.aerosol_concentration_file_path(;
                context = ClimaComms.context(Y.c),
            ),
            name,
            target_space;
            reference_date = start_date,
            regridder_type = :InterpolationsRegridder,
            regridder_kwargs = (; extrapolation_bc),
            method = LinearPeriodFillingInterpolation(Year(1)),
        ) for name in aerosol_names
    ]

    # Field is updated in the radiation callback
    prescribed_aerosols_field = similar(
        Y.c,
        NamedTuple{
            aerosol_names,
            NTuple{length(aerosol_names), eltype(Y.c.ρ)},
        },
    )
    prescribed_aerosol_timevaryinginputs =
        (; zip(aerosol_names, timevaryinginputs)...)
    return (; prescribed_aerosols_field, prescribed_aerosol_timevaryinginputs)
end

function tracer_cache(Y, atmos, start_date)
    aero_cache = aerosols_cache(atmos.aerosols, Y, start_date)
    o3_cache = ozone_cache(atmos.ozone, Y, start_date)
    return (; aero_cache..., o3_cache...)
end
