import Dates: Year, Date
import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput, PeriodicCalendar, LinearPeriodFillingInterpolation, LinearInterpolation
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
        method = LinearPeriodFillingInterpolation(Year(1)),
    )
    return (; o3, prescribed_o3_timevaryinginput)
end

function cloud_cache(Y, start_date)
    target_space = axes(Y.c)
    prescribed_cloud_names = ("cc", "clwc", "ciwc")
    prescribed_cloud_names_as_symbols = Symbol.(prescribed_cloud_names)
    extrapolation_bc = (Intp.Periodic(), Intp.Flat(), Intp.Flat())
    timevaryinginputs = [
        TimeVaryingInput(
            joinpath(
                @clima_artifact(
                    "era5_cloud",
                    ClimaComms.context(Y.c)
                ),
                "era5_cloud.nc",
            ),
            name,
            target_space;
            reference_date = start_date,
            regridder_type = :InterpolationsRegridder,
            regridder_kwargs = (; extrapolation_bc),
            method = LinearInterpolation(PeriodicCalendar(Year(1), Date(2010))),
        ) for name in prescribed_cloud_names
    ]

    prescribed_clouds_field = similar(
        Y.c,
        NamedTuple{
            prescribed_cloud_names_as_symbols,
            NTuple{
                length(prescribed_cloud_names_as_symbols),
                eltype(Y.c.ρ),
            },
        },
    )
    prescribed_cloud_timevaryinginputs =
        (; zip(prescribed_cloud_names_as_symbols, timevaryinginputs)...)
    return (; prescribed_clouds_field, prescribed_cloud_timevaryinginputs)
end

function tracer_cache(Y, atmos, prescribed_aerosol_names, start_date)
    if !isempty(prescribed_aerosol_names)
        target_space = axes(Y.c)

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
    o3_cache = ozone_cache(atmos.ozone, Y, start_date)
    clouds = cloud_cache(Y, start_date)
    return (; aerosol_cache..., o3_cache..., clouds...)
end
