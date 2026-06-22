
"""
    preprocess(var::OutputVar, ::PerfectAtmosModelInterface)

For a perfect model calibraiton, postprocess the simulation data in the form of
`ClimaAnalysis.OutputVar`.

Because it is a perfect model calibration, the function is used when generating
the observations (in `create_ekp_observations`), postprocessing in the
observation map (in `process_member_data!`), and analyzing the iterations (in
`plot_ensemble`).
"""
function preprocess(var::ClimaAnalysis.OutputVar, ::PerfectAtmosModelInterface)
    var_dates = ClimaAnalysis.dates(var)
    start_date = last(var_dates) - Dates.Hour(6) + Dates.Minute(20)
    var = ClimaAnalysis.window(
        var,
        "time",
        left = start_date,
        by = ClimaAnalysis.MatchValue(),
    )
    return var
end
