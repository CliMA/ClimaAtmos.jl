
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
    var = ClimaAnalysis.window(
        var,
        "time",
        left = 2, # throw away the first time point
        by = ClimaAnalysis.Index(),
    )
    return var
end
