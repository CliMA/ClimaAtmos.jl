"""
Smoke / debug: run one SOCRATES ClimaAtmos column case (in-memory forcing).

Usage (from experiment root):
  julia --project=. run_single_case.jl RF01_Obs
  julia --project=. run_single_case.jl RF01_Obs --short
"""

using YAML: YAML

include(joinpath(@__DIR__, "model_interface", "model_interface.jl"))

exp_cfg = YAML.load_file(joinpath(@__DIR__, "configs", "experiment_config.yml"))
config = joinpath(@__DIR__, exp_cfg["model_config"])
output_dir = joinpath(@__DIR__, exp_cfg["output_dir"], "smoke")

interface = SOCRATESAtmosModelInterface(config, output_dir, exp_cfg)


# cases = interface.cases

cases = SOCRATESCase[
#   SOCRATESCase(1, SSCF.ObsForcing()),
#   SOCRATESCase(1, SSCF.ERA5Forcing()),
  SOCRATESCase(9, SSCF.ObsForcing()),
#   SOCRATESCase(9, SSCF.ERA5Forcing()),
#   SOCRATESCase(10, SSCF.ObsForcing()),
#   SOCRATESCase(10, SSCF.ERA5Forcing()),
#   SOCRATESCase(11, SSCF.ERA5Forcing()),
#   SOCRATESCase(12, SSCF.ObsForcing()),
#   SOCRATESCase(12, SSCF.ERA5Forcing()),
#   SOCRATESCase(13, SSCF.ObsForcing()),
#   SOCRATESCase(13, SSCF.ERA5Forcing()),
]

out = Vector{String}(undef, length(cases))
@inbounds for i in eachindex(cases)
    case = cases[i]
    case_name = name(case)
    out[i] = run_single_case!(
        interface,
        case;
        name = case_name,
        output_dir = joinpath(output_dir, case_name),
    )
    @info "Simulation finished" case_name out[i]
end

