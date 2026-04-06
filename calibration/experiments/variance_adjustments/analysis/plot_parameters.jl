# Plot constrained parameter trajectories from EKI.
#
#   julia --project=. analysis/plot_parameters.jl <eki_file.jld2> <prior.toml>
#
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

import CairoMakie as M
import JLD2
import EnsembleKalmanProcesses as EKP
import ClimaCalibrate as CAL

exp_dir = joinpath(@__DIR__, "..")
eki_path = get(ARGS, 1, joinpath(exp_dir, "simulation_output", "calibration", "TRMM_LBA", "N_3", "varfix_off", "eki", "iteration_001", "eki_file.jld2"))
prior_path = get(ARGS, 2, joinpath(exp_dir, "prior.toml"))

!isfile(eki_path) && @warn "Missing eki file" eki_path
if isfile(eki_path)
    eki = JLD2.load_object(eki_path)
    prior = CAL.get_prior(prior_path)
    phi = EKP.get_ϕ(prior, eki)
    fig = M.Figure()
    ax = M.Axis(fig[1, 1]; xlabel = "iteration", ylabel = "parameter index (columns)")
    # Simple heatmap-style: iterations × dim
    u_mat = reduce(hcat, phi)
    M.heatmap!(ax, rotr90(u_mat))
    outp = joinpath(@__DIR__, "parameters.png")
    M.save(outp, fig)
    @info "Saved" outp
end
