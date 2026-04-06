# Plot RMSE vs iteration from saved EKI artifacts.
# Point `eki_file` to `eki_file.jld2` inside an iteration directory.
#
#   julia --project=. analysis/plot_losses.jl
#
import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

import CairoMakie as M
import JLD2
import EnsembleKalmanProcesses as EKP

eki_path = get(ARGS, 1, joinpath(@__DIR__, "..", "simulation_output", "calibration", "TRMM_LBA", "N_3", "varfix_off", "eki", "iteration_001", "eki_file.jld2"))
!isfile(eki_path) && @warn "EKI file not found; pass path as first argument" eki_path

if isfile(eki_path)
    eki = JLD2.load_object(eki_path)
    errs = map(eachindex(EKP.get_g(eki))) do i
        g = vec(EKP.get_g(eki, i))
        sqrt(sum(abs2, g))
    end
    fig = M.Figure()
    ax = M.Axis(fig[1, 1]; xlabel = "iteration", ylabel = "‖g‖₂ (placeholder)")
    M.lines!(ax, 0:(length(errs) - 1), errs)
    outp = joinpath(@__DIR__, "losses.png")
    M.save(outp, fig)
    @info "Saved" outp
end
