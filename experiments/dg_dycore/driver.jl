#=
Entry point for the DG dycore sandbox.

    julia --project=experiments/dg_dycore experiments/dg_dycore/driver.jl \
        experiments/dg_dycore/configs/balanced_flow_parity.yml

or from the REPL:

    include("experiments/dg_dycore/src/DGDycore.jl")
    using .DGDycore
    result = run!(DGSimulation(BaroclinicWaveFDDG(; helem = 10, zelem = 30,
        interface_flux = :roe, κ₄ = 0.0, dt = 60.0, t_end = 3 * 86400.0)))

The YAML config is a flat mirror of the BaroclinicWaveFDDG keywords.
=#

include(joinpath(@__DIR__, "src", "DGDycore.jl"))
using .DGDycore
import YAML

function problem_from_yaml(path)
    cfg = YAML.load_file(path)
    FT = get(cfg, "float_type", "Float64") == "Float32" ? Float32 : Float64
    core = Symbol(get(cfg, "core", "fddg"))   # :fddg or :vi
    core in (:fddg, :vi) || error("core must be fddg or vi")
    shared = (
        ("helem", :helem, Int),
        ("npoly", :npoly, Int),
        ("zelem", :zelem, Int),
        ("zmax", :zmax, FT),
        ("stepper", :stepper, Symbol),
        ("dt", :dt, FT),
        ("t_end", :t_end, FT),
        ("perturb", :perturb, Bool),
        ("kappa4", :κ₄, FT),
        ("sponge_tau", :sponge_τ, FT),
        ("sponge_depth", :sponge_depth, FT),
        ("sponge_uh", :sponge_uh, Bool),
        ("topography", :topography, Symbol),
        ("topography_damping_factor", :topography_damping_factor, FT),
        ("constants_mode", :constants_mode, Symbol),
        ("ic_source", :ic_source, Symbol),
        ("held_suarez", :held_suarez, Bool),
        ("output_dir", :output_dir, String),
        ("diag_period", :diag_period, FT),
        ("dt_save", :dt_save, FT),
        ("ndiag", :ndiag, Int),
    )
    fddg_only = (("interface_flux", :interface_flux, Symbol),)
    vi_only = (
        ("momentum_adv", :momentum_adv, Symbol),
        ("face_set", :face_set, Symbol),
        ("filter_Nc", :filter_Nc, Int),
    )
    kw = Dict{Symbol, Any}()
    for (key, field, conv) in
        (shared..., (core == :fddg ? fddg_only : vi_only)...)
        haskey(cfg, key) && (kw[field] = conv(cfg[key]))
    end
    if haskey(cfg, "zstretch")
        zs = cfg["zstretch"]
        kw[:zstretch] = (FT(zs[1]), FT(zs[2]))
    end
    return core == :fddg ? BaroclinicWaveFDDG{FT}(; kw...) :
           BaroclinicWaveDG{FT}(; kw...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error("usage: driver.jl <config.yml>")
    prob = problem_from_yaml(ARGS[1])
    @info "DGDycore run" prob
    result = run!(DGSimulation(prob))
    @info "done" result
end
