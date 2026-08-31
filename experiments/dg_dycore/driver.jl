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
        ("newton_max_iters", :newton_max_iters, Int),
        ("dt", :dt, FT),
        ("t_end", :t_end, FT),
        ("perturb", :perturb, Bool),
        ("kappa4", :κ₄, FT),
        ("kappa4_frac", :κ₄_frac, FT),
        ("sponge_tau", :sponge_τ, FT),
        ("sponge_depth", :sponge_depth, FT),
        ("sponge_uh", :sponge_uh, Bool),
        ("topography", :topography, Symbol),
        ("topography_damping_factor", :topography_damping_factor, FT),
        ("terrain_warp", :terrain_warp, Symbol),
        ("sleve_eta_h", :sleve_eta_h, FT),
        ("sleve_s", :sleve_s, FT),
        ("constants_mode", :constants_mode, Symbol),
        ("ic_source", :ic_source, Symbol),
        ("held_suarez", :held_suarez, Bool),
        ("output_dir", :output_dir, String),
        ("diag_period", :diag_period, FT),
        ("dt_save", :dt_save, FT),
        ("ndiag", :ndiag, Int),
    )
    fddg_only = (
        ("interface_flux", :interface_flux, Symbol),
        ("pgf", :pgf, Symbol),
        ("volume_flux", :volume_flux, Symbol),
        ("wb_gravity", :wb_gravity, Bool),
        ("wb_metric", :wb_metric, Symbol),
        ("entropy_correction", :entropy_correction, Bool),
        ("moisture", :moisture, Symbol),
        ("microphysics", :microphysics, Symbol),
        ("precip_timescale", :precip_timescale, FT),
        ("moisture_ic", :moisture_ic, Symbol),
        ("rh0", :rh0, FT),
        ("rh_max", :rh_max, FT),
        ("q_0", :q_0, FT),
        ("z_q1", :z_q1, FT),
        ("z_q2", :z_q2, FT),
    )
    vi_only = (
        ("momentum_adv", :momentum_adv, Symbol),
        ("face_set", :face_set, Symbol),
        ("terrain_u3", :terrain_u3, Symbol),
        ("nu_vert", :ν_vert, FT),
        ("nu_div_frac", :ν_div_frac, FT),
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
