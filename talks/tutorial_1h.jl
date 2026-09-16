# Participant script — ClimaAtmos.jl 1-hour hands-on tutorial
#
# From the ClimaAtmos.jl repository root:
#
#   julia +1.11 --project
#
# then copy-paste each section into the REPL. Keep that session open:
# the first `import` / `AtmosSimulation` / `solve_atmos!` compile a lot of
# code; restarting throws that work away.
#
# This hour is about running the model and looking at variables. The grids
# and durations below are deliberately coarse/short — they are not
# scientifically tuned.

import ClimaAtmos as CA

# Instantaneous snapshots so short runs still write NetCDF. Default
# diagnostics only start for runs of 1 hour or longer, and then only as
# time averages. `hus` is moist-only — do not request it on the dry wave.
column_diagnostics = CA.DiagnosticsConfig(;
    default = false,
    additional = (
        "ta" => (; period = "10mins"),
        "thetaa" => (; period = "10mins"),
        "ua" => (; period = "10mins"),
        "wa" => (; period = "10mins"),
        "hus" => (; period = "10mins"),
        "rhoa" => (; period = "10mins"),
    ),
)
sphere_diagnostics = CA.DiagnosticsConfig(;
    default = false,
    additional = (
        "ta" => (; period = "6hours"),
        "ua" => (; period = "6hours"),
        "va" => (; period = "6hours"),
        "wa" => (; period = "6hours"),
        "rhoa" => (; period = "6hours"),
        "pfull" => (; period = "6hours"),
    ),
)

# ---------------------------------------------------------------------------
# 3a. Single column — BOMEX (moist, no EDMF)
# ---------------------------------------------------------------------------
# 60 uniform levels to 3 km. 1 hour of model time at dt = 10 s.
column = CA.Presets.bomex(
    Float32;
    t_end = "1hours",
    job_id = "workshop_bomex",
    diagnostics = column_diagnostics,
)

Ycol = column.integrator.u
@propertynames Ycol.c
@propertynames Ycol.f
@show extrema(Ycol.c.ρ)
@show extrema(Ycol.c.ρe_tot)

CA.solve_atmos!(column)
@show column.integrator.t
@show column.output_dir

# ---------------------------------------------------------------------------
# 3b. Sphere — dry baroclinic wave
# ---------------------------------------------------------------------------
# Cubed sphere, h_elem = 6 (~550 km), z_elem = 10. 1 day of model time at
# dt = 10 min. The wave is only beginning to grow; we are exploring fields,
# not reproducing a published test.
sphere = CA.Presets.baroclinic_wave(
    Float32;
    t_end = "1days",
    job_id = "workshop_bcw",
    diagnostics = sphere_diagnostics,
)

Ysph = sphere.integrator.u
@propertynames Ysph.c
@propertynames Ysph.f
@show extrema(Ysph.c.ρ)

CA.solve_atmos!(sphere)
@show sphere.integrator.t
@show sphere.output_dir

# ---------------------------------------------------------------------------
# Optional: the same runs from YAML (config overlay)
# ---------------------------------------------------------------------------
# config = CA.AtmosConfig(
#     [
#         "config/model_configs/prognostic_edmfx_bomex_column.yml",
#         "talks/workshop_configs/short_column.yml",
#     ];
#     job_id = "workshop_bomex_yaml",
# )
# sim = CA.AtmosSimulation(config)
# CA.solve_atmos!(sim)
#
# config = CA.AtmosConfig(
#     [
#         "config/model_configs/baroclinic_wave.yml",
#         "talks/workshop_configs/short_sphere.yml",
#     ];
#     job_id = "workshop_bcw_yaml",
# )
# sim = CA.AtmosSimulation(config)
# CA.solve_atmos!(sim)

println("Runs finished.")
println("  column output: ", column.output_dir)
println("  sphere output: ", sphere.output_dir)
println("Plot with talks/tutorial_1h_plots.jl from a *default* Julia session.")
