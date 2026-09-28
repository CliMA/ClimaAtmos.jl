# Minimal driver for the chemistry timing test.
#
# Unlike .buildkite/ci_driver.jl this does NOT import Musica by default, and it
# skips all post-processing (plots / reproducibility), so it measures only the
# solve. `import Musica` here is what activates the ClimaAtmosMusica extension:
# without it, GasPhaseChem falls back to the no-op methods and chemistry costs
# ~nothing (the timing test would measure only extra-tracer advection).
#
# SYPD is logged by CA.solve_atmos! as `[ Info: sypd: <value>`. Note that
# solve_atmos! takes a warmup step and precompiles callbacks BEFORE the timed
# region, so SYPD excludes JIT compilation.
#
# Usage (from the repo root; note: no `--` separator when running a script file):
#   julia +1.11 --project=.buildkite timing_test/timing_driver.jl \
#       --config_file config/model_configs/timing_bomex_abba.yml --job_id timing_bomex_abba
import ClimaComms
ClimaComms.@import_required_backends

# Chemistry on/off switch (env var TIMING_CHEMISTRY, default "on").
#   on  -> import Musica, which activates the ClimaAtmosMusica extension and runs
#          the per-cell MICM solve every step.
#   off -> skip Musica. The tracer set is still read from `chemistry_config` by the
#          base package (chemistry_species_names), so all the ρq_gas_* tracers are
#          still created and ADVECTED -- only the MICM solve is deactivated
#          (update_chemistry! falls back to a no-op). This isolates the cost of
#          transporting the extra tracers from the cost of the chemistry solve.
if lowercase(get(ENV, "TIMING_CHEMISTRY", "on")) == "on"
    import Musica
    @info "TIMING_CHEMISTRY=on: Musica loaded, chemistry solve ACTIVE"
else
    @info "TIMING_CHEMISTRY=off: Musica NOT loaded, tracers advected but chemistry solve INACTIVE"
end

import ClimaAtmos as CA
import Random
Random.seed!(1234)

(; config_file, job_id) = CA.commandline_kwargs()
config = CA.AtmosConfig(config_file; job_id)
simulation = CA.get_simulation(config)
sol_res = CA.solve_atmos!(simulation)   # logs "sypd: ..."

if sol_res.ret_code == :simulation_crashed
    error("Simulation $(job_id) crashed; SYPD is not meaningful.")
end
