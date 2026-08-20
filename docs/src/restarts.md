# Restarting and Checkpointing

`ClimaAtmos` supports restarting simulations from previously saved checkpoints,
allowing you to split simulations across multiple runs. This feature is
useful for

  - **Performing long simulations on clusters:** Most supercomputers do not allow
    jobs to run for an unlimited amount of wall-time. Instead of running a
    multi-year simulation in a single run, you can break it down into shorter
    segments, restarting from the last saved state.

  - **Recovery from interruptions:** If a simulation is unexpectedly interrupted
    (e.g., due to a crash), you can resume it from the last saved checkpoint
    instead of starting over.

  - **Sensitivity experiments:** You can run a simulation to a certain point, then
    branch it off into multiple simulations with modified parameters or initial
    conditions, restarting from the common checkpoint.

!!! note

    In the current version, restarting a simulation will check if the `AtmosModel`
    used to produce the restart file is identical to the new one and throw a warning
    if that is not the case. When the warning is produced, it is your responsibility
    to ensure that what you are doing makes sense.

!!! note

    By default, the simulation cannot be restarted in a reproducible way. To
    enable reproducible restarts, you need to set `reproducible_restart` to `true`.
    When `reproducible_restart` is `true`, `ClimaAtmos` recalculates the grid-scale
    cloud fraction and uses it in the buoyancy gradient calculation to ensure deterministic
    behavior across restarts. We recommend disabling this option for production runs.

## Checkpointing and restarting a run

The minimal recipe is two configuration keys:

```yaml
# my_run.yml
dt_save_state_to_disk: "6hours"  # write a checkpoint every 6 hours
detect_restart_file: true        # on launch, resume from the latest checkpoint
```

Run the simulation as usual. If it is interrupted, launching the same
configuration again finds the most recent checkpoint in the output directory
and resumes from it; if none exists, a fresh simulation starts. To restart
from a specific checkpoint instead, set `restart_file: <path/to/dayD.S.hdf5>`.

The rest of this page explains what these options do.

## How Restarts Work

`ClimaAtmos` periodically saves the simulation state to a *restart file*, an
HDF5 file holding everything needed to resume from that point: the values of all
prognostic variables.

Each file is named for the elapsed simulation time, as `day$day.$sec.hdf5`. A
checkpoint written 10 days and 3600 seconds into a run is `day10.3600.hdf5`.
They are written to the run's output directory, which defaults to
`output/<job_id>`. Under the default `ActiveLink` style that directory holds a
numbered subdirectory per run — `output_0000`, `output_0001`, and so on — with an
`output_active` symlink pointing at the current one, so a checkpoint from the
first run lands in `output/<job_id>/output_0000/day10.3600.hdf5`.

Set how often they are written with `dt_save_state_to_disk` in a YAML
configuration, or with the `checkpoint_frequency` keyword when building an
`AtmosSimulation` from a script. Duration strings, including `"<N>months"`,
are accepted. The default is `Inf`, which writes no checkpoints.

Inside the file the state is stored under the name `Y`, alongside two attributes:
`time`, the simulation time in seconds, and `atmos_model_hash`, a hash of the
model configuration that a restart checks so a checkpoint cannot be loaded
silently into a different model.

On restart, a new simulation is prepared as specified by the new configuration,
but takes its state from the file. Non-prognostic variables are recomputed from
that state.

`ClimaAtmos` can automatically detect the latest restart file within a
structured output directory generated using the `ActiveLinkStyle`. When
`ClimaAtmos` is configured to do so (e.g., with the `detect_restart_file` option),
`ClimaAtmos` scans previous output directories for the most recent file that
matches the expected name for a restart file. If none is found, a new simulation
is started.

It is also possible to manually specify a restart file with the `restart_file`
configuration option, which overrides any automatically detected file.

`ClimaAtmos` also provides the configuration option `t_start` to change the initial
time of the simulation without changing the start date. This option can be
useful when manually restarting a simulation (e.g., by overwriting the initial
conditions). When a simulation is restarted from a checkpoint, the checkpoint
supplies the time and any `t_start` given is ignored, with a warning.

## Accumulated Diagnostics

At the moment, `ClimaAtmos` does not support working with accumulated
diagnostics across restarts. The present limitations are best illustrated with
an example.

Suppose you are saving 30-day averages and stop the simulation at day 45. You'll
find output for day 30 and the checkpoint at day 45. Then, if you
restart the simulation, you'll see that the next diagnostic output will be at
day 75, and not day 60: the counter starts from 0 with every restart.

!!! note

    If you care about accurate accumulated diagnostics, make sure to line up your
    checkpoint and diagnostic frequencies. ClimaAtmos checks this at setup and
    warns when the checkpoint frequency is not a multiple of every accumulation
    period.
