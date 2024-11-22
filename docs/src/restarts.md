## Restarting Simulations in ClimaAtmos

`ClimaAtmos` supports restarting simulations from previously saved checkpoints,
allowing you to split simulations across multiple runs. This feature is
particularly useful for

* **Performing long simulations on clusters:** Most supercomputers do not allow
  jobs to run for an unlimited amount of wall-time. So, instead of running
  multi-year simulation in a single run, you can break it down into shorter
  segments, restarting from the last saved state.

* **Recovery from interruptions:** If a simulation is unexpectedly interrupted
  (e.g., due to a crash), you can resume it from the last saved checkpoint
  instead of starting over.

* **Sensitivity experiments:** You can run a simulation to a certain point, then
  branch it off into multiple simulations with modified parameters or initial
  conditions, restarting from the common checkpoint.

!!! note

    In the current version, restarting a simulation will check if the `AtmosModel`
    used to produce the restart file is identical to the new one and throw a warning
    if that is not the case. When the warning is produced, it is your responsability
    to ensure that what you are doing makes sense.

### How Restarts Work

`ClimaAtmos` periodically saves the simulation state to a file called a "restart
file". This file contains all the necessary information to resume the simulation
from that point, including the values of all prognostic variables. The frequency
of saving restart files can be configured in the simulation settings using the
`dt_save_state_to_disk` option.

Restart files are HDF5 files that contain the state `Y` of the simulation at the
time of checkpoint. Then, the run restarted by preparing a new simulation as
specified by the new configuration, but using the state read from file. The
values of non-prognostic variables is computed again.

`ClimaAtmos` can automatically detect the latest restart file within a
structured output directory generated using the `ActiveLinkStyle`. When
`ClimaAtmos` is configured to do (e.g., with the `detect_restart_file` option),
`ClimaAtmos` will scan previous output directories for the most recent file that
matches the expected name for a restart file. If none is found, a new simulation
is started.

If is also possible to manually specify a restart file. In this case, this will
override any file automatically detected.

### Accumulated Diagnostics

At the moment, `ClimaAtmos` does not support working with accumulated
diagnostics across restarts. The present limitations are best illustrated with
an example.

Suppose you are saving 30-day averages and stop the simulation at day 45. If you
do so, you'll find output for day 30 and the checkpoint at day 45. Then, if you
restart the simulation, you'll see that the next diagnostic output will be at
day 75, and not day 60. In other words, the counter starts from 0 with every
restart.

!!! note

    If you care about accurate accumulated diagnostics, make sure to line up your
    checkpoint and diagnostic frequencies.
