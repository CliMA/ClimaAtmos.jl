# Buildkite longrun jobs

Here is a brief description of the buildkite longrun jobs. The authoritative
list is `.buildkite/longruns_gpu/pipeline.yml`, with configurations in
`config/longrun_configs/`.

```
longrun_hydrostatic_balance

Solid body rotation. Test if the dry dycore can maintain the no flow steady state.
```

```
longrun_dry_baroclinic_wave_he60

Dry baroclinic wave. Test the dry dycore initial value problem.
```

```
longrun_moist_baroclinic_wave_he60

Moist baroclinic wave. Test the moist dycore initial value problem.
```

```
longrun_dry_held_suarez

Dry Held Suarez. Test the dry dycore with an equilibrium state with sources and sinks.
```

```
longrun_moist_held_suarez

Moist Held Suarez. Test the moist dycore with an equilibrium state with sources and sinks.
```

```
longrun_aquaplanet_allsky_progedmf_0M

Aquaplanet with idealized insolation, all-sky radiation, PROPHET, and 0-moment
microphysics.
```

```
longrun_aquaplanet_allsky_progedmf_1M

Aquaplanet with idealized insolation, all-sky radiation, PROPHET, and 1-moment
microphysics.
```

```
longrun_aquaplanet_allsky_tvinsol_0M_slabocean

Aquaplanet with slab ocean with time-varying insolation, all-sky radiation
(with clear-sky diagnostics), and 0-moment microphysics.
Test if the coupled system conserves energy and water.
Test if the time-varying insolation yields reasonable results.
```

```
longrun_aquaplanet_allsky_1M

Aquaplanet with idealized insolation, all-sky radiation, ED-only EDMF, and
1-moment microphysics.
Use this job to test 1-moment microphysics related features.
```

```
amip_target

Global setup targeting AMIP (atmosphere-land simulation), with Earth topography,
prescribed aerosols, time-varying trace gases, non-orographic gravity-wave drag,
PROPHET, and 1-moment microphysics. This job includes all the working atmosphere
components required for AMIP and will be updated whenever new components are
ready.
```
