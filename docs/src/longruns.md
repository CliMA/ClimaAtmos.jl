# Buildkite longrun jobs

Here is a brief description of the buildkite longrun jobs.

```
longrun_hydrostatic_balance

Solid body rotation. Test if the dry dycore can maintain the no flow steady state.
```
```
longrun_dry_baroclinic_wave

Dry baroclinic wave. Test the dry dycore initial value problem.
```
```
longrun_dry_baroclinic_wave_he60

Dry baroclinic wave with a higher resolution.
```
```
longrun_moist_baroclinic_wave

Moist baroclinic wave. Test the moist dycore initial value problem.
```
```
longrun_moist_baroclinic_wave_he60

Moist baroclinic wave with a higher resolution.
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
longrun_aquaplanet_allsky_0M

Aquaplanet with idealized insolation, all-sky radiation, and 0-moment microphysics.
Use this job to test new physical components.
```
```
longrun_aquaplanet_allsky_diagedmf_0M

Aquaplanet with idealized insolation, all-sky radiation, diagnostic edmf
and 0-moment microphysics.
```
```
longrun_aquaplanet_allsky_progedmf_diffonly_0M

Aquaplanet with idealized insolation, all-sky radiation, prognostic edmf with diffusion only
and 0-moment microphysics.
```
```
longrun_aquaplanet_allsky_tvinsol_0M_slabocean

Aquaplanet with slab ocean with time-varying insolation, clear-sky radiation,
and 0-moment microphysics.
Test if the coupled system conserves energy and water.
Test if the time-varying insolation yields reasonable results.
```
```
longrun_aquaplanet_allsky_0M_earth

Aquaplanet with idealized insolation, all-sky radiation, 0-moment microphysics, and
Earth topography. Use this job to test topography related features.
```
```
longrun_aquaplanet_allsky_1M

Aquaplanet with idealized insolation, all-sky radiation, 1-moment microphysics.
Use this job to test 1-moment microphysics related features.
```
```
longrun_aquaplanet_dyamond

Aquaplanet setup for DYAMOND (global high-resolution simulation). This job includes
all the atmosphere components required for DYAMOND. It will be updated whenever new components are ready.
```
```
amip_target_diagedmf

Aquaplanet setup for AMIP (atmosphere-land simulation). This job includes all the
working atmosphere components required for AMIP. It will be updated whenever new components are ready.
```
```
amip_target_edonly

Aquaplanet setup for AMIP (atmosphere-land simulation) without convection. This job includes all the
working atmosphere components required for AMIP without convection. It will be updated whenever new components are ready.
```
