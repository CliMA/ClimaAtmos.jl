# Buildkite longrun jobs

Here is a brief description of the buildkite longrun jobs.
## GPU longruns
```
longrun_bw_rhoe_highres

Dry baroclinic wave. Test the dry dycore initial value problem.
```
```
longrun_bw_rhoe_equil_highres

Moist baroclinic wave. Test the moist dycore initial value problem.
```
```
longrun_hs_rhoe_dry_55km_nz63

Dry Held Suarez. Test the dry dycore with an equilibrium state with sources and sinks.
```
```
longrun_hs_rhoe_equil_55km_nz63_0M

Moist Held Suarez. Test the moist dycore with an equilibrium state with sources and sinks.
```
```
longrun_aquaplanet_rhoe_equil_55km_nz63_clearsky_0M

Aquaplanet with idealized insolation, clear-sky radiation, and 0-moment microphysics.
Use this job to test new physical components.
```
```
longrun_aquaplanet_rhoe_equil_55km_nz63_clearsky_diagedmf_0M

Aquaplanet with idealized insolation, clear-sky radiation, diagnostic edmf
and 0-moment microphysics.
```
longrun_aquaplanet_rhoe_equil_55km_nz63_allsky_diagedmf_0M

Aquaplanet with idealized insolation, all-sky radiation, diagnostic edmf
and 0-moment microphysics.
```

## CPU longruns
```
longrun_sphere_hydrostatic_balance_rhoe

Solid body rotation. Test if the dry dycore can maintain the no flow steady state.
```
```
longrun_zalesak_tracer_energy_bw_rhoe_equil_highres

Moist baroclinic wave with limiter and flux-corrected transport (zalesak).
```
```
longrun_ssp_bw_rhoe_equil_highres

Moist baroclinic wave with the SSP timestepper.
```
```
longrun_aquaplanet_rhoe_equil_55km_nz63_gray_0M

Aquaplanet with idealized insolation, gray radiation, and 0-moment microphysics.
```
```
longrun_aquaplanet_rhoe_equil_55km_nz63_clearsky_diagedmf_diffonly_0M

Aquaplanet with idealized insolation, clear-sky radiation, diagnostic edmf (diffusion only)
and 0-moment microphysics.
```
```
longrun_aquaplanet_rhoe_equil_55km_nz63_clearsky_tvinsol_0M_slabocean

Aquaplanet with slab ocean with time-varying insolation, clear-sky radiation, 
and 0-moment microphysics.
Test if the coupled system conserves energy and water.
Test if the time-varying insolation yields reasonable results.
```
```
longrun_aquaplanet_rhoe_equil_55km_nz63_clearsky_0M_earth

Aquaplanet with idealized insolation, clear-sky radiation, 0-moment microphysics, and
Earth topography. Use this job to test topography related features.
```
```
longrun_bw_rhoe_equil_highres_topography_earth

Moist baroclinic wave with Earth topography. Test moist dycore with topography.
```
```
longrun_aquaplanet_rhoe_equil_highres_clearsky_ft32_earth

Moist aquaplanet with topography. Will be removed soon.
```
```
longrun_aquaplanet_dyamond

Aquaplanet setup for DYAMOND (global high-resolution simulation). This job includes
all the atmosphere components required for DYAMOND.
```
```
longrun_aquaplanet_amip

Aquaplanet setup for AMIP (atmosphere-land simulation). This job includes all the
working atmosphere components required for AMIP. It can be unstable as some of them
are still under development. It will be updated whenever new components are ready.
```