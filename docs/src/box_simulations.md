# Running Box Cases

## Radiative Convective Equilibrium Setups

`ClimaAtmos` supports two versions of the cloud resolving model (CRM) versions of the Radiative Convective Equilibrium Model Intercomparison Projects (RCEMIP) develop by [Wing et. al. (2018)](https://doi.org/10.5194/gmd-11-793-2018) and [Wing et. al. (2024)](https://doi.org/10.5194/gmd-17-6195-2024).

!!! note "Under construction"

    This setup is in active development and subject to change!

### RCEMIP

Following [Wing et. al. (2018)](https://doi.org/10.5194/gmd-11-793-2018), we focus on the small CRM configurations: `RCE_small`. In the future, there may be implementations of the channel version of the RCEMIP CRM, `RCE_large`. `RCE_small` is designed to run with 1km resolution and 100km x 100km horizontal boundary. The vertical boundary is at 30km, and in the ClimaAtmos setup, this must be ran with a 1 second or less timestep as a consequence of representations of sound waves.

We use the vertical grid and initial sounding as specified by [Wing et. al. (2018)](https://doi.org/10.5194/gmd-11-793-2018).
For the original RCEMIP, we implement some noise in the bottom five layers of the sounding to induce convection. To allow for changes in the z-grid used, we employ a linear function for the amplitude of the noise. The amplitude decays linearly with height, beginning at 0.1 the lowest layer (37.0 m) and reaching zero at the sixth zmesh level (520.0 m):

```math
\text{noise}(z) =
\begin{cases}
\dfrac{0.1\,(520.0 - z)}{520.0 - 37.0} \cdot \xi & z < 520.0 \\[2ex]
0 & z \geq 520.0
\end{cases}
```

where $\xi \sim \mathcal{U}(-1, 1)$ is a random number drawn independently per column and per layer.

The surface is a uniform SST set at either `295`,`300`, or `305`, which can be accessed by choosing initial surface profiles `RCEMIPIProfile_295`, `RCEMIPIProfile_300`, and `RCEMIPIProfile_305`. Additionally, the `SST_mean` parameter must be set in the toml.

!!! note "Random noise in bottom five surface levels"

    For reproducibility, make sure to set Random.seed() in the runscript. If a different grid is used than the one suggested by [Wing et. al. (2018)](https://doi.org/10.5194/gmd-11-793-2018), there will still be noise, but only in height levels below 520.0 meters.

We use anisotropic explicit Smagorinsky turbulence, with a Smagorinsky parameter $c_{smag} = 0.2$ ([Smagorinsky (1963)](https://doi.org/10.1175/1520-0493(1963)091%3C0099:GCEWTP%3E2.3.CO;2)). The horizontal advection scheme used by CliMA does not preserve monotonicity so we must employ a technique to prevent the creation of negative tracers during horizontal advection. We recommend the `elementwise-constraint` option.

### RCEMIPII

Differing from the original RCEMIP, the RCEMIPII simulation is initialized with a sinusoidal wave in sea surface temperature as described by [Wing et. al. (2024)](https://doi.org/10.5194/gmd-17-6195-2024):

```math
    SST(x) = \braket{SST} - \frac{\Delta SST}{2} \cos\left(\frac{2 \pi x}{L_x} \right)
```

The `SST_mean` and `SST_delta` are set by parameters in the toml and through the choice of RCEMIPII profile: `RCEMIPIIProfile_295`, `RCEMIPIIProfile_300`, or `RCEMIPIIProfile_305`.
$L_x$ is set by the `SST_wavelength` parameter. $L_x$ is recommended to be the zonal size of the domain, which is 100km for `RCE_small`.

The addition of noise to the bottom five layers of the atmospheric initial profiles are neglected for RCEMIPII, as convection is induced by the SST gradient. All other aspects of RCEMIPII are the same as RCEMIP.
