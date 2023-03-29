# Gravity wave parameterization
Gravity waves have a great impact on the atmospheric circulation. They are usually generated from topography or convection, propagate upward and alter temperature and winds in the middle atmosphere, and influence tropospheric circulation through downward control. The horizontal wavelength for gravity waves ranges from several kilometers to hundreds of kilometers, which is smaller than typical GCM resolution and needs to be parameterized.

The gravity wave drag on the wind velocities (``\overline{\vec{v}}=(u,v)``) are 
```math
\frac{\partial \overline{\vec{v}}}{ \partial t} = ... - \underbrace{\frac{\partial \overline{\vec{v}'w'}}{\partial z}\Big|_{GW} }_{\vec{X}} 
``` 
with $\vec{X} = (X_\lambda, X_\phi)$ representing the subgrid scale zonal and meridional components of the gravity wave drag and is calculated with the parameterization. 

## Non-orographic gravity wave
The non-orographic gravity wave drag parameterization follows the spectra methods described in [alexander1999](@cite). The following assumptions are made for this parameterization to work:
* The wave spectrum consists of independent monochromatic waves, and wave-wave interaction is neglected when propagation and instability is computed.
* The gravity wave propagates vertically and conservatively to the breaking level and deposits all momentum flux into that level, as opposed to the method using saturation profile described in [lindzen1981](@cite).
* The wave breaking criterion is derived in a hydrostatic, non-rotating frame. Including nonhydrostatic and rotation is proved to have negligible impacts.
* The gravity wave is intermittent and the intermittency is computed as the ratio of the total long-term average source to the integral of the source momentum flux spectrum.

### Spetrum of the momentum flux sources
The source spectrum with respect to phase speed is prescribed in Eq. (17) in [alexander1999](@cite). We adapt the equation so that the spectrum writes as a combination of a wide and a narrow band as
```math
B_0(c) = \frac{F_{S0}(c)}{\rho_0} = sgn(c-u_0) \left( Bm\_w \exp\left[ -\left( \frac{c-c_0}{c_{w\_w}} \right) \ln{2} \right] + Bm\_n \exp\left[ -\left( \frac{c-c_0}{c_{w\_n}} \right) \ln{2} \right] \right)
```
where the subscript ``0`` denotes values at the source level. ``c_0`` is the phase speed with the maximum flux magnitude ``Bm``. ``c_w`` is the half-width at half-maximum of the Gaussian.  ``\_w`` and ``\_n`` represent the wide and narrow bands of the spectra.

### Upward propagation and wave breaking
Waves that are reflected will be removed from the spectrum. A wave that breaks at a level above the source will deposit all its momentum flux into that level and be removed from the spectrum.

The reflection frequency is defined as
```math
\omega_r(z) = (\frac{N(z)^2 k^2}{k^2+\alpha^2})^{1/2}
```	
where ``N(z)`` is the buoyancy frequency, ``k`` is the horizontal wavenumber that corresponds to a wavelength of 300 km, ``\alpha = 1/H`` where $H$ is the scale height. ``\omega_r(z)`` is used to determine for each monochromatic wave in the spectrum, whether it will be reflected at height ``z``.

The instability condition is defined as
```math
Q(z,c) = \frac{\rho_0}{\rho(z)} \frac{2N(z)B_0(c)}{k[c-u(z)]^3}
```
``Q(z,c)`` is used to determine whether the monochromatic wave of phase speed ``c`` gets unstable at height ``z``. 

* At the source level
  - if ``|\omega|=k|c-u_0| \geq \omega_r``, this wave would have undergone internal reflection somewhere below and is removed from the spectrum;
  - if ``Q(z_0, c) \geq 1``, it is also removed because it is not stable at the source level.

* At the levels above ``(z_n>z_0)``, ``|\omega(z_n)|=k|c-u(z_n)| \geq \omega_r(z_n)`` is removed from the spectrum. In the remaining speed, ``Q(z_n,c) \geq 1`` are breaking between level ``z_{n-1}`` and ``z_n``, and this portion of momentum flux is all deposited between ``z_{n-1}`` and ``z_n``, which yields
  ```math
  X(z_{n-1/2}) = \frac{\epsilon \rho_0}{\rho(z_{n-1/2)}}\Sigma_j (B_0)j.
  ```
  where ``\epsilon=F_{S0}/\rho_0/\Sigma B_0`` is the wave intermittency. In computing the intermittency, ``F_{S0}`` is the time average total momentum flux and is prescribed as latitude dependent properties. 
  And we get 
  ```math
  X(z_{n-1}) = 0.5*\left[X(z_{n-3/2}) +X(z_{n-1/2}) \right].
  ```
By applying the above parameterization on zonal and meridional winds, the forcing for the physical wind velocity is computed. We further transform them onto the Covariant vectors and that will be the tendencied added onto the momentum equation.
## Orographic gravity wave
