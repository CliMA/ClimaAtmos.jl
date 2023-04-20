# Gravity wave parameterization
Gravity waves have a great impact on the atmospheric circulation. They are usually generated from topography or convection, propagate upward and alter temperature and winds in the middle atmosphere, and influence tropospheric circulation through downward control. The horizontal wavelength for gravity waves ranges from several kilometers to hundreds of kilometers, which is smaller than typical GCM resolution and needs to be parameterized.

The gravity wave drag on the wind velocities (``\overline{\vec{v}}=(u,v)``) are 
```math
\frac{\partial \overline{\vec{v}}}{ \partial t} = ... - \underbrace{\frac{\partial \overline{\vec{v}'w'}}{\partial z}\Big|_{GW} }_{\vec{X}} 
``` 
with $\vec{X} = (X_\lambda, X_\phi)$ representing the sub-grid scale zonal and meridional components of the gravity wave drag and is calculated with the parameterization. 

## Non-orographic gravity wave
The non-orographic gravity wave drag parameterization follows the spectra methods described in [alexander1999](@cite). The following assumptions are made for this parameterization to work:
* The wave spectrum consists of independent monochromatic waves, and wave-wave interaction is neglected when propagation and instability is computed.
* The gravity wave propagates vertically and conservatively to the breaking level and deposits all momentum flux into that level, as opposed to the method using saturation profile described in [lindzen1981](@cite).
* The wave breaking criterion is derived in a hydrostatic, non-rotating frame. Including non-hydrostatic and rotation is proved to have negligible impacts.
* The gravity wave is intermittent and the intermittency is computed as the ratio of the total long-term average source to the integral of the source momentum flux spectrum.

### Spectrum of the momentum flux sources
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
The orographic gravity wave drag parameterization follows the methods described in [garner2005](@cite). The momentum drag from sub-grid scale mountains is divided into a non-propagating component and a propagating component. The non-propagating component forces momentum drag within the planetary boundary layer while the propagating component generate a `c=0` gravity wave which propagates upwards and deposit momentum flux to the layers where it breaks.

### PBL top
There are many ways to determine the PBL top. We implement the following simple criteria to find the PBL top level k as the highest level that satisfies
```math
^c p[k] \ge 0.5 ∗ ^f p[0]
```
and
```math
^c T[0] + T_{boost} - ^c T[k] > g/c_p * ( ^c z[k] - ^c z[0] )
```
where the superscripts ``f`` and ``c`` represent cell faces and cell centers in the vertical stencils. ``T_{boost} = 1.5 \mathrm{K}`` is the surface temperature boost to improve PBL height estimate.

### Orographic information
The orographic information needed in generating the base momentum flux for low-level flow encountering the sub-grid scale mountains. We compute the tensor ``\textbf{T}`` and the scalar ``h_{max}`` from the Earth elevation data (GFDL codes [here](https://caltech.box.com/s/w4szffattzofarpwyv9rmm5s77jgo3o4)).

#### Tensor ``\textbf{T}``
The tensor ``\textbf{T}``, which contains all relevant information including amplitude, variance, orientation, and anisotropy about topography, is computed as
```math
\textbf{T} = \nabla \chi (\nabla h)^T,
```
where ``h`` is the earth elevation, and ``\chi = - \frac{\rho N}{2\pi} \frac{h(x')}{|x-x'|} \int \int dx' dy'`` is the velocity potential.

#### ``h_max``
``h_max`` represents the relation between the local elevation with its surroundings. It is computed as the ``4th`` moments of the local elevation with a certain area.

### Base flux
The base momentum flux generated is computed and divided into the propagating and non-propagating components.

Let ``\overline{\cdot}`` represents the mean property of the low-level flow which can be obtained
as either the average within PBL or the value at the first cell center right above PBL top. Let ``\overline{V} = (\overline{u}, \overline{v})``, ``\overline{N}``, and ``\overline{\rho}`` represent the horizontal wind, buoyancy frequency, and density of the low-level flow. ``\overline{N}`` is computed as 
```math
\overline{N} ^2 = \frac{g}{\overline{T}} * \left( \overline{\frac{dT}{dz}} + \frac{g}{c_p} \right).
```

The base flux is computed as the linear drag following
```math
\tau = \overline{\rho} \overline{N} \langle \textbf{T} \rangle ^T \overline{V},
```
where ``\langle \textbf{T} \rangle = [t_{11}, t_{12}; t_{21}, t_{22}]`` is the tensor that contains orographic information. In the code, we compute the zonal and meridional components separately as 
```math
\tau_x = \overline{\rho} \overline{N} (t_{11} \overline{u} + t_{21} \overline{v}),
```
```math
\tau_y = \overline{\rho} \overline{N} (t_{12} \overline{u} + t_{22} \overline{v}).
```

The base flux is then corrected using Froude number and saturation velocity. Let
```math
V_{\tau} = \max(\epsilon_0, - \overline{V} \cdot \frac{\tau}{|\tau|}),
```
and given the orographic information ``h_{max}`` and ``h_{min}``, the max and min Froude numbers are computed as
```math
Fr_{max} = h_{max} \frac{\overline{N}}{V_{\tau}},
```
```math
Fr_{min} = h_{min} \frac{\overline{N}}{V_{\tau}}.
```
We also compute the med Froude number as
```math
Fr_{med} = Fr_{crit} + Fr_{int},
```
where ``Fr_{crit} = 0.7`` is the critical Froude number for nonlinear flow, ``Fr_{int} = 0.5`` is an arbitrary parameter.

The saturation velocity is computed as
```math
U_{sat} = \sqrt{\frac{\overline{\rho}}{\rho_0} \frac{V_{\pmb{\tau}}^3}{\overline{N} L_0}},
```
where ``\rho_0 = 1.2 \mathrm{kg/m^3}`` is the arbitrary density scale, and ``L_0 = 80e3 \mathrm{m}`` is the arbitrary horizontal length scale. 

The following set of intermediate variables (``FrU``'s) are computed to correct the linear base flux:
```math
FrU_{sat} = Fr_{crit} * U_{sat},
```
```math
FrU_{min} = Fr_{min} * U_{sat},
```
```math
FrU_{med} = Fr_{med} * U_{sat},
```
```math
FrU_{max} = \max(Fr_{max} * U_{sat}, FrU_{min} + \epsilon_0),
```
```math
FrU_{clp} = \min(FrU_{max}, \max(FrU_{min}, FrU_{sat})),
```
```math
FrU_0 = \frac{U_0}{V_{\tau}}U_{sat},
```
and ``U_0=1.0 \mathrm{m/s}`` is the arbitrary velocity scale.

Now the correct linear drag is computed as 
```math
\tau_l = \frac{FrU_{max}^{2+\gamma-\epsilon} - FrU_{min}^{2+\gamma-\epsilon}}{2+\gamma-\epsilon},
```
and the propagating and non-propagating parts of the drag are computed as
```math
\tau_p = a_0 \left[ \frac{FrU_{clp}^{2+\gamma-\epsilon} - FrU_{min}^{2+\gamma-\epsilon}}{2+\gamma-\epsilon} + FrU_{sat}^{\beta+2} \frac{FrU_{max}^{\gamma-\epsilon-\beta} - FrU_{clp}^{\gamma-\epsilon-\beta}}{\gamma-\epsilon-\beta} \right],
```
```math
\tau_{np} = a_1 \frac{U_{sat}}{1+\beta} \left[ \frac{FrU_{max}^{1+\gamma-\epsilon} - FrU_{clp}^{1+\gamma-\epsilon}}{1+\gamma-\epsilon} - FrU_{sat}^{\beta+1} \frac{FrU_{\max}^{\gamma-\epsilon-\beta} - FrU_{clp}^{\gamma-\epsilon-\beta}}{\gamma-\epsilon-\beta} \right].
```

Here, $(\gamma, \epsilon, \beta) = (0.4, 0.0, 0.5)$ are arbitrary parameters that describe the mountain shapes.

### Saturation profiles for the propagating component
The vertical profiles of saturated momentum flux ``\tau_{sat}`` is computed then so that momentum forcing can be obtained for ``d\overline{V}/dt = -\overline{\rho}^{-1}d\tau_{sat}/dz``. This only applies to the propagating part.

Similar to the base flux calculation but for the 3D fields, we computed ``N`` and ``V_\tau`` at cell faces as
```math
^f N[k]^2 = \frac{g}{^f T[k]} * ( ^f \overline{\frac{dT}{dz}}[k] + \frac{g}{cp}),
```
```math
^f V_{\tau}[k] = \max(\epsilon_0, - V[k] \cdot \frac{\tau}{|\tau|}).
```

Let ``L_1 = L_0 * \max(0.5, \min(2.0, 1.0-samp*V_\tau*d^2V_{\tau}/N^2))`` where ``samp=1.0`` is the correction for coarse sampling of ``d^2V/dz^2``, and
```math
^f d^2V_{\tau}[k] = - \frac{d^2 V}{dz^2}[k] \cdot \frac{\tau}{|\tau|}.
```
The saturated velocity ``U_{sat}`` is refined as follows and used to computed the intermediate ``FrU's``
```math
U_{sat} = \min(U_{sat}, \sqrt{\frac{^\mathrm{f} \rho}{\rho_0} \frac{^\mathrm{f} V_{\pmb{\tau}}^3}{^\mathrm{f}N L_1} }).
```

The ``FrU_{min}`` and ``FrU_{max}`` are inherited from the base flux calculation. Let's save the source level ``FrU_{sat}`` and ``FrU_{clp}`` into
```math
FrU_{sat0} = FrU_{sat},
```
```math
FrU_{clp0} = FrU_{clp},
```
and update ``FrU_{sat}`` and ``FrU_{clp}`` as 
```math
FrU_{sat} = Fr_{crit} * U_{sat},
```
```math
FrU_{clp} = \min(FrU_{\max}, \max(FrU_{\min}, FrU_{sat})).
```

Then, the saturated profile of propagating component of the momentum flux is 
```math
\tau_{sat} = a_0 \left[ \frac{FrU_{clp}^{2+\gamma-\epsilon} - FrU_{\min}^{2+\gamma-\epsilon}}{2+\gamma-\epsilon} + FrU_{sat}^2 FrU_{sat0}^\beta \frac{FrU_{\max}^{\gamma-\epsilon-\beta} - FrU_{clp0}^{\gamma-\epsilon-\beta}}{\gamma-\epsilon-\beta} + FrU_{sat}^2 \frac{FrU_{clp0}^{\gamma-\epsilon} - FrU_{clp}^{\gamma-\epsilon}}{\gamma-\epsilon} \right]
```

If the wave does not break and propagates all the way up to the model top, the residual momentum carried by this part will be redistributed throughout the column weighted by pressure to conserve momentum. That is,
```math
\tau_{sat}[k] = \tau_{sat}[k] - \tau_{sat}[end] \frac{^f p[1]-^f p[k]}{^f p[1]-^f p[end]}.
```

### Velocity tendencies due to the orographic drag
#### Propagating component
The forcing from the propagating part on the zonal and meridional wind are
```math
^c \left( \frac{du}{dt} \right) _p = - \frac{1}{^c \rho} \frac{\tau_x}{\tau_l} \frac{d\tau_{sat}}{dz},
```
```math
^c \left( \frac{dv}{dt} \right) _p = - \frac{1}{^c \rho} \frac{\tau_y}{\tau_l} \frac{d\tau_{sat}}{dz}.
```
Here, ``(\tau_x, \tau_y, \tau_l)`` is computed in the base flux calculation, and ``\tau_{sat}`` is calculated in the saturation flux profile. The propagating part functions throughout the entire column.

#### Non-propagating component
Let's first find the reference level ``kref`` below which the non-propagating part functions to decelerate the flow. Let ``k`` loops from PBL top upwards, initiate ``phase = 0.0`` and ``z_{last} = ^fz[k]``, if ``phase \leq \pi`` and ``k`` is below the top level, update ``phase`` and ``z_{last}`` as
```math
phase += \frac{\max(N_{min}, \min(N_{max}, ^fN[k]))}{\max(vvmin, ^f V_{\tau}[k])} * (^c z[k+1] - z_{last})
```
```math
z_{last} = ^c z[k+1]
```
and move one level up. Here, ``N_{min} = 0.7e-2, N_{max}=1.7e-2, vvmin = 1.0``; and ``(^f N, ^f V_{\tau})`` are computed during the saturation profile calculation. Let ``kref=k`` when the loop terminates.

The drag forcing due to non-propagating component functions from the PBL top to the level of ``kref`` and is weighted by pressure. The weights of each level are computed as
```math
weight[k] = ^c p[k] - ^f p[kref],
```
and the sum of the weights is 
```math
wtsum += \frac{^f p[k-1] - ^f p[k]}{weight[k]}.
```

For level ``k`` between PBL top and ``kref``, the forcing due to non-propagating component is
```math
^c \left( \frac{du}{dt} [k] \right)_{np} = g \frac{\tau_{np}}{\tau_l} \frac{weight[k]}{wtsum} \tau_x, 
```
```math
^c \left( \frac{dv}{dt} [k] \right)_{np} = g \frac{\tau_{np}}{\tau_l} \frac{weight[k]}{wtsum} \tau_y, 
```
where ``(\tau_x, \tau_y, \tau_l, \tau_{np})`` is computed in the base flux calculation.

### Constrain the forcings
Total drag from both components are
```math
^c \left( \frac{du}{dt} [k] \right)_{\tau} = ^c \left( \frac{du}{dt} [k] \right)_{p} + ^c \left( \frac{du}{dt} [k] \right)_{np},
```
```math
^c \left( \frac{dv}{dt} [k] \right)_{\tau} = ^c \left( \frac{dv}{dt} [k] \right)_{p} + ^c \left( \frac{dv}{dt} [k] \right)_{np}.
```
To avoid instability due to large tendencies from the forcing, let's constrain the forcing magnitude with ``\epsilon_V = 3e-3``, and let
```math
^c \left( \frac{du}{dt} [k] \right)_{\tau} = \max(-\epsilon_V, \min(\epsilon_V, ^c \left( \frac{du}{dt} [k] \right)_{\pmb{\tau}})),
```
```math
^c \left( \frac{dv}{dt} [k] \right)_{\tau} = \max(-\epsilon_V, \min(\epsilon_V, ^c \left( \frac{dv}{dt} [k] \right)_{\pmb{\tau}})).
```

Here we computed the forcing on the physical velocity (i.e., zonal and meridional wind). They are converted to the Covariant12Vector before being added to ``Y_t`` in the codes.