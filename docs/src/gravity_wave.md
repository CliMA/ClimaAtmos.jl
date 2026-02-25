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

The reference frame for the spectrum is latitude-dependent: it is ground-relative in the extra-tropics and relative to the source-level zonal wind in the tropics, with smoothing applied at the transition.

### Upward propagation and wave breaking
Waves that are reflected will be removed from the spectrum. A wave that breaks at a level above the source will deposit all its momentum flux into that level and be removed from the spectrum.

The reflection frequency is defined as
```math
\omega_r(z) = (\frac{N(z)^2 k^2}{k^2+\alpha^2})^{1/2}
```
where ``N(z)`` is the buoyancy frequency, ``k`` is the horizontal wavenumber that corresponds to a wavelength of 300 km, and ``\alpha = 1/(2H)`` where $H$ is the scale height. ``\omega_r(z)`` is used to determine for each monochromatic wave in the spectrum, whether it will be reflected at height ``z``.

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

To ensure momentum conservation, any momentum flux that propagates to the model top without breaking is re-deposited evenly throughout the damping layer (sponge layer).

And we get
```math
X(z_{n-1}) = 0.5*\left[X(z_{n-3/2}) +X(z_{n-1/2}) \right].
```

## Orographic gravity wave
The orographic gravity wave drag parameterization follows the methods described in [garner2005](@cite). The momentum drag from sub-grid scale mountains is divided into a non-propagating component and a propagating component. The non-propagating component forces momentum drag within the planetary boundary layer while the propagating component generate a stationary (``c = 0``, zero phase speed) gravity wave which propagates upwards and deposit momentum flux to the layers where it breaks.

### Planetary Boundary Layer (PBL) top
There are many ways to determine the PBL top. We implement the following simple criteria to find the PBL top level k as the highest level that satisfies
```math
^cp[k] \ge 0.5 ∗ ^cp[1]
```
and
```math
^cT[1] + T_{boost} - ^cT[k] > g/c_p * ( ^cz[k] - ^cz[1] )
```
where the superscript ``c`` represents cell centers in the vertical stencils, and ``[1]`` denotes the first (lowest) cell center level. ``T_{boost} = 1.5 \mathrm{K}`` is the surface temperature boost to improve PBL height estimate.

### Orographic information
The orographic information needed in generating the base momentum flux for low-level flow encountering the sub-grid scale mountains. We compute the tensor ``\textbf{T}`` and the scalar ``h_{max}`` from the Earth elevation data (GFDL code [here](https://github.com/NOAA-GFDL/atmos_phys/blob/main/atmos_param/topo_drag/topo_drag.F90); note: the original reference Fortran code used in the implementation of our drag computation requires [Caltech Box access](https://caltech.box.com/s/w4szffattzofarpwyv9rmm5s77jgo3o4)).

#### Tensor ``\textbf{T}``
The tensor ``\textbf{T}``, which contains all relevant information including amplitude, variance, orientation, and anisotropy about topography, is computed as
```math
\textbf{T} = \nabla \chi (\nabla h)^T,
```
where ``h`` is the earth elevation, and ``\chi = - \frac{\rho N}{2\pi} \frac{h(x')}{|x-x'|} \int \int dx' dy'`` is the velocity potential.

#### ``h_{max}``
``h_{max}`` represents the effective maximum height of the orographic features within a grid cell relative to the mean surface.

**Technical Note:** The definition of ``h_{max}`` depends on the input data mode. In both modes using pre-computed GFDL input and raw topography, it is derived from the 4th root of the distance-weighted 4th moment of subgrid elevation deviations from the local mean, i.e., ``h_{max} = \left( \sum_i w_i (h_i - \bar{h})^4 / \sum_i w_i \right)^{1/4}``, to estimate peak obstruction height.

### Base flux
The base momentum flux generated is computed and divided into the propagating and non-propagating components.

Let ``\overline{\cdot}`` represents the mean property of the low-level flow which can be obtained as either the average within PBL or the value at the first cell center right above PBL top. Let ``\overline{V} = (\overline{u}, \overline{v})``, ``\overline{N}``, and ``\overline{\rho}`` represent the horizontal wind, buoyancy frequency, and density of the low-level flow. ``\overline{N}`` is computed as
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
Here, ``Fr_{crit} = 0.7`` is the critical Froude number for nonlinear flow. ``Fr_{crit}`` acts as the primary tuning lever for partitioning drag between the low-level blocked flow and the upper-level wave breaking.

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
FrU_{max} = \max(Fr_{max} * U_{sat}, FrU_{min} + \epsilon_0),
```
```math
FrU_{clp} = \min(FrU_{max}, \max(FrU_{min}, FrU_{sat})).
```

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
The non-propagating drag is then scaled by the Froude number:
```math
\tau_{np} = \frac{\tau_{np}}{\max(Fr_{crit}, Fr_{max})}.
```

Here, ``(\gamma, \epsilon, \beta) = (0.4, 0.0, 0.5)`` are empirical shape parameters constrained by observations [Garner 2005]. Specifically, ``\gamma=0.4`` is derived from the observed scaling of mountain width versus height.

### Saturation profiles for the propagating component
The vertical profiles of saturated momentum flux ``\tau_{sat}`` is computed then so that momentum forcing can be obtained for ``d\overline{V}/dt = -\overline{\rho}^{-1}d\tau_{sat}/dz``. This only applies to the propagating part.

Similar to the base flux calculation but for the 3D fields, we compute ``V_\tau`` at cell centers as
```math
^c V_{\tau}[k] = \max(\epsilon_0, - V[k] \cdot \frac{\tau}{|\tau|}),
```
where ``\epsilon_0`` denotes a measure of floating-point precision.

Let ``L_1 = L_0 * \max(0.5, \min(2.0, 1.0 - 2 V_\tau \cdot d^2V_{\tau} / N^2))`` where the factor of 2 is a correction for coarse sampling of ``d^2V/dz^2``, and
```math
^c d^2V_{\tau}[k] = - \frac{d^2 V}{dz^2}[k] \cdot \frac{\tau}{|\tau|}.
```
The saturated velocity ``U_{sat}`` is refined as follows and used to compute the intermediate ``FrU``'s:
```math
U_{sat} = \min(U_{sat}, \sqrt{\frac{^c \rho}{\rho_0} \frac{^c V_{\pmb{\tau}}^3}{^c N \cdot L_1} }).
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
\tau_{sat}[k] = \tau_{sat}[k] - \tau_{sat}[end] \frac{^cp[1] - ^cp[k]}{^cp[1] - ^cp[end]}.
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
Let's first find the reference level ``z_{ref}`` below which the non-propagating part functions to decelerate the flow. Iterating over face levels above the PBL top, we accumulate phase as
```math
phase += (^fz[k] - z_{pbl}) \cdot \frac{\max(N_{min}, \min(N_{max}, ^f N[k]))}{\max(vvmin, ^f V_{\tau}[k])}
```
and set ``z_{ref} = ^f z[k]`` when ``phase > \pi``. Here, ``N_{min} = 0.7e-2, N_{max}=1.7e-2, vvmin = 1.0``; and ``(^f N, ^f V_{\tau})`` are computed during the saturation profile calculation. If phase never exceeds ``\pi``, ``z_{ref}`` defaults to the model top.

The drag forcing due to non-propagating component functions from the PBL top to ``z_{ref}`` and is weighted by pressure. The weights at each cell center are computed by interpolating face-level pressure differences:
```math
weight[k] = \overline{^f p - ^f p_{ref}}^c,
```
where ``^f p_{ref}`` is the face pressure at ``z_{ref}``, and the overbar denotes interpolation to cell centers. The pressure layer thickness is similarly interpolated:
```math
diff[k] = \overline{^f p[k-1] - ^f p[k]}^c,
```
and the sum of the weights is
```math
wtsum = \sum_{k \in \mathrm{mask}} \frac{diff[k]}{weight[k]}.
```
The mask selects cells that overlap with the interval ``[z_{pbl}, z_{ref})`` and have nonzero weights.

For masked levels, the forcing due to non-propagating component is
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


## Implementation Details - Orographic Gravity Wave

### End-to-End Pipeline

The orographic gravity wave implementation consists of two stages: an **offline preprocessing** stage that computes topographic information from raw elevation data, and an **online runtime** stage that uses that information to compute drag tendencies during simulation.

```
┌─────────────────────────────────────────────────────────────────┐
│                  OFFLINE PREPROCESSING                          │
│                                                                 │
│  ETOPO2022 elevation (NetCDF)                                   │
│       │                                                         │
│       ▼                                                         │
│  compute_OGW_info()                                             │
│   ├─ calc_hpoz_latlon()  ──→  hmax, hmin  (lat-lon grid)        │
│   ├─ calc_velocity_potential()  ──→  χ    (2D Hilbert transform)│
│   └─ calc_orographic_tensor()  ──→  T    (lat-lon grid)         │
│       │                                                         │
│       ▼                                                         │
│  regrid_OGW_info()  ──→  SpaceVaryingInput to spectral element  │
│       │                                                         │
│       ▼                                                         │
│  write_computed_drag!()  ──→  HDF5 artifact                     │
│       (hmax, hmin, t11, t12, t21, t22)                          │
└─────────────────────────────────────────────────────────────────┘
                            │
                    stored as ClimaArtifact
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ONLINE RUNTIME                                 │
│                                                                 │
│  Simulation startup:                                            │
│   compute_ogw_drag()  ──→  load HDF5 artifact                   │
│   orographic_gravity_wave_cache()  ──→  allocate ~30 fields     │
│                                                                 │
│  Every dt_ogw (callback):                                       │
│   orographic_gravity_wave_compute_tendency!()                   │
│   ├─ buoyancy frequency N                                       │
│   ├─ get_pbl_z!()                                               │
│   ├─ calc_base_flux!()                                          │
│   ├─ calc_saturation_profile!()                                 │
│   ├─ calc_propagate_forcing!()                                  │
│   ├─ calc_nonpropagating_forcing!()                             │
│   └─ clamp to ±3e-3 m/s²                                        │
│       │                                                         │
│       ▼  (stored in cache)                                      │
│                                                                 │
│  Every dt (integrator):                                         │
│   orographic_gravity_wave_apply_tendency!()                     │
│   └─ Yₜ.c.uₕ += Covariant12Vector(ᶜuforcing, ᶜvforcing)         │
└─────────────────────────────────────────────────────────────────┘
```

### Topography Preprocessing Pipeline

For Earth topography simulations, the orographic information is preprocessed offline before runtime. The preprocessing is driven by `test/parameterized_tendencies/gravity_wave/orographic_gravity_wave/compute_preprocessed_topography.jl` and the core computation lives in `src/parameterized_tendencies/gravity_wave_drag/orographic_gravity_wave_helper.jl`.

The pipeline proceeds as follows:

1. **Loading**: The ETOPO2022 elevation artifact is loaded onto a regular lat-lon grid (downsampled by a configurable `skip_pt` factor).

2. **Height statistics** (`calc_hpoz_latlon`): ``h_{max}`` is computed using a distance-weighted 4th-moment calculation over the sub-grid terrain within each GCM grid cell. This captures the effective peak obstruction height rather than the raw maximum. ``h_{min} = h_{frac} \cdot h_{max}`` where ``h_{frac}`` is a configurable fraction.

3. **Velocity potential** (`calc_velocity_potential`): The velocity potential ``\chi`` is computed via a 2D Hilbert transform following the GFDL Fortran implementation (`get_velpot.f90`). A Blackman window taper controls the smoothing scale, and the window size is determined by the GCM grid resolution.

4. **Orographic tensor** (`calc_orographic_tensor`): The drag tensor ``\textbf{T} = \nabla\chi (\nabla h)^T`` is computed from finite-difference gradients in spherical coordinates. A ``\cos^2`` polar taper is applied from ``|\mathrm{lat}| = 75°`` to ``90°`` to suppress spurious gradients from lat-lon grid convergence.

5. **Regridding** (`regrid_OGW_info`): The 6 topographic fields ``(h_{max}, h_{min}, t_{11}, t_{12}, t_{21}, t_{22})`` are interpolated from the lat-lon grid onto the CliMA spectral element grid using `SpaceVaryingInput`.

6. **Saving** (`write_computed_drag!`): The ClimaCore field is written to HDF5 with metadata attributes (topography type, smoothing, damping factor, ``h_{elem}``). Pre-computed artifacts for common ``h_{elem}`` values are available via `ClimaArtifacts`.

This separation of preprocessing from runtime avoids expensive tensor calculations during each simulation.

### Cache Initialization

The orographic gravity wave cache is initialized via `orographic_gravity_wave_cache()`, which pre-allocates approximately 30 fields for runtime computation. The topographic information is loaded via `get_topo_info()`, which follows one of three paths based on the `topo_info` configuration:

| Configuration | Description |
|--------------|-------------|
| `Val(:gfdl_restart)` | Load pre-computed orographic data from GFDL restart NetCDF and remap via `regrid_OGW_info` |
| `Val(:raw_topo)` | Load pre-computed HDF5 artifact (local file or ClimaArtifact), or compute tensor on-the-fly for analytical topographies |
| `Val(:linear)` | Use user-provided drag input directly as fields |

For Earth topography with `Val(:raw_topo)`, the runtime code first checks for a local HDF5 file, then falls back to fetching a lazy artifact via `ClimaArtifacts`. For analytical test cases (DCMIP200, Hughes2023, Agnesi, Schar, Cosine2d, Cosine3d), the tensor is computed on-the-fly using ClimaCore horizontal gradient operators.

### Runtime Computation

The orographic gravity wave uses a split compute/apply pattern:

**Compute step** (`orographic_gravity_wave_compute_tendency!`): Runs periodically at the `dt_ogw` interval via a callback (not every timestep). It executes the following pipeline and stores the result in the cache:

1. **Buoyancy frequency**: Compute ``N`` from the temperature profile at cell centers and faces.
2. **PBL detection**: Determine planetary boundary layer height using pressure and temperature criteria via `get_pbl_z!()`.
3. **Base flux**: Calculate the base momentum flux and Froude numbers via `calc_base_flux!()`.
4. **Saturation profile**: Build the vertical saturation flux profile via `calc_saturation_profile!()`.
5. **Propagating forcing**: Compute drag from vertically propagating waves via `calc_propagate_forcing!()`.
6. **Non-propagating forcing**: Compute drag from blocked flow below the reference level via `calc_nonpropagating_forcing!()`.

The computed tendencies are constrained to ``\pm 3 \times 10^{-3}`` m/s² to ensure numerical stability.

**Apply step** (`orographic_gravity_wave_apply_tendency!`): Runs every timestep as part of `remaining_tendency!`. It reads the cached forcing and adds it to the velocity tendency ``Y_t`` as a `Covariant12Vector`.

### Key Source Files

| File | Description |
|------|-------------|
| `src/parameterized_tendencies/gravity_wave_drag/orographic_gravity_wave.jl` | Runtime forcing computation: cache, compute/apply tendency, all physics subroutines |
| `src/parameterized_tendencies/gravity_wave_drag/orographic_gravity_wave_helper.jl` | Offline preprocessing: tensor computation, velocity potential, regridding, HDF5 I/O |
| `src/parameterized_tendencies/gravity_wave_drag/preprocess_topography.jl` | Preprocessing driver utilities: HDF5 writing, NetCDF diagnostics, plotting |
| `src/topography/topography.jl` | Analytical topography functions (DCMIP200, Hughes2023, Agnesi, Schar, Cosine) |
| `src/solver/types.jl` | Type definitions: `OrographicGravityWave`, `FullOrographicGravityWave`, `LinearOrographicGravityWave` |
| `src/callbacks/callbacks.jl` | `ogw_model_callback!` that triggers the compute step |

### Testing and Validation

The orographic gravity wave implementation is validated through:

- **Garner 2005 reproduction**: Unit tests that reproduce figures from the reference paper.
- **3D simulation tests**: Full atmospheric simulations with orographic forcing.
- **Base flux validation**: Comparison of computed fluxes against expected values for known topographies.
- **Preprocessed topography validation**: The script `test/parameterized_tendencies/gravity_wave/orographic_gravity_wave/test_ogw_computed_drag.jl` compares the preprocessed topographic fields (``h_{max}``, ``h_{min}``, ``t_{11}``, ``t_{12}``, ``t_{21}``, ``t_{22}``) against the GFDL restart data. It loads both the CliMA-computed drag artifact and the GFDL reference via `Val(:gfdl_restart)`, then generates side-by-side contour plots for visual comparison.

Test outputs and artifacts have been verified in [PR #4208](https://github.com/CliMA/ClimaAtmos.jl/pull/4208).