# Non-orographic Gravity Wave Parameterization

Gravity waves have a great impact on the atmospheric circulation. They are usually generated from topography or convection, propagate upward and alter temperature and winds in the middle atmosphere, and influence tropospheric circulation through downward control. The horizontal wavelength for gravity waves ranges from several kilometers to hundreds of kilometers, which is smaller than typical GCM resolution and needs to be parameterized.

The gravity wave drag on the wind velocities (``\overline{\vec{v}}=(u,v)``) are
```math
\frac{\partial \overline{\vec{v}}}{ \partial t} = ... - \underbrace{\frac{\partial \overline{\vec{v}'w'}}{\partial z}\Big|_{GW} }_{\vec{X}}
```
with $\vec{X} = (X_\lambda, X_\phi)$ representing the sub-grid scale zonal and meridional components of the gravity wave drag and is calculated with the parameterization.

## AD99 Gaussian Source Spectrum
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

## Beres (2004) Convective Source Spectrum

When `nogw_beres_source` is enabled, the AD99 Gaussian source spectrum is replaced in columns where convective heating from the EDMF parameterization exceeds activation thresholds. This couples the gravity wave source directly to resolved/parameterized convection, following [beres2004](@cite).

### Convective heating extraction

Convective heating properties are extracted from the EDMF updraft fields at each column. The convective heating rate ``Q_1`` is computed from the mass-flux divergence (Yanai ``Q_1``):
```math
Q_1 = -\frac{1}{\rho c_p} \frac{\partial}{\partial z} \left[ \sum_j \rho^j (w^j - \bar{w}) a^j (\text{mse}^j + K^j - h_{\text{tot}}) \right]
```
where ``\rho^j``, ``w^j``, ``a^j``, ``\text{mse}^j``, and ``K^j`` are the density, vertical velocity, area fraction, moist static energy, and kinetic energy of updraft ``j``, and ``h_{\text{tot}}`` is the grid-mean total enthalpy.

From the column profile of ``Q_1``, we extract:

- **Peak heating rate**: ``Q_0 = \max_z |Q_1(z)|`` — the maximum absolute convective heating rate in the column.
- **Heating depth**: ``h = z_{\text{top}} - z_{\text{bot}}`` — the vertical extent of the heating layer (levels where ``|Q_1| > 0``), clamped to a minimum of 1000 m.
- **Mean wind in heating layer**: density-weighted averages over active levels,
```math
\bar{u}_{\text{heat}} = \frac{\sum_k u_k |Q_1(z_k)| \rho_k}{\sum_k |Q_1(z_k)| \rho_k}, \quad \bar{N}_{\text{source}} = \frac{\sum_k N_k |Q_1(z_k)| \rho_k}{\sum_k |Q_1(z_k)| \rho_k}.
```

### Activation criteria

The Beres source activates in a column only when both conditions are met:
- ``Q_0 > Q_{0,\text{threshold}}`` (default ``1.157 \times 10^{-4}`` K/s, approximately 10 K/day)
- ``h > h_{\min}`` (default 3000 m)

This filters out shallow or weak convection that would produce unreliable wave spectra. In columns where Beres is inactive, the AD99 Gaussian source is used instead.

### Source spectrum ``B(c)``

The momentum flux spectrum at the source level is computed following Eqs. (23), (29)-(30) of [beres2004](@cite). The convective heating profile is assumed sinusoidal with depth ``h`` and amplitude ``Q_0``.

**Fourier decomposition of the heating:** For horizontal wavenumber ``k``, the spectral power of the heating is
```math
G_k^2 = \frac{Q_0^2 \sigma_x^2}{2} \exp\left( -\frac{k^2 \sigma_x^2}{2} \right)
```
where ``\sigma_x`` is the convective cell half-width (default 4000 m).

**Vertical wavenumber:** For intrinsic frequency ``\hat{\nu} = \nu - k \bar{u}_{\text{heat}}``,
```math
m^2 = k^2 \left( \frac{N^2}{\hat{\nu}^2} - 1 \right).
```
Waves exist only when ``m^2 > 0`` and ``|\hat{\nu}| < N``.

**Response function:** The atmospheric response to the sinusoidal heating is
```math
R = \frac{\pi m h \, \text{sinc}(mh - \pi)}{(mh + \pi)(N^2 - \hat{\nu}^2)}.
```

**Spectral flux density:** The momentum flux density in ``(k, \nu)`` space is
```math
F(k, \nu) = \frac{1}{\sqrt{2\pi}} \frac{\sqrt{N^2 - \hat{\nu}^2}}{|\hat{\nu}|} \, G_k^2 \, R^2.
```

**Phase speed spectrum:** The source spectrum ``B_0(c)`` is obtained by integrating over frequency ``\nu \in [\nu_{\min}, \nu_{\max}]`` with Boole's rule quadrature and applying the Jacobian ``\nu / c^2`` to transform from ``(k, \nu)`` to ``(c, \nu)`` space:
```math
B_0(c) = \text{sgn}(\hat{c}) \cdot \alpha \int_{\nu_{\min}}^{\nu_{\max}} F(k, \nu) \frac{\nu}{c^2} \, d\nu
```
where ``\hat{c} = c - \bar{u}_{\text{heat}}`` and ``\alpha`` is the `beres_scale_factor`.

### Heating depth averaging

Optionally, the spectrum can be averaged over ``n_{h,\text{avg}}`` values of ``h`` in the range ``[h - \Delta h, h + \Delta h]`` where ``\Delta h = \Delta h_{\text{frac}} \cdot h``. This smooths the resonance peaks in the response function that arise from the sinusoidal heating assumption, following the recommendation in Section 2, Figure 4 of [beres2004](@cite).

### Propagation and breaking

Once the Beres source spectrum ``B_0(c)`` replaces the AD99 Gaussian, the upward propagation, wave breaking, and momentum deposition logic is **identical** to the AD99 method described above. The same reflection criterion, instability condition, intermittency factor, and sponge layer redistribution apply.


## Implementation Details

### Runtime Pipeline

```
Every dt_nogw (callback: nogw_model_callback!):
  compute_beres_convective_heating!()  [if Beres enabled]
    ├─ EDMF mass-flux divergence → Q_conv column profile
    ├─ column_reduce! → Q0 (max-abs), h_heat, u_heat, N_source
    └─ activation flag: Q0 > threshold AND h > h_min
  non_orographic_gravity_wave_compute_tendency!()
    ├─ buoyancy frequency N(z) at all levels
    ├─ source level / damp level identification
    ├─ wave_source() → B_0(c) [AD99 Gaussian or Beres dispatch]
    ├─ upward propagation: reflection + breaking at each level
    └─ momentum deposition + sponge layer redistribution

Every dt (integrator):
  non_orographic_gravity_wave_apply_tendency!()
    └─ Y_t.c.uₕ += Covariant12Vector(uforcing, vforcing)
```

### Configuration

| Config key | Default | Description |
|---|---|---|
| `non_orographic_gravity_wave` | `false` | Enable NOGW parameterization |
| `nogw_beres_source` | `false` | Enable Beres convective source (requires EDMF turbconv) |
| `dt_nogw` | `1800secs` | Callback interval for NOGW computation |
| `beres_Q0_threshold` | `1.0e-5` | Min heating rate to activate (K/s, ~1 K/day) |
| `beres_scale_factor` | `2.0e-6` | Amplitude scaling for Beres momentum flux |
| `beres_sigma_x` | `4000.0` | Convective cell half-width (m) |
| `beres_nu_min` | `8.727e-4` | Min angular frequency for integration (rad/s) |
| `beres_nu_max` | `1.047e-2` | Max angular frequency for integration (rad/s) |
| `beres_n_nu` | `9` | Quadrature points for frequency integration (must be 4k+1) |
| `beres_h_heat_min` | `1000.0` | Min heating depth to activate (m) |

### Diagnostics

| Short name | Description |
|---|---|
| `utendnogw` | Eastward acceleration due to non-orographic GW drag (m/s²) |
| `vtendnogw` | Northward acceleration due to non-orographic GW drag (m/s²) |
| `nogw_Q0` | Peak convective heating rate (K/s, Beres-only diagnostic) |
| `nogw_h_heat` | Convective heating depth (m, Beres-only diagnostic) |

### Key Source Files

| File | Description |
|---|---|
| `src/parameterized_tendencies/gravity_wave_drag/non_orographic_gravity_wave.jl` | All NOGW runtime: cache, Beres heating extraction, compute/apply tendency, both AD99 and Beres `wave_source` dispatches |
| `src/solver/types.jl` | `NonOrographicGravityWave` and `BeresSourceParams` structs |
| `src/solver/model_getters.jl` | Config parsing and Beres parameter construction |
| `src/diagnostics/gravitywave_diagnostics.jl` | NOGW diagnostic variable definitions |
| `src/callbacks/callbacks.jl` | `nogw_model_callback!` that triggers the compute step |
