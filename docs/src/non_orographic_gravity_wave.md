# Non-orographic Gravity Wave Parameterization

Gravity waves have a great impact on the atmospheric circulation. They are usually generated from topography or convection, propagate upward and alter temperature and winds in the middle atmosphere, and influence tropospheric circulation through 'downward control' of the residual circulation by stratospheric wave forcing. The horizontal wavelength for gravity waves ranges from tens to thousands of kilometers, and the shorter end of this range up to a few hundred kilometers is unresolved at typical GCM resolution and must be parameterized.

The gravity wave drag on the wind velocities ($\overline{\vec{v}}=(u,v)$) are
```math
\frac{\partial \overline{\vec{v}}}{ \partial t} = ... + \underbrace{\left(-\frac{\partial \overline{\vec{v}'w'}}{\partial z}\Big|_{GW}\right)}_{\vec{X}}
```
with $\vec{X} = (X_\lambda, X_\phi)$ representing the sub-grid scale zonal and meridional components of the gravity wave drag and is calculated with the parameterization.

## AD99 Gaussian Source Spectrum
The non-orographic gravity wave drag parameterization follows the spectra methods described in [alexander1999](@cite). The following assumptions are made for this parameterization to work:
* The wave spectrum consists of independent monochromatic waves, and wave-wave interaction is neglected when propagation and instability is computed.
* The gravity wave propagates vertically and conservatively to the breaking level and deposits all momentum flux into that level, as opposed to the method using saturation profile described in [lindzen1981](@cite).
* The wave breaking criterion is derived hydrostatically and non-rotating, and AD99 show including these effects in the breaking criterion has small impact; non-hydrostatic effects are nevertheless retained in the reflection condition (cf. Eq. 6 of AD99).
* The gravity wave is intermittent and the intermittency is computed as the ratio of the total long-term average source to the integral of the source momentum flux spectrum.

### Spectrum of the momentum flux sources
The source spectrum with respect to phase speed is prescribed in Eq. (17) in [alexander1999](@cite). We adapt the equation so that the spectrum writes as a combination of a wide and a narrow band as
```math
B_0(c) = \frac{F_{S0}(c)}{\rho_0} = sgn(c-u_0) \left( Bm\_w \exp\left[ -\left( \frac{c-c_0}{c_{w\_w}} \right)^2 \ln{2} \right] + Bm\_n \exp\left[ -\left( \frac{c-c_0}{c_{w\_n}} \right) \ln{2} \right] \right)
```
where the subscript $0$ denotes values at the source level. $c_0$ is the phase speed with the maximum flux magnitude $Bm$. $c_w$ is the half-width at half-maximum of the Gaussian.  $\_w$ and $\_n$ represent the wide and narrow bands of the spectra.

The reference frame for the spectrum is latitude-dependent: it is ground-relative in the extra-tropics and relative to the source-level zonal wind in the tropics, with smoothing applied at the transition.

### Upward propagation and wave breaking
Waves that are reflected will be removed from the spectrum. A wave that breaks at any level above the source will deposit all its momentum flux into that level and be removed from the spectrum.

The reflection frequency is defined as
```math
\omega_r(z) = \left(\frac{N(z)^2 k^2}{k^2+\alpha^2}\right)^{1/2}
```
where $N(z)$ is the buoyancy frequency, $k$ is the horizontal wavenumber that corresponds to a default wavelength of 300 km, and $\alpha = 1/(2H)$ where $H$ is the scale height. $\omega_r(z)$ is used to determine, for each monochromatic wave in the spectrum, whether the wave will be reflected at height $z$.

The instability condition is defined as
```math
Q(z,c) = \frac{\rho_0}{\rho(z)} \frac{2N(z)B_0(c)}{k[c-u(z)]^3}
```
$Q(z,c)$ is used to determine whether the monochromatic wave of phase speed $c$ gets unstable at height $z$.

At the source level:
  - if $|\omega|=k|c-u_0| \geq \omega_r$, this wave would have undergone internal reflection somewhere below and is removed from the spectrum;
  - if $Q(z_0, c) \geq 1$, it is also removed because it is not stable at the source level.

At each level above the source $(z_n > z_0)$, waves are tested in order:
  - **Reflection:** waves with $|\omega(z_n)| = k|c - u(z_n)| \geq \omega_r(z_n)$ are removed from the spectrum.
  - **Breaking:** among the remaining waves, those with $Q(z_n, c) \geq 1$ break between levels $z_{n-1}$ and $z_n$, and their momentum flux is deposited entirely into that layer.

The gravity wave drag at half-levels is then
```math
X(z_{n-1/2}) = \frac{\epsilon \rho_0}{\rho(z_{n-1/2})\Delta z} \sum_j (B_0)_j
```
where the sum is over the breaking waves and $\epsilon = F_{S0} / (\rho_0 \sum |B_0|)$ is the wave intermittency. Here $F_{S0}$ is the prescribed time-averaged total momentum flux, specified as a latitude-dependent quantity.

Any momentum flux that propagates to the model top without breaking is re-deposited by averaging it across all levels above the damping level (defined by `damp_pressure`) to ensure momentum conservation.

The drag at full levels is obtained by averaging adjacent half-levels:
```math
X(z_{n}) = \frac{1}{2} \left[ X(z_{n-1/2}) + X(z_{n+1/2}) \right].
```

## Beres (2004) Convective Source Spectrum

When `nogw_beres_source` is enabled, the AD99 Gaussian source spectrum is replaced in columns where convective heating from the EDMF parameterization exceeds activation thresholds. This couples the gravity wave source directly to resolved/parameterized convection, following [beres2004](@cite).

### Convective heating extraction

Convective heating properties are extracted from the EDMF updraft fields at each column. The DSE-based mass-flux $Q_1$ (Yanai apparent heat source) is computed from the mass-flux divergence of dry static energy anomalies:
```math
Q_1 = -\frac{1}{\rho} \frac{\partial}{\partial z} \left[ \sum_j \rho^j (w^j - \bar{w}) a^j \, (T^j - \bar{T}) \right]
```
where $\rho^j$, $w^j$, $a^j$, and $T^j$ are the density, vertical velocity, area fraction, and temperature of updraft $j$, and $\bar{T}$ is the grid-mean temperature.

The dry static energy $s = c_p T + gz$ is used because $T^j$ is computed via saturation adjustment and already reflects the warming from condensation along the parcel trajectory. Since $s^j - \bar{s} = c_p(T^j - \bar{T})$ (the $gz$ terms are identical at a given level and cancel in the anomaly), no explicit $L_v(c-e)$ correction is needed.

From the column profile of $Q_1$, we extract:

- **Peak heating rate**: $Q_0 = \max_z |Q_1(z)|$ — the maximum absolute convective heating rate in the column.
- **Heating depth**: $h = z_{\text{top}} - z_{\text{bot}}$ — the vertical extent of the convective envelope, determined from updraft structure: $z_{\text{top}}$ is the highest level where the total updraft area fraction exceeds $10^{-3}$, $z_{\text{peak}}$ is the height of maximum updraft velocity, and $z_{\text{bot}} = \max(2 z_{\text{peak}} - z_{\text{top}},\; 3000\;\text{m})$.
- **Mean wind in heating layer**: mass-weighted averages over levels within the envelope $[z_{\text{bot}}, z_{\text{top}}]$,
```math
\bar{u}_{\text{heat}} = \frac{\sum_k u_k \rho_k \Delta z_k}{\sum_k \rho_k \Delta z_k}, \quad \bar{N}_{\text{source}} = \frac{\sum_k N_k \rho_k \Delta z_k}{\sum_k \rho_k \Delta z_k}.
```

**Design choices.**
The envelope detection makes several deliberate simplifications:
- **Symmetric envelope.** $z_{\text{bot}} = 2 z_{\text{peak}} - z_{\text{top}}$ mirrors the upper half of the plume around the velocity peak to produce a single envelope depth $h$ for the half-sine response function in Beres (2004). Asymmetric profiles (e.g., bottom-heavy congestus or top-heavy stratiform regimes) are approximated by this symmetric construction.
- **Velocity peak as envelope center.** $z_{\text{peak}}$ is the height of maximum updraft velocity, not the height of maximum $|Q_1|$. The velocity field is smoother than the $Q_1$ divergence field, making it a more robust proxy. It tends to sit slightly above the true heating peak because buoyancy integrates heating upward.
- **Shallow convection.** Columns with $z_{\text{top}} < 3000$ m produce $z_{\text{bot}} \geq z_{\text{top}}$ and thus $h \leq 0$, which is filtered out by the activation criterion $h > h_{\min}$ (see below). This is by design: the Beres spectrum is intended for deep convection only. (As the choice of $z_{\text{bot}}$ affects the activation threshold, it should be kept consistent with $h_{\min}$.)
- **Mass-weighted means.** $\bar{u}_{\text{heat}}$ and $\bar{N}_{\text{source}}$ are weighted by mass ($\rho \Delta z$) over the geometric envelope, not by $|Q_1|$. This is consistent with the geometric (rather than heating-based) envelope definition. Beres (2004) assumes constant $U$ over the heating depth, so either weighting is a defensible discretization choice; in sheared environments the difference can affect the asymmetry of the source spectrum.

### Activation criteria

The Beres source activates in a column only when both conditions are met:
- $Q_0 > Q_{0,\text{threshold}}$ (default $1.157 \times 10^{-4}$ K/s, approximately 10 K/day)
- $h > h_{\min}$ (default 3000 m)

This filters out shallow or weak convection. In columns where Beres is inactive, the AD99 Gaussian source is used instead.

### Source spectrum $B_0(c)$

The momentum flux spectrum at the source level is computed following the linear analysis of [beres2004](@cite). The convective heating profile is assumed half-sine in the vertical with depth $h$ and peak amplitude $Q_0$ (Beres Eq. 8), Gaussian in the horizontal with half-width $\sigma_x$ (Beres Eq. 7), and white-noise in time (see "Time spectrum" below). Only the nonstationary ($\nu > 0$) component is retained; the stationary, mountain-wave-like response from the interaction of $\bar{u}_{\text{heat}}$ with the steady part of the heating (Beres Eqs. 32–33) is **not** included in the present implementation. This is a deliberate simplification — Beres' §5 conclusions emphasize that both components can dominate depending on convective regime, so this omission is a known limitation of the current scheme.

**Fourier decomposition of the heating (Beres Eqs. 7, 11).** The horizontal heating profile in physical space is $q_x(x) = Q_0 \exp[-(x-x_0)^2/\sigma_x^2]$, a Gaussian bump with peak amplitude $Q_0$ and half-width $\sigma_x$. Its Fourier transform decomposes into horizontal wavenumbers $k$ as $Q_x(k) = Q_0 \, G_k$, where the shape factor $G_k$ depends only on $\sigma_x$:
```math
G_k = \frac{\sigma_x}{\sqrt{2}} \exp\!\left(-\frac{k^2 \sigma_x^2}{4}\right).
```
The default $\sigma_x = 4000$ m sits between Beres' two test values (2.5 km for narrow squall-line cells, 18 km for broader heating); the spectrum's east–west asymmetry under shear is sensitive to this choice (compare Beres Fig. 1a vs. 1b).

**Vertical wavenumber (Beres Eq. 18).** For ground-relative frequency $\nu$ and intrinsic frequency $\hat{\nu} = \nu - k \bar{u}_{\text{heat}}$,
```math
m^2 = k^2 \left( \frac{N^2}{\hat{\nu}^2} - 1 \right).
```
Vertically propagating waves require $|\hat{\nu}| < N$; outside this range the response vanishes.

**Response function.** The atmospheric response to the half-sine vertical heating profile is
```math
R(k, \nu) = \frac{\pi m h \, \text{sinc}(mh - \pi)}{(mh + \pi)(N^2 - \hat{\nu}^2)},
```
with $\text{sinc}(x) = \sin(x)/x$. This is the response factor extracted from $|B_{k\nu}|^2$ in Beres Eq. (23) after substituting the dispersion relation; the resonance at $mh = \pi$ corresponds to a vertical wavelength of $2h$, exactly matching the half-sine projection.

**Time spectrum.** Beres' analysis carries an additional factor $|Q_t(\nu)|^2$ — the squared temporal Fourier transform of the heating's time dependence. The present implementation sets $|Q_t(\nu)|^2 = 1$ (white noise), absorbing the overall normalization into the scale factor $\alpha$ below. Beres recommends (their §5) a red-noise time spectrum, which their Fig. 13 shows reproduces CRM-derived spectra noticeably better than white. Adding a frequency-dependent $|Q_t(\nu)|^2$ is a future improvement.

**Spectral flux density (Beres Eq. 30).** Combining the above, the momentum flux density in $(k, \nu)$ space is
```math
F(k, \nu) = \frac{1}{\sqrt{2\pi}} \, \frac{\sqrt{N^2 - \hat{\nu}^2}}{|\hat{\nu}|} \, Q_0^2 \, G_k^2 \, R^2(k, \nu) \qquad \text{for } 0 < |\hat{\nu}| < N,
```
and zero otherwise. Beres' prefactor of $\rho_0 / (L\tau)$, where $L$ and $\tau$ are reference horizontal and temporal scales, is absorbed into $\alpha$ below.

**Phase speed transformation.** The source spectrum $B_0(c)$ is obtained by changing variables from $(k, \nu)$ to $(c, \nu)$ at fixed $\nu$ via $k = \nu/c$, with Jacobian $|dk/dc| = \nu/c^2$, and integrating over frequency:
```math
B_0(c) = \text{sgn}(\hat{c}) \cdot \alpha \int_{\nu_{\min}}^{\nu_{\max}} F\!\left(k = \frac{\nu}{c},\; \nu\right) \frac{\nu}{c^2} \, d\nu,
```
where $\hat{c} = c - \bar{u}_{\text{heat}}$ is the intrinsic phase speed. The integration uses Boole's rule quadrature. The integration limits $\nu_{\min}$ and $\nu_{\max}$ should satisfy:

- $\nu_{\min} > 0$, both for numerical reasons (the integrand contains $\nu/c^2$ and $1/|\hat{\nu}|$) and to exclude the steady component noted above.
- $\nu_{\max}$ small enough that propagating waves exist over the resolved $c$ range; $|\hat{\nu}| \geq N$ contributes zero by construction.

The factor $\alpha$ (`beres_scale_factor`) bundles the dimensional prefactor $\rho_0/(L\tau)$ from Beres Eq. (30), the white-noise $|Q_t|^2$ normalization, and any empirical tuning multiplier.

**Sign convention.** The code applies $\text{sgn}(\hat{c})$ outside the integral, whereas Beres Eq. (30) places $\text{sgn}(\hat{\nu})$ inside the integrand. Since $k = \nu/c$ with $\nu > 0$, the two are related by $\text{sgn}(\hat{\nu}) = \text{sgn}(c) \cdot \text{sgn}(\hat{c})$, so the forms differ by a factor of $\text{sgn}(c)$ for negative phase speeds. The $\text{sgn}(\hat{c})$ convention is correct for the AD99 propagator, which interprets $B_0(c) < 0$ as westward momentum flux for westward-propagating waves ($c < 0$).

### Heating depth averaging

Optionally, the spectrum may be computed and averaged at $n_{h,\text{avg}}$ values of $h$ uniformly spaced in $[h - \Delta h, h + \Delta h]$, with $\Delta h = \Delta h_{\text{frac}} \cdot h$. This averaging smooths the resonance at $mh = \pi$ (vertical wavelength $\lambda_z = 2h$) in the response function $R$, which is sharp for any single $h$ but unphysical, as real columns contain cells of varying depth. [beres2004](@cite) applies the same regularization to the steady-component factor $\beta$ in §2.b, Fig. 4 (using $n_{h,\text{avg}} = 20$, $\Delta h_{\text{frac}} \approx 0.17$); the resonance has the same origin in the nonstationary spectrum described above, and so a similar averaging may be applied. Note that `n_h_avg = 1`, and heating depth averaging is off by default.

### Propagation and breaking

Once the Beres source spectrum $B_0(c)$ replaces the AD99 Gaussian, the upward propagation, wave breaking, and momentum deposition logic is **identical** to the AD99 method described above. The same reflection criterion, instability condition, intermittency factor, and sponge layer redistribution apply.


## Implementation Summary

The parameterization runs on a callback timer (`dt_nogw`) and applies the accumulated forcing every integrator step.

```
Every dt_nogw seconds:
  nogw_model_callback!
    └─ non_orographic_gravity_wave_compute_tendency!
        ├─ 1. Compute buoyancy frequency N(z) and identify source/damp levels
        ├─ 2. [Beres only] Extract convective heating from EDMF updrafts
        │       ├─ Compute Q₁ (mass-flux divergence of DSE anomalies)
        │       ├─ Detect convective envelope (z_bot, z_top, h)
        │       ├─ Extract heating-layer properties (Q₀, u_heat, N_source)
        │       └─ Activate if Q₀ > threshold AND h > h_min
        └─ 3. For each horizontal wavenumber k:
                ├─ AD99: Gaussian source spectrum B₀(c)
                ├─ [Beres active]: Beres source spectrum B₀(c) from Q₀, h, N
                ├─ Propagate upward: reflection / breaking / critical level
                ├─ Apply intermittency factor
                └─ Accumulate momentum flux deposit into uforcing, vforcing

Every dt (integrator step):
  non_orographic_gravity_wave_apply_tendency!
    └─ Clamp forcing, zero NaN/Inf, apply to wind tendencies
```