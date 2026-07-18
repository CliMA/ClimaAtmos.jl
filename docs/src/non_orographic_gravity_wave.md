# Non-orographic Gravity Wave Parameterization

Gravity waves have a great impact on the atmospheric circulation. They are usually generated from topography or convection, propagate upward and alter temperature and winds in the middle atmosphere, and influence tropospheric circulation through 'downward control' of the residual circulation by stratospheric wave forcing. The horizontal wavelength for gravity waves ranges from tens to thousands of kilometers, and the shorter end of this range up to a few hundred kilometers is unresolved at typical GCM resolution and must be parameterized.

The gravity wave drag on the wind velocities $\overline{\vec{v}}=(u,v)$ are

```math
\frac{\partial \overline{\vec{v}}}{ \partial t} = ... + \underbrace{\left(-\frac{\partial \overline{\vec{v}'w'}}{\partial z}\Big|_{GW}\right)}_{\vec{X}}
```

with $\vec{X} = (X_\lambda, X_\phi)$ representing the sub-grid scale zonal and meridional components of the gravity wave drag and is calculated with the parameterization.

## AD99 Gaussian Source Spectrum

The non-orographic gravity wave drag parameterization follows the spectra methods described in [alexander1999](@cite) (AD99). The following assumptions are made for this parameterization to work:

  - The wave spectrum consists of independent monochromatic waves, and wave-wave interaction is neglected when propagation and instability is computed.
  - The gravity wave propagates vertically and conservatively to the breaking level and deposits all momentum flux into that level, as opposed to the method using saturation profile described in [lindzen1981](@cite).
  - The wave breaking criterion is derived hydrostatically and non-rotating, and AD99 show including these effects in the breaking criterion has small impact. Non-hydrostatic effects are nevertheless retained in the reflection condition (cf. Eq. 6 of AD99).
  - The gravity wave is intermittent and the intermittency is computed as the ratio of the total long-term average source to the integral of the source momentum flux spectrum.

### Spectrum of the momentum flux sources

The source spectrum with respect to phase speed is prescribed in Eq. (17) in [alexander1999](@cite). We adapt the equation so that the spectrum writes as a combination of a wide and a narrow band as

```math
B_0(c) = \frac{F_{S0}(c)}{\rho_0} = sgn(c-u_0) \left( Bm\_w \exp\left[ -\left( \frac{c-c_0}{c_{w\_w}} \right)^2 \ln{2} \right] + Bm\_n \exp\left[ -\left( \frac{c-c_0}{c_{w\_n}} \right)^2 \ln{2} \right] \right)
```

where the subscript $0$ denotes values at the source level. $c_0$ is the phase speed with the maximum flux magnitude $Bm$, and $c_w$ is the half-width at half-maximum of the Gaussian.  $\_w$ and $\_n$ represent the wide and narrow bands of the spectra.

Intuitively, $B_0(c)$ is a recipe for how much momentum the wave field carries, sorted by how fast each wave travels. The two Gaussians say most waves move near a favored speed $c_0$, with waves much faster or slower being rare (a wide hump for the broad spread plus a narrow hump for the extra pile-up at the peak), and $c_w$ measures how quickly that falls off (one step of $c_w$ from the peak halves the flux). The $sgn(c-u_0)$ factor flips the sign so waves moving with the source-level wind push one way and waves moving against it push the other. The launch spectrum is therefore symmetric and carries no net momentum until the atmosphere filters one side out higher up (see below).

Because the background winds and convective sources differ between the tropics and the extra-tropics, the parameterization changes a few of its settings at the edge of the tropical band. The most important is the reference frame in which the phase-speed spectrum is defined: in the extra-tropics, phase speeds are measured relative to the ground, while in the tropics they are measured relative to the zonal wind at the source level. This frame, together with the narrow-band amplitude $Bm\_n$ and the wide-band width $c_w$, changes abruptly right at the band edge. The overall source strength $F_{S0}$, by contrast, is blended smoothly across the transition, so the total wave drag varies continuously with latitude rather than jumping. (With the default parameters $Bm\_n$ and $c_w$ are the same inside and outside the tropics, so in practice only the reference-frame switch and the smooth amplitude blend have any effect.)

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
  - **Breaking:** among the remaining waves, those that become convectively unstable, i.e., $Q(z_n, c) \geq 1$ or encounter a critical level, i.e., where the intrinsic phase speed has changed sign relative to the source $(c - u_0)(c - u(z_n)) \leq 0$, break between levels $z_{n-1}$ and $z_n$, and their momentum flux is deposited entirely into that layer.

The gravity wave drag at half-levels is then

```math
X(z_{n-1/2}) = \frac{\epsilon \rho_0}{\rho(z_{n-1/2})\Delta z} \sum_j (B_0)_j
```

where the sum is over the breaking waves and $\epsilon = F_{S0} / (\rho_0\, n_k \sum |B_0|)$ is the wave intermittency ($n_k$ is the number of horizontal wavenumber bands, one by default). Here $F_{S0}$ is the prescribed time-averaged total momentum flux, specified as a latitude-dependent quantity.

Any momentum flux that propagates to the model top without breaking is re-deposited by averaging it across all levels above the damping level (defined by `damp_pressure`) to ensure momentum conservation.

The drag at full levels is obtained by averaging adjacent half-levels:

```math
X(z_{n}) = \frac{1}{2} \left[ X(z_{n-1/2}) + X(z_{n+1/2}) \right].
```

## Beres (2004) Convective Source Spectrum

When `nogw_beres_source` is enabled, the AD99 Gaussian spectrum continues to act as an always-on background source, and an additional convective source (following [beres2004](@cite)) is launched on top of it in columns where convective heating from the EDMF parameterization exceeds activation thresholds. The two contributions are computed independently and their momentum-flux forcings are summed. The Beres term couples part of the gravity wave source directly to resolved/parameterized convection.

### Convective heating extraction

Convective heating properties are extracted from the EDMF updraft fields at each column. Two heating profiles are computed from the mass-flux divergence of dry static energy anomalies:

The grid-mean DSE-based mass-flux $Q_1$ (Yanai apparent heat source),

```math
Q_1 = -\frac{1}{\rho} \frac{\partial}{\partial z} \left[ \sum_j \rho^j (w^j - \bar{w}) a^j \, (T^j - \bar{T}) \right],
```

where $\rho^j$, $w^j$, $a^j$, and $T^j$ are the density, vertical velocity, area fraction, and temperature of updraft $j$, and $\bar{T}$ is the grid-mean temperature. The in-cloud (per-draft conditional-mean) heating is

```math
Q_{\text{ic}} = \frac{\sum_j \rho^j a^j \, Q_{\text{ic}}^j}{\sum_j \rho^j a^j}, \qquad
Q_{\text{ic}}^j = -\frac{1}{\rho^j} \frac{\partial}{\partial z} \left[ \rho^j (w^j - \bar{w}) (T^j - \bar{T}) \right],
```

which is the same construction without the area-fraction dilution. The distinction matters because Beres' linear theory is forced by the local heating of the convective cell (their squall-line reference value is $Q_0 \approx 0.004\;\mathrm{K\,s^{-1}}$, far above any grid-mean value): the spectrum amplitude is built from $Q_{\text{ic}}$, while envelope detection and activation gating use $Q_1$ (their thresholds are calibrated to grid-mean magnitudes). The reference WACCM/CAM implementation applies the analogous grid-mean → local conversion with a fixed assumed convective fraction (`CF = 20`, i.e. heating concentrated in 5% of the cell). Here, the EDMF area fraction supplies the conversion per column, and the corresponding coverage factor enters the deposition instead (see "Intermittency" below).

The dry static energy $s = c_{p,d} T + gz$ is used because $T^j$ is computed via saturation adjustment and already reflects the warming from condensation along the parcel trajectory. Since $s^j - \bar{s} = c_{p,d}(T^j - \bar{T})$ (the $gz$ terms are identical at a given level and cancel in the anomaly), no explicit $L_v(c-e)$ correction is needed.

**Alternative heating source: canonical latent heating (one-moment microphysics).** The DSE-based $Q_1$ is an apparent heat source: it is the heating convection imparts to the resolved environment, and because it is a transport (mass-flux divergence) term it carries the convective redistribution of dry static energy as well as the latent release. Under one-moment (non-equilibrium) microphysics with prognostic EDMF, the in-cloud heating that sets the source amplitude can instead be drawn from the transport-free latent heating (`nogw_beres_heating_latent`, off by default),

```math
Q_{\text{lat}}^j = \frac{1}{c_p^j} \sum_p L_p \, \mathcal{R}_p^j,
```

the per-draft sum of latent heat $L_p$ times net phase-conversion rate $\mathcal{R}_p^j$ (vapor–liquid, vapor–ice, liquid–ice) with the subdomain moist heat capacity $c_p^j$. The two heatings share the same column-integrated value, but $Q_{\text{lat}}$ is the pure diabatic profile (lower and more vertically compact than the transport-shifted $Q_1$), so selecting it lowers and compacts the launched source. This path is available only under one-moment microphysics, where the per-phase conversion rates are exposed. Equilibrium (zero-moment, saturation-adjustment) microphysics does not expose them, so there $Q_1$ is the only option. In either case the grid-mean $Q_1$, continues to drive envelope detection and triggers the activation of convective gravity-wave.

From the column profiles, we extract:

  - **Heating amplitude** $Q_0$: Beres (2004) assumes a half-sine heating profile of depth $h$ with peak amplitude $Q_0$. Rather than a noisy pointwise maximum, $Q_0$ is recovered from the depth-mean of the in-cloud heating $Q_{\text{ic}}$ over the envelope by inverting the half-sine mean ($\overline{\text{half-sine}} = (2/\pi)\,Q_0$):

```math
Q_0 = \frac{\pi}{2} \, \frac{1}{h} \int_{z_{\text{bot}}}^{z_{\text{top}}} Q_{\text{ic}}(z) \, dz, \qquad (\text{clamped to } Q_0 \geq 0).
```

  - **Heating depth** $h = z_{\text{top}} - z_{\text{bot}}$ is the vertical extent of the convective envelope. It is set by moment-matching a half-sine to the in-cloud heating $Q_{\text{ic}}$ through its first two vertical moments, namely the heating centroid $z_c$ and spread $\sigma$ (both taken over $z \geq z_{\text{bot,floor}}$, i.e., `nogw_beres_z_bot_floor`, to eliminate the boundary-layer / dry-thermal signal). A half-sine of depth $h$ has variance $h^2(\pi^2-8)/(4\pi^2)$, so $h = \sigma / \sqrt{(\pi^2-8)/(4\pi^2)}$, and $z_{\text{bot}}, z_{\text{top}} = z_c \mp h/2$ (with $z_{\text{top}}$ clamped to the domain top and $z_{\text{bot}} \geq 0$). This seats the envelope on the actual heating peak.
  - **Mean wind and stability in heating layer**: mass-weighted averages over the levels within the envelope $[z_{\text{bot}}, z_{\text{top}}]$,

```math
\bar{u}_{\text{heat}} = \frac{\sum_k u_k \rho_k \Delta z_k}{\sum_k \rho_k \Delta z_k}, \quad \bar{v}_{\text{heat}} = \frac{\sum_k v_k \rho_k \Delta z_k}{\sum_k \rho_k \Delta z_k}, \quad \bar{N}_{\text{source}} = \frac{\sum_k N_k \rho_k \Delta z_k}{\sum_k \rho_k \Delta z_k}.
```

Both horizontal wind components are carried so the convective source can force the zonal and meridional winds independently.

**Design choices.**
The envelope detection makes several deliberate simplifications:

  - **Single envelope depth.** Beres (2004)'s half-sine response function is parameterized by one heating depth $h$. The continuous envelope $[z_{\text{bot}}, z_{\text{top}}]$ collapses a possibly multi-peaked or asymmetric heating profile to a single depth. Bottom-heavy and top-heavy regimes are approximated by this single-depth construction.
  - **Altitude floor on $z_{\text{bot}}$.** The floor $z_{\text{bot,floor}}$ prevents the boundary-layer / dry-thermal $Q_1$ signal from anchoring the envelope at the surface. It works together with the activation threshold $h_{\min}$: convection that does not extend at least $h_{\min}$ above the floor is filtered out, so the Beres source acts on deep convection only.
  - **Mass-weighted means.** $\bar{u}_{\text{heat}}$, $\bar{v}_{\text{heat}}$, and $\bar{N}_{\text{source}}$ are weighted by mass ($\rho \Delta z$) over the geometric envelope, not by $Q_1$. This is consistent with the geometric (rather than heating-based) envelope definition. Beres (2004) assumes constant $U$ over the heating depth, so either weighting is a defensible discretization choice. In sheared environments the difference can affect the asymmetry of the source spectrum.

### Activation criteria

The Beres source activates in a column only when both conditions are met:

  - the grid-mean heating amplitude exceeds a threshold, $(\pi/2)\,h^{-1}\!\int_{z_{\text{bot}}}^{z_{\text{top}}} Q_1\,dz > Q_{0,\text{threshold}}$, of order $1\;\text{K/day}$ (`nogw_beres_Q0_threshold`). This gate deliberately uses the grid-mean $Q_1$ (not the in-cloud $Q_0$ that sets the launched amplitude) because its threshold is calibrated to grid-mean magnitudes.
  - $h > h_{\min}$, a heating-depth threshold of order $1\;\text{km}$ (`nogw_beres_h_heat_min`)

This filters out shallow or weak convection. In columns where the criteria are not met, the Beres contribution is zero and only the AD99 background source acts. Where they are met, the Beres flux is added on top of the AD99 background.

**Note:** The Beres physical parameters `nogw_beres_Q0_threshold`, `nogw_beres_h_heat_min`, and the spectrum and steady-component parameters used below are ClimaParams parameters. Their defaults are in the ClimaParams' `[nogw_beres_*]` / `[nogw_*]` entries. Only the toggle switches are in `default_config.yml`.

### Source spectrum $B_0(c)$

The momentum flux spectrum at the source level is computed following the linear analysis of [beres2004](@cite). The convective heating profile is assumed half-sine in the vertical with depth $h$ and peak amplitude $Q_0$ (Beres Eq. 8), Gaussian in the horizontal with half-width $\sigma_x$ (Beres Eq. 7), and white-noise in time (see "Time spectrum" below). The derivation below builds the transient ($\nu > 0$) spectrum, which is always computed. The stationary, mountain-wave-like response from the interaction of $\bar{u}_{\text{heat}}$ with the steady ($\nu = 0$) part of the heating (Beres Eqs. 32–33) is always computed, but its activation is toggled implicitly by the phase-speed grid, as described separately under "Steady component" below. Beres' §5 conclusions emphasize that both the transient and steady components can dominate depending on convective regime.

**Fourier decomposition of the heating (Beres Eqs. 7, 11).** The horizontal heating profile in physical space is $q_x(x) = Q_0 \exp[-(x-x_0)^2/\sigma_x^2]$, a Gaussian bump with peak amplitude $Q_0$ and half-width $\sigma_x$. Its Fourier transform decomposes into horizontal wavenumbers $k$ as $Q_x(k) = Q_0 \, G_k$, where the shape factor $G_k$ depends only on $\sigma_x$:

```math
G_k = \frac{\sigma_x}{\sqrt{2}} \exp\!\left(-\frac{k^2 \sigma_x^2}{4}\right).
```

The default $\sigma_x$ (≈ 4 km, ClimaParams parameter `nogw_beres_sigma_x`) lies between Beres' two test values (2.5 km for narrow squall-line cells, 18 km for broader heating). The spectrum's east–west asymmetry under shear is sensitive to this choice (cf. Beres Fig. 1a vs. 1b).

**Vertical wavenumber (Beres Eq. 18).** For ground-relative frequency $\nu$ and intrinsic frequency $\hat{\nu} = \nu - k \bar{u}_{\text{heat}}$,

```math
m^2 = k^2 \left( \frac{N^2}{\hat{\nu}^2} - 1 \right).
```

Vertically propagating waves require $|\hat{\nu}| < N$. Outside this range the response vanishes.

**Response function.** The atmospheric response to the half-sine vertical heating profile is

```math
R(k, \nu) = \frac{\pi m h \, \text{sinc}(mh - \pi)}{(mh + \pi)(N^2 - \hat{\nu}^2)},
```

with $\text{sinc}(x) = \sin(x)/x$. This is the response factor extracted from $|B_{k\nu}|^2$ in Beres Eq. (23) after substituting the dispersion relation. The resonance at $mh = \pi$ corresponds to a vertical wavelength of $2h$, exactly matching the half-sine projection.

**Time spectrum.** Beres' analysis carries an additional factor $|Q_t(\nu)|^2$, the squared temporal Fourier transform of the heating's time dependence. The present implementation sets $|Q_t(\nu)|^2 = 1$ (white noise), absorbing the overall normalization into the scale factor $\alpha$ below. Beres recommends (their §5) a red-noise time spectrum, which their Fig. 13 shows reproduces CRM-derived spectra noticeably better than white. Adding a frequency-dependent $|Q_t(\nu)|^2$ is a future improvement.

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

  - $\nu_{\min} > 0$, both for numerical reasons (the integrand contains $\nu/c^2$ and $1/|\hat{\nu}|$) and to keep the steady ($\nu = 0$) part out of the transient integral (it is handled separately, see "Steady component" below).
  - $\nu_{\max}$ small enough that propagating waves exist over the resolved $c$ range. The range $|\hat{\nu}| \geq N$ contributes zero by construction.

The factor $\alpha$ (`nogw_beres_scale_factor`) bundles the dimensional prefactor $\rho_0/(L\tau)$ from Beres Eq. (30), the white-noise $|Q_t|^2$ normalization, and any empirical tuning multiplier.

**Sign convention.** The code applies $\text{sgn}(\hat{c})$ outside the integral, whereas Beres Eq. (30) places $\text{sgn}(\hat{\nu})$ inside the integrand. Since $k = \nu/c$ with $\nu > 0$, the two are related by $\text{sgn}(\hat{\nu}) = \text{sgn}(c) \cdot \text{sgn}(\hat{c})$, so the forms differ by a factor of $\text{sgn}(c)$ for negative phase speeds. The $\text{sgn}(\hat{c})$ convention is correct for the AD99 propagator, which interprets $B_0(c) < 0$ as westward momentum flux for westward-propagating waves ($c < 0$).

### Heating depth averaging

Optionally, the spectrum may be computed and averaged at $n_{h,\text{avg}}$ values of $h$ uniformly spaced in $[h - \Delta h, h + \Delta h]$, with $\Delta h = \Delta h_{\text{frac}} \cdot h$. This averaging smooths the resonance at $mh = \pi$ (vertical wavelength $\lambda_z = 2h$) in the response function $R$, which is sharp for any single $h$ but unphysical, as real columns contain cells of varying depth. [beres2004](@cite) introduced this $h$-averaging for the steady (stationary) component, where the factor $\beta$ of their §2.b, Fig. 4, averaged over $n_{h,\text{avg}} = 20$ depths with $\Delta h_{\text{frac}} \approx 0.17$. Here, this averaging could instead also be applied to the transient (nonstationary) spectrum described above, whose response function $R$ carries a resonance of the same origin (the half-sine vertical projection), so the same averaging is warranted. Note that `n_h_avg = 1`, and heating depth averaging is off by default.

### Steady ($\nu = 0$) component

The transient spectrum above starts the frequency integral at $\nu_{\min} > 0$ and therefore omits the steady ($\nu = 0$) part of the heating. This steady part is always added as a single ground-stationary wave: the time-mean convective heating acts like a fixed obstacle in the mean wind $\bar{u}_{\text{heat}}$ and radiates one mountain-wave-like mode with vertical wavenumber $m_0 = N / |\bar{u}_{\text{heat}}|$ (Beres Eq. 31, the resulting steady flux is Beres Eqs. 32–34), signed to oppose $\bar{u}_{\text{heat}}$. It is deposited entirely into the $c \approx 0$ phase-speed bin, which the transient spectrum leaves empty, so there is no double-counting. The steady source has no config switch. It is controlled implicitly by the phase-speed grid. It deposits only when an exact $c = 0$ bin exists, i.e. when `cmax`/`dc` is an integer ($c[n] = (n-1)\,dc - cmax$). On a grid without one, it silently no-ops (a warning is emitted at construction). The default grid (`nogw_cmax` $= 100$, `nogw_dc` $= 0.8 \Rightarrow 125$) has a $c = 0$ bin, so the steady source is on by default. Its amplitude carries the same half-sine response factor and the same scale factor $\alpha$ as the transient spectrum, so the two are not independently tunable. Its only additional inputs are a zero-frequency temporal weight $|Q_t(0)|^2$ set by a DC-weight knob (`nogw_beres_steady_dc_frac`) and a largest convective-system scale (`nogw_beres_L_system`) that fixes the horizontal projection.

### Propagation and breaking

The Beres source spectrum $B_0(c)$ is propagated upward with the same reflection criterion, instability/critical-level breaking condition, momentum deposition, and sponge-layer redistribution as the AD99 method described above. Two differences from the AD99 background path are worth noting:

  - **Launch level.** $B_0(c)$ is the far-field flux radiated above the heating, so each column launches its Beres waves from the top of its convective envelope $z_{\text{top}}$, rather than from the fixed AD99 source level (`nogw_source_pressure`).
  - **Intermittency.** The AD99 forcing has an intermittency factor $\epsilon = F_{S0} / (\rho_0\, n_k \sum |B_0|)$ ($n_k$ horizontal wavenumber bands, one by default). The Beres $B_0(c)$ is already in physical momentum-flux units for the local (in-cloud) heating amplitude (set by $Q_0$, $\sigma_x$, and the scale factor $\alpha$), so it is not rescaled by the AD99 $\epsilon$. Instead, the deposited flux is diluted by the convective coverage $\bar{a}$, the mass-weighted envelope mean of the EDMF updraft area fraction: only the fraction $\bar{a}$ of the grid cell radiates, so the grid-mean flux is $\bar{a}$ times the local flux ($\propto \bar{a}\,Q_0^2$, linear in coverage, quadratic in local amplitude). This is the exact Beres analog of AD99's intermittency: $\bar{a}$ takes the place of the $F_{S0}/\sum|B_0|$ factor (the same $1/(\rho_0\, n_k)$ normalization is shared), diagnosed per column from EDMF rather than tuned. Breaking levels are still computed at the local amplitude ($B_0$ itself is not rescaled). Only the deposition is diluted. The flux is then distributed across the $n_k$ horizontal wavenumber bands (one by default), exactly as for the AD99 background.

The Beres forcing computed this way is added to the AD99 background forcing in each column. The two source spectra are propagated and accumulated independently.

## Implementation Summary

The parameterization runs on a callback timer (`dt_nogw`) and applies the accumulated forcing every integrator step.

```
Every dt_nogw seconds:
  nogw_model_callback!
    └─ non_orographic_gravity_wave_compute_tendency!
        ├─ 1. Compute buoyancy frequency N(z) and identify AD99 source/damp levels
        ├─ 2. [Beres only] Extract convective heating from EDMF updrafts
        │       ├─ Compute Q₁ (grid-mean) and Q_ic (in-cloud) mass-flux
        │       │     divergences of DSE anomalies
        │       ├─ Moment-match a half-sine to in-cloud Q_ic: envelope centroid
        │       │     z_c and spread σ (over z ≥ z_bot_floor) give
        │       │     h = σ/√((π²−8)/4π²), z_bot/z_top = z_c ∓ h/2
        │       ├─ Heating-layer props: Q₀ = (π/2)·mean(Q_ic), coverage ā,
        │       │     u_heat, v_heat, N_source
        │       ├─ Activate if grid-mean amplitude > threshold AND h > h_min
        │       └─ Set per-column Beres launch level = z_top
        └─ 3. For each horizontal wavenumber k:
                ├─ AD99 background (always on): Gaussian B₀(c)
                │     → propagate¹ from AD99 source level → ×intermittency ε
                │     → accumulate into uforcing, vforcing
                └─ [Beres configured]: convective B₀(c) from Q₀, h, N
                      (zero unless column is active)
                      → propagate¹ from z_top → ×coverage ā at deposition
                      → ADD into uforcing, vforcing
           ¹"propagate" = the shared upward-marching kernel (reflection → instability /
            critical-level breaking → model-top + sponge redistribution); identical
            for both sources.

Every dt (integrator step):
  non_orographic_gravity_wave_apply_tendency!
    └─ Clamp forcing, zero NaN/Inf, apply to wind tendencies
```
