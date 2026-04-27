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

Convective heating properties are extracted from the EDMF updraft fields at each column. The convective heating rate $Q_1$ is computed from the mass-flux divergence (Yanai $Q_1$):
```math
Q_1 = -\frac{1}{\rho c_p} \frac{\partial}{\partial z} \left[ \sum_j \rho^j (w^j - \bar{w}) a^j (\text{mse}^j + K^j - h_{\text{tot}}) \right]
```
where $\rho^j$, $w^j$, $a^j$, $\text{mse}^j$, and $K^j$ are the density, vertical velocity, area fraction, moist static energy, and kinetic energy of updraft $j$, and $h_{\text{tot}}$ is the grid-mean total enthalpy.

From the column profile of $Q_1$, we extract:

- **Peak heating rate**: $Q_0 = \max_z |Q_1(z)|$ — the maximum absolute convective heating rate in the column.
- **Heating depth**: $h = z_{\text{top}} - z_{\text{bot}}$ — the vertical extent of the heating layer (levels where $|Q_1| > 0$), clamped to a minimum of 1000 m.
- **Mean wind in heating layer**: density-weighted averages over active levels,
```math
\bar{u}_{\text{heat}} = \frac{\sum_k u_k |Q_1(z_k)| \rho_k}{\sum_k |Q_1(z_k)| \rho_k}, \quad \bar{N}_{\text{source}} = \frac{\sum_k N_k |Q_1(z_k)| \rho_k}{\sum_k |Q_1(z_k)| \rho_k}.
```

### Activation criteria

The Beres source activates in a column only when both conditions are met:
- $Q_0 > Q_{0,\text{threshold}}$ (default $1.157 \times 10^{-4}$ K/s, approximately 10 K/day)
- $h > h_{\min}$ (default 3000 m)

This filters out shallow or weak convection that would produce unreliable wave spectra. In columns where Beres is inactive, the AD99 Gaussian source is used instead.

### Source spectrum $B(c)$

The momentum flux spectrum at the source level is computed following Eqs. (23), (29)-(30) of [beres2004](@cite). The convective heating profile is assumed sinusoidal with depth $h$ and amplitude $Q_0$.

**Fourier decomposition of the heating:** For horizontal wavenumber $k$, the spectral power of the heating is
```math
G_k^2 = \frac{Q_0^2 \sigma_x^2}{2} \exp\left( -\frac{k^2 \sigma_x^2}{2} \right)
```
where $\sigma_x$ is the convective cell half-width (default 4000 m).

**Vertical wavenumber:** For intrinsic frequency $\hat{\nu} = \nu - k \bar{u}_{\text{heat}}$,
```math
m^2 = k^2 \left( \frac{N^2}{\hat{\nu}^2} - 1 \right).
```
Waves exist only when $m^2 > 0$ and $|\hat{\nu}| < N$.

**Response function:** The atmospheric response to the sinusoidal heating is
```math
R = \frac{\pi m h \, \text{sinc}(mh - \pi)}{(mh + \pi)(N^2 - \hat{\nu}^2)}.
```

**Spectral flux density:** The momentum flux density in $(k, \nu)$ space is
```math
F(k, \nu) = \frac{1}{\sqrt{2\pi}} \frac{\sqrt{N^2 - \hat{\nu}^2}}{|\hat{\nu}|} \, G_k^2 \, R^2.
```

**Phase speed spectrum:** The source spectrum $B_0(c)$ is obtained by integrating over frequency $\nu \in [\nu_{\min}, \nu_{\max}]$ with Boole's rule quadrature and applying the Jacobian $\nu / c^2$ to transform from $(k, \nu)$ to $(c, \nu)$ space:
```math
B_0(c) = \text{sgn}(\hat{c}) \cdot \alpha \int_{\nu_{\min}}^{\nu_{\max}} F(k, \nu) \frac{\nu}{c^2} \, d\nu
```
where $\hat{c} = c - \bar{u}_{\text{heat}}$ and $\alpha$ is the `beres_scale_factor`.

### Heating depth averaging

Optionally, the spectrum can be averaged over $n_{h,\text{avg}}$ values of $h$ in the range $[h - \Delta h, h + \Delta h]$ where $\Delta h = \Delta h_{\text{frac}} \cdot h$. This smooths the resonance peaks in the response function that arise from the sinusoidal heating assumption, following the recommendation in Section 2, Figure 4 of [beres2004](@cite).

### Propagation and breaking

Once the Beres source spectrum $B_0(c)$ replaces the AD99 Gaussian, the upward propagation, wave breaking, and momentum deposition logic is **identical** to the AD99 method described above. The same reflection criterion, instability condition, intermittency factor, and sponge layer redistribution apply.


## Implementation Details

### Structs and type dispatch

The Beres extension is gated at the type level. `NonOrographicGravityWave{FT, BS}` (`src/solver/types.jl`) carries a `beres_source::BS` field that is either `nothing` (AD99-only) or a `BeresSourceParams{FT}`:

```
BeresSourceParams{FT}
  Q0_threshold::FT        # K/s — min heating rate to activate
  beres_scale_factor::FT  # dimensionless amplitude scaling (α)
  σ_x::FT                 # m — convective cell half-width
  ν_min::FT               # rad/s — min frequency for quadrature
  ν_max::FT               # rad/s — max frequency for quadrature
  n_ν::Int                # quadrature points (must be 4k+1)
  n_h_avg::Int            # number of h values to average (1 = no averaging)
  Δh_frac::FT             # fractional half-range for h averaging
  h_heat_min::FT          # m — min heating depth to activate
```

When `BS = Nothing`, all Beres code paths are eliminated at compile time via `isnothing(gw_beres_source)`. Config parsing and construction happens in `src/solver/model_getters.jl`; the `nogw_beres_source` key requires `turbconv` to be `diagnostic_edmfx` or `prognostic_edmfx`.

### Cache fields

`non_orographic_gravity_wave_cache()` (`non_orographic_gravity_wave.jl`) allocates both AD99 and Beres fields. The Beres-specific fields are always allocated (to keep the cache type stable) but only populated when `gw_beres_source` is not `nothing`:

| Cache field | Shape | Description |
|---|---|---|
| `gw_Q0` | 2D (horizontal) | Peak convective heating rate $Q_0$ (K/s) |
| `gw_h_heat` | 2D | Heating depth $h$ (m) |
| `gw_u_heat` / `gw_v_heat` | 2D | Density-weighted mean wind in heating layer (m/s) |
| `gw_N_source` | 2D | Density-weighted mean buoyancy frequency in heating layer (1/s) |
| `gw_beres_active` | 2D | Activation flag (1 or 0) |
| `gw_zbot` / `gw_ztop` | 2D | Bottom/top of convective envelope (m) |
| `gw_Q_conv` | 3D (column) | Full $Q_1$ profile persisted from scratch |
| `gw_reduce_result` | 2D (6-tuple) | Scratch for `column_reduce!` passes |
| `gw_deep_count` / `gw_cb_count` | 2D | Counters: deep-convection events and callback invocations |

### Runtime pipeline (detailed stacktrace)

Abbreviation: `nogw.jl` = `src/parameterized_tendencies/gravity_wave_drag/non_orographic_gravity_wave.jl`.

```
Every dt_nogw seconds:
  nogw_model_callback!(integrator)                      [callbacks.jl]
    └─ non_orographic_gravity_wave_compute_tendency!()   [nogw.jl]
        │
        ├─ Step 1: Buoyancy frequency N(z)
        │   ᶜdTdz = vertical gradient of temperature
        │   N² = (g/T)(dT/dz + g/cp), clamped so N² ≥ 2.5e-5
        │
        ├─ Step 2: Source / damp level identification
        │   Column mode: closest level to source_height
        │   Sphere mode: highest level with p > source_pressure
        │   Damp level:  lowest level with p < damp_pressure
        │
        ├─ Step 3: Beres convective heating  [only when beres_source ≠ nothing]
        │   compute_beres_convective_heating!(Y, p)
        │   │
        │   ├─ 3a: Mass-flux divergence (Yanai Q₁)
        │   │   For each updraft j = 1..n_updrafts:
        │   │     ᶠu³_diff = ᶠu³ʲ - ᶠu³         (face velocity anomaly)
        │   │     ᶜa_scalar = (mseʲ + Kʲ - h_tot) × (ρaʲ / ρʲ)
        │   │     divergence via vertical_transport(ρʲ, ᶠu³_diff, ᶜa_scalar)
        │   │     ᶜQ_conv += divergence / (ρ · cp)
        │   │   Sources: ᶠu³ʲs, ᶜKʲs, ᶜρʲs, ᶜh_tot, ᶠu³ from p.precomputed
        │   │   Pattern mirrors edmfx_sgs_flux.jl SGS flux computation
        │   │
        │   ├─ 3b: Convective envelope detection
        │   │   Pass 1 (column_reduce! over updraft fields):
        │   │     z_peak = height of maximum updraft vertical velocity
        │   │     z_top  = highest level where area fraction > 1e-3
        │   │     z_bot  = max(2·z_peak − z_top,  3000 m)
        │   │     h      = z_top − z_bot
        │   │   Uses updraft structure (not Q₁ thresholding) for robustness
        │   │
        │   ├─ 3c: Heating layer properties
        │   │   Pass 2 (column_reduce! over [z_bot, z_top] envelope):
        │   │     Q_integral = Σ(Q_conv · Δz)
        │   │     u_heat, v_heat  = density-weighted mean horizontal winds
        │   │     N_source        = density-weighted mean buoyancy frequency
        │   │     Q₀ = (π/2) · Q_integral / h,  clamped ≥ 0
        │   │
        │   └─ 3d: Activation flag
        │       beres_active = (Q₀ > Q0_threshold) AND (h > h_heat_min)
        │
        ├─ Step 4: Zero forcing accumulators (uforcing, vforcing = 0)
        │
        └─ Step 5: Gravity wave forcing
            non_orographic_gravity_wave_forcing()
            │
            ├─ 5a: Shift fields up one level
            │   ρ, u, v, bf, z → ρ_p1, u_p1, v_p1, bf_p1, z_p1
            │   (needed for level-pair breaking computation)
            │
            ├─ 5b: Pack broadcast inputs
            │   input_u, input_v: 21-element tuples of all per-column fields
            │   Both AD99 and Beres fields included (zeros when Beres inactive)
            │
            └─ 5c: Wavenumber loop  (ink = 1..nk)
                For each horizontal wavenumber k = 2π / (30·10^ink km):
                │
                ├─ AD99 pass (always runs):
                │   waveforcing_column_accumulate!(..., Val(:ad99))
                │   │
                │   ├─ At level 1 — source spectrum:
                │   │   wave_source(c, u_source, Bw, Bn, cw, cn, c0, flag)
                │   │   B₀(c) = sign(c−u)·[Bw·exp(−ln2·((c−c₀)/cw)²)
                │   │                        + Bn·exp(−ln2·((c−c₀)/cn)²)]
                │   │   Reference frame: ground-relative extratropics,
                │   │   source-wind-relative tropics (flag controls blending)
                │   │
                │   ├─ At each level above source — propagation:
                │   │   Reflection:  |c−u|·k ≥ ωᵣ        → remove wave
                │   │   Breaking:    B₀/(c−u)³ ≥ ½(ρ/ρ₀)k/N → deposit flux
                │   │   Critical:    c−u sign flip         → deposit flux
                │   │   Model top:   all remaining flux deposited
                │   │
                │   ├─ Intermittency:
                │   │   ε = source_ampl / (ρ₀ · nk · Σ|B₀|)
                │   │   (prescribed latitude-dependent total flux)
                │   │
                │   └─ postprocess_and_accumulate!()
                │       gw_average!  (center → face interpolation)
                │       gw_deposit   (sponge layer: escaped flux above damp level)
                │       uforcing += u_waveforcing
                │
                └─ Beres pass (only when beres_source ≠ nothing):
                    waveforcing_column_accumulate!(..., Val(:beres))
                    │
                    ├─ At level 1 — source spectrum:
                    │   compute_beres_spectrum(beres, beres_active, ...)
                    │   If beres_active > 0.5:
                    │     wave_source(c, u_heat, Q₀, h, N, beres)
                    │       └─ _beres_spectrum_single_h()   [inner integral]
                    │            Boole's rule quadrature over ν ∈ [ν_min, ν_max]
                    │            weights = [7, 32, 12, 32, 7] × 2dν/45
                    │            For each frequency ν_j:
                    │              k   = ν/c
                    │              ν̂   = ν − k·u_heat        (intrinsic freq)
                    │              m²  = k²(N²/ν̂² − 1)      (vertical wavenumber)
                    │              R   = πmh·sinc(mh−π) / [(mh+π)(N²−ν̂²)]
                    │              Gk² = Q₀²σ_x²/2 · exp(−k²σ_x²/2)
                    │              F(k,ν) = (1/√2π)·√(N²−ν̂²)/|ν̂| · Gk²·R²
                    │            B₀(c) = sign(ĉ) · α · ∫ F·(ν/c²) dν
                    │       Optional h-averaging: repeat for n_h_avg values
                    │       in [h − Δh_frac·h,  h + Δh_frac·h], then average
                    │   Else (non-convecting column): all zeros
                    │
                    ├─ Propagation: same reflection/breaking logic as AD99
                    │
                    ├─ Intermittency:
                    │   ε = 1 / (ρ₀ · nk)
                    │   (scale factor α already baked into B₀)
                    │
                    └─ postprocess_and_accumulate!()  (same as AD99)

Every dt (integrator step):
  non_orographic_gravity_wave_apply_tendency!()
    ├─ Clamp forcing to ±3×10⁻³ m/s², zero any NaN/Inf
    └─ Yₜ.c.uₕ += Covariant12Vector(uforcing, vforcing)
```

### Key design decisions

- **Additive two-pass architecture.** AD99 and Beres forcing are computed in separate `column_accumulate!` passes within the same wavenumber loop and summed into the same `uforcing`/`vforcing` accumulators. In non-convecting columns the Beres pass contributes zero; in convecting columns both the AD99 background and Beres convective sources contribute.

- **Intermittency difference.** AD99 uses $\varepsilon = \text{source\_ampl} / (\rho_0 \cdot n_k \cdot \sum|B_0|)$ where `source_ampl` is a prescribed latitude-dependent total flux. Beres uses $\varepsilon = 1/(\rho_0 \cdot n_k)$ because the amplitude scaling factor $\alpha$ (`beres_scale_factor`) is already applied inside the spectrum $B_0(c)$.

- **Convective envelope from updraft structure.** The heating depth $h$ is determined from the EDMF updraft velocity peak and area fraction ($z_{\text{peak}}$, $z_{\text{top}}$) rather than thresholding the $Q_1$ profile directly. This is more robust because $Q_1$ (a divergence) can be noisy near column boundaries, while the updraft kinematic fields are smoother.

- **GPU compatibility.** All vertical operations use `column_accumulate!` / `column_reduce!` (ClimaCore operators). The `@noinline` annotation on `unrolled_reduce` prevents CPU compilation blowup for large phase-speed grids while having no effect on GPU (kernel code is always inlined). `StaticBitVector` stores the wave-breaking mask efficiently (8 booleans per `UInt8`, supporting up to 256 phase speeds).

- **Type dispatch for zero-cost abstraction.** `NonOrographicGravityWave{FT, BS}` is parameterised on `BS`. When `BS = Nothing` (no Beres), the `isnothing(gw_beres_source)` branches are eliminated at compile time, so AD99-only runs pay no overhead for the Beres code paths.

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
| `nogw_Q0` | Peak convective heating rate (K/s, Beres-only; gated by `beres_active`) |
| `nogw_h_heat` | Convective heating depth (m, Beres-only; gated by `beres_active`) |

### Key source files

| File | Description |
|---|---|
| `src/parameterized_tendencies/gravity_wave_drag/non_orographic_gravity_wave.jl` | All NOGW runtime: cache allocation, `compute_beres_convective_heating!`, `non_orographic_gravity_wave_forcing`, AD99 and Beres `wave_source` dispatches, `_beres_spectrum_single_h` quadrature |
| `src/solver/types.jl` | `NonOrographicGravityWave{FT,BS}` and `BeresSourceParams{FT}` struct definitions |
| `src/solver/model_getters.jl` | Config parsing, EDMF requirement check, `BeresSourceParams` construction |
| `src/diagnostics/gravitywave_diagnostics.jl` | `nogw_Q0` and `nogw_h_heat` diagnostic definitions (type-dispatched on `BeresSourceParams`) |
| `src/callbacks/callbacks.jl` | `nogw_model_callback!` — triggers `compute_tendency!` every `dt_nogw` |
