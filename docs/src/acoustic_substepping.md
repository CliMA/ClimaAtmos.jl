# Acoustic substepping

## Motivation

ClimaAtmos integrates the compressible equations with a horizontally-explicit, vertically-implicit (HEVI) split.
The vertical acoustic (sound-wave) terms are solved implicitly, so thin vertical layers do not restrict the timestep, while the horizontal acoustic terms are explicit and must satisfy the acoustic CFL condition

```math
\frac{c_\mathrm{sound}\, \Delta t}{\Delta x} \lesssim 1 .
```

At high horizontal resolution this forces a small timestep even though the advective and physical timescales, set by the wind speed ``|u| \ll c_\mathrm{sound}`` and by the parameterizations, permit a much larger one.
Most of the model runtime then goes to stepping fast only to keep the sound waves stable.
The sound waves themselves are not of interest; the goal is to treat them correctly without letting them limit the timestep of the slow dynamics.

Acoustic substepping (a split-explicit scheme) sub-cycles the fast horizontal acoustic terms at a small sub-step inside a larger outer step, while the slow dynamics and physics are evaluated once per outer step.
The outer step is then limited by the advective and physical timescales, and the parameterizations are evaluated far less often.
The intended benefit is higher throughput in the convective gray zone (roughly one to ten kilometres of horizontal spacing), where the per-step physics is the dominant cost.

## How it works

Per outer step the scheme:

1. evaluates the slow forcing (the explicit tendency minus the horizontal acoustic terms and the kinetic-energy gradients) and freezes it,
2. sub-cycles the acoustic system `n_sub` times.
   Each sub-step advances the horizontal acoustic terms and the kinetic-energy gradients explicitly and the vertical acoustic terms implicitly (reusing the existing implicit solver), with the frozen slow forcing added.

A first- or second-order outer combination of the sub-cycles is available.

## Resonance source and divergence damping

Freezing the slow forcing over an outer step and sub-cycling the acoustic system excites a parametric resonance with the grid-scale acoustic modes.
The resonance grows fastest at the wavenumber where the acoustic phase advances by ``\pi`` per outer step, and that wavenumber migrates to larger scales as the outer step grows, so a damper tuned to the grid scale weakens as the outer step increases.

The dominant source of the resonance is the kinetic-energy gradient.
The grid-mean momentum tendency includes the gradients ``\nabla_h(K + \Phi - \Phi_r)`` on ``u_h`` and ``\partial_z K`` on ``u_3``, and ``K = |u|^2/2`` oscillates at the acoustic frequency.
Holding these gradients in the frozen slow forcing would feed an acoustic-frequency signal into the sub-cycle once per outer step, the resonant drive.
The scheme re-evaluates these gradients at every sub-step, so they follow the sub-cycled state and no longer act as a fixed periodic source.
Re-evaluation leaves the executed sub-step tendency unchanged at the outer-step start, where the frozen forcing is sampled.

The remaining resonant terms (the rotational momentum flux, upwind corrections, and frozen tracer/EDMF flux divergences) are weaker at the acoustic frequency and are held by divergence damping on the horizontal momentum.
The damping adds ``\nu_d \, \nabla_h(\delta)`` to ``u_h``, with the damped divergence ``\delta`` selected by `acoustic_substep_damping_form`:

| Form | Damped divergence ``\delta`` |
|---|---|
| `3d` (default) | ``\nabla_h \cdot u + \partial_z u^3``, the three-dimensional divergence of the full velocity. |
| `horizontal` | ``\nabla_h \cdot u_h``, the horizontal divergence of the horizontal velocity. |

Taking the horizontal divergence of the damped ``u_h`` equation gives a term ``-\nu_d k_h^2 \delta``, so horizontal and mixed acoustic modes are damped in proportion to ``k_h^2`` while purely vertical modes (``k_h = 0``), the rotational flow, and balanced flow are untouched.

### Coefficient conventions

The viscosity is ``\nu_d = \beta_d \, c_\mathrm{ref}^2 \, \Delta t_\mathrm{sub}``, and the per-sub-step grid-mode damping is ``\varepsilon_\mathrm{grid} = 4 \beta_d C_s^2`` with the sub-step acoustic Courant number ``C_s = c_\mathrm{ref} \Delta t_\mathrm{sub}/\Delta x_\mathrm{node} \approx 0.5``.
Published split-explicit schemes use two normalizations, both of which map into this ``\beta_d``:

| Normalization | Definition | Operational ``\beta_d`` |
|---|---|---|
| Forward-weighted pressure filter (WRF `smdiv`) | ``\nu = \gamma_d \, c^2 \Delta\tau`` | ``\gamma_d \approx 0.1`` |
| Time-adjusted filter (MPAS / Klemp 2018) | ``\nu = a_d \, \Delta x^2/\Delta\tau`` | ``a_d/C_s^2 \approx 0.4`` at ``C_s = 0.5`` |

The two describe different filters, so both are correct in their own convention; expressed in the ``\nu_d = \beta_d c_\mathrm{ref}^2 \Delta t_\mathrm{sub}`` normalization used here they give a canonical band ``\beta_d \in [0.1, 0.6]`` (COSMO operational practice is near ``0.3``).
With `acoustic_substep_damping: auto` the coefficient resolves to ``0.4``.
The explicit-diffusion limit ``\varepsilon_\mathrm{grid} \le 1`` gives ``\beta_d \lesssim 2`` at ``C_s = 0.5``; with a manually set sub-step count at a higher per-sub-step Courant number, ``\beta_d`` must shrink in proportion to ``1/C_s^2``.

Because the kinetic-energy gradient is re-evaluated in the sub-cycle, the lower edge of the stable window is ``\beta_d = 0``, so divergence damping is not required for stability and a lighter coefficient is viable: on the dry box, halving the default to ``\beta_d = 0.1`` roughly halves the divergent-flow distortion while remaining stable to the same outer step.

### Outer-step limits

Two limits bound the outer step, both independent of the resonance.

#### Frozen explicit horizontal mixing

Any explicit horizontal mixing tendency is in the frozen slow forcing ``G``, so the outer combination integrates it effectively explicitly at the outer step ``\Delta t``.
The frozen form has no damping benefit inside the sub-cycle: each scheme's own explicit stability limit applies at the outer step.
Which scheme is limiting depends on the configuration:

| Scheme | Diffusivity | Explicit limit | Outer-step multiple |
|---|---|---|---|
| Hyperdiffusion (4th order) | ``\nu_4 = c_4 h^3`` | ``\Delta t \lesssim C_4/(\nu_4 k_\mathrm{max}^4) \propto h/c_4`` | resolution-independent; ``\approx 3\times`` measured on the dry box |
| Smagorinsky | ``\nu = (C_s \Delta x)^2 |S|`` | ``\Delta t \le 1/(4 C_s^2 |S|)`` | strain timescale ``1/|S|`` (resolution-independent); ``O(50\text{–}1000)\times``, compressing where ``|S|`` is large |
| EDMF horizontal | ``\nu \sim \ell \sqrt{\mathrm{TKE}},\ \ell \propto \Delta x`` | ``\Delta t \le \Delta x/(4 c_\ell \sqrt{\mathrm{TKE}})`` | ``\approx c/(4 c_\ell \sqrt{\mathrm{TKE}}\,\mathrm{CFL})``, ``O(50\text{–}100)\times`` |

The hyperdiffusivity scales as ``h^3`` while the baseline acoustic step scales as ``h/c``, so the hyperdiffusion limit expressed as an outer-step multiple is resolution-independent (at fixed polynomial order and the ``\nu_4 \propto h^3`` convention), and the same small multiple applies on production gray-zone grids.
On hyperdiffusion-on configurations the frozen hyperdiffusion is therefore the limiting term: it is stable at ``\approx 3\times`` and unstable at ``\approx 4\times`` on the dry box.
Because hyperdiffusion is a numerical filter, its coefficient can be reduced at a large outer step with the `hyperdiffusion_dt_limit_safety` option, which raises this limit in proportion to ``1/\nu_4``.
The Smagorinsky and EDMF limits are physical timescales (strain rate, eddy turnover), so their outer-step multiples are large; on hyperdiffusion-off gray-zone or LES configurations the frozen-mixing limit is far above the ``4\text{–}6\times`` target and the limiting constraint is elsewhere.
Because those limits depend on the local flow, they are attributed empirically per configuration.

#### Outer advective consistency

The second-order outer combination freezes advection over the outer step, so at a large outer step the advective Courant number of the frozen forcing approaches its stability limit.
Both outer orders are available.

## Inner/outer implicit split

By default the sub-cycle treats the whole implicit tendency implicitly at the sub-step size, so with prognostic EDMF, implicit vertical diffusion, or implicit microphysics active it re-solves the full implicit operator and re-factorizes its Jacobian every sub-step.
The inner/outer implicit split restricts the implicit solve inside the sub-cycle to the vertical grid-mean acoustic block: the vertical mass-flux divergence on ``\rho``, the central vertical transport of total enthalpy on ``\rho e_\mathrm{tot}`` and total water on ``\rho q_\mathrm{tot}``, and the vertical pressure gradient, gravity, and Rayleigh sponge on ``u_3``.
The restricted system has a dedicated sparse Jacobian whose scalar block is diagonal, so it factorizes with the direct block-arrowhead solver instead of the iterative solver the full Jacobian requires.

The remaining implicit terms (EDMF SGS, vertical diffusion, implicit microphysics, and sedimentation) are solved once per outer step by a separate outer implicit solve, rather than every sub-step.
The first-order outer combination performs one outer solve per step; the second-order symmetric combination performs two.

The split is additive: the full implicit tendency and its Jacobian are unchanged, and the restricted operator duplicates the acoustic subset rather than extracting it, so the mode-off and unsplit paths are unchanged.
The split requires `acoustic_substep_vertical: implicit`.

## Configuration

| Option | Meaning |
|---|---|
| `acoustic_substeps` | `0` disables the mode; a positive integer fixes the sub-step count; `auto` selects it from the horizontal acoustic CFL. |
| `acoustic_substep_order` | Order of the outer combination (`1` or `2`). |
| `acoustic_substep_vertical` | `implicit` (default) keeps the vertically-implicit acoustic solve; `explicit` advances it in the sub-cycle. |
| `acoustic_substep_damping_form` | Divergence used by the divergence damping: `3d` (default) or `horizontal`. |
| `acoustic_substep_damping` | Divergence-damping coefficient ``\beta_d`` (``\nu_d = \beta_d c_\mathrm{ref}^2 \Delta t_\mathrm{sub}``). `auto` (default) resolves to `0.4`; a number keeps its direct value. |
| `acoustic_substep_implicit_split` | Restrict the sub-cycle implicit solve to the vertical grid-mean acoustic block and solve the remaining implicit terms once per outer step (default `false`). Requires `acoustic_substep_vertical: implicit`. |

With `auto`, the sub-step count is

```math
n_\mathrm{sub} = \left\lceil \frac{\Delta t}{f \, \Delta x_\mathrm{node} / c_\mathrm{ref}} \right\rceil ,
```

where ``\Delta x_\mathrm{node}`` is the horizontal node spacing, ``c_\mathrm{ref}`` a reference sound speed, and ``f = 0.5`` a CFL safety factor, so enabling the mode requires no manual tuning of the sub-step count.

## Example

A box configuration that enables the mode with an automatic sub-step count:

```yaml
config: "box"
x_elem: 30
y_elem: 30
z_elem: 30
dt: "4secs"
acoustic_substeps: "auto"
```

The damping form and damping coefficient take their defaults (`3d` and `auto`), so no further keys are needed.
On a hyperdiffusion-on configuration at a large outer step, set `hyperdiffusion_dt_limit_safety` (recommended `2`) so the frozen hyperdiffusion stays within its explicit stability limit.

## When to use it

Use acoustic substepping when the horizontal acoustic CFL, not the advective CFL or the physics, is what limits the timestep, and when the per-step physics is expensive enough to amortize.
This is the gray-zone regime.
On configurations with inexpensive physics the sub-step count needed for stability can outweigh the saving, so the mode is not beneficial there.

## Limitations

- The sub-cycle treats the whole implicit tendency implicitly at the sub-step size.
  The mode is validated for configurations whose implicit tendency is the vertical-acoustic block; configurations that also place microphysics, turbulence, or vertical diffusion in the implicit tendency re-solve those processes every sub-step, unless `acoustic_substep_implicit_split` moves them to a per-outer-step solve.
  The conditioning of that outer solve at large outer steps is untested.
- The mode advances the vertical acoustic terms implicitly inside each sub-step; the explicit-vertical variant is a reference path limited by the vertical acoustic CFL on anisotropic grids.
- The sub-cycle refreshes only the acoustic precomputed quantities; diagnostics that read the full cache should refresh it after a step.
- The divergence-damping coefficient must lie within its stable range; the automatic default targets it.
- The stable outer step and the coefficient defaults are established on a flat, dry box.
  Terrain and moist convection are not yet exercised at large outer steps: the kinetic-energy gradient is evaluated in metric-aware form, but the reference-relative pressure-gradient cancellation over terrain is sensitive in `Float32`, and the moist stable outer step may be set by limits other than the resonance.
  The `horizontal` damping form is the documented fallback if the vertical divergence term destabilizes on strongly anisotropic grids.
