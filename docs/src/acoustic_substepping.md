# Acoustic substepping

## Motivation

ClimaAtmos integrates the compressible equations with a horizontally-explicit, vertically-implicit
(HEVI) split. The vertical acoustic (sound-wave) terms are solved implicitly, so thin vertical layers
do not restrict the timestep, while the horizontal acoustic terms are explicit and must satisfy the
acoustic CFL condition

```math
\frac{c_\mathrm{sound}\, \Delta t}{\Delta x} \lesssim 1 .
```

At high horizontal resolution this forces a small timestep even though the advective and physical
timescales — set by the wind speed ``|u| \ll c_\mathrm{sound}`` and by the parameterizations — permit a
much larger one. The model then spends most of its cost stepping fast only to keep the sound waves
stable. The sound waves themselves are not of interest; the goal is to treat them correctly without
letting them throttle the timestep of the slow dynamics.

Acoustic substepping (a split-explicit scheme) sub-cycles the fast horizontal acoustic terms at a small
sub-step inside a larger outer step, while the slow dynamics and physics are evaluated once per outer
step. The outer step is then limited by the advective and physical timescales, and the expensive
parameterizations are evaluated far less often. The intended benefit is higher throughput in the
convective gray zone (roughly one to ten kilometres of horizontal spacing), where the per-step physics
dominates the cost.

## How it works

Per outer step the scheme:

1. evaluates the slow forcing — the explicit tendency minus the horizontal acoustic terms — and freezes
   it,
2. sub-cycles the acoustic system `n_sub` times. Each sub-step advances the horizontal acoustic terms
   explicitly and the vertical acoustic terms implicitly (reusing the existing implicit solver), with
   the frozen slow forcing added.

A first- or second-order outer combination of the sub-cycles is available.

## Resonance source and divergence damping

Freezing the slow forcing over an outer step and sub-cycling the acoustic system excites a parametric
resonance with the grid-scale acoustic modes. The resonance grows fastest at the wavenumber where the
acoustic phase advances by ``\pi`` per outer step, and that wavenumber migrates to larger scales as the
outer step grows, so a damper tuned to the grid scale weakens as the outer step increases.

The dominant source of the resonance is the kinetic-energy gradient. The grid-mean momentum tendency
carries the gradients ``\nabla_h(K + \Phi - \Phi_r)`` on ``u_h`` and ``\partial_z K`` on ``u_3``, and
``K = |u|^2/2`` oscillates at the acoustic frequency. Holding these gradients in the frozen slow forcing
feeds an acoustic-frequency signal into the sub-cycle once per outer step, which is the resonant drive.
By default (`acoustic_substep_kinetic_energy: fast`) these gradients are re-evaluated at every sub-step,
so they follow the sub-cycled state and no longer act as a fixed periodic source; `slow` keeps the earlier
behavior of holding them in the frozen forcing. Re-evaluating the gradients leaves the executed sub-step
tendency unchanged at the outer-step start, where the frozen forcing is sampled, so the two treatments
agree to first order in the outer step and differ only in the sub-cycle interior.

The remaining resonant channels — the rotational momentum flux, upwind corrections, and frozen
tracer/EDMF flux divergences — are weaker at the acoustic frequency and are held by divergence damping on
the horizontal momentum. The damping adds ``\nu_d \, \nabla_h(\delta)`` to ``u_h``, with the damped
divergence ``\delta`` selected by `acoustic_substep_damping_form`:

| Form | Damped divergence ``\delta`` |
|---|---|
| `3d` (default) | ``\nabla_h \cdot u + \partial_z u^3`` — the three-dimensional divergence of the full velocity. |
| `3d_perturbation` | ``\delta - \delta^0``, the deviation from the divergence at the outer-step start ``\delta^0``. |
| `horizontal` | ``\nabla_h \cdot u_h`` — the horizontal divergence of the horizontal velocity. |

Taking the horizontal divergence of the damped ``u_h`` equation gives a term ``-\nu_d k_h^2 \delta``, so
horizontal and mixed acoustic modes are damped in proportion to ``k_h^2`` while purely vertical modes
(``k_h = 0``), the rotational flow, and balanced flow are untouched.

### Coefficient conventions

The viscosity is ``\nu_d = \beta_d \, c_\mathrm{ref}^2 \, \Delta t_\mathrm{sub}``, and the per-sub-step
grid-mode damping is ``\varepsilon_\mathrm{grid} = 4 \beta_d C_s^2`` with the sub-step acoustic Courant
number ``C_s = c_\mathrm{ref} \Delta t_\mathrm{sub}/\Delta x_\mathrm{node} \approx 0.5``. Published
split-explicit schemes use two normalizations, both of which map into this ``\beta_d``:

| Normalization | Definition | Operational ``\beta_d`` |
|---|---|---|
| Forward-weighted pressure filter (WRF `smdiv`) | ``\nu = \gamma_d \, c^2 \Delta\tau`` | ``\gamma_d \approx 0.1`` |
| Time-adjusted filter (MPAS / Klemp 2018) | ``\nu = a_d \, \Delta x^2/\Delta\tau`` | ``a_d/C_s^2 \approx 0.4`` at ``C_s = 0.5`` |

The two describe different filters, so both are correct in their own convention; expressed in the
``\nu_d = \beta_d c_\mathrm{ref}^2 \Delta t_\mathrm{sub}`` normalization used here they give a canonical
band ``\beta_d \in [0.1, 0.6]`` (COSMO operational practice sits near ``0.3``). With
`acoustic_substep_damping: auto` the coefficient resolves to ``0.4`` with `fast` kinetic energy and to
``1.5`` with `slow`. The explicit-diffusion limit ``\varepsilon_\mathrm{grid} \le 1`` gives
``\beta_d \lesssim 2`` at ``C_s = 0.5``; with a hand-set sub-step count at a higher per-sub-step Courant
number, ``\beta_d`` must shrink in proportion to ``1/C_s^2``.

With the resonance source removed the lower edge of the stable window is ``\beta_d = 0``, so divergence
damping is no longer required for stability and a lighter coefficient is viable: on the dry box halving the
default to ``\beta_d = 0.1`` roughly halves the divergent-flow distortion while remaining stable to the
same outer step. The `3d_perturbation` form conserves energy slightly better than `3d` at the same
settings, but does not reduce the distortion enough to change the default (`3d`).

### Outer-step ceilings

Once the resonance source is removed, two limits bound the outer step, both
independent of the resonance.

**Frozen explicit horizontal mixing.** Any explicit horizontal mixing tendency
lives in the frozen slow forcing ``G``, so the outer combination integrates it
effectively explicitly at the outer step ``\Delta t``. The frozen form carries no
damping benefit inside the sub-cycle: each scheme's own explicit stability limit
applies at the outer step. Which scheme binds depends on the configuration:

| Scheme | Diffusivity | Explicit limit | Outer-step multiple |
|---|---|---|---|
| Hyperdiffusion (4th order) | ``\nu_4 = c_4 h^3`` | ``\Delta t \lesssim C_4/(\nu_4 k_\mathrm{max}^4) \propto h/c_4`` | resolution-independent; ``\approx 3\times`` measured on the dry box |
| Smagorinsky | ``\nu = (C_s \Delta x)^2 |S|`` | ``\Delta t \le 1/(4 C_s^2 |S|)`` | strain timescale ``1/|S|`` (resolution-independent); ``O(50\text{–}1000)\times``, compressing where ``|S|`` spikes |
| EDMF horizontal | ``\nu \sim \ell \sqrt{\mathrm{TKE}},\ \ell \propto \Delta x`` | ``\Delta t \le \Delta x/(4 c_\ell \sqrt{\mathrm{TKE}})`` | ``\approx c/(4 c_\ell \sqrt{\mathrm{TKE}}\,\mathrm{CFL})``, ``O(50\text{–}100)\times`` |

The hyperdiffusivity scales as ``h^3`` while the baseline acoustic step scales as
``h/c``, so the hyperdiffusion ceiling expressed as an outer-step multiple is
resolution-independent (at fixed polynomial order and the ``\nu_4 \propto h^3``
convention) — the same small multiple applies on production gray-zone grids. On
hyperdiffusion-on configurations the frozen hyperdiffusion is therefore the
binding constraint: the resonance is de-sourced to a ``\approx 6\times`` ceiling,
but the frozen hyperdiffusion binds at ``\approx 3\times`` (stable at ``3\times``,
unstable at ``4\times`` on the dry box). The resolution is a documented outer-step
limit, not a scheme change; the follow-up lever is to sub-cycle or implicitly
treat the horizontal hyperdiffusion (out of scope here). The Smagorinsky and EDMF
limits are physical timescales (strain rate, eddy turnover), so their outer-step
multiples are large; on hyperdiffusion-off gray-zone or LES configurations the
frozen-mixing ceiling sits far above the ``4\text{–}6\times`` target and the
binding constraint is expected elsewhere. Because those limits depend on the local
flow, they are attributed empirically per configuration.

**Hyperdiffusion coefficient scaling.** The hyperdiffusion is a numerical filter
rather than a physical mixing term, so its coefficient can be scaled with the
outer step to keep the frozen filter within its explicit limit. The forward-Euler
limit is ``\Delta t_\mathrm{hd} = 2 \Delta x_\mathrm{node} / (F\, c_4\, \beta^4)``,
where ``c_4`` is the vorticity hyperdiffusion coefficient,
``F = \max(\text{divergence damping factor}, 1/\text{Prandtl number})`` is the
strongest of the momentum and scalar channels, and ``\beta = 4`` is a
maximum-wavenumber prefactor calibrated for degree-3 spectral elements from the
measured limit ``\Delta t_\mathrm{hd} \approx 0.95`` s at
``\Delta x_\mathrm{node} = 113`` m. With `acoustic_substep_hyperdiffusion_scaling:
auto` (the default) the coefficient is multiplied by
``\min(1, \Delta t_\mathrm{hd} / (\text{safety}\cdot\Delta t))`` with a safety
factor of ``2``, so the ceiling rises in proportion to ``1/c_4`` (measured on the
dry box: ``3\times`` unscaled, ``6\times`` at half strength, and the
hyperdiffusion-off ceiling at one third). Because the coefficient falls as the
outer step grows, the per-step filter strength ``c_4 k^4 \Delta t`` is held
roughly fixed while the filter integrates stably. A real value scales the
coefficient directly, with `1` reproducing the unscaled coefficient. Scaling
applies only under substepping and only to the hyperdiffusion filter; the physical
Smagorinsky and EDMF mixing coefficients are never scaled.

**Outer advective consistency.** The second-order outer combination freezes
advection over the outer step, so at a large outer step the advective Courant
number of the frozen forcing approaches its stability limit. Both outer orders are
available.

## Inner/outer implicit split

By default the sub-cycle treats the whole implicit tendency implicitly at the sub-step size, so with
prognostic EDMF, implicit vertical diffusion, or implicit microphysics active it re-solves the full
implicit operator and re-factorizes its Jacobian every sub-step. The inner/outer implicit split restricts
the implicit solve inside the sub-cycle to the vertical grid-mean acoustic block: the vertical mass-flux
divergence on ``\rho``, the central vertical transport of total enthalpy on ``\rho e_\mathrm{tot}`` and
total water on ``\rho q_\mathrm{tot}``, and the vertical pressure gradient, gravity, and Rayleigh sponge
on ``u_3``. The restricted system carries a dedicated sparse Jacobian whose scalar block is diagonal, so
it factorizes with the direct block-arrowhead solver instead of the iterative solver the full Jacobian
requires.

The remaining implicit terms — EDMF SGS, vertical diffusion, implicit microphysics, and sedimentation —
are solved once per outer step by a separate outer implicit solve, rather than every sub-step. The
first-order outer combination performs one outer solve per step; the second-order symmetric combination
performs two.

The split is additive: the full implicit tendency and its Jacobian are unchanged, and the restricted
operator duplicates the acoustic subset rather than extracting it, so the mode-off and unsplit paths are
unchanged. The split requires `acoustic_substep_vertical: implicit`.

## Configuration

| Option | Meaning |
|---|---|
| `acoustic_substeps` | `0` disables the mode; a positive integer fixes the sub-step count; `auto` selects it from the horizontal acoustic CFL. |
| `acoustic_substep_order` | Order of the outer combination (`1` or `2`). |
| `acoustic_substep_vertical` | `implicit` (default) keeps the vertically-implicit acoustic solve; `explicit` advances it in the sub-cycle. |
| `acoustic_substep_kinetic_energy` | `fast` (default) re-evaluates the kinetic-energy and reference-relative geopotential gradients every sub-step; `slow` holds them in the frozen forcing. |
| `acoustic_substep_damping_form` | Divergence used by the divergence damping: `3d` (default), `3d_perturbation`, or `horizontal`. |
| `acoustic_substep_damping` | Divergence-damping coefficient ``\beta_d`` (``\nu_d = \beta_d c_\mathrm{ref}^2 \Delta t_\mathrm{sub}``). `auto` (default) resolves to `0.4` for `fast` kinetic energy and `1.5` for `slow`; a number keeps its direct value. |
| `acoustic_substep_implicit_split` | Restrict the sub-cycle implicit solve to the vertical grid-mean acoustic block and solve the remaining implicit terms once per outer step (default `false`). Requires `acoustic_substep_vertical: implicit`. |
| `acoustic_substep_hyperdiffusion_scaling` | Scaling of the vorticity hyperdiffusion coefficient under substepping. `auto` (default) reduces it to keep the frozen filter within its explicit limit; a number scales it directly, with `1` reproducing the unscaled coefficient. Ignored when `acoustic_substeps: 0`. |

With `auto`, the sub-step count is

```math
n_\mathrm{sub} = \left\lceil \frac{\Delta t}{f \, \Delta x_\mathrm{node} / c_\mathrm{ref}} \right\rceil ,
```

where ``\Delta x_\mathrm{node}`` is the horizontal node spacing, ``c_\mathrm{ref}`` a reference sound
speed, and ``f = 0.5`` a CFL safety factor, so enabling the mode requires no hand-tuning of the
sub-step count.

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

The kinetic-energy treatment, damping form, and damping coefficient take their defaults (`fast`, `3d`,
and `auto`), so no further keys are needed. The previously validated configuration is recovered with
`acoustic_substep_kinetic_energy: slow`, `acoustic_substep_damping_form: horizontal`, and
`acoustic_substep_damping: 1.5`.

## When to use it

Use acoustic substepping when the horizontal acoustic CFL — not the advective CFL or the physics — is
what limits the timestep, and when the per-step physics is expensive enough to amortize. This is the
gray-zone regime. On configurations with cheap physics the sub-step count needed for stability can
outweigh the saving, so the mode is not beneficial there.

## Limitations

- By default the sub-cycle treats the whole implicit tendency implicitly at the sub-step size, so
  configurations that place microphysics, turbulence, or vertical diffusion in the implicit tendency
  re-solve those processes every sub-step. The inner/outer implicit split moves them to a per-outer-step
  solve; its conditioning at large outer steps is untested.
- The mode advances the vertical acoustic terms implicitly inside each sub-step; the explicit-vertical
  variant is a reference path limited by the vertical acoustic CFL on anisotropic grids.
- The sub-cycle refreshes only the acoustic precomputed quantities; diagnostics that read the full cache
  should refresh it after a step.
- The divergence-damping coefficient must lie within its stable range; the automatic default targets it.
- The stable outer step and the coefficient defaults are established on a flat, dry box. Terrain and
  moist convection are not yet exercised at large outer steps: the kinetic-energy gradient is evaluated
  in metric-aware form, but the reference-relative pressure-gradient cancellation over terrain is
  sensitive in `Float32`, and the moist stable outer step may be set by limits other than the resonance.
  The `horizontal` damping form is the documented fallback if the vertical divergence term destabilizes
  on strongly anisotropic grids.
