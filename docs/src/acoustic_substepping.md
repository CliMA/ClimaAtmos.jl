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

Sub-cycling a frozen slow forcing excites a parametric resonance with the grid-scale acoustic modes (the
kinetic-energy gradient in the slow forcing oscillates at the acoustic frequency). The scheme suppresses
it with divergence damping on the horizontal momentum, which damps the divergent (acoustic) modes while
leaving the rotational flow untouched. The damping coefficient has a stable range: too little leaves the
resonance, too much over-damps the resolved divergent flow.

## Configuration

| Option | Meaning |
|---|---|
| `acoustic_substeps` | `0` disables the mode; a positive integer fixes the sub-step count; `auto` selects it from the horizontal acoustic CFL. |
| `acoustic_substep_order` | Order of the outer combination (`1` or `2`). |
| `acoustic_substep_vertical` | `implicit` (default) keeps the vertically-implicit acoustic solve; `explicit` advances it in the sub-cycle. |
| `acoustic_substep_damping` | Divergence-damping coefficient (default `1.5`). |

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
acoustic_substep_damping: 1.5
```

## When to use it

Use acoustic substepping when the horizontal acoustic CFL — not the advective CFL or the physics — is
what limits the timestep, and when the per-step physics is expensive enough to amortize. This is the
gray-zone regime. On configurations with cheap physics the sub-step count needed for stability can
outweigh the saving, so the mode is not beneficial there.

## Limitations

- The sub-cycle treats the whole implicit tendency implicitly at the sub-step size. The mode is
  validated for configurations whose implicit tendency is the vertical-acoustic block; configurations
  that also place microphysics, turbulence, or vertical diffusion in the implicit tendency re-solve
  those processes every sub-step.
- The mode advances the vertical acoustic terms implicitly inside each sub-step; the explicit-vertical
  variant is a reference path limited by the vertical acoustic CFL on anisotropic grids.
- The sub-cycle refreshes only the acoustic precomputed quantities; diagnostics that read the full cache
  should refresh it after a step.
- The divergence-damping coefficient must lie within its stable range; the automatic default targets it.
