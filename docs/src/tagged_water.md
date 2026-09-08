# Tagged Water Tracers

Tagged water tracers decompose total water ``\rho q_\mathrm{tot}`` into labeled
prognostic components, so that the water at a point can be attributed to where
or how it entered the atmosphere. Each tag is an ordinary grid-scale tracer
`Y.c.ρq_tag_<name>`, transported by the automatic tracer machinery (see
[Tracers](passive_tracers.md)).

They are the water counterpart of the [Tagged Energy
Tracers](tagged_tracers.md) and share their region masks, configuration schema
and restart handling. Two things differ, and both matter — read
[Attribution](#Attribution) before using the output.

!!! warning "Total water is not water vapor"

    The `hus` diagnostic in this repository is named "Specific Humidity" but
    computes ``\rho q_\mathrm{tot}/\rho``, i.e. the mass of **all** water
    phases; `husv` is the vapor-only counterpart. The tagged names do not
    inherit that ambiguity: `q_tag_<name>` is total water and `qv_tag_<name>` is
    vapor. In the CliMA formulation ``q_t = q_v + q_l + q_i``, with ``q_l`` and
    ``q_i`` including precipitation.

## Enabling tags

```yaml
microphysics_model: "0M"
water_tracers:
  - name: tropics
    region: tropics
  - name: extratropics
    region: extratropics
  - name: evap
    source: surface_flux
  - name: evap_tropics
    region: tropics
    source: surface_flux
```

Each entry needs a unique `name` and a `region`, a `source`, or both. The
default (`water_tracers: ~`) disables the feature entirely: no extra state
fields, cache entries, or runtime cost. The region types and their `inside: false` / `above: false` complements are exactly those documented for the
[energy tags](tagged_tracers.md#Region-tags): `everywhere`, `tanh_altitude`,
`tanh_latitude`, `tanh_box`, `tanh_polygon`.

See [Configuring Tracers](tracer_configuration.md) for the full schema, the
named regions (`tropics`, `extratropics`, `everywhere`) used above, and the
`water_closure_check` block, which reduces `q_tag_res` to a pair of numbers on a
period of its own and warns while the run goes.

A region tag is initialized to ``\rho q_\mathrm{tot} \, M(x)``; a tag with a
`source` starts at zero.

!!! note "One partition at a time"

    The closure diagnostic `q_tag_res` sums **all** pure region tags, so
    configure exactly one partition of unity per run (a region and its
    complement). A warning is emitted at initialization when the pure region
    masks do not sum to 1.

## Attribution

For each attributed process, the increment ``\Delta`` that it adds to
``\rho q_\mathrm{tot}`` is split into gross production and gross loss,
``\Delta^{+} = \max(\Delta, 0)`` and ``\Delta^{-} = \max(-\Delta, 0)``, and the
two halves are attributed by **different rules**:

```math
\Delta\!\left(\rho q_{\mathrm{tag},k}\right)
= M_k \, \Delta^{+} - \varphi_k \, \Delta^{-},
\qquad
\varphi_k = \frac{\rho q_{\mathrm{tag},k}}{\rho q_\mathrm{tot}}.
```

  - **Production is mask-weighted.** ``M_k`` is the tag's region mask (1 for a
    region-less source tag, 0 if the tag does not list this process). New water
    carries the label of where it entered.
  - **Loss is donor-proportional.** Water leaves in proportion to what is
    actually present, and **every** tag is depleted — including source tags,
    whatever processes they list. This is what makes ``\rho q_{\mathrm{tag},k}``
    an actual water mass rather than a running source integral.

This is the one place where the water tags deliberately depart from the energy
tags, which attribute the whole increment by mask. A mask-weighted *loss* would
remove water a tag does not own and can drive tags negative. The rule here is
the tendency form of the relative scaling ``\chi \mathrel{*}= (1 + \dot q\, \Delta t / q)`` used by the MESSy `H2OEMIS` submodel.

Two consequences worth stating:

  - **Closure.** With ``\sum_k M_k = 1`` and ``\sum_k \rho q_{\mathrm{tag},k} = \rho q_\mathrm{tot}`` we have ``\sum_k \varphi_k = 1``, so
    ``\sum_k \Delta_k = \Delta^{+} - \Delta^{-} = \Delta`` exactly, per process.
    If the configured tags are a strict subset (say a single "Atlantic
    evaporation" tag), then ``\sum_k \varphi_k < 1`` and the untagged remainder
    absorbs the rest — also correct, just not a partition.
  - **Positivity.** A tag update is
    ``\rho q_{\mathrm{tag},k}\,(1 - \Delta^{-}\Delta t / \rho q_\mathrm{tot})``,
    so tags stay non-negative under the same step restriction that keeps
    ``\rho q_\mathrm{tot}`` non-negative, which the 0-moment sink already
    enforces.

### Taggable processes

| Group     | `source` label          | Process                                                             |
|:--------- |:----------------------- |:------------------------------------------------------------------- |
| `surface` | `surface_flux`          | Turbulent surface moisture flux (evaporation, or dew when negative) |
| *(none)*  | `microphysics`          | The 0-moment total-water sink                                       |
| `forcing` | `large_scale_advection` | Prescribed large-scale advective moistening or drying               |
| `forcing` | `subsidence`            | Prescribed large-scale subsidence                                   |
| `forcing` | `external_forcing`      | Externally prescribed (e.g. GCM-driven) forcing and `q_tot` nudging |

The group `all` expands to every process in the table. Note that `microphysics`
belongs to **no named group**: `source: surface` selects `surface_flux` only, so
a tag written that way follows evaporation but not the 0-moment sink. `all` is
the only group that includes `microphysics`; to follow both without the
forcings, list them explicitly as `source: [surface_flux, microphysics]`.

This is a *different, smaller* set than the energy tags': `radiation` and
`held_suarez` do not move water.

Splitting a net increment by sign is exact only where production and loss are
mutually exclusive at a point, which holds for the two that matter most — the
surface flux is evaporation or dew, and the 0-moment tendency is a sink by
construction. For the prescribed forcings it is an assumption.

### What is *not* taggable, and why

  - **Transport**: advection, hyperdiffusion, sponges, interior vertical
    diffusion and LES SGS diffusion all act on each tag in its own right, so
    attributing the ``\rho q_\mathrm{tot}`` version on top would count transport
    twice. This is the central correctness constraint of the design.
  - **Phase changes**: condensation, evaporation, freezing and melting conserve
    ``q_t``, so they are invisible to a total-water tag by construction. This is
    why no per-transfer ledger is needed — and why a vapor-only passive tracer
    would be the wrong design, since it would lose provenance at every phase
    change.
  - **Precipitation sedimentation**: with 0-moment microphysics there are no
    prognostic condensate species to sediment, so the term does not exist. With
    1-moment it is a flux divergence between levels rather than a local source,
    so it is not attributed but *mirrored* — see
    [Sedimentation with 1-moment microphysics](@ref).
  - **Numerical corrections** are handled separately, by
    `rescale_water_tags!`: the tags are excluded from both tracer limiters,
    because limiting each independently has no reason to reproduce the parent's
    adjustment and would break ``\sum_i \rho q_{\mathrm{tag},i} = \rho q_\mathrm{tot}``. Instead every tag is scaled by the parent's relative
    change, which preserves both the sum and non-negativity exactly, and the
    signed water moved is recorded in `q_tag_fix_<name>`.

### Sedimentation with 1-moment microphysics

With `microphysics_model: "1M"` the condensate species are prognostic and fall,
so sedimentation moves ``\rho q_\mathrm{tot}`` between levels. This is the one
water process that is neither attributed nor ignored, because it is a flux
divergence rather than a local source: its net increment in a cell mixes water
arriving from above with water leaving below, and attributing that increment
would label the arriving water with the receiving cell's mask and drain the
departing water in proportion to the total-water composition when the falling
condensate's composition is what actually leaves.

Instead the flux itself is *mirrored*. `sediment_water_tags!` is called once per
sedimenting species from inside the species loop of
`vertical_advection_of_water_tendency!`, and builds each tag's flux from the very
same specific content ``q``, terminal velocity ``w`` and face density ``\rho_f``
as the parent — the same donor-cell (`ᶠtop_bias`) reconstruction — scaled by
the tag's share of the local water. Because only the share differs, the tagged
fluxes sum to the parent flux exactly, level by level, and surface precipitation
is tagged.

The share differs between the two kinds of tag:

  - **Partition tags** (a region, no sources) use the *renormalized* clamped
    donor share ``\hat\varphi_k = \varphi_k / \sum_j \varphi_j``, with
    ``\varphi_k = \mathrm{clamp}(\rho q_{\mathrm{tag},k} / \rho q_\mathrm{tot}, 0, 1)``. The renormalization is what preserves exact closure: unlimited
    transport lets a tag drift slightly out of the partition — a few percent of
    ``\max(\rho q_\mathrm{tot})`` below zero on a sphere — after which the
    clamped shares no longer sum to one and ``\sum_k \mathrm{vtt}_k = \mathrm{vtt}`` fails by the size of that drift. Dividing by the sum restores
    it, and since each clamped share is one of the non-negative terms of the
    denominator, ``\hat\varphi_k \in [0, 1]`` however small the denominator gets.
  - **Source tags** are not members of the partition — they start at zero and
    accumulate one process — so no closure constraint applies and their share is
    the unnormalized ``\varphi_k``. Their water is real water that falls out like
    any other, under the same donor rule the loss half of the attribution uses.

Where no tagged water is present the share is zero rather than undefined; if
``\rho q_\mathrm{tot}`` is nonzero there, closure genuinely cannot hold and the
discrepancy surfaces in `q_tag_res` as it should.

Sedimentation is stepped implicitly, so the tags also enter the Jacobian:
`update_sedimentation_jacobian!` allocates and fills their diagonal blocks using
the analytic derivative of the share. For a partition tag that derivative carries
a ``(1 - \hat\varphi_k)`` factor, because a tag that already owns all the local
water cannot increase its share.

!!! note "Phases are well mixed within a cell"

    The mirror assumes the sedimenting condensate carries the cell's *total*-water
    tag composition, since the tags partition ``q_t`` and hold no phase
    information of their own. This is the same assumption `qv_tag` rests on.

Under 1-moment this is the only microphysical writer of ``\rho q_\mathrm{tot}``
— `microphysics_tendency!` moves mass between species only — so the
`microphysics` attribution bracket is a no-op there, and the `precipitation`
label that the energy tags carry has no water counterpart.

## Diagnostics and closure

  - `q_tag_<name>`: tagged **total** water ``\rho q_\mathrm{tag}/\rho``;
  - `qv_tag_<name>`: tagged **vapor**, ``q_\mathrm{tag} \, q_v / q_t``;
  - `q_tag_res`: the closure residual ``(\rho q_\mathrm{tot} - \sum_i \rho q_{\mathrm{tag},i})/\rho``, summed over the pure region tags;
  - `q_tag_fix_<name>`: water moved into or out of the tag by the limiters and
    state constraints, cumulative since the start of the simulation segment (and
    reset on restart), so a budget over an interval is the difference of two
    outputs, and a time *average* of it is not meaningful.

!!! note "What `q_tag_fix` includes"

    Two mechanisms write to the ledger. `repair_water_tag_partition!` runs every
    step and contributes wherever transport drove a partition tag negative, so
    `q_tag_fix_<name>` is generally nonzero even under stock settings — it is a
    useful direct measure of how much the tags are drifting.
    `rescale_water_tags!` contributes only when something actually corrects
    ``\rho q_\mathrm{tot}``: `apply_sem_quasimonotone_limiter: true`,
    `tracer_nonnegativity_method: vertical_water_borrowing`, an elementwise
    tracer nonnegativity constraint, or a `PrescribedFlow` setup. With none of
    those configured, everything in this field is partition repair.

!!! note "The vapor split is an assumption"

    `qv_tag` assumes the water phases are well mixed within a grid cell: the
    tags partition total water and carry no phase information of their own, and
    with 0-moment microphysics ``q_l`` and ``q_i`` are the saturation-adjustment
    diagnosis of the grid mean. This is stated in the diagnostic's `comments`
    field as well, so it travels with the output.

`q_tag_res` is a **monitored residual**, not a machine-precision identity — but
a much tighter one than the energy tags' `e_tag_res`. The tags use *identical*
vertical diffusion (unscaled ``K_h``) and hyperdiffusion (unscaled
``\nu_4``) operators to ``\rho q_\mathrm{tot}``, because ``\rho q_\mathrm{tot}``
is likewise absent from the sedimenting-species list that receives the scaled
coefficients. The dominant contributor is the vertical advection split: ``\rho q_\mathrm{tot}`` is advected implicitly with a post-Newton upwind correction,
while the tags ride the explicit passive-tracer path. Subtract `q_tag_fix_*` to
separate that operator disagreement from numerical corrections.

It is not the *only* contributor, though. Any tendency that writes
``\rho q_\mathrm{tot}`` by name without an attribution bracket and without a
tagged counterpart also lands here — see the Caveats below for the two known
cases (`PrognosticEDMFX` SGS mass flux, and the `PrescribedFlow` surface water
inflow). If `q_tag_res` grows faster than expected, check those before
concluding the advection split is responsible.

A sharper *process closure* check is available by splitting a source tag across
a partition: with `evap`, `evap_tropics` and `evap_extratropics`, linearity of
production, loss, transport and the limiter rescale implies
``q_\mathrm{tag,evap\_tropics} + q_\mathrm{tag,evap\_extratropics} = q_\mathrm{tag,evap}`` to near machine precision at all times — any violation
indicates a bug rather than expected leakage.
`config/model_configs/baroclinic_wave_tagged_water.yml` and the integration test use
this identity.

## Scope

Water tagging supports `microphysics_model: "0M"` and `"1M"`, and
`check_water_tagging_supported` errors otherwise.

  - **0-moment**: every writer of ``\rho q_\mathrm{tot}`` is a local source or
    sink, so bracketed attribution alone is exact and nothing sediments.
  - **1-moment**: phase changes are *not* an obstacle — those conserve
    ``\rho q_\mathrm{tot}`` and are invisible to the tags. Sedimentation is,
    and it is handled by mirroring the flux per tag rather than attributing it;
    see [Sedimentation with 1-moment microphysics](@ref).
  - **Dry**: there is no ``\rho q_\mathrm{tot}`` in the state to partition.
  - **2-moment and P3** remain unsupported: they additionally carry prognostic
    number concentrations, whose provenance is a separate question from the mass
    provenance these tags partition, and mirroring only the mass flux would leave
    the number field untagged and the two inconsistent.

## Caveats

  - Tags are **grid-scale only**: they have no sub-grid (updraft) counterpart.
    With `PrognosticEDMFX` the SGS mass flux moves ``\rho q_\mathrm{tot}`` in a
    way the tags never receive, so `q_tag_res` grows; the grid-mean surface
    evaporation is still attributed correctly. Nothing rejects this combination
    at configuration time, so watch `q_tag_res` if you enable it.
  - With a **`PrescribedFlow`** setup (e.g. `ShipwayHill2012`), the surface
    water inflow imposed as a vertical-transport boundary condition adds to
    ``\rho q_\mathrm{tot}`` outside every attribution bracket and has no tagged
    counterpart. That water enters the domain untagged and `q_tag_res` drifts
    monotonically. The combination is accepted by
    `check_water_tagging_supported` (`ShipwayHill2012` is 1-moment), and
    `prescribe_flow!` does rescale the tags after its clip — so the tags stay
    consistent with each other, they are just collectively short of
    ``\rho q_\mathrm{tot}`` by the injected amount.
  - The energy tags' `microphysics` label still fires only when microphysics is
    stepped explicitly. The water tags are bracketed on the implicit path too,
    because `implicit_microphysics` defaults to `true` and that is where the
    0-moment water sink lives.
  - Tagged state is carried through restarts like any other prognostic field;
    the masks are rebuilt from the configuration, so the `water_tracers` block
    must match the one used to write the checkpoint. The `q_tag_fix` ledger is
    cache-resident and restarts at zero.

## Interpretation limit

Exact closure establishes internally consistent contribution accounting; it does
not turn the tags into counterfactual sensitivities. The donor-fraction loss
rule, the well-mixed-phases assumption behind `qv_tag`, and the limiter rescale
policy are modeling choices, and conclusions are conditional on them.

See `config/model_configs/baroclinic_wave_tagged_water.yml` for a complete example,
and `test/tagged_water_integration.jl` for the closure assertions.

## API

Rendered here so that the `@ref` links in these docstrings resolve; Documenter
resolves `@ref` only against docstrings a `@docs` block splices into a page.

```@docs
ClimaAtmos.WaterTaggingModel
ClimaAtmos.WaterTag
ClimaAtmos.KNOWN_WATER_TAG_SOURCES
ClimaAtmos.WATER_TAG_SOURCE_GROUPS
ClimaAtmos.water_tag_fraction
ClimaAtmos.water_tag_share_norm!
ClimaAtmos.water_tag_sediment_share
ClimaAtmos.sediment_water_tags!
ClimaAtmos.snapshot_tagged_ρq_tot!
ClimaAtmos.attribute_tagged_ρq_tot!
ClimaAtmos.rescale_water_tags!
ClimaAtmos.repair_water_tag_partition!
```
