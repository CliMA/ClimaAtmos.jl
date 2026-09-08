#####
##### Tagged prognostic water tracers
#####
##### Each tag adds one grid-scale prognostic field `Y.c.ρq_tag_<name>` holding
##### part of the total water `ρq_tot`, so a tag records where water came from.
##### Regions, masks, state entries and restarts are shared with the energy tags
##### in `tagged_tracers.jl`. Config parsing lives in `config/tracer_config.jl`
##### under the `water_tracers` key. The physics is written up in
##### `docs/src/tagged_water.md`.
#####
##### Tag names are ρ-weighted, so `gs_tracer_names(Y)` picks them up and the
##### usual tracer machinery supplies advection, hyperdiffusion, sponges and
##### vertical eddy diffusion. Leave transport to that machinery. Attributing it
##### here would count it twice. Only the processes in `KNOWN_WATER_TAG_SOURCES`
##### are attributed.
#####
##### Attribution follows two rules. Local sources and sinks come from a
##### bracketed increment, where production is shared by region mask and loss in
##### proportion to what each tag already holds. Sedimentation is a flux
##### divergence, so each tag sediments with its own donor-cell flux built from
##### the parent's terminal velocity. See `sediment_water_tags!`.
#####
##### `ρq_tot` is advected implicitly and the tags explicitly. That split is the
##### one unavoidable source of closure drift, and `q_tag_res` measures it.

# ============================================================================
# Names, state, and initial values
# ============================================================================

# Build a single-entry NamedTuple `(; ρq_tag_<name> = value)`. As for the energy
# tags, the field name is computed at compile time from the tag's type
# parameter, so this is type-stable and GPU-compatible.
@generated function tag_entry(::WaterTag{name}, value) where {name}
    field_name = Symbol(:ρq_tag_, name)
    return :(NamedTuple{($(QuoteNode(field_name)),)}((value,)))
end

# Compile-time lookup of the entry `ρq_tag_<name>` in a state or tendency
# `Field` (e.g. `Yₜ.c`) or in a keyed cache `NamedTuple`.
@generated tag_field(obj, ::WaterTag{name}) where {name} =
    :(obj.$(Symbol(:ρq_tag_, name)))

# The tag's state field as a `MatrixFields.FieldName` relative to `Y.c`, the
# form the Jacobian indexes blocks by. `@name` needs a literal, so the name is
# built from the tag's type parameter instead; the result is a singleton type,
# so this stays a compile-time constant.
@generated function water_tag_field_name(::WaterTag{name}) where {name}
    field_name = Symbol(:ρq_tag_, name)
    return :(MatrixFields.FieldName($(QuoteNode(field_name))))
end

"""
    water_tagging_variables(ρq_tot, local_geometry, water_tagging_model)

NamedTuple of tagged prognostic water fields `(; ρq_tag_<name₁> = ..., ...)` for
a single grid point, to be splatted into the center prognostic state alongside
the other grid-scale variables. Returns `(;)` when water tagging is disabled
(`water_tagging_model === nothing`).

Must be evaluated with the same `ρq_tot` that `moisture_variables` puts into the
state, so that a partition-of-unity set of region tags sums to `ρq_tot` exactly
at `t = 0`.
"""
water_tagging_variables(ρq_tot, local_geometry, ::Nothing) = (;)
water_tagging_variables(ρq_tot, local_geometry, model::WaterTaggingModel) =
    _tag_variables(ρq_tot, local_geometry.coordinates, model.tags)

"""
    water_tag_state_names(model::WaterTaggingModel)

`Tuple` of the state-field `Symbol`s (`:ρq_tag_<name>`) of every water tag.
"""
water_tag_state_names(model::WaterTaggingModel) =
    Tuple(Symbol(:ρq_tag_, tag_name(tag)) for tag in model.tags)

"""
    water_region_tag_state_names(model::WaterTaggingModel)

`Tuple` of the state-field `Symbol`s of the *pure region* water tags: those with
a region and no sources. These are the tags whose sum is expected to track
`ρq_tot`, and the ones `q_tag_res` sums over. A tag that also carries a `source`
starts at zero and accumulates only that source, so including it would
double-count.
"""
water_region_tag_state_names(model::WaterTaggingModel) = Tuple(
    Symbol(:ρq_tag_, tag_name(tag)) for
    tag in model.tags if !isnothing(tag.region) && isempty(tag.sources)
)

# ============================================================================
# Config parsing
# ============================================================================

"""
    KNOWN_WATER_TAG_SOURCES

`Tuple` of the process labels that can be attributed to a tagged water tracer.
This is a *different, smaller* set than [`KNOWN_TAG_SOURCES`](@ref), because it
lists only the processes that actually move total water:

  - `:surface_flux`: turbulent surface moisture flux (evaporation, or dew when
    the flux is negative)
  - `:microphysics`: the 0-moment total-water sink (`dq_tot_dt ≤ 0`)
  - `:large_scale_advection`: prescribed large-scale advective moistening/drying
  - `:subsidence`: prescribed large-scale subsidence
  - `:external_forcing`: externally prescribed (e.g. GCM-driven) forcing and
    nudging of `q_tot`

Three deliberate absences:

  - `:radiation` and `:held_suarez` appear in `KNOWN_TAG_SOURCES` but do not move
    water. The shared attribution bracket still runs at those call sites; the
    water kernel simply returns without computing an increment.
  - `:precipitation` is absent by design, and stays absent now that 1-moment is
    supported. With 0-moment microphysics
    `vertical_advection_of_water_tendency!` has no sedimenting species to loop
    over and is a no-op. With 1-moment it is a flux divergence between levels,
    which is *mirrored* with the tags' own fluxes rather than attributed as a net
    source — attributing it would both double-count and mislabel the water
    arriving from the cell above. See [`sediment_water_tags!`](@ref).
  - `tracer_nonnegativity_vapor_tendency!` moves water between condensate species
    and diagnostic vapor and leaves `ρq_tot` unchanged.

Transport-like tendencies (advection, hyperdiffusion, sponges, vertical and SGS
diffusion) are never attributed: each tag is transported in its own right, so
attributing the `ρq_tot` version on top would count transport twice.

See also [`WATER_TAG_SOURCE_GROUPS`](@ref).
"""
const KNOWN_WATER_TAG_SOURCES = (
    :surface_flux,
    :microphysics,
    :large_scale_advection,
    :subsidence,
    :external_forcing,
)

"""
    WATER_TAG_SOURCE_GROUPS

Named groups of water process labels, usable wherever a `source` is expected:

  - `:surface`: turbulent exchange with the surface
  - `:forcing`: prescribed/idealized forcings (large-scale advection,
    subsidence, external forcing and nudging)
  - `:all`: every process in [`KNOWN_WATER_TAG_SOURCES`](@ref)

Groups expand at configuration time.
"""
const WATER_TAG_SOURCE_GROUPS = (;
    surface = (:surface_flux,),
    forcing = (:large_scale_advection, :subsidence, :external_forcing),
    all = KNOWN_WATER_TAG_SOURCES,
)

"""
    check_water_tagging_supported(microphysics_model)

Throw a descriptive error unless `microphysics_model` is one the water tags can
close against.

  - `EquilibriumMicrophysics0M`: every writer of `ρq_tot` is a *local* source or
    sink, so bracketed attribution alone is exact and nothing sediments.
  - `NonEquilibriumMicrophysics1M`: the condensate species are prognostic and
    sediment, moving `ρq_tot` between levels. That is a flux divergence rather
    than a local source, so it is not attributed but *mirrored*: each tag
    sediments with its own donor-cell flux (see [`sediment_water_tags!`](@ref)).
    Note that under 1M this is the only microphysical writer of `ρq_tot` —
    `microphysics_tendency!` moves mass between species only — so the
    `:microphysics` attribution bracket is a no-op there.

2-moment and P3 remain unsupported: they add number-concentration provenance,
which is a separate question from the mass provenance the tags carry.
"""
check_water_tagging_supported(::EquilibriumMicrophysics0M, key = "water_tracers") =
    nothing
check_water_tagging_supported(
    ::NonEquilibriumMicrophysics1M,
    key = "water_tracers",
) = nothing
check_water_tagging_supported(::DryModel, key = "water_tracers") = error(
    "`$key` requires a moist model: with `microphysics_model: dry` " *
    "there is no `ρq_tot` in the prognostic state to partition.",
)
check_water_tagging_supported(model, key = "water_tracers") = error(
    "`$key` supports `microphysics_model: 0M` and `1M` only (got " *
    "$(nameof(typeof(model)))). 2-moment and P3 schemes additionally carry " *
    "prognostic number concentrations, whose provenance is a separate question " *
    "from the water mass provenance these tags partition; mirroring only the " *
    "mass flux would leave the number field untagged and the two inconsistent.",
)

# ============================================================================
# Cache
# ============================================================================

_tag_mask_entry(ᶜcoord, ::WaterTag{name, Nothing}) where {name} = (;)
_tag_mask_entry(ᶜcoord, tag::WaterTag) =
    tag_entry(tag, region_mask.(Ref(tag.region), ᶜcoord))

_water_fix_fields(ᶜρ, ::Tuple{}) = (;)
_water_fix_fields(ᶜρ, tags::Tuple) = merge(
    tag_entry(first(tags), zero.(ᶜρ)),
    _water_fix_fields(ᶜρ, Base.tail(tags)),
)

"""
    _water_tagging_cache(Y, water_tagging_model)

Cache entries used by water-tag attribution, merged into `p.tagging`; `nothing`
when water tagging is disabled. Contains:

  - `ᶜwater_masks`: one static center `Field` per region tag holding the smooth
    spatial mask of that tag's region, keyed like the state (`ρq_tag_<name>`).
    Masks are evaluated once here and never inside a per-timestep broadcast.
  - `ᶜwater_fix`: one center `Field` per tag accumulating the water that the
    limiters and state constraints have moved into or out of that tag (see
    [`rescale_water_tags!`](@ref) and [`repair_water_tag_partition!`](@ref)).
    Cumulative since the start of the simulation segment, and reset on restart,
    so a budget over an interval is the difference of two outputs.
  - `ᶜrepair_pos`, `ᶜrepair_neg`: the positive and negative parts of the
    partition sum, used by [`repair_water_tag_partition!`](@ref). They live in
    the cache rather than `p.scratch` because the repair runs on the real state
    in `constrain_state!`, never inside a dual-typed tendency evaluation.

The `ρq_tot` snapshot that [`snapshot_tagged_ρq_tot!`](@ref) records lives in
`p.scratch` instead, because the implicit tendency is evaluated with
`ForwardDiff.Dual` numbers when an automatic-differentiation Jacobian is used and
only `p.precomputed` and `p.scratch` are converted to dual-typed fields.
"""
_water_tagging_cache(Y, ::Nothing) = nothing
function _water_tagging_cache(Y, model::WaterTaggingModel)
    ᶜwater_masks = _tag_masks(Fields.coordinate_field(Y.c), model.tags)
    _check_region_partition(
        ᶜwater_masks,
        water_region_tag_state_names(model),
        "q_tag_res",
        "ρq_tag",
    )
    ᶜwater_fix = _water_fix_fields(Y.c.ρ, model.tags)
    ᶜrepair_pos = zero.(Y.c.ρ)
    ᶜrepair_neg = zero.(Y.c.ρ)
    return (; ᶜwater_masks, ᶜwater_fix, ᶜrepair_pos, ᶜrepair_neg)
end

# ============================================================================
# Source attribution
# ============================================================================

"""
    water_tag_fraction(ρq_tag, ρq_tot)

The donor share `φ = ρq_tag / ρq_tot` of a tag in the local total water, clamped
to `[0, 1]` and defined to be zero where there is no water. The clamp keeps the
rule well posed when the tags have drifted slightly out of partition through
transport leakage, and when `ρq_tot` is positive but negligible.
"""
@inline water_tag_fraction(ρq_tag, ρq_tot) =
    ρq_tot > zero(ρq_tot) ?
    min(max(ρq_tag / ρq_tot, zero(ρq_tot)), one(ρq_tot)) : zero(ρq_tot)

"""
    water_tag_rescale_ratio(ρq_tot_after, ρq_tot_before)

The factor by which [`rescale_water_tags!`](@ref) scales every tag when a
limiter or state constraint has changed `ρq_tot`. It is
`ρq_tot_after / ρq_tot_before`, floored at zero, and zero where there was no
positive water to begin with.

Deliberately *not* clamped above 1: both limiters move water between cells, so a
cell that was clipped up legitimately needs its tags scaled up. The result stays
consistent because `ρq_tag ≤ ρq_tot_before` implies
`ρq_tag · ratio ≤ ρq_tot_after`.

The `ρq_tot_before ≤ 0` branch returns **zero**, not one. That branch is reached
precisely when a nonnegativity constraint clips a negative `ρq_tot` up, which is
the most common correction of all, and the tags of such a cell are themselves
negative — the donor rule scaled them by the same negative parent. Returning one
left them negative while the parent became zero, breaking both invariants this
function claims to preserve, and recorded `ρq_tag · (1 - 1) = 0` in the ledger,
so `q_tag_fix_<name>` reported that the limiter had done nothing — exactly the
conflation the ledger exists to prevent. Returning zero empties the tags along
with the parent, keeps `Σᵢ ρq_tag_i = ρq_tot` exact when the parent is clipped to
zero, and logs the removal honestly. Where the correction instead adds water to a
cell that had none, the tags stay at zero and the new water surfaces in
`q_tag_res` rather than being invented into a tag — the original intent, which
zero also satisfies.
"""
@inline water_tag_rescale_ratio(ρq_tot_after, ρq_tot_before) =
    ρq_tot_before > zero(ρq_tot_before) ?
    max(ρq_tot_after / ρq_tot_before, zero(ρq_tot_before)) :
    zero(ρq_tot_before)

"""
    snapshot_tagged_ρq_tot!(p, Yₜ)

Record the current value of `Yₜ.c.ρq_tot` in `p.scratch`, opening a water
attribution bracket. A no-op when water tagging is disabled.

Together with [`attribute_tagged_ρq_tot!`](@ref) this brackets a block of
tendency calls: whatever the block adds to `Yₜ.c.ρq_tot` is attributed to the
tagged water tracers without modifying the process itself. Brackets must not be
nested.
"""
snapshot_tagged_ρq_tot!(p, Yₜ) =
    _snapshot_tagged_ρq_tot!(p, Yₜ, p.atmos.water_tagging_model)
_snapshot_tagged_ρq_tot!(p, Yₜ, ::Nothing) = nothing
function _snapshot_tagged_ρq_tot!(p, Yₜ, ::WaterTaggingModel)
    p.scratch.ᶜtagging_q_snapshot .= Yₜ.c.ρq_tot
    return nothing
end

"""
    attribute_tagged_ρq_tot!(Yₜ, Y, p, source::Symbol)

Close a bracket opened by [`snapshot_tagged_ρq_tot!`](@ref): compute the
increment `Δ = Yₜ.c.ρq_tot - snapshot` produced by the bracketed process
(labeled `source`) and add `M_k·Δ⁺ - φ_k·Δ⁻` to each tag's tendency, where `M_k`
is the tag's mask and `φ_k` its donor share of the local water.

Production reaches a tag only when it lists `source` (pure region tags list none
and so receive every source); loss reaches *every* tag. A no-op when water
tagging is disabled, or when `source` is not a water process.

`Y` is needed in addition to `Yₜ` because the donor share is a property of the
current state.
"""
attribute_tagged_ρq_tot!(Yₜ, Y, p, source::Symbol) =
    _attribute_tagged_ρq_tot!(Yₜ, Y, p, source, p.atmos.water_tagging_model)
_attribute_tagged_ρq_tot!(Yₜ, Y, p, source, ::Nothing) = nothing
function _attribute_tagged_ρq_tot!(Yₜ, Y, p, source, model::WaterTaggingModel)
    source in KNOWN_WATER_TAG_SOURCES || return nothing
    (; ᶜwater_masks) = p.tagging
    ᶜρq_tot_snapshot = p.scratch.ᶜtagging_q_snapshot
    ᶜΔρq_tot = @. lazy(Yₜ.c.ρq_tot - ᶜρq_tot_snapshot)
    _accumulate_water_tags!(
        Yₜ.c,
        Y.c,
        ᶜwater_masks,
        ᶜΔρq_tot,
        source,
        model.tags,
    )
    return nothing
end

# The split `Δ = Δ⁺ - Δ⁻` appears below as `max(Δ, 0)` and `min(Δ, 0)`. The loss
# term is written `+ min(Δ, 0) * φ` instead of the equivalent `- max(-Δ, 0) * φ`
# for two reasons. It keeps the whole update in one broadcast over `ᶜΔ`. And a
# `-` directly before a modifier letter such as `ᶜ` parses as the suffixed
# operator `-ᶜ`, which Julia leaves undefined.
_accumulate_water_tags!(ᶜYₜ, ᶜY, ᶜmasks, ᶜΔ, source, ::Tuple{}) = nothing
function _accumulate_water_tags!(ᶜYₜ, ᶜY, ᶜmasks, ᶜΔ, source, tags::Tuple)
    _accumulate_water_tag!(ᶜYₜ, ᶜY, ᶜmasks, ᶜΔ, source, first(tags))
    return _accumulate_water_tags!(
        ᶜYₜ,
        ᶜY,
        ᶜmasks,
        ᶜΔ,
        source,
        Base.tail(tags),
    )
end

# Region-less tag: production weight is 1 wherever the tag receives this source.
function _accumulate_water_tag!(
    ᶜYₜ,
    ᶜY,
    ᶜmasks,
    ᶜΔ,
    source,
    tag::WaterTag{name, Nothing},
) where {name}
    ᶜρq_tagₜ = tag_field(ᶜYₜ, tag)
    ᶜρq_tag = tag_field(ᶜY, tag)
    if tag_receives_source(tag, source)
        @. ᶜρq_tagₜ +=
            max(ᶜΔ, 0) +
            min(ᶜΔ, 0) * water_tag_fraction(ᶜρq_tag, ᶜY.ρq_tot)
    else
        @. ᶜρq_tagₜ += min(ᶜΔ, 0) * water_tag_fraction(ᶜρq_tag, ᶜY.ρq_tot)
    end
    return nothing
end

# Tag with a region: production is masked. Loss stays donor-proportional, so
# water leaves from wherever the tag is holding it.
function _accumulate_water_tag!(ᶜYₜ, ᶜY, ᶜmasks, ᶜΔ, source, tag::WaterTag)
    ᶜρq_tagₜ = tag_field(ᶜYₜ, tag)
    ᶜρq_tag = tag_field(ᶜY, tag)
    ᶜmask = tag_field(ᶜmasks, tag)
    if tag_receives_source(tag, source)
        @. ᶜρq_tagₜ +=
            ᶜmask * max(ᶜΔ, 0) +
            min(ᶜΔ, 0) * water_tag_fraction(ᶜρq_tag, ᶜY.ρq_tot)
    else
        @. ᶜρq_tagₜ += min(ᶜΔ, 0) * water_tag_fraction(ᶜρq_tag, ᶜY.ρq_tot)
    end
    return nothing
end

# ============================================================================
# Sedimentation (1-moment and higher)
# ============================================================================

##### Each tag sediments with its own copy of the parent flux. For a sedimenting
##### species `s` with terminal velocity `wₛ` and specific content `qₛ`,
##### `vertical_advection_of_water_tendency!` adds
#####
#####     vtt = -ᶜprecipdivᵥ(ᶠρ * ᶠtop_bias(WVector(-wₛ) * qₛ))
#####
##### to `Yₜ.c.ρ` and `Yₜ.c.ρq_tot`. A tag gets the same expression with its
##### donor share `φ̂ₖ` placed inside the reconstruction:
#####
#####     vttₖ = -ᶜprecipdivᵥ(ᶠρ * ᶠtop_bias(WVector(-wₛ) * qₛ * φ̂ₖ))
#####
##### That placement buys three things. Closure is exact, because both operators
##### are linear, so shares summing to 1 pointwise give `Σₖ vttₖ = vtt` to
##### roundoff. Provenance is right, because `ᶠtop_bias` samples the cell the
##### water falls from. Surface removal comes for free, because `ᶜprecipdivᵥ`
##### leaves the bottom face as free outflow.
#####
##### Only the grid-mean flux is mirrored. The `PrognosticEDMFX` subdomain
##### corrections apply to the energy flux, and the tags are grid-scale only.
##### `docs/src/tagged_water.md` works through the shares and the Jacobian.

# A pure region tag has a region and no sources. Those tags form the partition
# whose shares are renormalized to sum to 1. Both properties are type
# parameters, so this resolves at compile time. It mirrors the runtime filter in
# `water_region_tag_state_names`.
_is_partition_tag(
    ::WaterTag{name, R, Tuple{}},
) where {name, R <: AbstractTagRegion} = true
_is_partition_tag(::WaterTag) = false

"""
    water_tag_sediment_share(ρq_tag, ρq_tot, norm)

Fraction of a sedimenting species' mass carried by a *partition* tag: its
clamped donor share `φ = ρq_tag / ρq_tot` divided by `norm`, the sum of the
clamped shares of every partition tag.

The renormalization is what preserves exact closure. Unlimited transport lets a
tag drift slightly out of the partition — measurably so on a sphere, where tags
reach a few percent of `max(ρq_tot)` below zero — and `water_tag_fraction` then
clamps, after which the raw shares no longer sum to 1 and `Σₖ vttₖ = vtt` fails
by the size of the drift. Dividing by `norm` restores it.

The result is always in `[0, 1]`: each clamped share is one of the non-negative
terms of `norm`, so no amplification is possible however small `norm` becomes.
Where `norm` is zero there is no tagged water to sediment and the share is zero
— closure cannot hold in a cell whose tags are all empty but whose `ρq_tot` is
not, and that discrepancy correctly surfaces in `q_tag_res`.
"""
@inline water_tag_sediment_share(ρq_tag, ρq_tot, norm) =
    norm > zero(norm) ? water_tag_fraction(ρq_tag, ρq_tot) / norm : zero(norm)

"""
    water_tag_source_sediment_share(ρq_tag, ρq_tot)

Fraction of a sedimenting species' mass carried by a *source* tag: its own
clamped donor share, unnormalized.

Source tags are not members of the partition — they start at zero and accumulate
one process — so no closure constraint applies to them and there is nothing to
renormalize against. Their water is real water that falls out like any other, in
proportion to what the tag actually holds, which is the same donor rule the loss
half of [`attribute_tagged_ρq_tot!`](@ref) uses.
"""
@inline water_tag_source_sediment_share(ρq_tag, ρq_tot) =
    water_tag_fraction(ρq_tag, ρq_tot)

"""
    water_tag_sediment_dshare(ρq_tag, ρq_tot, norm)
    water_tag_source_sediment_dshare(ρq_tag, ρq_tot)

Derivative of the corresponding share with respect to `ρq_tag`, for the implicit
sedimentation Jacobian (see `update_sedimentation_jacobian!`).

For a partition tag `norm` depends on `ρq_tag` too, which contributes the
`(1 - φ̂)` factor: a tag that already owns all of the local water cannot increase
its share, so the derivative vanishes there rather than staying at `1/(ρq_tot ⋅ norm)`. Both derivatives are zero wherever the `[0, 1]` clamp is active, matching
the tendency's own piecewise behavior.
"""
@inline function water_tag_sediment_dshare(ρq_tag, ρq_tot, norm)
    (ρq_tot > zero(ρq_tot) && norm > zero(norm)) || return zero(ρq_tot)
    φ = ρq_tag / ρq_tot
    (φ > zero(φ) && φ < one(φ)) || return zero(ρq_tot)
    return (one(φ) - φ / norm) / (ρq_tot * norm)
end

@inline function water_tag_source_sediment_dshare(ρq_tag, ρq_tot)
    ρq_tot > zero(ρq_tot) || return zero(ρq_tot)
    φ = ρq_tag / ρq_tot
    (φ > zero(φ) && φ < one(φ)) || return zero(ρq_tot)
    return inv(ρq_tot)
end

"""
    water_tag_sediment_dshare_field(Y, p, tag)

Lazy field of `∂φ̂/∂ρq_tag` for `tag`, selecting the partition or source form.
`_is_partition_tag` resolves on the tag's type, so the branch folds away and the
return type is inferred.

Requires [`water_tag_share_norm!`](@ref) to have been evaluated for the current
state.
"""
function water_tag_sediment_dshare_field(Y, p, tag)
    ᶜρq_tag = tag_field(Y.c, tag)
    ᶜρq_tot = Y.c.ρq_tot
    if _is_partition_tag(tag)
        ᶜnorm = p.scratch.ᶜtagging_q_share_norm
        return @. lazy(water_tag_sediment_dshare(ᶜρq_tag, ᶜρq_tot, ᶜnorm))
    else
        return @. lazy(water_tag_source_sediment_dshare(ᶜρq_tag, ᶜρq_tot))
    end
end

"""
    water_tag_share_norm!(p, Y)

Fill `p.scratch.ᶜtagging_q_share_norm` with `Σⱼ clamp(ρq_tag_j / ρq_tot)` over
the *partition* tags, the denominator [`water_tag_sediment_share`](@ref) divides
by. A no-op when water tagging is disabled.

Lives in `p.scratch` rather than `p.tagging` for the same reason as the `ρq_tot`
snapshot: the implicit tendency is evaluated with `ForwardDiff.Dual` numbers when
an automatic-differentiation Jacobian is used, and only `p.precomputed` and
`p.scratch` are converted to dual-typed fields.

Must be recomputed whenever the state changes — once per tendency evaluation and
once per Jacobian update — because it is a property of the current `Y`.
"""
water_tag_share_norm!(p, Y) =
    _water_tag_share_norm!(p, Y, p.atmos.water_tagging_model)
_water_tag_share_norm!(p, Y, ::Nothing) = nothing
function _water_tag_share_norm!(p, Y, model::WaterTaggingModel)
    ᶜnorm = p.scratch.ᶜtagging_q_share_norm
    ᶜnorm .= zero(eltype(ᶜnorm))
    _accumulate_share_norm!(ᶜnorm, Y.c, model.tags)
    return nothing
end

_accumulate_share_norm!(ᶜnorm, ᶜY, ::Tuple{}) = nothing
function _accumulate_share_norm!(ᶜnorm, ᶜY, tags::Tuple)
    tag = first(tags)
    if _is_partition_tag(tag)
        ᶜρq_tag = tag_field(ᶜY, tag)
        @. ᶜnorm += water_tag_fraction(ᶜρq_tag, ᶜY.ρq_tot)
    end
    return _accumulate_share_norm!(ᶜnorm, ᶜY, Base.tail(tags))
end

"""
    sediment_water_tags!(Yₜ, Y, p, ᶜq, ᶜw, ᶠρ)

Add one sedimenting species' mirrored flux divergence to every tagged water
tracer, where `ᶜq` is that species' specific content, `ᶜw` its terminal velocity
and `ᶠρ` the face-interpolated density — the same three quantities the parent
`ρq_tot` flux is built from in `vertical_advection_of_water_tendency!`, so that
the tagged fluxes sum to it exactly.

Call once per species, from inside that function's species loop, with
[`water_tag_share_norm!`](@ref) already evaluated for the current state. A no-op
when water tagging is disabled.
"""
sediment_water_tags!(Yₜ, Y, p, ᶜq, ᶜw, ᶠρ) =
    _sediment_water_tags!(Yₜ, Y, p, ᶜq, ᶜw, ᶠρ, p.atmos.water_tagging_model)
_sediment_water_tags!(Yₜ, Y, p, ᶜq, ᶜw, ᶠρ, ::Nothing) = nothing
function _sediment_water_tags!(
    Yₜ,
    Y,
    p,
    ᶜq,
    ᶜw,
    ᶠρ,
    model::WaterTaggingModel,
)
    _sediment_water_tags!(
        Yₜ.c,
        Y.c,
        p.scratch.ᶜtagging_q_share_norm,
        ᶜq,
        ᶜw,
        ᶠρ,
        model.tags,
    )
    return nothing
end

_sediment_water_tags!(ᶜYₜ, ᶜY, ᶜnorm, ᶜq, ᶜw, ᶠρ, ::Tuple{}) = nothing
function _sediment_water_tags!(ᶜYₜ, ᶜY, ᶜnorm, ᶜq, ᶜw, ᶠρ, tags::Tuple)
    tag = first(tags)
    ᶜρq_tagₜ = tag_field(ᶜYₜ, tag)
    ᶜρq_tag = tag_field(ᶜY, tag)
    # `-(ᶜw)` is parenthesized, as in `vertical_advection_of_water_tendency!`.
    # A `-` directly before a modifier letter parses as the suffixed operator
    # `-ᶜ`, which Julia leaves undefined.
    if _is_partition_tag(tag)
        @. ᶜρq_tagₜ +=
            -1 * ᶜprecipdivᵥ(
                ᶠρ * ᶠtop_bias(
                    Geometry.WVector(-(ᶜw)) *
                    ᶜq *
                    water_tag_sediment_share(ᶜρq_tag, ᶜY.ρq_tot, ᶜnorm),
                ),
            )
    else
        @. ᶜρq_tagₜ +=
            -1 * ᶜprecipdivᵥ(
                ᶠρ * ᶠtop_bias(
                    Geometry.WVector(-(ᶜw)) *
                    ᶜq *
                    water_tag_source_sediment_share(ᶜρq_tag, ᶜY.ρq_tot),
                ),
            )
    end
    return _sediment_water_tags!(
        ᶜYₜ,
        ᶜY,
        ᶜnorm,
        ᶜq,
        ᶜw,
        ᶠρ,
        Base.tail(tags),
    )
end

"""
    sedimenting_water_tag_names(Y)

`Tuple` of `@name`s (relative to `Y.c`) of the tagged water tracers that mirror
the sedimentation flux, and so need their own implicit sedimentation Jacobian
block.

Empty when water tagging is disabled, and empty under 0-moment microphysics,
where there is no sedimenting species to mirror and the tags behave like any
other passive tracer. Used by both the Jacobian block allocation and the
Jacobian update so the two cannot disagree about which blocks exist.
"""
sedimenting_water_tag_names(Y) =
    isempty(sedimenting_mass_names(Y)) ? () :
    unrolled_filter(is_water_tag_name, gs_tracer_names(Y))

# ============================================================================
# Combined brackets
# ============================================================================

"""
    snapshot_tags!(p, Yₜ, source::Symbol)
    attribute_tags!(Yₜ, Y, p, source::Symbol)

Open and close one attribution bracket for every tagging family at once, and
for the process records. The energy half is unconditional (every bracketed
process is an energy process); the water half fires only for
`source in KNOWN_WATER_TAG_SOURCES`, so bracketing e.g. radiation costs a
water-tagged run nothing.

Each half is a no-op when its own model is `nothing`, so a run with only one
family enabled pays only for that family.

The same bracket also feeds the process records, which difference the same two
fields to record what the bracketed process applied. See
[`snapshot_process_record!`](@ref).
"""
function snapshot_tags!(p, Yₜ, source::Symbol)
    snapshot_tagged_ρe_tot!(p, Yₜ)
    if source in KNOWN_WATER_TAG_SOURCES
        snapshot_tagged_ρq_tot!(p, Yₜ)
    end
    snapshot_energy_source_tags!(p, Yₜ)
    snapshot_process_record!(p, Yₜ, source)
    return nothing
end

function attribute_tags!(Yₜ, Y, p, source::Symbol)
    attribute_tagged_ρe_tot!(Yₜ, p, source)
    attribute_tagged_ρq_tot!(Yₜ, Y, p, source)
    attribute_energy_source_tags!(Yₜ, Y, p, source)
    accumulate_process_record!(Yₜ, p, source)
    return nothing
end

# ============================================================================
# Numerical corrections
# ============================================================================

"""
    rescale_water_tags!(Y, p, ᶜρq_tot_before)

Follow a correction that the limiters or state constraints applied to `ρq_tot`
by scaling every water tag by the parent's relative change,

    ρq_tag_k *= ρq_tot_after / ρq_tot_before,

which is the donor rule again: numerical corrections add or remove water in
proportion to the local composition. This preserves both `Σᵢ ρq_tag_i = ρq_tot`
and non-negativity exactly, which limiting each tag independently would not — a
shape-preserving adjustment applied per tag has no reason to sum to the parent's.
That is why water tags are excluded from the tracer limiters by
[`is_tagged_tracer_name`](@ref) and corrected here instead.

`ᶜρq_tot_before` holds `ρq_tot` from before the correction; `Y.c.ρq_tot` already
holds the corrected value. The signed water moved is accumulated into
`p.tagging.ᶜwater_fix` and reported by the `q_tag_fix_<name>` diagnostic, so that
"the limiters moved water" stays distinguishable from "the transport operators
disagree", which `q_tag_res` alone would conflate.

A no-op when water tagging is disabled.
"""
rescale_water_tags!(Y, p, ᶜρq_tot_before) =
    _rescale_water_tags!(Y, p, ᶜρq_tot_before, p.atmos.water_tagging_model)
_rescale_water_tags!(Y, p, ᶜρq_tot_before, ::Nothing) = nothing
function _rescale_water_tags!(Y, p, ᶜρq_tot_before, model::WaterTaggingModel)
    (; ᶜwater_fix) = p.tagging
    _rescale_water_tags!(Y.c, ᶜwater_fix, ᶜρq_tot_before, model.tags)
    return nothing
end

_rescale_water_tags!(ᶜY, ᶜwater_fix, ᶜρq_tot_before, ::Tuple{}) = nothing
function _rescale_water_tags!(ᶜY, ᶜwater_fix, ᶜρq_tot_before, tags::Tuple)
    tag = first(tags)
    ᶜρq_tag = tag_field(ᶜY, tag)
    ᶜfix = tag_field(ᶜwater_fix, tag)
    # Accumulate the signed change before applying it, so the ledger records the
    # correction itself and not its effect on an already-corrected tag. The
    # ratio is recomputed on the spot. Two comparisons and a divide cost less
    # than a scratch field, and this stays allocation free.
    @. ᶜfix +=
        ᶜρq_tag * (water_tag_rescale_ratio(ᶜY.ρq_tot, ᶜρq_tot_before) - 1)
    @. ᶜρq_tag *= water_tag_rescale_ratio(ᶜY.ρq_tot, ᶜρq_tot_before)
    return _rescale_water_tags!(
        ᶜY,
        ᶜwater_fix,
        ᶜρq_tot_before,
        Base.tail(tags),
    )
end

# ============================================================================
# Partition repair
# ============================================================================

"""
    water_tag_repair_factor(pos, neg)

The common factor that [`repair_water_tag_partition!`](@ref) applies to the
non-negative part of each partition tag: `max(pos + neg, 0) / pos`, and zero
where there is no positive water to redistribute into.
"""
@inline water_tag_repair_factor(pos, neg) =
    pos > zero(pos) ? max(pos + neg, zero(pos)) / pos : zero(pos)

"""
    repair_water_tag_partition!(Y, p)

Restore non-negativity of the partition tags without changing their sum.

Unlimited transport lets the partition tags drift out of the partition: they ride
the explicit passive-tracer path with a nonlinear flux limiter applied per field,
which does not satisfy `Σ F(χᵢ) = F(Σ χᵢ)`, so individual tags reach a few
percent of `max(ρq_tot)` below zero on a sphere.

A negative tag is not merely cosmetic. [`water_tag_fraction`](@ref) clamps it to
a zero donor share, which shrinks the renormalization denominator `norm`; that
lets [`water_tag_sediment_share`](@ref) hand a surviving tag a share far larger
than the water it actually holds — the share is bounded by 1, but the mass
removed *relative to what the tag owns* is amplified by `1/norm` — driving that
tag negative in turn. The feedback compounds every step, and it takes the
sedimentation Jacobian with it, since `∂φ̂/∂ρq_tag` grows like
`1/Σⱼ ρq_tag_j` with only a `norm > 0` guard.

For each cell, writing `S⁺ = Σₖ max(ρq_tagₖ, 0)` and `S⁻ = Σₖ min(ρq_tagₖ, 0)`
over the partition tags, this sets

    ρq_tagₖ ← max(ρq_tagₖ, 0) · max(S⁺ + S⁻, 0) / S⁺,

absorbing the negative water into the positive tags in proportion to what each
holds. The sum `S⁺ + S⁻` is preserved exactly and every tag ends non-negative.
Where the negatives outweigh the positives (`S⁺ + S⁻ < 0`) no non-negative
partition can have that sum, so every tag is zeroed and the deficit surfaces in
`q_tag_res` rather than being hidden.

This deliberately does **not** renormalize the tags onto `ρq_tot`. Forcing
`Σᵢ ρq_tag_i = ρq_tot` every step would drive `q_tag_res` to zero by
construction and destroy the leakage monitor it exists to provide. The repair
removes only the negativity, leaving genuine transport leakage visible and
bounding the clamped-share denominator by that leakage instead of by how far a
tag has gone negative.

Source tags are left alone: they are not members of the partition, carry no
closure obligation, and are already excluded from the renormalization
denominator by `_accumulate_share_norm!`.

The signed water moved is accumulated into `p.tagging.ᶜwater_fix` alongside the
limiter corrections, so it is reported by the `q_tag_fix_<name>` diagnostic.

Called from `constrain_state!` after the limiters and state constraints have
finished with `ρq_tot`. A no-op when water tagging is disabled.
"""
repair_water_tag_partition!(Y, p) =
    _repair_water_tag_partition!(Y, p, p.atmos.water_tagging_model)
_repair_water_tag_partition!(Y, p, ::Nothing) = nothing
function _repair_water_tag_partition!(Y, p, model::WaterTaggingModel)
    (; ᶜwater_fix, ᶜrepair_pos, ᶜrepair_neg) = p.tagging
    ᶜrepair_pos .= zero(eltype(ᶜrepair_pos))
    ᶜrepair_neg .= zero(eltype(ᶜrepair_neg))
    _accumulate_partition_parts!(ᶜrepair_pos, ᶜrepair_neg, Y.c, model.tags)
    _apply_partition_repair!(
        Y.c,
        ᶜwater_fix,
        ᶜrepair_pos,
        ᶜrepair_neg,
        model.tags,
    )
    return nothing
end

_accumulate_partition_parts!(ᶜpos, ᶜneg, ᶜY, ::Tuple{}) = nothing
function _accumulate_partition_parts!(ᶜpos, ᶜneg, ᶜY, tags::Tuple)
    tag = first(tags)
    if _is_partition_tag(tag)
        ᶜρq_tag = tag_field(ᶜY, tag)
        @. ᶜpos += max(ᶜρq_tag, 0)
        @. ᶜneg += min(ᶜρq_tag, 0)
    end
    return _accumulate_partition_parts!(ᶜpos, ᶜneg, ᶜY, Base.tail(tags))
end

# `ᶜpos` and `ᶜneg` are read-only here and come from the pre-repair state, so
# each tag can be rewritten in place and a later tag's factor still holds.
_apply_partition_repair!(ᶜY, ᶜwater_fix, ᶜpos, ᶜneg, ::Tuple{}) = nothing
function _apply_partition_repair!(ᶜY, ᶜwater_fix, ᶜpos, ᶜneg, tags::Tuple)
    tag = first(tags)
    if _is_partition_tag(tag)
        ᶜρq_tag = tag_field(ᶜY, tag)
        ᶜfix = tag_field(ᶜwater_fix, tag)
        # Ledger first, so it records the correction itself and not its effect
        # on an already-corrected tag. This matches `rescale_water_tags!`.
        @. ᶜfix +=
            max(ᶜρq_tag, 0) * water_tag_repair_factor(ᶜpos, ᶜneg) - ᶜρq_tag
        @. ᶜρq_tag = max(ᶜρq_tag, 0) * water_tag_repair_factor(ᶜpos, ᶜneg)
    end
    return _apply_partition_repair!(
        ᶜY,
        ᶜwater_fix,
        ᶜpos,
        ᶜneg,
        Base.tail(tags),
    )
end
