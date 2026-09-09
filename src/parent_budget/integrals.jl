#####
##### Parent-budget ledger: the authoritative integrals
#####
##### The three parent quantities are defined once, here, and both ends of every
##### transaction use these definitions, so a change of definition cannot be
##### applied to one end and not the other.
#####
##### Everything in this file is local. Nothing communicates. A global total is
##### assembled by `reduction.jl`, which packs the local values of a whole step
##### into one buffer and reduces that buffer once. Keeping the definitions free
##### of communication is what makes one collective per accepted step possible.
#####
##### All three integrals are linear extensive functionals of authoritative
##### prognostic state, which is what makes exact additive process attribution
##### well defined. That is a property of these particular integrals, so changing
##### one reopens the question. See `docs/src/parent_budget/contract.md`.

"""
    BUDGET_ACCOUNTING_TYPE

The float type the ledger does its own arithmetic in, `Float64`.

Deliberately independent of the state's float type. A residual is what survives
subtracting two large global totals, and a `Float32` subtraction destroys it, so
the contract fixes the accounting precision at no less than `Float64` whatever
the model runs in.

The precision has to be reached **before** anything is accumulated. A global
mass carried in `Float32` holds about seven significant digits, so a per-step
change eight orders below it is already gone by the time a `Float32` sum
completes, and converting the finished sum afterwards recovers nothing. Every
integral below therefore converts pointwise, inside the expression that is
reduced.
"""
const BUDGET_ACCOUNTING_TYPE = Float64

"""
    BUDGET_QUANTITIES

The three parent quantities, in the order the packet layout uses them.

Each has its own definition, applicability, residual and tolerance. They share a
journal so that a coupled exchange stays coordinated across the three. They are
never interchangeable.
"""
const BUDGET_QUANTITIES = (:mass, :water, :energy)

# ============================================================================
# Pointwise conversion
# ============================================================================
#
# These are plain functions broadcast over fields, which is how the model itself
# writes pointwise arithmetic. Broadcasting them means the conversion happens at
# each degree of freedom and the reduction accumulates in the accounting type.

"""
    to_accounting(x)

Widen one value to `BUDGET_ACCOUNTING_TYPE`. Broadcast over a field so
that the accumulation, not just the result, is in the accounting type.
"""
to_accounting(x) = BUDGET_ACCOUNTING_TYPE(x)

"""
    dry_air_density(ρ, ρq_tot)

Return the pointwise dry-air density, widened before the subtraction so the
cancellation happens in the accounting type.
"""
dry_air_density(ρ, ρq_tot) =
    BUDGET_ACCOUNTING_TYPE(ρ) - BUDGET_ACCOUNTING_TYPE(ρq_tot)

"""
    boundary_areal_density(x, Δz)

Return the pointwise integrand of a horizontal integral at a boundary level.

The 2D space carries no vertical metric, so the area element is reconstructed
from the level's `Δz`, matching
`horizontal_integral_at_boundary`. Widened pointwise for the same reason
as the volume integrands.
"""
boundary_areal_density(x, Δz) =
    BUDGET_ACCOUNTING_TYPE(x) / BUDGET_ACCOUNTING_TYPE(Δz) * 2

# ============================================================================
# Local integration
# ============================================================================

"""
    local_volume_integral(field)

Integrate `field` over the part of the domain this rank owns, in accounting
precision, without communicating.

`Fields.local_sum` applies the space's weighted Jacobian and sums the result. It
accepts a lazy `Base.Broadcast.broadcasted` expression, so the widening costs no
materialized copy: the widened element promotes the product with the geometric
weight and the accumulator follows.

# Ownership

Elements are partitioned across ranks, and `Fields.field_values` sees only the
elements this rank owns. Spectral-element quadrature weights are per element, so
a node shared between two elements is counted once with each element's own
weight, which is the correct integral and needs no ownership mask. DSS makes the
*values* at such a node agree between ranks; it does not duplicate the weights.
Summing every rank's `local_volume_integral` is the same decomposition
`Base.sum` uses, evaluated in the accounting type.
"""
local_volume_integral(field) =
    Fields.local_sum(Base.Broadcast.broadcasted(to_accounting, field))

"""
    local_boundary_integral(field)

Integrate a 2D spectral-element field over the part of the boundary this rank
owns, in accounting precision, without communicating.

The horizontal counterpart of `local_volume_integral`, reconstructing the
area element the same way `horizontal_integral_at_boundary` does.
"""
function local_boundary_integral(field)
    space = axes(field)
    @assert space isa Spaces.SpectralElementSpace2D
    return Fields.local_sum(
        Base.Broadcast.broadcasted(
            boundary_areal_density,
            field,
            Fields.Δz_field(space),
        ),
    )
end

"""
    reduce_accounting_sums!(context, values)

Sum `values` across every rank of `context`, in place, with **one** collective.

This is the only place the ledger communicates. Everything upstream of it is
local, and everything downstream reads a reduced buffer, which is what keeps the
cost at one collective per accepted step however many quantities and legs a step
carries.
"""
function reduce_accounting_sums!(
    context,
    values::Vector{BUDGET_ACCOUNTING_TYPE},
)
    ClimaComms.allreduce!(context, values, +)
    return values
end

"""
    budget_context(Y)

Return the communications context the ledger reduces over, taken from the state
it is measuring rather than from a global default.
"""
budget_context(Y) = ClimaComms.context(axes(Y.c))

# ============================================================================
# Atmospheric integrals
# ============================================================================

"""
    local_atmosphere_mass(Y)

Return this rank's share of the total modeled atmospheric mass, `∫ρ`, in kg.

`Y.c.ρ` is moist air density and carries the whole of `ρq_tot`, precipitating
water included.

`ρ` moves with `ρq_tot` on the paths that move air and water together: 0-moment
removal, sedimentation, the surface flux, the viscous sponge, and
`enforce_mass_energy_consistency!`. It does not move with it on the prescribed
forcing paths, which write `ρq_tot` and `ρe_tot` and no `ρ` term at all. See
`atmosphere_dry_mass`.
"""
local_atmosphere_mass(Y) = local_volume_integral(Y.c.ρ)

"""
    local_atmosphere_water(Y)

Return this rank's share of the total atmospheric water, `∫ρq_tot`, in kg.

`ρq_tot` is total water and already contains precipitation. The thermodynamic
state is built from `q_liq = q_lcl + q_rai` and `q_ice = q_icl + q_sno`, with
`q_tot ≥ q_liq + q_ice`, so rain and snow sit inside `q_tot` exactly as cloud
water does.

The category fields therefore **partition** this integral and are never added to
it. `ρq_lcl`, `ρq_icl`, `ρq_rai` and `ρq_sno` are all deliberately absent:
adding any of them would invent water every time cloud condensate became rain,
because one-moment microphysics moves the categories while applying no source at
all to `ρq_tot`.

Callers guard on the microphysics model rather than on field presence. See
`owns_atmosphere_water`.
"""
local_atmosphere_water(Y) = local_volume_integral(Y.c.ρq_tot)

"""
    local_atmosphere_energy(Y)

Return this rank's share of the total atmospheric energy, `∫ρe_tot`, in J.

Total energy is prognostic, so this is authoritative and nothing is
reconstructed from momentum and thermodynamic state. `ρtke`, the tagged tracers
`ρe_tag_*`, the source tags `ρe_src_*`, and the process records `prc_e_*` are
all excluded: they are diagnostics of the energy, not additional energy.
"""
local_atmosphere_energy(Y) = local_volume_integral(Y.c.ρe_tot)

"""
    local_atmosphere_dry_mass(Y)

Return this rank's share of the dry-air mass, `∫(ρ − ρq_tot)`, in kg.

Written as one integral rather than a difference of two totals so the
cancellation happens pointwise, where it is exact, instead of between two large
sums. See `atmosphere_dry_mass` for what the quantity is and is not.
"""
function local_atmosphere_dry_mass(Y)
    hasproperty(Y.c, :ρq_tot) || return local_atmosphere_mass(Y)
    return Fields.local_sum(
        Base.Broadcast.broadcasted(dry_air_density, Y.c.ρ, Y.c.ρq_tot),
    )
end

"""
    atmosphere_dry_mass(Y)

Return the dry-air mass `∫(ρ − ρq_tot)` over the whole domain, in kg.

This is a **diagnostic derived from state, not a parent quantity and not a
conservation invariant.** Prescribed forcing adds water to a column without
adding air to it, so dry air is not conserved in a forced run and this quantity
moves by minus the added water. Comparing it against the mass and water budgets
is how those two are checked against each other; it is not a closure the ledger
claims.

Equals the total mass for a dry model, which has no water state.

Performs one global reduction of its own. It is a diagnostic, not part of the
accounting path, so it is not in the packed per-step buffer.
"""
function atmosphere_dry_mass(Y)
    values = [local_atmosphere_dry_mass(Y)]
    reduce_accounting_sums!(budget_context(Y), values)
    return @inbounds values[1]
end

# ============================================================================
# Applicability
# ============================================================================
#
# Applicability of a parent quantity is decided by the configuration, never by
# field presence. The dry-mass diagnostic above is the one exception, and it is
# not a parent quantity. The state cannot answer these questions:
# `surface_prognostic_variables` builds a slab as `(; T, water = FT(0))` whatever
# the moisture model is, so `Y.sfc.water` exists in a dry run and holds a
# permanent zero. Dispatching on presence would report that zero as a measured
# budget, which is exactly the confusion between a measured zero and an
# inapplicable quantity that the ledger exists to prevent.

"""
    owns_atmosphere_water(microphysics_model) -> Bool

Return whether the atmosphere owns a water budget in this configuration.
"""
owns_atmosphere_water(::DryModel) = false
owns_atmosphere_water(::AbstractMicrophysicsModel) = true

"""
    has_surface_reservoir(surface_temperature) -> Bool

Return whether the configuration has a prognostic surface reservoir at all.

Only a `SurfaceConditions.SlabOceanTemperature` owns surface state. Every other
surface temperature is prescribed or diagnosed, which makes it exterior to the
model rather than a reservoir of it, so a flux into it is a boundary crossing.
"""
has_surface_reservoir(::SurfaceConditions.SurfaceTemperature) = false
has_surface_reservoir(::SurfaceConditions.SlabOceanTemperature) = true

"""
    owns_surface_water(surface_temperature, microphysics_model) -> Bool

Return whether the surface reservoir owns water, and therefore the mass that
goes with it.

Both arguments are required. A slab in a dry run has a `Y.sfc.water` field
holding a permanent zero, and a moist run without a slab has no surface state at
all, so neither question can be answered from one of them alone.
"""
owns_surface_water(surface_temperature, microphysics_model) =
    has_surface_reservoir(surface_temperature) &&
    owns_atmosphere_water(microphysics_model)

# ============================================================================
# Surface integrals
# ============================================================================

"""
    slab_heat_capacity(slab)

Return the areal heat capacity `ρ_ocean · cp_ocean · depth_ocean` of a
`SurfaceConditions.SlabOceanTemperature`, in J m⁻² K⁻¹.

Constant, which is what makes `local_surface_energy` linear in
`Y.sfc.T`.
"""
slab_heat_capacity(slab::SurfaceConditions.SlabOceanTemperature) =
    slab.ρ_ocean * slab.cp_ocean * slab.depth_ocean

"""
    local_surface_energy(Y, slab)

Return this rank's share of the energy held by the prognostic surface reservoir,
in J.

Only defined for a `SurfaceConditions.SlabOceanTemperature`. Callers check
`has_surface_reservoir` first; there is deliberately no method that
returns a zero for a surface that does not exist, because a zero is a budget the
identity would then have to reconcile.
"""
local_surface_energy(Y, slab::SurfaceConditions.SlabOceanTemperature) =
    local_boundary_integral(Y.sfc.T) * to_accounting(slab_heat_capacity(slab))

"""
    local_surface_water(Y, slab)

Return this rank's share of the water held by the prognostic surface reservoir,
in kg.

Callers check `owns_surface_water` first.

`Y.sfc.water` is an **accounting accumulator**, not physically complete
hydrology. It integrates what the atmosphere has delivered to the surface. It
has no runoff, no freezing, no soil or snow store, and nothing reads it back
into a surface flux, so it does not constrain evaporation. Treating it as a
reservoir is right for the accounting, because the water in it did leave the
atmosphere and a coupled control volume has to see where it went. Presenting it
as surface hydrology is not.
"""
local_surface_water(Y, ::SurfaceConditions.SlabOceanTemperature) =
    local_boundary_integral(Y.sfc.water)

"""
    local_surface_mass(Y, slab)

Return this rank's share of the mass held by the prognostic surface reservoir,
in kg.

The slab owns mass as well as water. What it gains left the atmosphere as
`ρq_tot`, and `ρ` carries the whole of `ρq_tot`, so the same deposition is a
mass leg and a water leg of the same size.

**These are two projections of one endpoint, not two measurements.** This
returns `local_surface_water` unchanged, so the two values cannot disagree and
no test of them can discover anything. `local_endpoint_packet` therefore sums
`Y.sfc.water` once and writes the same number into both slots, rather than
calling this and paying for the sum twice.

Independent measurement is a property of the *transfer legs*: the atmospheric
side of a surface exchange and the surface side of it are collected separately,
from different quadratures, and must be allowed to disagree. That is where a
coupling mismatch becomes visible. It is not a property of this endpoint, and a
leg is never created by negating its counterparty.
"""
local_surface_mass(Y, slab::SurfaceConditions.SlabOceanTemperature) =
    local_surface_water(Y, slab)
