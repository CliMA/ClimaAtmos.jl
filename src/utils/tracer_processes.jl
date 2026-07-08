#####
##### Process-based tracer classification
#####
##### Central definitions of which tracers participate in which physical
##### process (sedimentation, diffusion, vertical advection, entrainment),
##### and of per-tracer process properties (sedimentation velocity names,
##### condensate phase). All name lists are filtered by the fields actually
##### present in the state `Y`, so a tracer that is added to the state
##### automatically opts into the processes that act on it.
#####
##### Grid-scale names are relative to `Y.c` (e.g. `@name(ρq_rai)`), matching
##### `gs_tracer_names`. SGS names are relative to the first updraft
##### `Y.c.sgsʲs.:(1)` (e.g. `@name(q_rai)`), matching `sgs_tracer_names`.
##### Sedimentation velocity names are relative to `p.precomputed`.
#####

import Thermodynamics as TD
import ClimaCore.MatrixFields
import ClimaCore.MatrixFields: @name
import UnrolledUtilities: unrolled_filter, unrolled_map

# ============================================================================
# Sedimentation
# ============================================================================

"""
    sedimentation_velocity_name(ρχ_name)

`FieldName` (relative to `p.precomputed`) of the sedimentation/terminal
velocity associated with the grid-scale tracer `ρχ_name` (relative to `Y.c`),
or `nothing` if the tracer does not sediment.
"""
sedimentation_velocity_name(ρχ_name) =
    ρχ_name == @name(ρq_lcl) ? @name(ᶜwₗ) :
    ρχ_name == @name(ρq_icl) ? @name(ᶜwᵢ) :
    ρχ_name == @name(ρq_rai) ? @name(ᶜwᵣ) :
    ρχ_name == @name(ρq_sno) ? @name(ᶜwₛ) :
    ρχ_name == @name(ρn_lcl) ? @name(ᶜwₙₗ) :
    ρχ_name == @name(ρn_rai) ? @name(ᶜwₙᵣ) :
    ρχ_name == @name(ρn_ice) ? @name(ᶜwnᵢ) :
    ρχ_name == @name(ρq_rim) ? @name(ᶜwᵢ) :
    ρχ_name == @name(ρb_rim) ? @name(ᶜwᵢ) : nothing

"""
    sgs_sedimentation_velocity_name(χ_name)

`FieldName` (relative to `p.precomputed`) of the sedimentation/terminal
velocity in the first updraft associated with the SGS tracer `χ_name`
(relative to `Y.c.sgsʲs.:(1)`), or `nothing` if the tracer does not sediment.
"""
sgs_sedimentation_velocity_name(χ_name) =
    χ_name == @name(q_lcl) ? @name(ᶜwₗʲs.:(1)) :
    χ_name == @name(q_icl) ? @name(ᶜwᵢʲs.:(1)) :
    χ_name == @name(q_rai) ? @name(ᶜwᵣʲs.:(1)) :
    χ_name == @name(q_sno) ? @name(ᶜwₛʲs.:(1)) :
    χ_name == @name(n_lcl) ? @name(ᶜwₙₗʲs.:(1)) :
    χ_name == @name(n_rai) ? @name(ᶜwₙᵣʲs.:(1)) : nothing

# Candidate lists fix a deterministic order (masses, then number
# concentrations, then P3 rime quantities) for all process loops.
const gs_sedimenting_tracer_candidates = (
    @name(ρq_lcl),
    @name(ρq_icl),
    @name(ρq_rai),
    @name(ρq_sno),
    @name(ρn_lcl),
    @name(ρn_rai),
    @name(ρn_ice),
    @name(ρq_rim),
    @name(ρb_rim),
)
const gs_sedimenting_mass_candidates =
    (@name(ρq_lcl), @name(ρq_icl), @name(ρq_rai), @name(ρq_sno))

const sgs_sedimenting_tracer_candidates = (
    @name(q_lcl),
    @name(q_icl),
    @name(q_rai),
    @name(q_sno),
    @name(n_lcl),
    @name(n_rai),
)
const sgs_sedimenting_mass_candidates =
    (@name(q_lcl), @name(q_icl), @name(q_rai), @name(q_sno))

"""
    sedimenting_tracer_names(Y)

`Tuple` of `@name`s (relative to `Y.c`) of the grid-scale tracers in `Y` that
sediment with a precomputed terminal velocity (see
[`sedimentation_velocity_name`](@ref)).
"""
sedimenting_tracer_names(Y) =
    unrolled_filter(
        name -> MatrixFields.has_field(Y.c, name),
        gs_sedimenting_tracer_candidates,
    )

"""
    sedimenting_mass_names(Y)

`Tuple` of `@name`s (relative to `Y.c`) of the sedimenting condensate *mass*
tracers in `Y`. These are the tracers whose sedimentation also transports
total water and enthalpy, so they couple to `ρq_tot` and `ρe_tot`.
"""
sedimenting_mass_names(Y) =
    unrolled_filter(
        name -> MatrixFields.has_field(Y.c, name),
        gs_sedimenting_mass_candidates,
    )

"""
    sedimenting_sgs_tracer_names(Y)

`Tuple` of `@name`s (relative to `Y.c.sgsʲs.:(1)`) of the SGS tracers in the
first updraft that sediment with a precomputed terminal velocity (see
[`sgs_sedimentation_velocity_name`](@ref)). Returns `()` when prognostic EDMF
is not active.
"""
sedimenting_sgs_tracer_names(Y) =
    _sedimenting_sgs_names(
        Val(hasproperty(Y.c, :sgsʲs)),
        Y,
        sgs_sedimenting_tracer_candidates,
    )

"""
    sedimenting_sgs_mass_names(Y)

`Tuple` of `@name`s (relative to `Y.c.sgsʲs.:(1)`) of the sedimenting SGS
condensate *mass* tracers, whose sedimentation couples to the updraft `q_tot`.
Returns `()` when prognostic EDMF is not active.
"""
sedimenting_sgs_mass_names(Y) =
    _sedimenting_sgs_names(
        Val(hasproperty(Y.c, :sgsʲs)),
        Y,
        sgs_sedimenting_mass_candidates,
    )

_sedimenting_sgs_names(::Val{false}, Y, candidates) = ()
_sedimenting_sgs_names(::Val{true}, Y, candidates) =
    unrolled_filter(
        name -> MatrixFields.has_field(Y.c.sgsʲs.:(1), name),
        candidates,
    )

# ============================================================================
# Condensate phase properties
# ============================================================================

"""
    condensate_phase(ρχ_name_or_χ_name)

Thermodynamic phase (`TD.Liquid()` or `TD.Ice()`) of a condensate mass tracer,
given either its grid-scale name (relative to `Y.c`, e.g. `@name(ρq_rai)`) or
its SGS/specific name (e.g. `@name(q_rai)`). Returns `nothing` for tracers
that are not condensate masses (e.g. number concentrations).
"""
condensate_phase(name) =
    (name == @name(ρq_lcl) || name == @name(ρq_rai)) ? TD.Liquid() :
    (name == @name(ρq_icl) || name == @name(ρq_sno)) ? TD.Ice() :
    (name == @name(q_lcl) || name == @name(q_rai)) ? TD.Liquid() :
    (name == @name(q_icl) || name == @name(q_sno)) ? TD.Ice() : nothing

"""
    internal_energy_function(phase)

The `Thermodynamics` function that computes the specific internal energy of a
condensate with the given phase (`TD.Liquid()` or `TD.Ice()`).
"""
internal_energy_function(::TD.Liquid) = TD.internal_energy_liquid
internal_energy_function(::TD.Ice) = TD.internal_energy_ice

"""
    condensate_e_int_offset(phase, params)

Reference internal energy offset `e_int_χ0` of a condensate phase, used in
derivatives of pressure and internal energy with respect to condensate mass.
"""
condensate_e_int_offset(::TD.Liquid, params) = eltype(params)(CAP.e_int_v0(params))
condensate_e_int_offset(::TD.Ice, params) =
    eltype(params)(CAP.e_int_i0(params)) + eltype(params)(CAP.e_int_v0(params))

"""
    condensate_cv_difference(phase, params)

Difference between the isobaric specific heat of a condensate phase and the
isochoric specific heat of water vapor, `cp_χ - cv_v`, used in derivatives of
pressure and internal energy with respect to condensate mass.
"""
condensate_cv_difference(::TD.Liquid, params) =
    eltype(params)(CAP.cp_l(params) - CAP.cv_v(params))
condensate_cv_difference(::TD.Ice, params) =
    eltype(params)(CAP.cp_i(params) - CAP.cv_v(params))

# ============================================================================
# Other processes
# ============================================================================

"""
    advected_gs_scalar_names(Y)

`Tuple` of `@name`s (relative to `Y.c`) of the "active" grid-scale scalars
whose vertical advection with the grid-mean velocity is treated implicitly.
"""
advected_gs_scalar_names(Y) = (
    @name(ρ),
    @name(ρe_tot),
    (MatrixFields.has_field(Y.c, @name(ρq_tot)) ? (@name(ρq_tot),) : ())...,
)

"""
    microphysics_tracer_names(Y)

`Tuple` of `@name`s (relative to `Y.c`) of the grid-scale water tracers:
`ρq_tot` plus all sedimenting condensate tracers. These are the tracers that
participate in the implicit moisture processes (sedimentation, diffusion, and
SGS mass flux).
"""
microphysics_tracer_names(Y) = (
    (MatrixFields.has_field(Y.c, @name(ρq_tot)) ? (@name(ρq_tot),) : ())...,
    sedimenting_tracer_names(Y)...,
)

"""
    diffused_gs_scalar_names(Y)

`Tuple` of `@name`s (relative to `Y.c`) of the grid-scale scalars that are
diffused implicitly by the vertical eddy diffusivity (excluding `ρtke`, which
receives additional dissipation terms and is treated separately).
"""
diffused_gs_scalar_names(Y) = (@name(ρe_tot), microphysics_tracer_names(Y)...)

"""
    passive_sgs_tracer_names(Y)

`Tuple` of `@name`s (relative to `Y.c.sgsʲs.:(1)`) of the SGS tracers that do
not sediment (e.g. passive chemistry tracers like `q_gas_A`). Returns `()`
when prognostic EDMF is not active.
"""
passive_sgs_tracer_names(Y) =
    unrolled_filter(
        name -> isnothing(sgs_sedimentation_velocity_name(name)),
        sgs_tracer_names(Y),
    )

"""
    advected_sgs_scalar_names(Y)

`Tuple` of `@name`s (relative to `Y.c.sgsʲs.:(1)`) of all SGS scalars whose
vertical advection with the updraft velocity is treated implicitly:
sedimenting tracers, `q_tot`, `mse`, and passive tracers. Returns `()` when
prognostic EDMF is not active.
"""
advected_sgs_scalar_names(Y) =
    hasproperty(Y.c, :sgsʲs) ?
    (
        sedimenting_sgs_tracer_names(Y)...,
        @name(q_tot),
        @name(mse),
        passive_sgs_tracer_names(Y)...,
    ) : ()

# ============================================================================
# Name lifting to full state names
# ============================================================================

"""
    center_state_name(name)

Lift a `@name` relative to `Y.c` (e.g. `@name(ρq_rai)`) to the corresponding
full state name (e.g. `@name(c.ρq_rai)`), as used for `FieldMatrix` keys.
"""
center_state_name(name) = MatrixFields.append_internal_name(@name(c), name)

"""
    sgs_state_name(name)

Lift a `@name` relative to `Y.c.sgsʲs.:(1)` (e.g. `@name(q_rai)`) to the
corresponding full state name (e.g. `@name(c.sgsʲs.:(1).q_rai)`), as used for
`FieldMatrix` keys.
"""
sgs_state_name(name) =
    MatrixFields.append_internal_name(@name(c.sgsʲs.:(1)), name)
