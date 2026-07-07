#####
##### Tendencies due to prescribed subsidence
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators

"""
    subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, scheme::Val)

Computes the tendency contribution to a density-weighted scalar `ρχ` due to
subsidence (vertical advection by a prescribed large-scale vertical velocity `ᶠu³`).

This function is dispatched based on the `scheme` argument to use different
numerical methods for reconstructing the advective flux `wχ` at cell faces:

  - `::Val{:none}`: Uses a centered reconstruction (`ᶠu³ * ᶠinterp(ᶜχ)`).
  - `::Val{:first_order}`: Uses a first-order upwind reconstruction (`ᶠupwind1(ᶠu³, ᶜχ)`).
  - `::Val{:third_order}`: Uses a third-order upwind reconstruction (`ᶠupwind3(ᶠu³, ᶜχ)`).

The formulation `ᶜρ * (ᶜadvdivᵥ(Flux) - ᶜχ * ᶜadvdivᵥ(ᶠu³))` is equivalent to
`ᶜρ * (ᶠu³ ⋅ ∇ᶜχ)`, implementing the advective form. The result is subtracted
from `ᶜρχₜ`, effectively adding `ρ * (-ᶠu³ ⋅ ∇ᶜχ)` to it.

Both divergences use `ᶜadvdivᵥ`, which zeroes the flux through the top and
bottom faces. In the advective form this implements a zero-gradient boundary
condition: for inflow through the lid (`w < 0` aloft, the usual subsidence
case), the upwinded top-face flux `w χ_top` and the compensating `χ_top w`
term cancel exactly, so zeroing both faces is equivalent to prescribing
`χ = χ_top` above the lid, and the top-cell advective tendency vanishes. For
outflow (`w > 0`), the top cell sees a one-sided upwind derivative. For
uniform `χ`, the tendency vanishes identically at every level, so the
`q ≡ 1` tracer-mass consistency test holds structurally.

Arguments:

  - `ᶜρχₜ`: Field for the tendency of the density-weighted scalar `ρχ`, modified in place.
  - `ᶜρ`: Cell-center density field.
  - `ᶠu³`: Face-valued field of prescribed vertical velocity (subsidence velocity `w`).
    Typically, `w < 0` for subsidence in an upward `z` coordinate.
  - `ᶜχ`: Cell-center field of the specific scalar quantity `χ` being advected.
  - `scheme`: A `Val` type specifying the advection scheme (e.g., `Val{:first_order}()`).
"""
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:none}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜadvdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)) # Centered difference ρ * (-w * ∂χ/∂z)
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:first_order}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜadvdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)) # 1st-order upwind ρ * (-w * ∂χ/∂z)
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:third_order}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜadvdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)) # 3rd-order upwind ρ * (-w * ∂χ/∂z)


"""
    subsidence_tendency!(Yₜ, Y, p, t, subsidence_model::Subsidence)

Applies subsidence tendencies to total energy (`ρe_tot`), total specific humidity
(`ρq_tot`), and other moisture species (`ρq_lcl`, `ρq_icl`) if a `NonEquilibriumMicrophysics`
is used.

The subsidence velocity profile `w_sub(z)` is obtained from `subsidence_model.prof`.
This profile is used to construct a face-valued vertical velocity field `ᶠsubsidence³`.
The `subsidence!` helper function is then called (currently with a first-order
upwind scheme) to compute and apply the vertical advective tendency for each relevant
scalar quantity `χ`.

Arguments:

  - `Yₜ`: The tendency state vector, modified in place.
  - `Y`: The current state vector (used for `Y.c.ρ`).
  - `p`: Cache containing parameters, precomputed fields (`ᶜh_tot`),
    atmospheric model configurations (`p.atmos.microphysics_model`, `p.atmos.subsidence`),
    and scratch space.
  - `t`: Current simulation time (unused by this specific tendency calculation).
  - `subsidence_model`: A `Subsidence` object containing the subsidence profile function.

If `subsidence_model` is `Nothing`, no subsidence tendency is applied.
"""
subsidence_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing    # No subsidence

function subsidence_tendency!(Yₜ, Y, p, t, subsidence::Subsidence)
    (; microphysics_model) = p.atmos
    subsidence_profile = subsidence.prof
    (; ᶜh_tot) = p.precomputed

    ᶠz = Fields.coordinate_field(axes(Y.f)).z
    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶠsubsidence³ = p.scratch.ᶠtemp_CT3
    @. ᶠsubsidence³ =
        subsidence_profile(ᶠz) * CT3(unit_basis_vector_data(CT3, ᶠlg))

    # LS Subsidence
    subsidence!(Yₜ.c.ρe_tot, Y.c.ρ, ᶠsubsidence³, ᶜh_tot, Val{:first_order}())

    if !(microphysics_model isa DryModel)
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
        subsidence!(Yₜ.c.ρq_tot, Y.c.ρ, ᶠsubsidence³, ᶜq_tot, Val{:first_order}())
        if microphysics_model isa NonEquilibriumMicrophysics
            ᶜq_lcl = @. lazy(specific(Y.c.ρq_lcl, Y.c.ρ))
            subsidence!(
                Yₜ.c.ρq_lcl,
                Y.c.ρ,
                ᶠsubsidence³,
                ᶜq_lcl,
                Val{:first_order}(),
            )
            ᶜq_icl = @. lazy(specific(Y.c.ρq_icl, Y.c.ρ))
            subsidence!(
                Yₜ.c.ρq_icl,
                Y.c.ρ,
                ᶠsubsidence³,
                ᶜq_icl,
                Val{:first_order}(),
            )
        end
    end

    return nothing
end
