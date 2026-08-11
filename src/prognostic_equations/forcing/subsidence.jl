#####
##### Tendencies due to prescribed subsidence
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators

"""
    subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, scheme::Val)

Add the subsidence tendency of the density-weighted scalar `ρχ`, i.e., vertical
advection by a prescribed large-scale vertical velocity.

The advective form `ρ (w ⋅ ∇χ)` is discretized as
`ᶜρ * (ᶜadvdivᵥ(flux) - ᶜχ * ᶜadvdivᵥ(ᶠu³))` and subtracted from `ᶜρχₜ`, so the
tendency added is `-ρ (w ⋅ ∇χ)`. The face reconstruction of the flux `wχ` depends
on `scheme`:

  - `Val{:none}()`: centered, `ᶠu³ * ᶠinterp(ᶜχ)`.
  - `Val{:first_order}()`: first-order upwind, `ᶠupwind1(ᶠu³, ᶜχ)`.
  - `Val{:third_order}()`: third-order upwind, `ᶠupwind3(ᶠu³, ᶜχ)`.

Both divergences use `ᶜadvdivᵥ`, which zeroes the flux through the top and bottom
faces. In the advective form this acts as a zero-gradient boundary condition: for
inflow through the lid (`w < 0` aloft, the usual subsidence case), the upwinded
top-face flux `w χ_top` and the compensating `χ_top w` term cancel exactly, so
zeroing both faces is equivalent to prescribing `χ = χ_top` above the lid and the
top-cell tendency vanishes; for outflow (`w > 0`) the top cell sees a one-sided
upwind derivative. For uniform `χ` the tendency vanishes at every level, so the
`q ≡ 1` tracer-mass consistency test holds structurally.

# Arguments

  - `ᶜρχₜ`: Tendency of the density-weighted scalar, modified in place.
  - `ᶜρ`: Cell-center density [kg/m³].
  - `ᶠu³`: Face contravariant vertical velocity of the prescribed subsidence;
    negative for subsidence in an upward `z` coordinate.
  - `ᶜχ`: Cell-center specific scalar being advected.
  - `scheme`: `Val` selecting the face reconstruction, as above.
"""
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:none}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜadvdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)) # Centered difference ρ * (-w * ∂χ/∂z)
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:first_order}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜadvdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)) # 1st-order upwind ρ * (-w * ∂χ/∂z)
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:third_order}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜadvdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)) # 3rd-order upwind ρ * (-w * ∂χ/∂z)


"""
    subsidence_tendency!(Yₜ, Y, p, t, subsidence)
    subsidence_tendency!(Yₜ, Y, p, t, subsidence::LargeScaleSubsidence)

Add the tendencies from a prescribed large-scale subsidence profile.

The method for `nothing` is a no-op. For a `LargeScaleSubsidence`, the profile
`subsidence.prof` is evaluated at the face heights to build the vertical velocity
field `ᶠsubsidence³`, and `subsidence!` is called with a first-order upwind scheme
for each advected quantity, incrementing:

  - `Yₜ.c.ρe_tot`, advecting the total enthalpy `ᶜh_tot`.
  - `Yₜ.c.ρq_tot`, unless the microphysics model is a `DryModel`.
  - `Yₜ.c.ρq_lcl` and `Yₜ.c.ρq_icl`, for a `NonEquilibriumMicrophysics` model. Rain
    and snow are not subsided.

Reads `Y.c.ρ`, the precomputed `ᶜh_tot`, and scratch space; `t` is unused. Called
from `additional_tendency!`. Returns `nothing`.
"""
subsidence_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing    # No subsidence

function subsidence_tendency!(Yₜ, Y, p, t, subsidence::LargeScaleSubsidence)
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
