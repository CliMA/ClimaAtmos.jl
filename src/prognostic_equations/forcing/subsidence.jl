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

The formulation `ᶜρ * (ᶜsubdivᵥ(Flux) - ᶜχ * ᶜsubdivᵥ(ᶠu³))` is equivalent to
`ᶜρ * (ᶠu³ ⋅ ∇ᶜχ)`, implementing the advective form. The result is subtracted
from `ᶜρχₜ`, effectively adding `ρ * (-ᶠu³ ⋅ ∇ᶜχ)` to it.

Arguments:
- `ᶜρχₜ`: Field for the tendency of the density-weighted scalar `ρχ`, modified in place.
- `ᶜρ`: Cell-center density field.
- `ᶠu³`: Face-valued field of prescribed vertical velocity (subsidence velocity `w`).
        Typically, `w < 0` for subsidence in an upward `z` coordinate.
- `ᶜχ`: Cell-center field of the specific scalar quantity `χ` being advected.
- `scheme`: A `Val` type specifying the advection scheme (e.g., `Val{:first_order}()`).
"""
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:none}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜsubdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³)) # Centered difference ρ * (-w * ∂χ/∂z)
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:first_order}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜsubdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³)) # 1st-order upwind ρ * (-w * ∂χ/∂z)
subsidence!(ᶜρχₜ, ᶜρ, ᶠu³, ᶜχ, ::Val{:third_order}) =
    @. ᶜρχₜ -= ᶜρ * (ᶜsubdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜsubdivᵥ(ᶠu³)) # 3rd-order upwind ρ * (-w * ∂χ/∂z)


"""
    subsidence_tendency!(Yₜ, Y, p, t, subsidence_model::Subsidence)

Applies subsidence tendencies to total energy (`ρe_tot`), total specific humidity
(`ρq_tot`), and other moisture species (`ρq_liq`, `ρq_ice`) if a `NonEquilibriumMicrophysics`
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


"""
    subsidence_tendency!(Yₜ, Y, p, t, subsidence_model::Subsidence)

Applies subsidence tendencies to total energy (`ρe_tot`), total specific humidity
(`ρq_tot`), and other moisture species (`ρq_liq`, `ρq_ice`) if a `NonEquilibriumMicrophysics`
is used.

The subsidence velocity profile `w_sub(z)` is obtained from `subsidence_model.prof`.
This profile is used to construct a face-valued vertical velocity field `ᶠsubsidence³`.
The `subsidence!` helper function is then called (currently with a first-order
upwind scheme) to compute and apply the vertical advective tendency for each relevant 
scalar quantity `χ`.

Arguments:
- `Yₜ`: The tendency state vector, modified in place.
- `Y`: The current state vector, used for density (`ρ`).
- `p`: Cache containing parameters, and the subsidence model object.
- `t`: Current simulation time.
- `subsidence`: The subsidence model object.
"""
function subsidence_tendency!(Yₜ, Y, p, t, ::Subsidence)
    (; microphysics_model) = p.atmos
    subsidence_profile = p.atmos.subsidence.prof
    thermo_params = CAP.thermodynamics_params(p.params)
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
            ᶜq_liq = @. lazy(specific(Y.c.ρq_liq, Y.c.ρ))
            subsidence!(
                Yₜ.c.ρq_liq,
                Y.c.ρ,
                ᶠsubsidence³,
                ᶜq_liq,
                Val{:first_order}(),
            )
            ᶜq_ice = @. lazy(specific(Y.c.ρq_ice, Y.c.ρ))
            subsidence!(
                Yₜ.c.ρq_ice,
                Y.c.ρ,
                ᶠsubsidence³,
                ᶜq_ice,
                Val{:first_order}(),
            )
        end
    end

    return nothing
end
