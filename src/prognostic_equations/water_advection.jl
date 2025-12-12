#####
##### Applies tendencies to density, total energy, and total specific humidity 
##### due to the sedimentation/precipitation of water and its associated enthalpy
#####


import ClimaCore: Fields

"""
    vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

Computes and applies tendencies to grid-mean density (`ρ`), total energy (`ρe_tot`),
and total specific humidity (`ρq_tot`) due to sedimentation/precipitation of total water
and its associated enthalpy.

This function is active only if the atmospheric model includes moisture (i.e.,
is not a `DryModel`). The tendencies are calculated in a conservative form,
`-∇ ⋅ F`, where `F` represents the vertical fluxes.

Fluxes at cell faces are reconstructed using a first-order upwind scheme.
Specifically, face-valued density (`ᶠρ`) is multiplied by a right-biased,
negated cell-centered specific flux term (e.g., `ᶠright_bias(-ᶜwₜqₜ)` for total water,
where `ᶜwₜqₜ` represents `w q_t` at cell centers). The resulting face flux
is then diverged using the `ᶜprecipdivᵥ` operator.

The precomputed terms `p.precomputed.ᶜwₜqₜ` and `p.precomputed.ᶜwₕhₜ` represent 
cell-centered specific vertical fluxes (e.g., `w ⋅ q_t` and `w ⋅ h_t`,
respectively, where `w` is a vertical velocity component).

Additionally, when using prognostic EDMF with non-equilibrium moisture and microphysics,
this function also applies advective tendencies to the subdomain moist static energy
(`mse`) and total specific humidity (`q_tot`) fields within each mass flux
subdomain. These tendencies use sedimentation velocities computed from the precomputed
fields `ᶜwₕʲs` and `ᶜwₜʲs` for each subdomain j.

Arguments:
- `Yₜ`: The tendency state vector, modified in place.
- `Y`: The current state vector.
- `p`: Cache containing parameters, precomputed fields (like `ᶜwₜqₜ`, `ᶜwₕhₜ`, `ᶜwₕʲs`, `ᶜwₜʲs`),
       and atmospheric model configurations.
- `t`: Current simulation time (not directly used in these calculations).

Modifies:
- `Yₜ.c.ρ`, `Yₜ.c.ρe_tot`, and `Yₜ.c.ρq_tot` (always when moisture is present)
- `Yₜ.c.sgsʲs.:(j).mse` and `Yₜ.c.sgsʲs.:(j).q_tot` (when using prognostic EDMFX with 
  non-equilibrium moisture and multi-moment microphysics)
"""
function vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

    ᶠρ = ᶠface_density(Y.c.ρ)
    (; ᶜwₜqₜ, ᶜwₕhₜ) = p.precomputed

    if !(p.atmos.moisture_model isa DryModel)
        @. Yₜ.c.ρ -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-(ᶜwₜqₜ)))
        @. Yₜ.c.ρe_tot -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-(ᶜwₕhₜ)))
        @. Yₜ.c.ρq_tot -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(-(ᶜwₜqₜ)))
    end

    return nothing
end
