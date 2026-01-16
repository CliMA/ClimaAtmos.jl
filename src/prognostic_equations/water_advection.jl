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
negated cell-centered specific flux term. The resulting face flux
is then diverged using the `ᶜprecipdivᵥ` operator.

Arguments:
- `Yₜ`: The tendency state vector, modified in place.
- `Y`: The current state vector.
- `p`: Cache containing parameters, precomputed fields (like `ᶜwₜqₜ`, `ᶜwₕhₜ`, `ᶜwₕʲs`, `ᶜwₜʲs`),
       and atmospheric model configurations.
- `t`: Current simulation time (not directly used in these calculations).

Modifies:
- `Yₜ.c.ρ`, `Yₜ.c.ρe_tot`, and `Yₜ.c.ρq_tot` (always when moisture is present)
"""
function vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

    (; params) = p
    (; ᶜΦ) = p.core
    (; ᶜu, ᶜts) = p.precomputed
    thp = CAP.thermodynamics_params(params)

    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J

    microphysics_tracers = (
        (@name(ρq_liq), @name(ᶜwₗ)),
        (@name(ρq_ice), @name(ᶜwᵢ)),
        (@name(ρq_rai), @name(ᶜwᵣ)),
        (@name(ρq_sno), @name(ᶜwₛ)),
    )
    MatrixFields.unrolled_foreach(microphysics_tracers) do (ρq_name, w_name)
        MatrixFields.has_field(Y.c, ρq_name) || return

        ᶜρq = MatrixFields.get_field(Y.c, ρq_name)
        ᶜw = MatrixFields.get_field(p.precomputed, w_name)
        vtt = @.lazy(
            -1 * ᶜprecipdivᵥ(
                ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
                    Geometry.WVector(-(ᶜw)) * specific(ᶜρq, Y.c.ρ),
                ),
            ),
        )
        @. Yₜ.c.ρ += vtt
        @. Yₜ.c.ρq_tot += vtt

        e_int_func = internal_energy_func(ρq_name)
        @. Yₜ.c.ρe_tot -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
                Geometry.WVector(-(ᶜw)) * specific(ᶜρq, Y.c.ρ) *
                (e_int_func(thp, ᶜts) + ᶜΦ + $(Kin(ᶜw, ᶜu))),
            ),
        )
    end

    # For prognostic edmf, augment the energy tendencies with the additional energy contributions
    # so that the total-grid energy flux remains consistent. Specifically, we enforce that the 
    # grid-mean energy flux equals the sum of the subdomain (updraft/environment) energy fluxes 
    # by accounting for the energy carried by sedimenting tracers.
    if p.atmos.turbconv_model isa PrognosticEDMFX
        (; ᶜρʲs, ᶜtsʲs, ᶜts⁰) = p.precomputed

        ᶜρ⁰ = @. lazy(TD.air_density(thp, ᶜts⁰))

        # TODO the following code works for only one updraft 
        sgs_microphysics_tracers = (
            (@name(q_liq), @name(ᶜwₗʲs.:(1)), @name(ᶜwₗ)),
            (@name(q_ice), @name(ᶜwᵢʲs.:(1)), @name(ᶜwᵢ)),
            (@name(q_rai), @name(ᶜwᵣʲs.:(1)), @name(ᶜwᵣ)),
            (@name(q_sno), @name(ᶜwₛʲs.:(1)), @name(ᶜwₛ)),
        )
        MatrixFields.unrolled_foreach(sgs_microphysics_tracers) do (q_name, wʲ_name, w_name)
            MatrixFields.has_field(Y.c.sgsʲs.:(1), q_name) || return

            ᶜqʲ = MatrixFields.get_field(Y.c.sgsʲs.:(1), q_name)
            ᶜwʲ = MatrixFields.get_field(p.precomputed, wʲ_name)
            ᶜρq = MatrixFields.get_field(Y.c, get_ρχ_name(q_name))
            ᶜw = MatrixFields.get_field(p.precomputed, w_name)

            e_int_func = internal_energy_func(q_name)
            # TODO do we need to add kinetic energy of subdomains?
            @. Yₜ.c.ρe_tot -=
                ᶜprecipdivᵥ(
                    ᶠinterp(ᶜρʲs.:(1) * ᶜJ) / ᶠJ * ᶠright_bias(
                        Geometry.WVector(-(ᶜwʲ)) *
                        draft_area(Y.c.sgsʲs.:(1).ρa, ᶜρʲs.:(1)) * ᶜqʲ *
                        (
                            e_int_func(thp, ᶜtsʲs.:(1)) - e_int_func(thp, ᶜts) -
                            $(Kin(ᶜw, ᶜu))
                        ),
                    ),
                )

            ᶜwaq⁰ = @. lazy((ᶜρq * ᶜw - Y.c.sgsʲs.:(1).ρa * ᶜqʲ * ᶜwʲ) / ᶜρ⁰)
            @. Yₜ.c.ρe_tot -=
                -1 * ᶜprecipdivᵥ(
                    ᶠinterp(ᶜρ⁰ * ᶜJ) / ᶠJ * ᶠright_bias(
                        Geometry.WVector(-(ᶜwaq⁰)) *
                        (e_int_func(thp, ᶜts⁰) - e_int_func(thp, ᶜts) - $(Kin(ᶜw, ᶜu))),
                    ),
                )
        end
    end

    return nothing
end
