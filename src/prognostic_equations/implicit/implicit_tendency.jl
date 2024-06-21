#####
##### Implicit tendencies
#####

import ClimaCore: Fields, Geometry

NVTX.@annotate function implicit_tendency!(Yₜ, Y, p, t)
    fill_with_nans!(p)
    Yₜ .= zero(eltype(Yₜ))
    implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    if p.atmos.sgs_adv_mode == Implicit()
        edmfx_sgs_vertical_advection_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.turbconv_model,
        )
    end

    if p.atmos.diff_mode == Implicit()
        vertical_diffusion_boundary_layer_tendency!(
            Yₜ,
            Y,
            p,
            t,
            p.atmos.vert_diff,
        )
        edmfx_sgs_diffusive_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    # NOTE: All ρa tendencies should be applied before calling this function
    pressure_work_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)

    # NOTE: This will zero out all monmentum tendencies in the edmfx advection test
    # please DO NOT add additional velocity tendencies after this function
    zero_velocity_tendency!(Yₜ, Y, p, t)

    # NOTE: This will zero out all tendencies
    # please DO NOT add additional tendencies after this function
    zero_tendency!(Yₜ, Y, p, t, p.atmos.tendency_model, p.atmos.turbconv_model)
    return nothing
end

# TODO: All of these should use dtγ instead of dt, but dtγ is not available in
# the implicit tendency function. Since dt >= dtγ, we can safely use dt for now.
# TODO: Can we rewrite ᶠfct_boris_book and ᶠfct_zalesak so that their broadcast
# expressions are less convoluted?
vertical_transport!(ᶜρχₜ, ᶜJ, ᶜρ, ᶠu³, ᶜχ, dt, upwinding::Val, ᶜdivᵥ) =
    vertical_transport!(1, ᶜρχₜ, ᶜJ, ᶜρ, ᶠu³, ᶜχ, dt, upwinding, ᶜdivᵥ)
vertical_transport!(ᶜρχₜ, ᶜJ, ᶜρ, ᶠu³, ᶜχ, dt, upwinding::Val) =
    vertical_transport!(1, ᶜρχₜ, ᶜJ, ᶜρ, ᶠu³, ᶜχ, dt, upwinding, ᶜadvdivᵥ)
vertical_transport!(
    coeff::Int,
    ᶜρχₜ,
    ᶜJ,
    ᶜρ,
    ᶠu³,
    ᶜχ,
    dt::Real,
    upwinding::Val,
) = vertical_transport!(coeff, ᶜρχₜ, ᶜJ, ᶜρ, ᶠu³, ᶜχ, dt, upwinding, ᶜadvdivᵥ)

vertical_transport!(coeff, ᶜρχₜ, ᶜJ, ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:none}, ᶜdivᵥ) =
    @. ᶜρχₜ += -coeff * (ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠu³ * ᶠinterp(ᶜχ)))
vertical_transport!(
    coeff,
    ᶜρχₜ,
    ᶜJ,
    ᶜρ,
    ᶠu³,
    ᶜχ,
    dt,
    ::Val{:first_order},
    ᶜdivᵥ,
) = @. ᶜρχₜ += -coeff * (ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(ᶠu³, ᶜχ)))
vertical_transport!(
    coeff,
    ᶜρχₜ,
    ᶜJ,
    ᶜρ,
    ᶠu³,
    ᶜχ,
    dt,
    ::Val{:third_order},
    ᶜdivᵥ,
) = @. ᶜρχₜ += -coeff * (ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind3(ᶠu³, ᶜχ)))
vertical_transport!(
    coeff,
    ᶜρχₜ,
    ᶜJ,
    ᶜρ,
    ᶠu³,
    ᶜχ,
    dt,
    ::Val{:boris_book},
    ᶜdivᵥ,
) = @. ᶜρχₜ +=
    -coeff * (ᶜdivᵥ(
        ᶠwinterp(ᶜJ, ᶜρ) * (
            ᶠupwind1(ᶠu³, ᶜχ) + ᶠfct_boris_book(
                ᶠupwind3(ᶠu³, ᶜχ) - ᶠupwind1(ᶠu³, ᶜχ),
                ᶜχ / dt - ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(ᶠu³, ᶜχ)) / ᶜρ,
            )
        ),
    ))
vertical_transport!(coeff, ᶜρχₜ, ᶜJ, ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:zalesak}, ᶜdivᵥ) =
    @. ᶜρχₜ +=
        -coeff * (ᶜdivᵥ(
            ᶠwinterp(ᶜJ, ᶜρ) * (
                ᶠupwind1(ᶠu³, ᶜχ) + ᶠfct_zalesak(
                    ᶠupwind3(ᶠu³, ᶜχ) - ᶠupwind1(ᶠu³, ᶜχ),
                    ᶜχ / dt,
                    ᶜχ / dt - ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(ᶠu³, ᶜχ)) / ᶜρ,
                )
            ),
        ))

vertical_advection!(ᶜρχₜ, ᶠu³, ᶜχ, ::Val{:none}) =
    @. ᶜρχₜ -= ᶜadvdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)
vertical_advection!(ᶜρχₜ, ᶠu³, ᶜχ, ::Val{:first_order}) =
    @. ᶜρχₜ -= ᶜadvdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)
vertical_advection!(ᶜρχₜ, ᶠu³, ᶜχ, ::Val{:third_order}) =
    @. ᶜρχₜ -= ᶜadvdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)

function implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    (; moisture_model, turbconv_model, rayleigh_sponge, precip_model) = p.atmos
    (; dt) = p
    n = n_mass_flux_subdomains(turbconv_model)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜh_tot, ᶜspecific, ᶠu³, ᶜp) = p.precomputed

    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠwinterp(ᶜJ, Y.c.ρ) * ᶠu³)

    # Central advection of active tracers (e_tot and q_tot)
    vertical_transport!(Yₜ.c.ρe_tot, ᶜJ, Y.c.ρ, ᶠu³, ᶜh_tot, dt, Val(:none))
    if !(moisture_model isa DryModel)
        vertical_transport!(
            Yₜ.c.ρq_tot,
            ᶜJ,
            Y.c.ρ,
            ᶠu³,
            ᶜspecific.q_tot,
            dt,
            Val(:none),
        )
    end

    if precip_model isa Microphysics1Moment
        # Advection of precipitation with the mean flow
        # is done with other passive tracers in the explicit tendency.
        # Here we add the advection with precipitation terminal velocity
        # using downward biasing and free outflow bottom boundary condition
        (; ᶜwᵣ, ᶜwₛ) = p.precomputed
        @. Yₜ.c.ρq_rai -= ᶜprecipdivᵥ(
            ᶠwinterp(ᶜJ, Y.c.ρ) *
            ᶠright_bias(Geometry.WVector(-(ᶜwᵣ)) * ᶜspecific.q_rai),
        )
        @. Yₜ.c.ρq_sno -= ᶜprecipdivᵥ(
            ᶠwinterp(ᶜJ, Y.c.ρ) *
            ᶠright_bias(Geometry.WVector(-(ᶜwₛ)) * ᶜspecific.q_sno),
        )
    end
    if precip_model isa Microphysics2Moment
        (; ᶜwᵣ, ᶜw_nᵣ) = p.precomputed
        @. Yₜ.c.ρq_rai -= ᶜprecipdivᵥ(
            ᶠwinterp(ᶜJ, Y.c.ρ) *
            ᶠright_bias(Geometry.WVector(-(ᶜwᵣ)) * ᶜspecific.q_rai),
        )
        @. Yₜ.c.ρn_rai -= ᶜprecipdivᵥ(
            ᶠwinterp(ᶜJ, Y.c.ρ) *
            ᶠright_bias(Geometry.WVector(-(ᶜw_nᵣ)) * ᶜspecific.n_rai),
        )
    end

    @. Yₜ.f.u₃ -= ᶠgradᵥ(ᶜp) / ᶠinterp(Y.c.ρ) + ᶠgradᵥ_ᶜΦ

    if rayleigh_sponge isa RayleighSponge
        (; ᶠβ_rayleigh_w) = p.rayleigh_sponge
        @. Yₜ.f.u₃ -= ᶠβ_rayleigh_w * Y.f.u₃
        if turbconv_model isa PrognosticEDMFX
            for j in 1:n
                @. Yₜ.f.sgsʲs.:($$j).u₃ -= ᶠβ_rayleigh_w * Y.f.sgsʲs.:($$j).u₃
            end
        end
    end
    return nothing
end
