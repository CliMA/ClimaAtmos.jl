#####
##### Implicit tendencies
#####

import ClimaCore: Fields, Geometry

NVTX.@annotate function implicit_tendency!(Yₜ, Y, p, t)
    fill_with_nans!(p)
    Yₜ .= zero(eltype(Yₜ))
    Fields.bycolumn(axes(Y.c)) do colidx
        implicit_vertical_advection_tendency!(Yₜ, Y, p, t, colidx)
        if p.atmos.diff_mode == Implicit()
            vertical_diffusion_boundary_layer_tendency!(
                Yₜ,
                Y,
                p,
                t,
                colidx,
                p.atmos.vert_diff,
            )
            edmfx_sgs_diffusive_flux_tendency!(
                Yₜ,
                Y,
                p,
                t,
                colidx,
                p.atmos.turbconv_model,
            )
        end
        # NOTE: All ρa tendencies should be applied before calling this function
        pressure_work_tendency!(Yₜ, Y, p, t, colidx, p.atmos.turbconv_model)

        # NOTE: This will zero out all monmentum tendencies in the edmfx advection test
        # please DO NOT add additional velocity tendencies after this function
        zero_velocity_tendency!(Yₜ, Y, p, t, colidx)
    end
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

function implicit_vertical_advection_tendency!(Yₜ, Y, p, t, colidx)
    (; moisture_model, turbconv_model, rayleigh_sponge, precip_model) = p.atmos
    (; dt) = p
    n = n_mass_flux_subdomains(turbconv_model)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; ᶠgradᵥ_ᶜΦ, ᶜρ_ref, ᶜp_ref) = p.core
    (; ᶜh_tot, ᶜspecific, ᶠu³, ᶜp) = p.precomputed
    (; precip_upwinding) = p.atmos.numerics

    @. Yₜ.c.ρ[colidx] -=
        ᶜdivᵥ(ᶠwinterp(ᶜJ[colidx], Y.c.ρ[colidx]) * ᶠu³[colidx])

    # Central advection of active tracers (e_tot and q_tot)
    vertical_transport!(
        Yₜ.c.ρe_tot[colidx],
        ᶜJ[colidx],
        Y.c.ρ[colidx],
        ᶠu³[colidx],
        ᶜh_tot[colidx],
        dt,
        Val(:none),
    )
    if !(moisture_model isa DryModel)
        vertical_transport!(
            Yₜ.c.ρq_tot[colidx],
            ᶜJ[colidx],
            Y.c.ρ[colidx],
            ᶠu³[colidx],
            ᶜspecific.q_tot[colidx],
            dt,
            Val(:none),
        )
    end

    if precip_model isa Microphysics1Moment
        # Advection of precipitation with the mean flow
        # is done with other passive tracers in the explicit tendency.
        # Here we add the advection with precipitation terminal velocity
        # using first order upwind and free outflow bottom boundary condition

        ᶠu³ₚ = p.scratch.ᶠtemp_CT3
        ᶠlg = Fields.local_geometry_field(Y.f)

        @. ᶠu³ₚ[colidx] =
            ᶠinterp(-p.precomputed.ᶜwᵣ[colidx]) *
            CT3(unit_basis_vector_data(CT3, ᶠlg[colidx]))
        vertical_transport!(
            Yₜ.c.ρq_rai[colidx],
            ᶜJ[colidx],
            Y.c.ρ[colidx],
            ᶠu³ₚ[colidx],
            ᶜspecific.q_rai[colidx],
            dt,
            precip_upwinding,
            ᶜprecipdivᵥ,
        )

        @. ᶠu³ₚ[colidx] =
            ᶠinterp(-p.precomputed.ᶜwₛ[colidx]) *
            CT3(unit_basis_vector_data(CT3, ᶠlg[colidx]))
        vertical_transport!(
            Yₜ.c.ρq_sno[colidx],
            ᶜJ[colidx],
            Y.c.ρ[colidx],
            ᶠu³ₚ[colidx],
            ᶜspecific.q_sno[colidx],
            dt,
            precip_upwinding,
            ᶜprecipdivᵥ,
        )
    end

    @. Yₜ.f.u₃[colidx] +=
        -(
            ᶠgradᵥ(ᶜp[colidx] - ᶜp_ref[colidx]) +
            ᶠinterp(Y.c.ρ[colidx] - ᶜρ_ref[colidx]) * ᶠgradᵥ_ᶜΦ[colidx]
        ) / ᶠinterp(Y.c.ρ[colidx])

    if rayleigh_sponge isa RayleighSponge
        (; ᶠβ_rayleigh_w) = p.rayleigh_sponge
        @. Yₜ.f.u₃[colidx] -= ᶠβ_rayleigh_w[colidx] * Y.f.u₃[colidx]
        if turbconv_model isa PrognosticEDMFX
            for j in 1:n
                @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] -=
                    ᶠβ_rayleigh_w[colidx] * Y.f.sgsʲs.:($$j).u₃[colidx]
            end
        end
    end
    return nothing
end
