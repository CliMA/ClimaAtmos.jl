#####
##### Implicit tendencies
#####

import ClimaCore: Fields

# TODO: All of these should use dtγ instead of dt, but dtγ is not available in
# the implicit tendency function. Since dt >= dtγ, we can safely use dt for now.
function vertical_transport!(ᶜρcₜ, ᶠu³, ᶜρ, ᶜρc, p, ::Val{:none})
    (; dt) = p.simulation
    ᶜJ = Fields.local_geometry_field(axes(ᶜρc)).J
    @. ᶜρcₜ = -(ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠu³ * ᶠinterp(ᶜρc / ᶜρ)))
end
function vertical_transport!(ᶜρcₜ, ᶠu³, ᶜρ, ᶜρc, p, ::Val{:first_order})
    (; dt) = p.simulation
    ᶜJ = Fields.local_geometry_field(axes(ᶜρc)).J
    @. ᶜρcₜ =
        -(ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠset_upwind_bcs(ᶠupwind1(ᶠu³, ᶜρc / ᶜρ))))
end
function vertical_transport!(ᶜρcₜ, ᶠu³, ᶜρ, ᶜρc, p, ::Val{:third_order})
    (; dt) = p.simulation
    ᶜJ = Fields.local_geometry_field(axes(ᶜρc)).J
    @. ᶜρcₜ =
        -(ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠset_upwind_bcs(ᶠupwind3(ᶠu³, ᶜρc / ᶜρ))))
end
function vertical_transport!(ᶜρcₜ, ᶠu³, ᶜρ, ᶜρc, p, ::Val{:boris_book})
    (; dt) = p.simulation
    ᶜJ = Fields.local_geometry_field(axes(ᶜρc)).J
    @. ᶜρcₜ = -(ᶜdivᵥ(
        ᶠwinterp(ᶜJ, ᶜρ) * ᶠset_upwind_bcs(
            ᶠupwind1(ᶠu³, ᶜρc / ᶜρ) + ᶠfct_boris_book(
                ᶠupwind3(ᶠu³, ᶜρc / ᶜρ) - ᶠupwind1(ᶠu³, ᶜρc / ᶜρ),
                (ᶜρc / dt - ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(ᶠu³, ᶜρc / ᶜρ))) / ᶜρ,
            ),
        ),
    ))
end
function vertical_transport!(ᶜρcₜ, ᶠu³, ᶜρ, ᶜρc, p, ::Val{:zalesak})
    (; dt) = p.simulation
    ᶜJ = Fields.local_geometry_field(axes(ᶜρc)).J
    @. ᶜρcₜ = -(ᶜdivᵥ(
        ᶠwinterp(ᶜJ, ᶜρ) * ᶠset_upwind_bcs(
            ᶠupwind1(ᶠu³, ᶜρc / ᶜρ) + ᶠfct_zalesak(
                ᶠupwind3(ᶠu³, ᶜρc / ᶜρ) - ᶠupwind1(ᶠu³, ᶜρc / ᶜρ),
                ᶜρc / ᶜρ / dt,
                (ᶜρc / dt - ᶜdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(ᶠu³, ᶜρc / ᶜρ))) / ᶜρ,
            ),
        ),
    ))
end

function implicit_vertical_advection_tendency!(Yₜ, Y, p, t, colidx)
    ᶜρ = Y.c.ρ
    (; ᶠgradᵥ_ᶜΦ, ᶜp, ᶠu³, ᶜρ_ref, ᶜp_ref) = p
    (; energy_upwinding, tracer_upwinding, density_upwinding) = p

    vertical_transport!(
        Yₜ.c.ρ[colidx],
        ᶠu³[colidx],
        ᶜρ[colidx],
        ᶜρ[colidx],
        p,
        density_upwinding,
    )

    if :ρθ in propertynames(Y.c)
        vertical_transport!(
            Yₜ.c.ρθ[colidx],
            ᶠu³[colidx],
            ᶜρ[colidx],
            Y.c.ρθ[colidx],
            p,
            energy_upwinding,
        )
    elseif :ρe_tot in propertynames(Y.c)
        ᶜρh = p.ᶜtemp_scalar
        @. ᶜρh[colidx] = Y.c.ρe_tot[colidx] + ᶜp[colidx]
        vertical_transport!(
            Yₜ.c.ρe_tot[colidx],
            ᶠu³[colidx],
            ᶜρ[colidx],
            ᶜρh[colidx],
            p,
            energy_upwinding,
        )
    end

    @. Yₜ.c.uₕ[colidx] = zero(Yₜ.c.uₕ[colidx])

    @. Yₜ.f.w[colidx] = -(
        ᶠgradᵥ(ᶜp[colidx] - ᶜp_ref[colidx]) / ᶠinterp(ᶜρ[colidx]) +
        (ᶠinterp(ᶜρ[colidx] - ᶜρ_ref[colidx])) / ᶠinterp(ᶜρ[colidx]) *
        ᶠgradᵥ_ᶜΦ[colidx]
    )
    if p.atmos.rayleigh_sponge isa RayleighSponge
        @. Yₜ.f.w[colidx] -= p.ᶠβ_rayleigh_w[colidx] * Y.f.w[colidx]
    end

    for ᶜρc_name in filter(is_tracer_var, propertynames(Y.c))
        ᶜρcₜ = getproperty(Yₜ.c, ᶜρc_name)
        ᶜρc = getproperty(Y.c, ᶜρc_name)
        vertical_transport!(
            ᶜρcₜ[colidx],
            ᶠu³[colidx],
            ᶜρ[colidx],
            ᶜρc[colidx],
            p,
            tracer_upwinding,
        )
    end
    return nothing
end
