#####
##### Implicit tendencies
#####

import ClimaCore: Fields, Geometry

function implicit_tendency!(Yₜ, Y, p, t)
    p.test isa TestDycoreConsistency && fill_with_nans!(p)
    @nvtx "implicit tendency" color = colorant"yellow" begin
        Yₜ .= zero(eltype(Yₜ))
        @nvtx "precomputed quantities" color = colorant"orange" begin
            set_precomputed_quantities!(Y, p, t)
        end
        Fields.bycolumn(axes(Y.c)) do colidx
            implicit_vertical_advection_tendency!(Yₜ, Y, p, t, colidx)
            if p.turbconv_model isa TurbulenceConvection.EDMFModel
                implicit_sgs_flux_tendency!(
                    Yₜ,
                    Y,
                    p,
                    t,
                    colidx,
                    p.turbconv_model,
                )
            end
            # NOTE: All ρa tendencies should be applied before calling this function
            pressure_work_tendency!(Yₜ, Y, p, t, colidx, p.atmos.turbconv_model)

            # NOTE: This will zero out all monmentum tendencies in the edmfx advection test
            # please DO NOT add additional velocity tendencies after this function
            zero_velocity_tendency!(Yₜ, Y, p, t, colidx)
        end
    end
end

# TODO: All of these should use dtγ instead of dt, but dtγ is not available in
# the implicit tendency function. Since dt >= dtγ, we can safely use dt for now.
# TODO: Can we rewrite ᶠfct_boris_book and ᶠfct_zalesak so that their broadcast
# expressions are less convoluted?
vertical_transport!(ᶜρχₜ, ᶜJ, ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:none}) =
    @. ᶜρχₜ += -(ᶜadvdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠu³ * ᶠinterp(ᶜχ)))
vertical_transport!(ᶜρχₜ, ᶜJ, ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:first_order}) =
    @. ᶜρχₜ += -(ᶜadvdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(ᶠu³, ᶜχ)))
vertical_transport!(ᶜρχₜ, ᶜJ, ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:third_order}) =
    @. ᶜρχₜ += -(ᶜadvdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind3(ᶠu³, ᶜχ)))
vertical_transport!(ᶜρχₜ, ᶜJ, ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:boris_book}) =
    @. ᶜρχₜ += -(ᶜadvdivᵥ(
        ᶠwinterp(ᶜJ, ᶜρ) * (
            ᶠupwind1(ᶠu³, ᶜχ) + ᶠfct_boris_book(
                ᶠupwind3(ᶠu³, ᶜχ) - ᶠupwind1(ᶠu³, ᶜχ),
                ᶜχ / dt - ᶜadvdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(ᶠu³, ᶜχ)) / ᶜρ,
            )
        ),
    ))
vertical_transport!(ᶜρχₜ, ᶜJ, ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:zalesak}) =
    @. ᶜρχₜ += -(ᶜadvdivᵥ(
        ᶠwinterp(ᶜJ, ᶜρ) * (
            ᶠupwind1(ᶠu³, ᶜχ) + ᶠfct_zalesak(
                ᶠupwind3(ᶠu³, ᶜχ) - ᶠupwind1(ᶠu³, ᶜχ),
                ᶜχ / dt,
                ᶜχ / dt - ᶜadvdivᵥ(ᶠwinterp(ᶜJ, ᶜρ) * ᶠupwind1(ᶠu³, ᶜχ)) / ᶜρ,
            )
        ),
    ))

function implicit_vertical_advection_tendency!(Yₜ, Y, p, t, colidx)
    (; energy_upwinding, tracer_upwinding, density_upwinding, edmfx_upwinding) =
        p
    (; turbconv_model, rayleigh_sponge) = p.atmos
    (; dt) = p.simulation
    n = n_mass_flux_subdomains(turbconv_model)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    (; ᶜspecific, ᶠu³, ᶜp, ᶠgradᵥ_ᶜΦ, ᶜρ_ref, ᶜp_ref, ᶜtemp_scalar) = p
    if turbconv_model isa EDMFX
        (; ᶜspecific⁰, ᶜρa⁰, ᶠu³⁰, ᶜspecificʲs, ᶠu³ʲs, ᶜρʲs) = p
    end

    ᶜ1 = ᶜtemp_scalar
    @. ᶜ1[colidx] = one(Y.c.ρ[colidx])
    vertical_transport!(
        Yₜ.c.ρ[colidx],
        ᶜJ[colidx],
        Y.c.ρ[colidx],
        ᶠu³[colidx],
        ᶜ1[colidx],
        dt,
        density_upwinding,
    )
    if turbconv_model isa EDMFX
        for j in 1:n
            vertical_transport!(
                Yₜ.c.sgsʲs.:($j).ρa[colidx],
                ᶜJ[colidx],
                Y.c.sgsʲs.:($j).ρa[colidx],
                ᶠu³ʲs.:($j)[colidx],
                ᶜ1[colidx],
                dt,
                edmfx_upwinding,
            )
        end
    end

    if :ρe_tot in propertynames(Yₜ.c)
        (; ᶜh_tot) = p
        vertical_transport!(
            Yₜ.c.ρe_tot[colidx],
            ᶜJ[colidx],
            Y.c.ρ[colidx],
            ᶠu³[colidx],
            ᶜh_tot[colidx],
            dt,
            energy_upwinding,
        )
    end
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        vertical_transport!(
            ᶜρχₜ[colidx],
            ᶜJ[colidx],
            Y.c.ρ[colidx],
            ᶠu³[colidx],
            ᶜχ[colidx],
            dt,
            χ_name == :θ ? energy_upwinding : tracer_upwinding,
        )
    end
    if turbconv_model isa EDMFX
        for (ᶜρaχ⁰ₜ, ᶜχ⁰, _) in matching_subfields(Yₜ.c.sgs⁰, ᶜspecific⁰)
            vertical_transport!(
                ᶜρaχ⁰ₜ[colidx],
                ᶜJ[colidx],
                ᶜρa⁰[colidx],
                ᶠu³⁰[colidx],
                ᶜχ⁰[colidx],
                dt,
                edmfx_upwinding,
            )
        end
    end
    if turbconv_model isa EDMFX
        for j in 1:n
            if :ρae_tot in propertynames(Yₜ.c.sgsʲs.:($j))
                (; ᶜh_totʲs) = p
                vertical_transport!(
                    Yₜ.c.sgsʲs.:($j).ρae_tot[colidx],
                    ᶜJ[colidx],
                    Y.c.sgsʲs.:($j).ρa[colidx],
                    ᶠu³ʲs.:($j)[colidx],
                    ᶜh_totʲs.:($j)[colidx],
                    dt,
                    edmfx_upwinding,
                )
            end
            for (ᶜρaχʲₜ, ᶜχʲ, χ_name) in
                matching_subfields(Yₜ.c.sgsʲs.:($j), ᶜspecificʲs.:($j))
                χ_name == :e_tot && continue
                vertical_transport!(
                    ᶜρaχʲₜ[colidx],
                    ᶜJ[colidx],
                    Y.c.sgsʲs.:($j).ρa[colidx],
                    ᶠu³ʲs.:($j)[colidx],
                    ᶜχʲ[colidx],
                    dt,
                    edmfx_upwinding,
                )
            end
        end
    end

    @. Yₜ.c.uₕ[colidx] = zero(Yₜ.c.uₕ[colidx])

    @. Yₜ.f.u₃[colidx] =
        -(
            ᶠgradᵥ(ᶜp[colidx] - ᶜp_ref[colidx]) +
            ᶠinterp(Y.c.ρ[colidx] - ᶜρ_ref[colidx]) * ᶠgradᵥ_ᶜΦ[colidx]
        ) / ᶠinterp(Y.c.ρ[colidx])
    if turbconv_model isa EDMFX
        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] =
                -(
                    ᶠgradᵥ(ᶜp[colidx] - ᶜp_ref[colidx]) +
                    ᶠinterp(ᶜρʲs.:($$j)[colidx] - ᶜρ_ref[colidx]) *
                    ᶠgradᵥ_ᶜΦ[colidx]
                ) / ᶠinterp(ᶜρʲs.:($$j)[colidx])
        end
    end

    if rayleigh_sponge isa RayleighSponge
        (; ᶠβ_rayleigh_w) = p
        @. Yₜ.f.u₃[colidx] -= ᶠβ_rayleigh_w[colidx] * Y.f.u₃[colidx]
        if turbconv_model isa EDMFX
            for j in 1:n
                @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] -=
                    ᶠβ_rayleigh_w[colidx] * Y.f.sgsʲs.:($$j).u₃[colidx]
            end
        end
    end
end
