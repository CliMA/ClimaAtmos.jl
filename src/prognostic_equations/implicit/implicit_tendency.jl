#####
##### Implicit tendencies
#####

import ClimaCore
import ClimaCore: Fields, Geometry

import ClimaCore.MatrixFields: @name


NVTX.@annotate function implicit_tendency!(Yₜ, Y, p, t)
    fill_with_nans!(p)
    Yₜ .= zero(eltype(Yₜ))

    #@assert sum(isnan, Yₜ.c.ρq_rai) == 0
    #@assert sum(isnan, Y.c.ρq_rai) == 0

    for name in (@name(c.ρq_rai), @name(c.ρq_sno), @name(c.ρ), @name(c.ρq_liq), @name(c.ρq_ice), @name(c.ρq_tot), @name(c.uₕ), @name(f.u₃))
        if sum(isnan, parent(MatrixFields.get_field(Yₜ, name))) != 0
            @info("beginning of implicit_tendency! ", name)
        end
        if sum(isnan, parent(MatrixFields.get_field(Y, name))) != 0
            @info("beginning of implicit_tendency! ", name)
        end
    end

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
    for name in (@name(c.ρq_rai), @name(c.ρq_sno), @name(c.ρ), @name(c.ρq_liq), @name(c.ρq_ice), @name(c.ρq_tot), @name(c.uₕ), @name(f.u₃))
        if sum(isnan, parent(MatrixFields.get_field(Yₜ, name))) != 0
            @info("1: ", name)
        end
        if sum(isnan, parent(MatrixFields.get_field(Y, name))) != 0
            @info("1: ", name)
        end
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

    for name in (@name(c.ρq_rai), @name(c.ρq_sno), @name(c.ρ), @name(c.ρq_liq), @name(c.ρq_ice), @name(c.ρq_tot), @name(c.uₕ), @name(f.u₃))
        if sum(isnan, parent(MatrixFields.get_field(Yₜ, name))) != 0
            @info("2: ", name)
        end
        if sum(isnan, parent(MatrixFields.get_field(Y, name))) != 0
            @info("2: ", name)
        end
    end


    if p.atmos.sgs_entr_detr_mode == Implicit()
        edmfx_entr_detr_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    for name in (@name(c.ρq_rai), @name(c.ρq_sno), @name(c.ρ), @name(c.ρq_liq), @name(c.ρq_ice), @name(c.ρq_tot), @name(c.uₕ), @name(f.u₃))
        if sum(isnan, parent(MatrixFields.get_field(Yₜ, name))) != 0
            @info("3: ", name)
        end
        if sum(isnan, parent(MatrixFields.get_field(Y, name))) != 0
            @info("3: ", name)
        end
    end


    if p.atmos.sgs_mf_mode == Implicit()
        edmfx_sgs_mass_flux_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    for name in (@name(c.ρq_rai), @name(c.ρq_sno), @name(c.ρ), @name(c.ρq_liq), @name(c.ρq_ice), @name(c.ρq_tot), @name(c.uₕ), @name(f.u₃))
        if sum(isnan, parent(MatrixFields.get_field(Yₜ, name))) != 0
            @info("4: ", name)
        end
        if sum(isnan, parent(MatrixFields.get_field(Y, name))) != 0
            @info("4: ", name)
        end
    end


    if p.atmos.sgs_nh_pressure_mode == Implicit()
        edmfx_nh_pressure_drag_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    end
    for name in (@name(c.ρq_rai), @name(c.ρq_sno), @name(c.ρ), @name(c.ρq_liq), @name(c.ρq_ice), @name(c.ρq_tot), @name(c.uₕ), @name(f.u₃))
        if sum(isnan, parent(MatrixFields.get_field(Yₜ, name))) != 0
            @info("5: ", name)
        end
        if sum(isnan, parent(MatrixFields.get_field(Y, name))) != 0
            @info("5: ", name)
        end
    end


    # NOTE: All ρa tendencies should be applied before calling this function
    pressure_work_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
    for name in (@name(c.ρq_rai), @name(c.ρq_sno), @name(c.ρ), @name(c.ρq_liq), @name(c.ρq_ice), @name(c.ρq_tot), @name(c.uₕ), @name(f.u₃))
        if sum(isnan, parent(MatrixFields.get_field(Yₜ, name))) != 0
            @info("6: ", name)
        end
        if sum(isnan, parent(MatrixFields.get_field(Y, name))) != 0
            @info("6: ", name)
        end
    end


    # NOTE: This will zero out all momentum tendencies in the edmfx advection test
    # please DO NOT add additional velocity tendencies after this function
    zero_velocity_tendency!(Yₜ, Y, p, t)

    for name in (@name(c.ρq_rai), @name(c.ρq_sno), @name(c.ρ), @name(c.ρq_liq), @name(c.ρq_ice), @name(c.ρq_tot), @name(c.uₕ), @name(f.u₃))
        if sum(isnan, parent(MatrixFields.get_field(Yₜ, name))) != 0
            @info("end of implicit_tendency! ", name)
        end
        if sum(isnan, parent(MatrixFields.get_field(Y, name))) != 0
            @info("end of implicit_tendency! ", name)
        end
    end


    #@assert sum(isnan, Yₜ.c.ρq_rai) == 0
    #@assert sum(isnan, Y.c.ρq_rai) == 0

    return nothing
end

# TODO: All of these should use dtγ instead of dt, but dtγ is not available in
# the implicit tendency function. Since dt >= dtγ, we can safely use dt for now.
# TODO: Can we rewrite ᶠfct_boris_book and ᶠfct_zalesak so that their broadcast
# expressions are less convoluted?

function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:none})
    ᶜJ = Fields.local_geometry_field(ᶜρ).J
    ᶠJ = Fields.local_geometry_field(ᶠu³).J
    return @. lazy(-(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠu³ * ᶠinterp(ᶜχ))))
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:first_order})
    ᶜJ = Fields.local_geometry_field(ᶜρ).J
    ᶠJ = Fields.local_geometry_field(ᶠu³).J
    return @. lazy(-(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠupwind1(ᶠu³, ᶜχ))))
end
@static if pkgversion(ClimaCore) ≥ v"0.14.22"
    function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:vanleer_limiter})
        ᶜJ = Fields.local_geometry_field(ᶜρ).J
        ᶠJ = Fields.local_geometry_field(ᶠu³).J
        return @. lazy(
            -(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠlin_vanleer(ᶠu³, ᶜχ, dt))),
        )
    end
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:third_order})
    ᶜJ = Fields.local_geometry_field(ᶜρ).J
    ᶠJ = Fields.local_geometry_field(ᶠu³).J
    return @. lazy(-(ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠupwind3(ᶠu³, ᶜχ))))
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:boris_book})
    ᶜJ = Fields.local_geometry_field(ᶜρ).J
    ᶠJ = Fields.local_geometry_field(ᶠu³).J
    return @. lazy(
        -(ᶜadvdivᵥ(
            ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * (
                ᶠupwind1(ᶠu³, ᶜχ) + ᶠfct_boris_book(
                    ᶠupwind3(ᶠu³, ᶜχ) - ᶠupwind1(ᶠu³, ᶜχ),
                    ᶜχ / dt -
                    ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠupwind1(ᶠu³, ᶜχ)) / ᶜρ,
                )
            ),
        )),
    )
end
function vertical_transport(ᶜρ, ᶠu³, ᶜχ, dt, ::Val{:zalesak})
    ᶜJ = Fields.local_geometry_field(ᶜρ).J
    ᶠJ = Fields.local_geometry_field(ᶠu³).J
    return @. lazy(
        -(ᶜadvdivᵥ(
            ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * (
                ᶠupwind1(ᶠu³, ᶜχ) + ᶠfct_zalesak(
                    ᶠupwind3(ᶠu³, ᶜχ) - ᶠupwind1(ᶠu³, ᶜχ),
                    ᶜχ / dt,
                    ᶜχ / dt -
                    ᶜadvdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠupwind1(ᶠu³, ᶜχ)) / ᶜρ,
                )
            ),
        )),
    )
end

vertical_advection(ᶠu³, ᶜχ, ::Val{:none}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))
vertical_advection(ᶠu³, ᶜχ, ::Val{:first_order}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))
vertical_advection(ᶠu³, ᶜχ, ::Val{:third_order}) =
    @. lazy(-(ᶜadvdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))

function implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
    #@assert sum(isnan, Yₜ.c.ρq_rai) == 0
    #@assert sum(isnan, Y.c.ρq_rai) == 0

    (; moisture_model, turbconv_model, rayleigh_sponge, precip_model) = p.atmos
    (; dt) = p
    n = n_mass_flux_subdomains(turbconv_model)
    ᶜJ = Fields.local_geometry_field(Y.c).J
    ᶠJ = Fields.local_geometry_field(Y.f).J
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜh_tot, ᶜspecific, ᶠu³, ᶜp) = p.precomputed

    @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠu³)

    # Central vertical advection of active tracers (e_tot and q_tot)
    vtt = vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, dt, Val(:none))
    @. Yₜ.c.ρe_tot += vtt
    if !(moisture_model isa DryModel)
        vtt = vertical_transport(Y.c.ρ, ᶠu³, ᶜspecific.q_tot, dt, Val(:none))
        @. Yₜ.c.ρq_tot += vtt
    end

    # Vertical advection of passive tracers with the mean flow
    # is done in the explicit tendency.
    # Here we add the vertical advection with precipitation terminal velocity
    # using downward biasing and free outflow bottom boundary condition
    if moisture_model isa NonEquilMoistModel
        (; ᶜwₗ, ᶜwᵢ) = p.precomputed
        @. Yₜ.c.ρq_liq -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ *
            ᶠright_bias(Geometry.WVector(-(ᶜwₗ)) * max(zero(Y.c.ρ), Y.c.ρq_liq / Y.c.ρ)),
        )
        @. Yₜ.c.ρq_ice -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ *
            ᶠright_bias(Geometry.WVector(-(ᶜwᵢ)) * max(zero(Y.c.ρ), Y.c.ρq_ice / Y.c.ρ)),
        )
    end
    if precip_model isa Microphysics1Moment
        (; ᶜwᵣ, ᶜwₛ) = p.precomputed

        #@assert sum(isnan, Yₜ.c.ρq_rai) == 0
        #@assert sum(isnan, Y.c.ρq_rai) == 0
        #@assert sum(isnan, Y.c.ρ) == 0
        #@assert sum(isnan, ᶜwᵣ) == 0

        #@assert sum(isinf, Yₜ.c.ρq_rai) == 0
        #@assert sum(isinf, Y.c.ρq_rai) == 0
        #@assert sum(isinf, Y.c.ρ) == 0
        #@assert sum(isinf, ᶜwᵣ) == 0

        @. Yₜ.c.ρq_rai -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ *
            ᶠright_bias(Geometry.WVector(-(ᶜwᵣ)) * max(zero(Y.c.ρ), Y.c.ρq_rai / Y.c.ρ)),
        )
        @. Yₜ.c.ρq_sno -= ᶜprecipdivᵥ(
            ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ *
            ᶠright_bias(Geometry.WVector(-(ᶜwₛ)) * max(zero(Y.c.ρ), Y.c.ρq_sno / Y.c.ρ)),
        )

        #@assert sum(isnan, Yₜ.c.ρq_rai) == 0
    end

    # TODO - decide if this needs to be explicit or implicit
    #vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

    @. Yₜ.f.u₃ -= ᶠgradᵥ(ᶜp) / ᶠinterp(Y.c.ρ) + ᶠgradᵥ_ᶜΦ

    if rayleigh_sponge isa RayleighSponge
        ᶠz = Fields.coordinate_field(Y.f).z
        zmax = z_max(axes(Y.f))
        rs = rayleigh_sponge
        @. Yₜ.f.u₃ -= β_rayleigh_w(rs, ᶠz, zmax) * Y.f.u₃
        if turbconv_model isa PrognosticEDMFX
            for j in 1:n
                @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                    β_rayleigh_w(rs, ᶠz, zmax) * Y.f.sgsʲs.:($$j).u₃
            end
        end
    end
    return nothing
end
