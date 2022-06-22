hyperdiffusion_cache(
    Y;
    κ₄ = FT(0),
    divergence_damping_factor = FT(1),
    use_tempest_mode = false,
) =
    (:ρq_tot in propertynames(Y.c)) ?
    merge(
        (;
            ᶜχ = similar(Y.c, FT),
            ᶜχρq_tot = similar(Y.c, FT),
            ᶜχuₕ = similar(Y.c, Geometry.Covariant12Vector{FT}),
            κ₄,
            divergence_damping_factor,
            use_tempest_mode,
        ),
        use_tempest_mode ? (; ᶠχw_data = similar(Y.F, FT)) : NamedTuple(),
    ) :
    merge(
        (;
            ᶜχ = similar(Y.c, FT),
            ᶜχuₕ = similar(Y.c, Geometry.Covariant12Vector{FT}),
            κ₄,
            divergence_damping_factor,
            use_tempest_mode,
        ),
        use_tempest_mode ? (; ᶠχw_data = similar(Y.F, FT)) : NamedTuple(),
    )

function hyperdiffusion_tendency!(Yₜ, Y, P, t)
    return P.use_tempest_mode ? hyperdiffusion_tendency_tempest!(Yₜ, Y, P, t) :
           hyperdiffusion_tendency_clima!(Yₜ, Y, P, t)
end

function hyperdiffusion_tendency_clima!(Yₜ, Y, p, t)
    NVTX.isactive() && (
        profile_hyperdiffusion_tendency = NVTX.range_start(;
            message = "hyperdiffusion tendency",
            color = colorant"yellow",
        )
    )
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    (; ᶜp, ᶜχ, ᶜχuₕ) = p # assume ᶜp has been updated
    (; ghost_buffer, κ₄, divergence_damping_factor, use_tempest_mode) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)
    is_ρq_tot = :ρq_tot in propertynames(Y.c)
    is_2d_pt = point_type <: Geometry.Abstract2DPoint
    is_3d_pt = !is_2d_pt

    ᶜρs =
        :ρe_tot in propertynames(Y.c) ? Y.c.ρe_tot :
        (:ρe_int in propertynames(Y.c) ? Y.c.ρe_int : Y.c.ρθ)
    ᵗρs =
        :ρe_tot in propertynames(Y.c) ? Yₜ.c.ρe_tot :
        (:ρe_int in propertynames(Y.c) ? Yₜ.c.ρe_int : Yₜ.c.ρθ)

    (:ρθ in propertynames(Y.c)) && (@. ᶜχ = wdivₕ(gradₕ(ᶜρs / ᶜρ)))
    !(:ρθ in propertynames(Y.c)) && (@. ᶜχ = wdivₕ(gradₕ((ᶜρs + ᶜp) / ᶜρ)))

    if is_ρq_tot
        (; ᶜχρq_tot) = p
        @. ᶜχρq_tot = wdivₕ(gradₕ(Y.c.ρq_tot / ᶜρ))
    end

    is_3d_pt && (@. ᶜχuₕ =
        wgradₕ(divₕ(ᶜuₕ)) - Geometry.Covariant12Vector(
            wcurlₕ(Geometry.Covariant3Vector(curlₕ(ᶜuₕ))),
        ))
    is_2d_pt && (@. ᶜχuₕ = Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜuₕ))))

    NVTX.isactive() && (
        dss_hyperdiffusion_tendency = NVTX.range_start(;
            message = "dss_hyperdiffusion_tendency",
            color = colorant"green",
        )
    )
    Spaces.weighted_dss_start!(ᶜχ, ghost_buffer.χ)
    is_ρq_tot && (Spaces.weighted_dss_start!(ᶜχρq_tot, ghost_buffer.ᶜχρq_tot))
    Spaces.weighted_dss_start!(ᶜχuₕ, ghost_buffer.χuₕ)

    Spaces.weighted_dss_internal!(ᶜχ, ghost_buffer.χ)
    is_ρq_tot &&
        (Spaces.weighted_dss_internal!(ᶜχρq_tot, ghost_buffer.ᶜχρq_tot))
    Spaces.weighted_dss_internal!(ᶜχuₕ, ghost_buffer.χuₕ)

    Spaces.weighted_dss_ghost!(ᶜχ, ghost_buffer.χ)
    is_ρq_tot && (Spaces.weighted_dss_ghost!(ᶜχρq_tot, ghost_buffer.ᶜχρq_tot))
    Spaces.weighted_dss_ghost!(ᶜχuₕ, ghost_buffer.χuₕ)
    NVTX.isactive() && NVTX.range_end(dss_hyperdiffusion_tendency)

    @. ᵗρs -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
    if is_ρq_tot
        @. Yₜ.c.ρq_tot -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχρq_tot))
        @. Yₜ.c.ρ -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχρq_tot))
    end
    if is_3d_pt
        @. Yₜ.c.uₕ -=
            κ₄ * (
                divergence_damping_factor * wgradₕ(divₕ(ᶜχuₕ)) -
                Geometry.Covariant12Vector(
                    wcurlₕ(Geometry.Covariant3Vector(curlₕ(ᶜχuₕ))),
                )
            )
    elseif is_2d_pt
        @. Yₜ.c.uₕ -=
            κ₄ *
            divergence_damping_factor *
            Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜχuₕ)))
    end
    NVTX.isactive() && NVTX.range_end(profile_hyperdiffusion_tendency)
    return nothing
end

function hyperdiffusion_tendency_tempest!(Yₜ, Y, p, t)
    NVTX.isactive() && (
        profile_hyperdiffusion_tendency = NVTX.range_start(;
            message = "hyperdiffusion tendency",
            color = colorant"yellow",
        )
    )
    !(:ρθ in propertynames(Y.c)) &&
        (error("use_tempest_mode must be false when not using ρθ"))
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    (; ᶜp, ᶜχ, ᶜχuₕ) = p # assume ᶜp has been updated
    (; ghost_buffer, κ₄, divergence_damping_factor, use_tempest_mode) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)
    is_ρq_tot = :ρq_tot in propertynames(Y.c)
    is_2d_pt = point_type <: Geometry.Abstract2DPoint
    is_3d_pt = !is_2d_pt

    @. ᶜχ = wdivₕ(gradₕ(ᶜρ)) # ᶜχρ
    Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
    @. Yₜ.c.ρ -= κ₄ * wdivₕ(gradₕ(ᶜχ))

    @. ᶜχ = wdivₕ(gradₕ(Y.c.ρθ)) # ᶜχρθ
    Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
    @. Yₜ.c.ρθ -= κ₄ * wdivₕ(gradₕ(ᶜχ))

    (; ᶠχw_data) = p
    @. ᶠχw_data = wdivₕ(gradₕ(Y.f.w.components.data.:1))
    Spaces.weighted_dss!(ᶠχw_data, ghost_buffer.χ)
    @. Yₜ.f.w.components.data.:1 -= κ₄ * wdivₕ(gradₕ(ᶠχw_data))

    if is_ρq_tot
        (; ᶜχρq_tot) = p
        @. ᶜχρq_tot = wdivₕ(gradₕ(Y.c.ρq_tot / ᶜρ))
        Spaces.weighted_dss!(ᶜχρq_tot, ghost_buffer.χ)
        @. Yₜ.c.ρq_tot -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχρq_tot))
        @. Yₜ.c.ρ -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχρq_tot))
    end

    if is_3d_pt
        @. ᶜχuₕ =
            wgradₕ(divₕ(ᶜuₕ)) - Geometry.Covariant12Vector(
                wcurlₕ(Geometry.Covariant3Vector(curlₕ(ᶜuₕ))),
            )
        Spaces.weighted_dss!(ᶜχuₕ, ghost_buffer.χuₕ)
        @. Yₜ.c.uₕ -=
            κ₄ * (
                divergence_damping_factor * wgradₕ(divₕ(ᶜχuₕ)) -
                Geometry.Covariant12Vector(
                    wcurlₕ(Geometry.Covariant3Vector(curlₕ(ᶜχuₕ))),
                )
            )
    elseif is_2d_pt
        @. ᶜχuₕ = Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜuₕ)))
        Spaces.weighted_dss!(ᶜχuₕ, ghost_buffer.χuₕ)
        @. Yₜ.c.uₕ -=
            κ₄ *
            divergence_damping_factor *
            Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜχuₕ)))
    end
    NVTX.isactive() && NVTX.range_end(profile_hyperdiffusion_tendency)
    return nothing
end
