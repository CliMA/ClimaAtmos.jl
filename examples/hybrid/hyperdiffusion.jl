hyperdiffusion_cache(
    ᶜlocal_geometry,
    ᶠlocal_geometry;
    κ₄ = FT(0),
    divergence_damping_factor = FT(1),
    use_tempest_mode = false,
) = merge(
    (;
        ᶜχ = similar(ᶜlocal_geometry, FT),
        ᶜχuₕ = similar(ᶜlocal_geometry, Geometry.Covariant12Vector{FT}),
        κ₄,
        divergence_damping_factor,
        use_tempest_mode,
    ),
    use_tempest_mode ? (; ᶠχw_data = similar(ᶠlocal_geometry, FT)) :
        NamedTuple(),
)

function hyperdiffusion_tendency!(Yₜ, Y, p, t, comms_ctx = nothing)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    (; ᶜp, ᶜχ, ᶜχuₕ) = p # assume that ᶜp has been updated
    (; comms_ctx, κ₄, divergence_damping_factor, use_tempest_mode) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)

    if use_tempest_mode
        @. ᶜχ = wdivₕ(gradₕ(ᶜρ)) # ᶜχρ
        Spaces.weighted_dss!(ᶜχ, comms_ctx)
        @. Yₜ.c.ρ -= κ₄ * wdivₕ(gradₕ(ᶜχ))

        if :ρθ in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ(Y.c.ρθ)) # ᶜχρθ
            Spaces.weighted_dss!(ᶜχ, comms_ctx)
            @. Yₜ.c.ρθ -= κ₄ * wdivₕ(gradₕ(ᶜχ))
        else
            error("use_tempest_mode must be false when not using ρθ")
        end

        (; ᶠχw_data) = p
        @. ᶠχw_data = wdivₕ(gradₕ(Y.f.w.components.data.:1))
        Spaces.weighted_dss!(ᶠχw_data, comms_ctx)
        @. Yₜ.f.w.components.data.:1 -= κ₄ * wdivₕ(gradₕ(ᶠχw_data))
    else
        if :ρθ in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ(Y.c.ρθ / ᶜρ)) # ᶜχθ
            Spaces.weighted_dss!(ᶜχ, comms_ctx)
            @. Yₜ.c.ρθ -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        elseif :ρe in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ((Y.c.ρe + ᶜp) / ᶜρ)) # ᶜχe
            Spaces.weighted_dss!(ᶜχ, comms_ctx)
            @. Yₜ.c.ρe -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        elseif :ρe_int in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ((Y.c.ρe_int + ᶜp) / ᶜρ)) # ᶜχe_int
            Spaces.weighted_dss!(ᶜχ, comms_ctx)
            @. Yₜ.c.ρe_int -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        end
    end

    if point_type <: Geometry.Abstract3DPoint
        @. ᶜχuₕ =
            wgradₕ(divₕ(ᶜuₕ)) -
            Geometry.Covariant12Vector(wcurlₕ(Geometry.Covariant3Vector(curlₕ(
                ᶜuₕ,
            ))),)
        Spaces.weighted_dss!(ᶜχuₕ, comms_ctx)
        @. Yₜ.c.uₕ -=
            κ₄ * (
                divergence_damping_factor * wgradₕ(divₕ(ᶜχuₕ)) -
                Geometry.Covariant12Vector(wcurlₕ(Geometry.Covariant3Vector(curlₕ(
                    ᶜχuₕ,
                ))),)
            )
    elseif point_type <: Geometry.Abstract2DPoint
        @. ᶜχuₕ = Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜuₕ)))
        Spaces.weighted_dss!(ᶜχuₕ, comms_ctx)
        @. Yₜ.c.uₕ -=
            κ₄ *
            divergence_damping_factor *
            Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜχuₕ)))
    end
end
