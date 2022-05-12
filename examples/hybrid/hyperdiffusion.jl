hyperdiffusion_cache(
    Y;
    κ₄ = FT(0),
    divergence_damping_factor = FT(1),
    use_tempest_mode = false,
) = merge(
    (;
        ᶜχ = similar(Y.c, FT),
        ᶜχuₕ = similar(Y.c, Geometry.Covariant12Vector{FT}),
        κ₄,
        divergence_damping_factor,
        use_tempest_mode,
    ),
    use_tempest_mode ? (; ᶠχw_data = similar(Y.F, FT)) : NamedTuple(),
)

function hyperdiffusion_tendency!(Yₜ, Y, p, t)
    ᶜρ = Y.c.ρ
    ᶜuₕ = Y.c.uₕ
    (; ᶜp, ᶜχ, ᶜχuₕ) = p # assume ᶜp has been updated
    (; ghost_buffer, κ₄, divergence_damping_factor, use_tempest_mode) = p
    point_type = eltype(Fields.local_geometry_field(axes(Y.c)).coordinates)

    if use_tempest_mode
        @. ᶜχ = wdivₕ(gradₕ(ᶜρ)) # ᶜχρ
        Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
        @. Yₜ.c.ρ -= κ₄ * wdivₕ(gradₕ(ᶜχ))

        if :ρθ in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ(Y.c.ρθ)) # ᶜχρθ
            Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
            @. Yₜ.c.ρθ -= κ₄ * wdivₕ(gradₕ(ᶜχ))
        else
            error("use_tempest_mode must be false when not using ρθ")
        end

        (; ᶠχw_data) = p
        @. ᶠχw_data = wdivₕ(gradₕ(Y.f.w.components.data.:1))
        Spaces.weighted_dss!(ᶠχw_data, ghost_buffer.χ)
        @. Yₜ.f.w.components.data.:1 -= κ₄ * wdivₕ(gradₕ(ᶠχw_data))
    else
        if :ρθ in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ(Y.c.ρθ / ᶜρ)) # ᶜχθ
            Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
            @. Yₜ.c.ρθ -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        elseif :ρe in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ((Y.c.ρe + ᶜp) / ᶜρ)) # ᶜχe
            Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
            @. Yₜ.c.ρe -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        elseif :ρe_int in propertynames(Y.c)
            @. ᶜχ = wdivₕ(gradₕ((Y.c.ρe_int + ᶜp) / ᶜρ)) # ᶜχe_int
            Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
            @. Yₜ.c.ρe_int -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        end
    end

    if :ρq_tot in propertynames(Y.c)
        @. ᶜχ = wdivₕ(gradₕ(Y.c.ρq_tot / ᶜρ))
        Spaces.weighted_dss!(ᶜχ, ghost_buffer.χ)
        @. Yₜ.c.ρq_tot -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
        @. Yₜ.c.ρ -= κ₄ * wdivₕ(ᶜρ * gradₕ(ᶜχ))
    end

    if point_type <: Geometry.Abstract3DPoint
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
    elseif point_type <: Geometry.Abstract2DPoint
        @. ᶜχuₕ = Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜuₕ)))
        Spaces.weighted_dss!(ᶜχuₕ, ghost_buffer.χuₕ)
        @. Yₜ.c.uₕ -=
            κ₄ *
            divergence_damping_factor *
            Geometry.Covariant12Vector(wgradₕ(divₕ(ᶜχuₕ)))
    end
end
