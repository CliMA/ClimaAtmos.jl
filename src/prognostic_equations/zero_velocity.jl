###
### edmfx advection test 
###

function zero_velocity_tendency!(Yₜ, Y, p, t, colidx)
    # turn off all monmentum tendencies in the advection test
    if p.atmos.edmfx_adv_test
        FT = eltype(Y)
        n = n_mass_flux_subdomains(p.atmos.turbconv_model)

        @. Yₜ.c.uₕ[colidx] = C12(FT(0), FT(0))
        @. Yₜ.f.u₃[colidx] = Geometry.Covariant3Vector(FT(0))
        if p.atmos.turbconv_model isa EDMFX
            for j in 1:n
                @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] =
                    Geometry.Covariant3Vector(FT(0))
            end
        end
    end
    return nothing
end
