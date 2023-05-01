#####
##### Pressure work
#####

function pressure_work_tendency!(Yₜ, Y, p, t, colidx, ::EDMFX)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜp) = p
    for j in 1:n
        if :ρe_tot in propertynames(Y.c)
            @. Yₜ.c.sgsʲs.:($$j).ρae_tot[colidx] -=
                ᶜp[colidx] / Y.c.ρ[colidx] * Yₜ.c.sgsʲs.:($$j).ρa[colidx]
        end
    end
end

pressure_work_tendency!(Yₜ, Y, p, t, colidx, ::Any) = nothing
