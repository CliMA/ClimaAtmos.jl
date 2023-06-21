#####
##### Pressure work
#####

function pressure_work_tendency!(Yₜ, Y, p, t, ::EDMFX)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)
    (; ᶜp) = p
    for j in 1:n
        if :ρe_tot in propertynames(Y.c)
            @. Yₜ.c.sgsʲs.:($$j).ρae_tot -= ᶜp / Y.c.ρ * Yₜ.c.sgsʲs.:($$j).ρa
        end
    end
end

pressure_work_tendency!(Yₜ, Y, p, t, ::Any) = nothing
