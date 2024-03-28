###
### simple edmfx test 
###

function zero_gridscale_tendency!(Yₜ, Y, p, t, colidx)
    # turn off all grid-scale tendencies in the simple edmfx test
    if !p.atmos.gs_tendency
        @. Yₜ.c.ρ[colidx] = 0
        @. Yₜ.c.uₕ[colidx] = C12(0, 0)
        @. Yₜ.f.u₃[colidx] = C3(0)
        @. Yₜ.c.ρe_tot[colidx] = 0
        for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
            @. Yₜ.c.:($$ρχ_name)[colidx] = 0
        end
    end
    return nothing
end

function zero_subgridscale_tendency!(Yₜ, Y, p, t, colidx)
    # turn off all subgrid-scale tendencies
    n = n_prognostic_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @. Yₜ.c.sgsʲs.:($$j).ρa[colidx] = 0
        @. Yₜ.f.sgsʲs.:($$j).u₃[colidx] = C3(0)
        @. Yₜ.c.sgsʲs.:($$j).mse[colidx] = 0
        @. Yₜ.c.sgsʲs.:($$j).q_tot[colidx] = 0
    end
    return nothing
end