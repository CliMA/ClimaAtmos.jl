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
