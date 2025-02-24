###
### Zero grid-scale or subgrid-scale tendencies
###

zero_tendency!(Yₜ, Y, p, t, _, _) = nothing

function zero_tendency!(Yₜ, Y, p, t, ::NoGridScaleTendency, _)
    # turn off all grid-scale tendencies
    @. Yₜ.c.ρ = 0
    @. Yₜ.c.uₕ = C12(0, 0)
    @. Yₜ.f.u₃ = C3(0)
    @. Yₜ.c.ρe_tot = 0
    for ρχ_name in filter(is_tracer_var, propertynames(Y.c))
        @. Yₜ.c.:($$ρχ_name) = 0
    end
    return nothing
end

function zero_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NoSubgridScaleTendency,
    ::PrognosticEDMFX,
)
    # turn off all subgrid-scale tendencies
    n = n_prognostic_mass_flux_subdomains(p.atmos.turbconv_model)
    for j in 1:n
        @. Yₜ.c.sgsʲs.:($$j).ρa = 0
        @. Yₜ.f.sgsʲs.:($$j).u₃ = C3(0)
        @. Yₜ.c.sgsʲs.:($$j).mse = 0
        @. Yₜ.c.sgsʲs.:($$j).q_tot = 0

        if p.atmos.moisture_model isa NonEquilMoistModel
            @. Yₜ.c.sgsʲs.:($$j).q_liq = 0
            @. Yₜ.c.sgsʲs.:($$j).q_ice = 0
        end
    end
    return nothing
end
