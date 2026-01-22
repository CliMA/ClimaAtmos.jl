import ClimaCore.MatrixFields as MF
import CloudMicrophysics.ThermodynamicsInterface as TDI

function tracer_nonnegativity_vapor_tendency(q, qᵥ, dt)
    FT = eltype(q)
    return triangle_inequality_limiter(-min(FT(0), q / dt), limit(qᵥ, dt, 5), FT(0))
end

tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t, _, _) = nothing

function tracer_nonnegativity_vapor_tendency!(Yₜ, Y, p, t,
    ::NonEquilMoistModel, ::Union{Microphysics1Moment, Microphysics2Moment},
)
    p.atmos.water.tracer_nonnegativity_method isa TracerNonnegativityVaporTendency || return

    moisture_species = (
        MF.@name(ρq_liq), MF.@name(ρq_ice),
        MF.@name(ρq_rai), MF.@name(ρq_sno),
    )
    ρqᵥ = @. lazy(TDI.q_vap(Y.c.ρq_tot, Y.c.ρq_liq, Y.c.ρq_ice, Y.c.ρq_rai, Y.c.ρq_sno))
    qᵥ = @. lazy(specific(ρqᵥ, Y.c.ρ))

    MF.unrolled_foreach(moisture_species) do ρq_name
        ᶜρq = MF.get_field(Y.c, ρq_name)
        ᶜρqₜ = MF.get_field(Yₜ.c, ρq_name)
        ᶜq = @. lazy(specific(ᶜρq, Y.c.ρ))
        # Increase the grid mean small tracers if negative, using mass from grid mean vapor.
        @. ᶜρqₜ += Y.c.ρ * tracer_nonnegativity_vapor_tendency(ᶜq, qᵥ, p.dt)
    end
end
