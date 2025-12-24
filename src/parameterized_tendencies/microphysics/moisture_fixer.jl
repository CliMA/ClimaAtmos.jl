import ClimaCore.MatrixFields as MF
import CloudMicrophysics.ThermodynamicsInterface as TDI

moisture_fixer_tendency!(Yₜ, Y, p, t, _, _) = nothing

function moisture_fixer_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::NonEquilMoistModel,
    ::Union{Microphysics1Moment, Microphysics2Moment},
)
    if p.atmos.water.moisture_fixer
        moisture_species = (
            MF.@name(c.ρq_liq), MF.@name(c.ρq_ice),
            MF.@name(c.ρq_rai), MF.@name(c.ρq_sno),
        )
        qᵥ = @. lazy(
            specific(
                TDI.q_vap(Y.c.ρq_tot, Y.c.ρq_liq, Y.c.ρq_ice, Y.c.ρq_rai, Y.c.ρq_sno),
                Y.c.ρ,
            ),
        )

        MF.unrolled_foreach(moisture_species) do (ρq_name)
            ᶜρq = MF.get_field(Y, ρq_name)
            ᶜρqₜ = MF.get_field(Yₜ, ρq_name)
            ᶜq = @. lazy(specific(ᶜρq, Y.c.ρ))
            # Increase the grid mean small tracers if negative,
            # using mass from grid mean vapor.
            @. ᶜρqₜ += Y.c.ρ * moisture_fixer(ᶜq, qᵥ, p.dt)
        end
    end
end


microphysics_limiter_tendency!(Yₜ, Y, p, t, _, _) = nothing

function microphysics_limiter_tendency!(Yₜ, Y, p, t,
    ::NonEquilMoistModel, ::Union{Microphysics1Moment, Microphysics2Moment},
)
    moisture_species = (
        MF.@name(c.ρq_liq), MF.@name(c.ρq_ice),
        MF.@name(c.ρq_rai), MF.@name(c.ρq_sno),
    )
    MF.unrolled_foreach(moisture_species) do ρq_name
        ᶜρq = MF.get_field(Y, ρq_name)
        ᶜρqₜ = MF.get_field(Yₜ, ρq_name)
        # Limit the tracer tendencies so that:
        # - if ρqₜ < 0, then at most `ρq/dt` is removed
        # - if ρqₜ > 0, ρq > 0, then the tendency remains unchanged
        # - if ρq < 0, then at least `ρq/dt` is added
        @. ᶜρqₜ = max(ᶜρqₜ, -(ᶜρq / p.dt))
        # @. ᶜρqₜ = ifelse(
        #     ᶜρqₜ < 0,
        #     max(ᶜρqₜ, -(ᶜρq / p.dt)),  # evaporating
        #     max(ᶜρqₜ, -(ᶜρqᵥ / p.dt)), # condensing
        # )
    end

    return nothing
end
