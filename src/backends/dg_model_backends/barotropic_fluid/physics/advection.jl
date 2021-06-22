struct NonlinearAdvection{𝒯} <: AbstractTerm end

@inline calc_component!(flux, ::Nothing, _...) = nothing
@inline calc_component!(flux, ::AbstractTerm, _...) = nothing

@inline function calc_component!(flux, ::NonlinearAdvection{(:ρ, :ρu, :ρθ)}, state, aux, physics)
    ρ  = state.ρ
    ρu = state.ρu
    ρθ = state.ρθ
    eos = physics.eos
    parameters = physics.parameters
    
    u = ρu / ρ
    p = calc_pressure(eos, state, aux, parameters)

    flux.ρ  += ρu
    flux.ρu += ρu ⊗ u + p * I
    flux.ρθ += ρθ * u

    nothing
end