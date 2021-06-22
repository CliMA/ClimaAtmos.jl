abstract type AbstractCoriolis <: AbstractTerm end

struct DeepShellCoriolis <: AbstractCoriolis end

@inline calc_component!(source, ::Nothing, state, _...) = nothing
@inline calc_component!(source, ::AbstractTerm, _...) = nothing

@inline function calc_component!(source, ::DeepShellCoriolis, state, aux, physics)
    ρu = state.ρu

    Ω  = @SVector [-0, -0, physics.parameters.Ω]

    source.ρu -= 2Ω × ρu

    nothing
end