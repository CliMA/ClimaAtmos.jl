@inline calc_component!(::ThreeDimensionalCompressibleEulerWithBarotropicFluid, _...) = nothing

@inline function calc_source!(
        flux,
        balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid, 
        ::Coriolis, 
        state, 
        aux
    )
    ρu = state.ρu

    Ω  = @SVector [-0, -0, balance_law.parameters.Ω]

    flux.ρu += -2Ω × ρu

    nothing
end