@inline calc_source!(source, ::ThreeDimensionalCompressibleEulerWithBarotropicFluid, _...) = nothing

@inline function calc_source!(
        source,
        balance_law::ThreeDimensionalCompressibleEulerWithBarotropicFluid, 
        ::DeepShellCoriolis, 
        state, 
        aux
    )
    ρu = state.ρu

    Ω  = @SVector [-0, -0, balance_law.parameters.Ω]

    source.ρu += -2Ω × ρu

    nothing
end
