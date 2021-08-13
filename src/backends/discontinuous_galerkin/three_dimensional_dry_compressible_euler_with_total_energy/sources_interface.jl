@inline calc_source!(source, ::ThreeDimensionalDryCompressibleEulerWithTotalEnergy, _...) = nothing

@inline function calc_source!(
        source,
        balance_law::ThreeDimensionalDryCompressibleEulerWithTotalEnergy, 
        ::DeepShellCoriolis, 
        state, 
        aux
    )
    ρu = state.ρu

    Ω  = @SVector [-0, -0, balance_law.parameters.Ω]

    source.ρu += -2Ω × ρu

    nothing
end