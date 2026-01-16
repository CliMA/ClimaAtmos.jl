using Test
import ClimaComms
ClimaComms.@import_required_backends
using Random
Random.seed!(1234)
import ClimaAtmos as CA
import ClimaAtmos.RRTMGPInterface as RRTMGPI
import ClimaParams as CP
import RRTMGP.Parameters.RRTMGPParameters

param_dict = CP.create_toml_dict(Float64)
params = RRTMGPParameters(param_dict)

### RRTMGP Interface Tests
# Includes tests for functions defined within the RRTMGPInterface.jl file.
# Assesses that interp / extrap functions are correctly defined.

using Statistics

include("test_helpers.jl")

@testset "Array <-> field" begin
    (; bubble_space) = get_spherical_spaces()
    test_field = ones(bubble_space)
    idx = size(parent(test_field)) # IJFH layout
    npts = idx[1] * idx[2] * idx[4]
    @assert test_field isa Fields.Field
    test_array = Fields.field2array(test_field)
    @test test_array isa Array
    @test length(test_array) == npts
    new_field = Fields.array2field(test_array, bubble_space)
    @test new_field == test_field
    @test size(parent(new_field)) == idx
end

@testset "Arithmetic ops" begin
    FloatType = (Float32, Float64)
    for FT in FloatType
        _T = [FT(1)]
        _Tꜛ = [FT(1)]
        _Tꜜ = [FT(2)]
        _p = [FT(1 // 2)]
        _p₁ = [FT(2)]
        _p₂ = [FT(3)]
        _Tₛ = [FT(5)]
        XT = similar(_T)
        XP = similar(_p)
        @test RRTMGPI.uniform_z_p.(_T, _p₁, _Tꜜ, _p₁, _Tꜛ) == _p₁
        @test RRTMGPI.uniform_z_p.(_T, _p₁, _Tꜜ, _p₁, _Tꜛ) == _p₁

        RRTMGPI.interp!(RRTMGPI.ArithmeticMean(), XP, XT, _p₁, _Tꜜ, _p₂, _Tꜛ)
        @test XT[1] == (_Tꜜ[1] + _Tꜛ[1]) / 2
        @test XP[1] == (_p₁[1] + _p₂[1]) / 2

        RRTMGPI.interp!(RRTMGPI.GeometricMean(), XP, XT, _p₁, _Tꜜ, _p₂, _Tꜛ)
        @test XT[1] == sqrt(_Tꜜ[1] * _Tꜛ[1])
        @test XP[1] == sqrt(_p₁[1] * _p₂[1])

        RRTMGPI.interp!(RRTMGPI.UniformZ(), XP, XT, _p₁, _Tꜜ, _p₂, _Tꜛ)
        @test XT[1] == (_Tꜜ[1] + _Tꜛ[1]) / 2
        @test XP[1] == RRTMGPI.uniform_z_p.(XT, _p₁, _Tꜜ, _p₂, _Tꜛ)[]

        RRTMGPI.interp!(RRTMGPI.UniformP(), XP, XT, _p₁, _Tꜜ, _p₂, _Tꜛ)
        @test XP[1] == (_p₁[1] + _p₂[1]) / 2
        # The second result entry has the same function form as uniform_z_p, 
        # with arguments swapped
        @test XT[1] == RRTMGPI.uniform_z_p.(XP, _Tꜜ, _p₁, _Tꜛ, _p₂)[]

        RRTMGPI.extrap!(
            RRTMGPI.ArithmeticMean(),
            XP,
            XT,
            _p₁,
            _Tꜜ,
            _p₂,
            _Tꜛ,
            _Tₛ,
            params,
        )
        @test XP[1] == @. (3 * _p₁[1] - _p₂[1]) / 2
        @test XT[1] == @. (3 * _Tꜜ[1] - _Tꜛ[1]) / 2

        RRTMGPI.extrap!(
            RRTMGPI.GeometricMean(),
            XP,
            XT,
            _p₁,
            _Tꜜ,
            _p₂,
            _Tꜛ,
            _Tₛ,
            params,
        )
        @test XP[1] == @. sqrt(_p₁[1]^3 / _p₂[1])
        @test XT[1] == @. sqrt(_Tꜜ[1]^3 / _Tꜛ[1])

        RRTMGPI.extrap!(
            RRTMGPI.ArithmeticMean(),
            XP,
            XT,
            _p₁,
            _Tꜜ,
            _p₂,
            _Tꜛ,
            _Tₛ,
            params,
        )
        _XP = XP
        RRTMGPI.extrap!(
            RRTMGPI.UniformZ(),
            XP,
            XT,
            _p₁,
            _Tꜜ,
            _p₂,
            _Tꜛ,
            _Tₛ,
            params,
        )
        @test _XP[1] == XP[1]
        # The result entry has the same function form as uniform_z_p, 
        # with arguments swapped
        @test XT[1] == RRTMGPI.uniform_z_p.(XP, _Tꜜ, _p₁, _Tꜛ, _p₂)[]

        RRTMGPI.extrap!(
            RRTMGPI.ArithmeticMean(),
            XP,
            XT,
            _p₁,
            _Tꜜ,
            _p₂,
            _Tꜛ,
            _Tₛ,
            params,
        )
        _XT = XT
        RRTMGPI.extrap!(
            RRTMGPI.UniformP(),
            XP,
            XT,
            _p₁,
            _Tꜜ,
            _p₂,
            _Tꜛ,
            _Tₛ,
            params,
        )
        @test XP[1] == RRTMGPI.uniform_z_p.(XT, _p₁, _Tꜜ, _p₂, _Tꜛ)[]
        @test _XT[1] == XT[1]

    end
end
