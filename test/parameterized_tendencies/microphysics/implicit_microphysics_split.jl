#=
Unit tests for the implicit-microphysics parameter split and the
`ᶜmp_tendency_implicit` cache populated by a second CM BMT call with
`microphysics_1m_implicit_params`.

Covers:
- `_split_formation` sends the user's `cloud_liquid_formation` /
  `cloud_ice_formation` to the bulk or implicit parameter set based on
  `implicit_microphysics`.
- The two 1M parameter sets `ClimaAtmosParameters` builds satisfy the
  contract: at most one of `(bulk, implicit)` carries each formation.
=#

using Test
import ClimaParams as CP
import CloudMicrophysics.Parameters as CMP
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaAtmos: _split_formation

# ClimaAtmosParameters is reexported from ClimaAtmos; use that path to pass
# the top-level kwargs the constructor accepts.

@testset "`_split_formation` contract" begin
    @test _split_formation(nothing, false) == (nothing, nothing)
    @test _split_formation(nothing, true) == (nothing, nothing)

    clf = CMP.CloudLiquidFormation()
    @test _split_formation(clf, false) == (clf, nothing)
    @test _split_formation(clf, true) == (nothing, clf)

    cif = CMP.TemperatureDependentIceNumber()
    @test _split_formation(cif, false) == (cif, nothing)
    @test _split_formation(cif, true) == (nothing, cif)

    # For every (opt, is_implicit), at most one of (bulk, implicit) is set.
    for opt in (nothing, CMP.CloudLiquidFormation(), CMP.ConstantTimescale())
        for is_implicit in (false, true)
            bulk, impl = _split_formation(opt, is_implicit)
            @test isnothing(bulk) || isnothing(impl)
        end
    end
end

@testset "`ClimaAtmosParameters` 1M params split" begin
    for FT in (Float32, Float64), is_implicit in (false, true)
        @testset "FT = $FT, implicit_microphysics = $is_implicit" begin
            toml_dict = CP.create_toml_dict(FT)
            mp_model = CA.NonEquilibriumMicrophysics1M()
            params = CA.ClimaAtmosParameters(
                toml_dict;
                microphysics_model = mp_model,
                implicit_microphysics = is_implicit,
            )
            bulk = CAP.microphysics_1m_params(params)
            impl = CAP.microphysics_1m_implicit_params(params)

            if is_implicit
                @test isnothing(bulk.processes.cloud_liquid_formation)
                @test isnothing(bulk.processes.cloud_ice_formation)
                @test !isnothing(impl.processes.cloud_liquid_formation)
                @test !isnothing(impl.processes.cloud_ice_formation)
            else
                @test !isnothing(bulk.processes.cloud_liquid_formation)
                @test !isnothing(bulk.processes.cloud_ice_formation)
                @test isnothing(impl.processes.cloud_liquid_formation)
                @test isnothing(impl.processes.cloud_ice_formation)
            end
            # The implicit set carries only the two formation processes;
            # every other process is nulled so CM BMT returns only cond/subl.
            # `vapor_relaxation` is a coupling mode (not a process); it cannot
            # be `nothing` and is inert when no vapor-driven phase change
            # process is active.
            for pn in propertynames(impl.processes)
                pn in (
                    :cloud_liquid_formation,
                    :cloud_ice_formation,
                    :vapor_relaxation,
                ) && continue
                @test isnothing(getproperty(impl.processes, pn))
            end
        end
    end
end
