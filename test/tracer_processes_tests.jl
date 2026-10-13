#=
Unit tests for the process-based tracer classification utilities in
src/utils/tracer_processes.jl.
=#

using Test
import ClimaAtmos as CA
import Thermodynamics as TD
import ClimaCore: Fields
import ClimaCore.MatrixFields: @name

include("test_helpers.jl")

@testset "Sedimentation velocity names" begin
    @test CA.sedimentation_velocity_name(@name(ρq_lcl)) == @name(ᶜwₗ)
    @test CA.sedimentation_velocity_name(@name(ρq_icl)) == @name(ᶜwᵢ)
    @test CA.sedimentation_velocity_name(@name(ρq_rai)) == @name(ᶜwᵣ)
    @test CA.sedimentation_velocity_name(@name(ρq_sno)) == @name(ᶜwₛ)
    @test CA.sedimentation_velocity_name(@name(ρn_lcl)) == @name(ᶜwₙₗ)
    @test CA.sedimentation_velocity_name(@name(ρn_rai)) == @name(ᶜwₙᵣ)
    # P3 rime quantities sediment with the ice velocity
    @test CA.sedimentation_velocity_name(@name(ρq_rim)) == @name(ᶜwᵢ)
    @test CA.sedimentation_velocity_name(@name(ρb_rim)) == @name(ᶜwᵢ)
    # non-sedimenting tracers
    @test isnothing(CA.sedimentation_velocity_name(@name(ρq_tot)))
    @test isnothing(CA.sedimentation_velocity_name(@name(ρq_gas_A)))

    @test CA.sgs_sedimentation_velocity_name(@name(q_lcl)) ==
          @name(ᶜwₗʲs.:(1))
    @test CA.sgs_sedimentation_velocity_name(@name(n_rai)) ==
          @name(ᶜwₙᵣʲs.:(1))
    @test isnothing(CA.sgs_sedimentation_velocity_name(@name(q_tot)))
    @test isnothing(CA.sgs_sedimentation_velocity_name(@name(q_gas_A)))
end

@testset "Condensate phase properties" begin
    @test CA.condensate_phase(@name(ρq_lcl)) == TD.Liquid()
    @test CA.condensate_phase(@name(ρq_rai)) == TD.Liquid()
    @test CA.condensate_phase(@name(ρq_icl)) == TD.Ice()
    @test CA.condensate_phase(@name(ρq_sno)) == TD.Ice()
    # SGS / specific names
    @test CA.condensate_phase(@name(q_rai)) == TD.Liquid()
    @test CA.condensate_phase(@name(q_sno)) == TD.Ice()
    # number concentrations and non-condensates have no phase
    @test isnothing(CA.condensate_phase(@name(ρn_lcl)))
    @test isnothing(CA.condensate_phase(@name(ρq_tot)))

    @test CA.internal_energy_function(TD.Liquid()) ==
          TD.internal_energy_liquid
    @test CA.internal_energy_function(TD.Ice()) == TD.internal_energy_ice

    FT = Float32
    params = CA.ClimaAtmosParameters(FT)
    CAP = CA.Parameters
    @test CA.condensate_e_int_offset(TD.Liquid(), params) ==
          FT(CAP.e_int_v0(params))
    @test CA.condensate_e_int_offset(TD.Ice(), params) ==
          FT(CAP.e_int_i0(params)) + FT(CAP.e_int_v0(params))
    @test CA.condensate_cv_difference(TD.Liquid(), params) ==
          FT(CAP.cp_l(params) - CAP.cv_v(params))
    @test CA.condensate_cv_difference(TD.Ice(), params) ==
          FT(CAP.cp_i(params) - CAP.cv_v(params))
    @test CA.condensate_e_int_offset(TD.Liquid(), params) isa FT
    @test CA.condensate_cv_difference(TD.Ice(), params) isa FT
end

@testset "Process-based tracer lists" begin
    (; cent_space) = get_cartesian_spaces()
    FT = Float32
    coords = Fields.coordinate_field(cent_space)

    # Dry state: no tracers participate in any moisture process
    Y_dry = (;
        c = similar(coords, NamedTuple{(:ρ, :ρe_tot), Tuple{FT, FT}}),
    )
    @test CA.sedimenting_tracer_names(Y_dry) == ()
    @test CA.sedimenting_mass_names(Y_dry) == ()
    @test CA.microphysics_tracer_names(Y_dry) == ()
    @test CA.diffused_gs_scalar_names(Y_dry) == (@name(ρe_tot),)
    @test CA.advected_gs_scalar_names(Y_dry) == (@name(ρ), @name(ρe_tot))
    @test CA.sedimenting_sgs_tracer_names(Y_dry) == ()
    @test CA.advected_sgs_scalar_names(Y_dry) == ()

    # 1M moist state with a passive chemistry tracer and prognostic EDMF
    sgs_type = NamedTuple{
        (:ρa, :mse, :q_tot, :q_lcl, :q_icl, :q_rai, :q_sno, :q_gas_A),
        NTuple{8, FT},
    }
    Y_1m = (;
        c = similar(
            coords,
            NamedTuple{
                (
                    :ρ,
                    :ρe_tot,
                    :ρq_tot,
                    :ρq_lcl,
                    :ρq_icl,
                    :ρq_rai,
                    :ρq_sno,
                    :ρq_gas_A,
                    :sgsʲs,
                ),
                Tuple{FT, FT, FT, FT, FT, FT, FT, FT, Tuple{sgs_type}},
            },
        ),
    )

    @test CA.sedimenting_tracer_names(Y_1m) ==
          (@name(ρq_lcl), @name(ρq_icl), @name(ρq_rai), @name(ρq_sno))
    @test CA.sedimenting_mass_names(Y_1m) ==
          (@name(ρq_lcl), @name(ρq_icl), @name(ρq_rai), @name(ρq_sno))
    # passive tracers (ρq_gas_A) are excluded from the moisture processes
    @test CA.microphysics_tracer_names(Y_1m) == (
        @name(ρq_tot),
        @name(ρq_lcl),
        @name(ρq_icl),
        @name(ρq_rai),
        @name(ρq_sno),
    )
    @test CA.diffused_gs_scalar_names(Y_1m) ==
          (@name(ρe_tot), CA.microphysics_tracer_names(Y_1m)...)
    @test CA.advected_gs_scalar_names(Y_1m) ==
          (@name(ρ), @name(ρe_tot), @name(ρq_tot))

    @test CA.sedimenting_sgs_tracer_names(Y_1m) ==
          (@name(q_lcl), @name(q_icl), @name(q_rai), @name(q_sno))
    @test CA.sedimenting_sgs_mass_names(Y_1m) ==
          (@name(q_lcl), @name(q_icl), @name(q_rai), @name(q_sno))
    @test CA.passive_sgs_tracer_names(Y_1m) == (@name(q_gas_A),)
    @test CA.advected_sgs_scalar_names(Y_1m) == (
        @name(q_lcl),
        @name(q_icl),
        @name(q_rai),
        @name(q_sno),
        @name(q_tot),
        @name(mse),
        @name(q_gas_A),
    )

    # every sedimenting tracer must have a velocity, every sedimenting mass
    # must have a phase
    for name in CA.sedimenting_tracer_names(Y_1m)
        @test !isnothing(CA.sedimentation_velocity_name(name))
    end
    for name in CA.sedimenting_mass_names(Y_1m)
        @test !isnothing(CA.condensate_phase(name))
    end
    for name in CA.sedimenting_sgs_tracer_names(Y_1m)
        @test !isnothing(CA.sgs_sedimentation_velocity_name(name))
    end
end

@testset "Name lifting" begin
    @test CA.center_state_name(@name(ρq_rai)) == @name(c.ρq_rai)
    @test CA.sgs_state_name(@name(q_rai)) == @name(c.sgsʲs.:(1).q_rai)
end
