#=
Hyperdiffusion: the microphysics species of the one-moment, two-moment, and P3
schemes are in the set that the generic ∇⁴ tracer tendency skips, so each takes
its share of the total-water tendency or receives nothing. A species missing
from the set would receive the full-strength ∇⁴ tendency on top of its share.

This asserts on the name set rather than on the tendency because two-moment and
P3 microphysics are disabled in `set_precomputed_quantities!`, so a state
carrying `ρn_ice`, `ρq_rim`, or `ρb_rim` cannot be built. Assert on the
tendency once that restriction lifts.
=#
using Test
using ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, MatrixFields, Spaces
const TD = CA.TD

include("../test_helpers.jl")

@testset "Microphysics species are excluded from the generic ∇⁴ tendency" begin
    state = CA.Setups.physical_state(; T = 300.0, p = 1e5)
    for model in (
        CA.NonEquilibriumMicrophysics1M(),
        CA.NonEquilibriumMicrophysics2M(),
        CA.NonEquilibriumMicrophysics2MP3(),
    )
        species = keys(CA.Setups.precip_variables(1.0, state, model))
        @test !isempty(species)
        for name in species
            @test MatrixFields.FieldName(name) in CA.hyperdiffusion_excluded_gs_names
        end
    end
    # The cloud masses are excluded as well; they take their share of the
    # total-water tendency.
    for name in (:ρq_lcl, :ρq_icl)
        @test MatrixFields.FieldName(name) in CA.hyperdiffusion_excluded_gs_names
    end
    # A passive tracer is not excluded.
    @test !(MatrixFields.FieldName(:ρq_gas_A) in CA.hyperdiffusion_excluded_gs_names)
end

@testset "Hyperdiffusion Kirchhoff T_0 invariance (grid-mean and PrognosticEDMFX)" begin
    config = CA.AtmosConfig(
        Dict{String, Any}(
            "config" => "box",
            "initial_condition" => "Bomex",
            "FLOAT_TYPE" => "Float64",
            "turbconv" => "prognostic_edmfx",
            "edmfx_entr_model" => "Generalized",
            "edmfx_detr_model" => "Generalized",
            "edmfx_sgs_mass_flux" => true,
            "edmfx_sgs_diffusive_flux" => true,
            "edmfx_vertical_diffusion" => true,
            "edmfx_nh_pressure" => true,
            "prognostic_tke" => true,
            "microphysics_model" => "1M",
            "x_max" => 6400.0,
            "x_elem" => 2,
            "y_max" => 6400.0,
            "y_elem" => 2,
            "z_max" => 3000.0,
            "z_elem" => 6,
            "z_stretch" => false,
            "dt" => "1secs",
            "t_end" => "10secs",
            "output_default_diagnostics" => false,
        );
        job_id = "hyperdiffusion_t0_invariance_test",
    )
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    FT = eltype(Y)

    ᶜx = Fields.coordinate_field(Y.c).x
    ᶜy = Fields.coordinate_field(Y.c).y
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜpert = @. 1 +
       FT(0.15) *
       sin(FT(2π) * ᶜx / FT(6400)) *
       cos(FT(2π) * ᶜy / FT(6400)) *
       cos(FT(π) * ᶜz / FT(3000))
    @. Y.c.ρq_lcl = FT(2e-4) * Y.c.ρ * ᶜpert
    @. Y.c.ρq_icl = FT(1e-4) * Y.c.ρ * ᶜpert
    @. Y.c.ρq_rai = FT(5e-5) * Y.c.ρ * ᶜpert
    @. Y.c.ρq_sno = FT(2e-5) * Y.c.ρ * ᶜpert
    @. Y.c.ρq_tot = FT(8e-3) * Y.c.ρ * ᶜpert + Y.c.ρq_rai + Y.c.ρq_sno
    @. Y.c.sgsʲs.:(1).ρa = FT(0.08) * Y.c.ρ * ᶜpert
    @. Y.c.sgsʲs.:(1).q_lcl = FT(3e-4) * ᶜpert
    @. Y.c.sgsʲs.:(1).q_icl = FT(1.5e-4) * ᶜpert
    @. Y.c.sgsʲs.:(1).q_rai = FT(8e-5) * ᶜpert
    @. Y.c.sgsʲs.:(1).q_sno = FT(3e-5) * ᶜpert
    @. Y.c.sgsʲs.:(1).q_tot =
        FT(1e-2) * ᶜpert + Y.c.sgsʲs.:(1).q_rai + Y.c.sgsʲs.:(1).q_sno
    @. Y.c.sgsʲs.:(1).mse = FT(3.05e5) + FT(3e3) * (ᶜpert - 1)
    CA.set_precomputed_quantities!(Y, p, t)

    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.hyperdiffusion_tendency!(Yₜ, Yₜ, Y, p, t)
    @test maximum(abs, parent(Yₜ.c.ρe_tot)) > 0
    @test maximum(abs, parent(Yₜ.c.sgsʲs.:(1).mse)) > 0

    δT_0 = FT(25.0)
    toml_shifted = deepcopy(config.toml_dict)
    toml_shifted.data["thermodynamics_temperature_reference"]["value"] += δT_0
    toml_shifted.data["latent_heat_vaporization_at_reference"]["value"] +=
        (CAP.cp_v(p.params) - CAP.cp_l(p.params)) * δT_0
    toml_shifted.data["latent_heat_sublimation_at_reference"]["value"] +=
        (CAP.cp_v(p.params) - CAP.cp_i(p.params)) * δT_0
    params_shifted = CA.ClimaAtmosParameters(
        toml_shifted;
        microphysics_model = p.atmos.microphysics_model,
    )
    tp_0 = CAP.thermodynamics_params(p.params)
    tp_s = CAP.thermodynamics_params(params_shifted)
    δh_d =
        TD.enthalpy_dry(tp_s, FT(280)) - TD.enthalpy_dry(tp_0, FT(280))
    δh_water =
        TD.enthalpy_vapor(tp_s, FT(280)) - TD.enthalpy_vapor(tp_0, FT(280))

    Y_s = deepcopy(Y)
    @. Y_s.c.ρe_tot =
        Y.c.ρe_tot + (Y.c.ρ - Y.c.ρq_tot) * δh_d + Y.c.ρq_tot * δh_water
    @. Y_s.c.sgsʲs.:(1).mse =
        Y.c.sgsʲs.:(1).mse +
        (1 - Y.c.sgsʲs.:(1).q_tot) * δh_d +
        Y.c.sgsʲs.:(1).q_tot * δh_water
    p_s = CA.AtmosCache(
        (
            k == :params ? params_shifted : getfield(p, k) for
            k in fieldnames(typeof(p))
        )...,
    )
    CA.set_precomputed_quantities!(Y_s, p_s, t)
    Yₜ_s = similar(Y)
    Yₜ_s .= zero(eltype(Yₜ_s))
    CA.hyperdiffusion_tendency!(Yₜ_s, Yₜ_s, Y_s, p_s, t)

    ᶜerr_gm = similar(Y.c.ρ)
    @. ᶜerr_gm = (Yₜ_s.c.ρe_tot - Yₜ.c.ρe_tot) - δh_water * Yₜ.c.ρq_tot
    @test maximum(abs, parent(ᶜerr_gm)) <=
          FT(1e-10) * maximum(abs, parent(Yₜ.c.ρe_tot))

    ᶜerr_up = similar(Y.c.ρ)
    @. ᶜerr_up =
        (Yₜ_s.c.sgsʲs.:(1).mse - Yₜ.c.sgsʲs.:(1).mse) -
        (δh_water - δh_d) * Yₜ.c.sgsʲs.:(1).q_tot
    @test maximum(abs, parent(ᶜerr_up)) <=
          FT(1e-10) * maximum(abs, parent(Yₜ.c.sgsʲs.:(1).mse))

    # With only microphysics SGS tracers active, no passive SGS tracer buffer is
    # allocated or DSSed.
    @test isempty(CA.hyperdiffused_sgs_tracer_names(Y))
    @test !hasproperty(p.hyperdiff, :ᶜ∇²sgs_tracerʲs)
end

@noinline function _alloc_sgs_hyperdiff_dss!(p)
    return @allocated begin
        pairs = CA.dss_hyperdiffusion_tendency_pairs(p)
        Spaces.weighted_dss!(last(pairs))
    end
end

@testset "Passive SGS tracer hyperdiffusion batched DSS and 0 B allocations" begin
    for FT in (Float32, Float64)
        config = CA.AtmosConfig(
            Dict{String, Any}(
                "config" => "box",
                "initial_condition" => "Bomex",
                "FLOAT_TYPE" => string(FT),
                "turbconv" => "prognostic_edmfx",
                "edmfx_entr_model" => "Generalized",
                "edmfx_detr_model" => "Generalized",
                "edmfx_sgs_mass_flux" => true,
                "edmfx_sgs_diffusive_flux" => true,
                "edmfx_vertical_diffusion" => true,
                "edmfx_nh_pressure" => true,
                "prognostic_tke" => true,
                "microphysics_model" => "0M",
                "chemistry_model" => "passive",
                "x_max" => 6400.0,
                "x_elem" => 2,
                "y_max" => 6400.0,
                "y_elem" => 2,
                "z_max" => 3000.0,
                "z_elem" => 6,
                "z_stretch" => false,
                "dt" => "1secs",
                "t_end" => "10secs",
                "output_default_diagnostics" => false,
            );
            job_id = "hyperdiffusion_passive_sgs_$FT",
        )
        (; Y, p, simulation) = generate_test_simulation(config)
        t = simulation.integrator.t
        @test CA.do_dss(axes(Y.c))
        @test CA.hyperdiffused_sgs_tracer_names(Y) == (MatrixFields.@name(q_gas_A),)
        @test hasproperty(p.hyperdiff, :ᶜ∇²sgs_tracerʲs)

        ᶜcoord = Fields.coordinate_field(Y.c)
        @. Y.c.ρq_gas_A =
            Y.c.ρ *
            FT(1e-5) *
            (
                FT(1) +
                FT(0.3) *
                sin(FT(2π) * ᶜcoord.x / FT(6400)) *
                cos(FT(4π) * ᶜcoord.y / FT(6400)) *
                (FT(1) + ᶜcoord.z / FT(3000))
            )
        @. Y.c.sgsʲs.:(1).q_gas_A =
            FT(2e-5) * (
                FT(1) +
                FT(0.4) *
                cos(FT(2π) * ᶜcoord.x / FT(6400)) *
                sin(FT(2π) * ᶜcoord.y / FT(6400)) *
                cos(FT(π) * ᶜcoord.z / FT(3000))
            )
        CA.set_precomputed_quantities!(Y, p, t)

        Yₜ = zero(Y)
        Yₜ_lim = zero(Y)
        CA.hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)

        # Reference single-tracer two-pass ∇⁴ operator with standalone DSS.
        (; ν₄_scalar) = CA.ν₄(p.atmos.hyperdiff, Y)
        ref_lap = similar(Y.c, FT)
        ref_buf = Spaces.create_dss_buffer(ref_lap)
        @. ref_lap = CA.wdivₕ(CA.gradₕ(Y.c.sgsʲs.:(1).q_gas_A))
        Spaces.weighted_dss!(ref_lap => ref_buf)
        ref_tend = similar(Y.c, FT)
        ref_tend .= zero(FT)
        @. ref_tend -= ν₄_scalar * CA.wdivₕ(CA.gradₕ(ref_lap))

        @test maximum(abs, parent(ref_tend)) > zero(FT)
        @test all(parent(Yₜ_lim.c.sgsʲs.:(1).q_gas_A) .=== parent(ref_tend))

        @test last(CA.dss_hyperdiffusion_tendency_pairs(p)) === (
            p.hyperdiff.ᶜ∇²sgs_tracerʲs =>
                p.hyperdiff.hyperdiffusion_ghost_buffer.ᶜ∇²sgs_tracerʲs
        )
        if Base.JLOptions().check_bounds != 1
            _alloc_sgs_hyperdiff_dss!(p)
            @test _alloc_sgs_hyperdiff_dss!(p) == 0
        end
    end
end
