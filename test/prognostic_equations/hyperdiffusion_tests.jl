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

# The Laplacians of the packed field `p.hyperdiff.ᶜ∇²`, each computed with its
# own broadcast over the whole space and DSSed on its own.
function reference_laplacians(Y, p)
    thermo_params = CAP.thermodynamics_params(p.params)
    (; ᶜu, ᶜT, ᶜp, ᶜuʲs, ᶜTʲs) = p.precomputed
    (; ᶜΦ) = p.core
    FT = eltype(Y)
    ᶜT_r = @. CA.air_temperature_reference(thermo_params, ᶜp)
    ᶜq_tot_r = @. CA.q_tot_r(thermo_params, ᶜp)
    ref = (;
        u = @.(
            CA.C123(CA.wgradₕ(CA.divₕ(ᶜu))) -
            CA.C123(CA.wcurlₕ(CA.C123(CA.curlₕ(ᶜu))))
        ),
        T = @.(CA.wdivₕ(CA.gradₕ(ᶜT - ᶜT_r))),
        Φ = @.(CA.wdivₕ(CA.gradₕ(ᶜΦ - CA.phi_r(thermo_params, ᶜp)))),
        q_tot_eff = @.(
            CA.wdivₕ(CA.gradₕ(CA.specific(Y.c.ρq_tot, Y.c.ρ) - ᶜq_tot_r))
        ),
        tke = @.(CA.wdivₕ(CA.gradₕ(CA.specific(Y.c.ρtke, Y.c.ρ)))),
        q_gas_A = @.(CA.wdivₕ(CA.gradₕ(CA.specific(Y.c.ρq_gas_A, Y.c.ρ)))),
        u¹ = @.(
            CA.C123(CA.wgradₕ(CA.divₕ(ᶜuʲs.:(1)))) -
            CA.C123(CA.wcurlₕ(CA.C123(CA.curlₕ(ᶜuʲs.:(1)))))
        ),
        T¹ = @.(CA.wdivₕ(CA.gradₕ(ᶜTʲs.:(1) - ᶜT_r))),
        q_tot_eff¹ = @.(CA.wdivₕ(CA.gradₕ(Y.c.sgsʲs.:(1).q_tot - ᶜq_tot_r))),
        q_gas_A¹ = @.(CA.wdivₕ(CA.gradₕ(Y.c.sgsʲs.:(1).q_gas_A))),
    )
    foreach(field -> Spaces.weighted_dss!(field), ref)
    return ref
end

bitwise_equal(a, b) = all(parent(a) .=== parent(b))

@noinline _alloc_start_hyperdiffusion!(Y, p) =
    @allocated CA.start_hyperdiffusion_tendency!(Y, p)
@noinline function _alloc_finish_hyperdiffusion_dss!(p)
    (; ᶜ∇², hyperdiffusion_ghost_buffer) = p.hyperdiff
    return @allocated begin
        Spaces.weighted_dss_internal!(ᶜ∇², hyperdiffusion_ghost_buffer)
        Spaces.weighted_dss_ghost!(ᶜ∇², hyperdiffusion_ghost_buffer)
    end
end

@testset "Packed hyperdiffusion Laplacians: one DSS, bitwise parity, 0 B allocations" begin
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
            job_id = "hyperdiffusion_packed_$FT",
        )
        (; Y, p, simulation) = generate_test_simulation(config)
        t = simulation.integrator.t
        @test CA.do_dss(axes(Y.c))
        @test CA.hyperdiffused_gs_tracer_names(Y) == (MatrixFields.@name(ρq_gas_A),)
        @test CA.hyperdiffused_sgs_tracer_names(Y) == (MatrixFields.@name(q_gas_A),)
        @test propertynames(p.hyperdiff.ᶜ∇²) == (
            :u, :T, :Φ, :q_tot_eff, :specific_tracers, :tke,
            :uʲs, :Tʲs, :q_tot_effʲs, :sgs_tracerʲs,
        )
        # The named Laplacians are views of the packed field.
        p.hyperdiff.ᶜ∇²T .= FT(1)
        @test all(isequal(FT(1)), parent(p.hyperdiff.ᶜ∇².T))

        # Perturb every input along all three dimensions, so that a Laplacian
        # read from the wrong component or level would not match.
        ᶜcoord = Fields.coordinate_field(Y.c)
        ᶜpert = @. FT(1) +
           FT(0.02) *
           sin(FT(2π) * ᶜcoord.x / FT(6400)) *
           cos(FT(4π) * ᶜcoord.y / FT(6400)) *
           (FT(1) + ᶜcoord.z / FT(3000))
        @. Y.c.ρe_tot *= ᶜpert
        @. Y.c.ρq_tot *= FT(2) - ᶜpert
        @. Y.c.ρtke = Y.c.ρ * FT(0.5) * ᶜpert
        @. Y.c.uₕ *= ᶜpert
        @. Y.c.ρq_gas_A = Y.c.ρ * FT(1e-5) * ᶜpert^2
        @. Y.c.sgsʲs.:(1).q_gas_A =
            FT(2e-5) * (
                FT(1) +
                FT(0.4) *
                cos(FT(2π) * ᶜcoord.x / FT(6400)) *
                sin(FT(2π) * ᶜcoord.y / FT(6400)) *
                cos(FT(π) * ᶜcoord.z / FT(3000))
            )
        @. Y.c.sgsʲs.:(1).q_tot *= ᶜpert
        @. Y.c.sgsʲs.:(1).mse *= ᶜpert
        CA.set_precomputed_quantities!(Y, p, t)

        # Compare the DSSed Laplacians before the apply functions, which
        # overwrite `∇²u` and `∇²uʲs` with `∇⁴u` and `∇⁴uʲs`.
        CA.start_hyperdiffusion_tendency!(Y, p)
        (; ᶜ∇², hyperdiffusion_ghost_buffer) = p.hyperdiff
        Spaces.weighted_dss_internal!(ᶜ∇², hyperdiffusion_ghost_buffer)
        Spaces.weighted_dss_ghost!(ᶜ∇², hyperdiffusion_ghost_buffer)
        ref = reference_laplacians(Y, p)
        @test maximum(abs, parent(ref.q_gas_A¹)) > zero(FT)
        @test bitwise_equal(ᶜ∇².u, ref.u)
        @test bitwise_equal(ᶜ∇².T, ref.T)
        @test bitwise_equal(ᶜ∇².Φ, ref.Φ)
        @test bitwise_equal(ᶜ∇².q_tot_eff, ref.q_tot_eff)
        @test bitwise_equal(ᶜ∇².tke, ref.tke)
        @test bitwise_equal(ᶜ∇².specific_tracers.q_gas_A, ref.q_gas_A)
        @test bitwise_equal(ᶜ∇².uʲs.:(1), ref.u¹)
        @test bitwise_equal(ᶜ∇².Tʲs.:(1), ref.T¹)
        @test bitwise_equal(ᶜ∇².q_tot_effʲs.:(1), ref.q_tot_eff¹)
        @test bitwise_equal(ᶜ∇².sgs_tracerʲs.q_gas_A.:(1), ref.q_gas_A¹)

        Yₜ = zero(Y)
        Yₜ_lim = zero(Y)
        CA.hyperdiffusion_tendency!(Yₜ, Yₜ_lim, Y, p, t)

        # Reference SGS tracer tendency from the separately DSSed Laplacian.
        (; ν₄_scalar) = CA.ν₄(p.atmos.hyperdiff, Y)
        ref_tend = similar(Y.c, FT)
        ref_tend .= zero(FT)
        @. ref_tend -= ν₄_scalar * CA.wdivₕ(CA.gradₕ(ref.q_gas_A¹))
        @test maximum(abs, parent(ref_tend)) > zero(FT)
        @test bitwise_equal(Yₜ_lim.c.sgsʲs.:(1).q_gas_A, ref_tend)

        if Base.JLOptions().check_bounds != 1
            _alloc_start_hyperdiffusion!(Y, p)
            _alloc_finish_hyperdiffusion_dss!(p)
            @test _alloc_start_hyperdiffusion!(Y, p) == 0
            @test _alloc_finish_hyperdiffusion_dss!(p) == 0
        end
    end
end
