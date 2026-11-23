using Test
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore.MatrixFields as MatrixFields
import ClimaTimeSteppers as CTS
import Thermodynamics as TD

@testset "GP entrainment/detrainment and hydrometeor shedding" begin
    @testset "SparseMultiHeadMaternGP32 table, OOD, TTL taper, and validation" begin
        @test isbitstype(CA.SparseMultiHeadMaternGP32{6, 10, 60, 55})
        @test sizeof(CA.SparseMultiHeadMaternGP32{6, 10, 60, 55}) == 672

        p0 = CA.TurbulenceConvectionParameters(Float32)
        p1 = CA.TurbulenceConvectionParameters(
            Float32,
            (;
                edmf_gp_entr_weights = fill(0.3f0, 10),
                edmf_gp_detr_buoy_weights = fill(-0.2f0, 10),
                edmf_gp_detr_mix_weights = fill(0.1f0, 10),
                edmf_detr_mix_coeff = 0.08f0,
                edmf_condensate_detr_coeff = 0.5f0,
                edmf_precip_detr_coeff = 0.5f0,
            ),
        )
        @test typeof(p0) === typeof(p1)
        gp0 = CAP.gp_entr_detr(p0)
        gp1 = CAP.gp_entr_detr(p1)

        x = (0.1f0, 0.2f0, 0.17f0, 0.13f0, 0.27f0, 0.02f0)
        r0 = CA.eval_gp_mu3_var(gp0, x)
        @test r0[1] === 0.0f0 && r0[2] === 0.0f0 && r0[3] === 0.0f0
        @test 0.0f0 <= r0[4] <= 1.0f0

        x_anchor1 = ntuple(d -> gp1.mu_x[d] + gp1.Z[d] / gp1.inv_std_x[d], Val(6))
        @test CA.eval_gp_mu3_var(gp1, x_anchor1)[4] < 1.0f-3
        @test CA.eval_gp_mu3_var(
            gp1,
            (1.0f0, 1.0f0, 1.0f0, 1.0f0, 50.0f0, 1.0f0),
        )[4] > 0.999f0

        @test CA._gp_ttl_taper(1.90f0, 1.90f0, 2.40f0) === 1.0f0
        @test CA._gp_ttl_taper(2.40f0, 1.90f0, 2.40f0) === 0.0f0
        @test CA._gp_ttl_taper(3.00f0, 1.90f0, 2.40f0) === 0.0f0

        thp = TD.Parameters.ThermodynamicsParameters(Float32)
        for em in (
                CA.InvZEntrainment(),
                CA.GPInvZEntrainment(),
                CA.PiGroupsEntrainment(),
            ),
            dm in (CA.BuoyancyVelocityDetrainment(), CA.GPBuoyancyDetrainment())

            rts = Base.return_types(
                CA.gp_entr_detr_closures,
                (
                    typeof(thp),
                    typeof(p1),
                    ntuple(_ -> Float32, Val(14))...,
                    typeof(em),
                    typeof(dm),
                ),
            )
            @test rts == [Tuple{Float32, Float32, Float32}]
        end

        @test_throws ArgumentError CA.TurbulenceConvectionParameters(
            Float32,
            (; edmf_condensate_detr_coeff = -0.1f0),
        )
        @test_throws ArgumentError CA.TurbulenceConvectionParameters(
            Float32,
            (; edmf_precip_detr_coeff = -0.1f0),
        )
        @test_throws ArgumentError CA.TurbulenceConvectionParameters(
            Float32,
            (; edmf_detr_mix_coeff = -0.1f0),
        )
        @test_throws ArgumentError CA.TurbulenceConvectionParameters(
            Float32,
            (; edmf_gp_log_cap = 0.0f0),
        )
        @test_throws ArgumentError CA.TurbulenceConvectionParameters(
            Float32,
            (; edmf_gp_ood_scale = 0.0f0),
        )
        @test_throws ArgumentError CA.TurbulenceConvectionParameters(
            Float32,
            (; edmf_gp_entr_weights = zeros(Float32, 9)),
        )
        @test CA.get_entrainment_model(Dict("edmfx_entr_model" => "GPInvZ")) isa
              CA.GPInvZEntrainment
        @test CA.get_detrainment_model(Dict("edmfx_detr_model" => "GPBuoyancy")) isa
              CA.GPBuoyancyDetrainment
    end

    function _make_col_sim(
        toml_path;
        entr = "Generalized",
        detr = "Generalized",
        n_picard = 1,
        auto_jac = false,
    )
        cfg_dict = Dict(
            "config" => "column",
            "initial_condition" => "MPACE",
            "surface_setup" => "MPACE",
            "microphysics_model" => "1M",
            "snow_deposition_sublimation" => "DepositionAndSublimation",
            "cloud_ice_terminal_velocity" => "Chen2022",
            "rain_terminal_velocity" => "Chen2022",
            "snow_terminal_velocity" => "Chen2022",
            "turbconv" => "prognostic_edmfx",
            "edmfx_entr_model" => entr,
            "edmfx_detr_model" => detr,
            "edmfx_sgs_mass_flux" => true,
            "edmfx_sgs_diffusive_flux" => true,
            "edmfx_nh_pressure" => true,
            "edmfx_filter" => true,
            "edmfx_updraft_picard_rounds" => n_picard,
            "prognostic_tke" => true,
            "use_sgs_quadrature" => true,
            "sgs_distribution" => "gaussian",
            "quadrature_order" => 3,
            "FLOAT_TYPE" => "Float32",
            "z_elem" => 30,
            "z_max" => 10000.0,
            "z_stretch" => false,
            "dt" => "120secs",
            "t_end" => "240secs",
            "output_default_diagnostics" => false,
        )
        isnothing(toml_path) || (cfg_dict["toml"] = [toml_path])
        auto_jac && (cfg_dict["auto_jacobian"] = true)
        return CA.get_simulation(CA.AtmosConfig(cfg_dict))
    end

    @testset "0-ULP parity, picard=0 sweep, and conservation" begin
        sim_gen = _make_col_sim(nothing; entr = "Generalized", detr = "Generalized")
        sim_gp0 = _make_col_sim(nothing; entr = "GPInvZ", detr = "GPBuoyancy")
        CTS.step!(sim_gen.integrator)
        CTS.step!(sim_gp0.integrator)
        @test Array(parent(sim_gen.integrator.u.c)) ==
              Array(parent(sim_gp0.integrator.u.c))
        @test Array(parent(sim_gen.integrator.u.f)) ==
              Array(parent(sim_gp0.integrator.u.f))

        mktemp() do toml_path, io
            write(
                io,
                """
                [edmf_gp_entr_weights]
                value = [0.2, -0.1, 0.15, -0.3, 0.1, 0.0, -0.2, 0.25, -0.1, 0.1]
                type = "float"
                [edmf_gp_detr_buoy_weights]
                value = [-0.2, 0.2, -0.2, -0.4, 0.2, 0.1, 0.1, 0.2, 0.15, 0.3]
                type = "float"
                [edmf_gp_detr_mix_weights]
                value = [0.0, -0.2, -0.3, -0.1, 0.2, 0.1, 0.0, 0.4, 0.3, 0.2]
                type = "float"
                [edmf_detr_mix_coeff]
                value = 0.08
                type = "float"
                [edmf_condensate_detr_coeff]
                value = 0.5
                type = "float"
                [edmf_precip_detr_coeff]
                value = 0.5
                type = "float"
                """,
            )
            flush(io)
            # Verify updraft_picard_rounds == 0 shedding sweep runs cleanly
            sim_p0 =
                _make_col_sim(toml_path; entr = "GPInvZ", detr = "GPBuoyancy", n_picard = 0)
            CTS.step!(sim_p0.integrator)
            @test all(isfinite, Array(parent(sim_p0.integrator.u.c.sgsʲs.:(1).q_tot)))
            @test all(isfinite, Array(parent(sim_p0.integrator.u.c.sgsʲs.:(1).mse)))

            # Verify grid-mean conservation in edmfx_entr_detr_tendency!
            Y = deepcopy(sim_p0.integrator.u)
            p = sim_p0.integrator.p
            t = sim_p0.integrator.t
            @. Y.c.sgsʲs.:(1).q_lcl += 5.0f-4
            @. Y.c.sgsʲs.:(1).q_tot += 5.0f-4
            @. Y.c.ρq_lcl += Y.c.sgsʲs.:(1).ρa * 5.0f-4
            @. Y.c.ρq_tot += Y.c.sgsʲs.:(1).ρa * 5.0f-4
            CA.set_precomputed_quantities!(Y, p, t)
            Yₜ = zero(Y)
            CA.edmfx_entr_detr_tendency!(Yₜ, Y, p, t, p.atmos.turbconv_model)
            @test all(iszero, Array(parent(Yₜ.c.ρ)))
            @test all(iszero, Array(parent(Yₜ.c.ρq_tot)))
            @test all(iszero, Array(parent(Yₜ.c.ρe_tot)))
            @test all(iszero, Array(parent(Yₜ.c.ρq_lcl)))
        end
    end
end
