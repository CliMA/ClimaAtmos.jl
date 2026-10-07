#=
Single-column tests of the below-cloud evaporation / sublimation of the
0-moment precipitation flux (`set_precipitation_evaporation_cache!`,
`precipitation_evaporation_tendency!`, the 0M surface fluxes, and the `prevap` /
`tnhusevp` diagnostics), without EDMF and with PrognosticEDMFX:

  - `precipitation_evaporation_coefficient = 0` (the default) is bit-for-bit;
  - the water, mass and energy column budgets close with the surface fluxes;
  - the top-down scan matches a plain reference loop over the column;
  - updraft prognostics are untouched (the environment receives the water);
  - the cache update does not allocate.
=#
using Test
using Dates
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, Operators
import ClimaCore.Utilities: half
import ClimaParams as CP
import Thermodynamics as TD

const FT = Float64
const grid = CA.ColumnGrid(FT; z_elem = 30, z_max = 15e3, z_stretch = false)

tm_pedmfx(params) = (;
    turbconv_model = CA.PrognosticEDMFX(;
        area_fraction = CAP.turbconv_params(params).min_area,
    ),
    edmfx_model = CA.EDMFXModel(;
        entr_model = CA.InvZEntrainment(),
        detr_model = CA.BuoyancyVelocityDetrainment(),
        scale_blending_method = CA.SmoothMinimumBlending(),
    ),
)

const EVAP_ON = Dict(
    "precipitation_evaporation_coefficient" =>
        Dict("value" => 5.44e-4, "type" => "float"),
    "precipitation_evaporation_area_fraction" =>
        Dict("value" => 0.3, "type" => "float"),
)
# Other evaporation keys set, but the coefficient zero: must still be off.
const EVAP_ZERO = Dict(
    "precipitation_evaporation_coefficient" =>
        Dict("value" => 0.0, "type" => "float"),
    "precipitation_evaporation_rh_crit" =>
        Dict("value" => 0.8, "type" => "float"),
)

"""
Build a 0M column (optionally with PrognosticEDMFX) with a cloudy layer at
3-6 km (q_tot = 1.5 q_sat) over dry air (q_tot = 0.4 q_sat), and set all
precomputed quantities.
"""
function build(edmf::Bool; overrides = Dict())
    params = CA.ClimaAtmosParameters(
        CP.create_toml_dict(FT; override_file = overrides),
    )
    model = CA.AtmosModel(
        grid;
        params,
        setup = CA.Setups.DecayingProfile(; params),
        microphysics_model = CA.EquilibriumMicrophysics0M(),
        (edmf ? tm_pedmfx(params) : (;))...,
    )
    Y = CA.initial_state(model)
    p = CA.build_cache(Y, model, params, FT(150), DateTime(2010, 1, 1), nothing)
    CA.set_precomputed_quantities!(Y, p, FT(0))
    thp = CAP.thermodynamics_params(params)
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜq_sat = @. TD.q_vap_saturation(thp, p.precomputed.ᶜT, Y.c.ρ)
    @. Y.c.ρq_tot =
        Y.c.ρ * ifelse((ᶜz > 3e3) & (ᶜz < 6e3), 1.5 * ᶜq_sat, 0.4 * ᶜq_sat)
    CA.set_precomputed_quantities!(Y, p, FT(0))
    return Y, p
end

function mp_tendency(Y, p)
    Yₜ = zero(Y)
    CA.microphysics_tendency!(
        Yₜ, Y, p, FT(0), p.atmos.microphysics_model, p.atmos.turbconv_model,
    )
    return Yₜ
end

function colint(Y, f)
    out = zeros(axes(Fields.level(Y.f, half)))
    Operators.column_integral_definite!(out, f)
    return out
end
scalar(f) = only(parent(f))
col(f) = vec(Array(parent(f)))
getd(name, Y, p) = Base.materialize(
    CA.Diagnostics.get_diagnostic_variable(name).compute(Y, p, FT(0)),
)

@testset "0M precipitation evaporation (single column)" begin
    for edmf in (false, true)
        @testset "PrognosticEDMFX = $edmf" begin
            (Y0, p0) = build(edmf)
            (Yz, pz) = build(edmf; overrides = EVAP_ZERO)
            (Y1, p1) = build(edmf; overrides = EVAP_ON)
            @test parent(Y0) == parent(Yz) == parent(Y1)
            @test !CA.precipitation_evaporation_active(p0.params)
            @test !CA.precipitation_evaporation_active(pz.params)
            @test CA.precipitation_evaporation_active(p1.params)
            sfc(p) = (
                scalar(p.precomputed.surface_rain_flux),
                scalar(p.precomputed.surface_snow_flux),
                scalar(p.precomputed.col_integrated_precip_energy_tendency),
            )

            @testset "k_E = 0 is bit-for-bit" begin
                Yₜ0 = mp_tendency(Y0, p0)
                Yₜz = mp_tendency(Yz, pz)
                @test parent(Yₜ0) == parent(Yₜz)
                @test sfc(p0) === sfc(pz)
                @test all(iszero, parent(p0.precomputed.ᶜprecip_evap))
                @test all(iszero, parent(pz.precomputed.ᶜprecip_evap))
                @test all(iszero, parent(getd("prevap", Y0, p0)))
                @test all(iszero, parent(getd("tnhusevp", Y0, p0)))
                # The old (pre-evaporation) formulas, written out.
                (; ᶜρ_dq_tot_dt, ᶜρ_de_tot_dt, ᶜT) = p0.precomputed
                T_f = TD.Parameters.T_freeze(CAP.thermodynamics_params(p0.params))
                rain = colint(Y0, @. ifelse(ᶜT >= T_f, ᶜρ_dq_tot_dt, FT(0)))
                snow = colint(Y0, @. ifelse(ᶜT < T_f, ᶜρ_dq_tot_dt, FT(0)))
                @test sfc(p0) ===
                      (scalar(rain), scalar(snow), scalar(colint(Y0, ᶜρ_de_tot_dt)))
                if !edmf
                    (; ᶜmp_tendency) = p0.precomputed
                    ρdq = @. Y0.c.ρ * ᶜmp_tendency.dq_tot_dt
                    @test parent(Yₜ0.c.ρq_tot) == parent(ρdq)
                    @test parent(Yₜ0.c.ρ) == parent(ρdq)
                    @test parent(Yₜ0.c.ρe_tot) ==
                          parent(@. ρdq * ᶜmp_tendency.e_tot_hlpr)
                else
                    # The grid-mean EDMF sinks are the cached aggregates.
                    @test parent(Yₜ0.c.ρq_tot) == parent(ᶜρ_dq_tot_dt)
                    @test parent(Yₜ0.c.ρ) == parent(ᶜρ_dq_tot_dt)
                    @test parent(Yₜ0.c.ρe_tot) == parent(ᶜρ_de_tot_dt)
                end
            end

            @testset "evaporation and column budgets" begin
                Yₜ0 = mp_tendency(Y0, p0)
                Yₜ1 = mp_tendency(Y1, p1)
                # Same production; only evaporation differs.
                @test parent(p1.precomputed.ᶜρ_dq_tot_dt) ==
                      parent(p0.precomputed.ᶜρ_dq_tot_dt)
                (; ᶜprecip_evap) = p1.precomputed
                ᶜρ_evap = @. ᶜprecip_evap.ρ_evap_rai + ᶜprecip_evap.ρ_evap_sno
                ᶜz = Fields.coordinate_field(Y1.c).z
                prod = -scalar(colint(Y1, p1.precomputed.ᶜρ_dq_tot_dt))
                prevap = scalar(getd("prevap", Y1, p1))
                @test prod > 0
                @test 0 < prevap < prod
                @test prevap ≈ scalar(colint(Y1, ᶜρ_evap))
                @test all(>=(0), col(ᶜprecip_evap.ρ_evap_rai))
                @test all(>=(0), col(ᶜprecip_evap.ρ_evap_sno))
                # Nothing evaporates above the cloud (no flux from above);
                # most of it below cloud base.
                @test all(iszero, col(ᶜρ_evap)[col(ᶜz) .> 6e3])
                @test scalar(colint(Y1, @. ᶜρ_evap * (ᶜz < 3e3))) > prevap / 2
                tnhusevp = getd("tnhusevp", Y1, p1)
                # d(ρq_tot/ρ)/dt, with the water added to both ρq_tot and ρ.
                q_tot = col(Y1.c.ρq_tot) ./ col(Y1.c.ρ)
                @test col(tnhusevp) ≈ (1 .- q_tot) .* col(ᶜρ_evap) ./ col(Y1.c.ρ)
                @test col(tnhusevp) ≈
                      (col(@. Y1.c.ρq_tot + 1e-3 * ᶜρ_evap) ./
                       col(@. Y1.c.ρ + 1e-3 * ᶜρ_evap) .- q_tot) ./ 1e-3 rtol = 1e-4
                @test maximum(col(tnhusevp)) > 0

                (rain1, snow1, energy1) = sfc(p1)
                (rain0, snow0, energy0) = sfc(p0)
                @test rain1 <= 0 && snow1 <= 0
                # Less reaches the surface, by exactly what evaporates.
                @test (rain1 + snow1) - (rain0 + snow0) ≈ prevap rtol = 1e-10
                # Column budgets close with the surface fluxes.
                @test scalar(colint(Y1, Yₜ1.c.ρq_tot)) ≈ rain1 + snow1 rtol = 1e-10
                @test scalar(colint(Y1, Yₜ1.c.ρ)) ≈ rain1 + snow1 rtol = 1e-10
                @test scalar(colint(Y1, Yₜ1.c.ρe_tot)) ≈ energy1 rtol = 1e-10
                Δρq = @. Yₜ1.c.ρq_tot - Yₜ0.c.ρq_tot
                @test col(Δρq) ≈ col(ᶜρ_evap)
                @test col(@. Yₜ1.c.ρe_tot - Yₜ0.c.ρe_tot) ≈
                      col(ᶜprecip_evap.ρe_evap)
                if edmf
                    # Updraft prognostics untouched: the environment gets it.
                    @test parent(Yₜ1.c.sgsʲs) == parent(Yₜ0.c.sgsʲs)
                end
            end

            @testset "top-down scan matches a reference loop" begin
                (; ᶜT, ᶜp, ᶜρ_dq_tot_dt, ᶜρ_de_tot_dt, ᶜprecip_evap) =
                    p1.precomputed
                thp = CAP.thermodynamics_params(p1.params)
                pe = CA.precipitation_evaporation_params(p1.params)
                T_f = TD.Parameters.T_freeze(thp)
                (ρ_env, ρa_env, T_env, q_vap) = map(
                    f -> col(Base.materialize(f)),
                    CA.precipitation_evaporation_air(
                        Y1, p1, p1.atmos.turbconv_model,
                    ),
                )
                Δz = col(Fields.Δz_field(Y1.c))
                T, pr, S, Φ = col(ᶜT), col(ᶜp), -col(ᶜρ_dq_tot_dt), col(p1.core.ᶜΦ)
                ρde = col(ᶜρ_de_tot_dt)
                n = length(T)
                ref = zeros(n, 3)
                F = zeros(n + 1, 4)  # F[k + 1] = fluxes entering cell k
                state = ntuple(_ -> zero(FT), 7)
                for k in n:-1:1
                    λ_src = CA.precipitation_source_liquid_fraction(
                        thp, -S[k], ρde[k], T[k], Φ[k],
                    )
                    inp = (
                        T[k] >= T_f ? S[k] : zero(FT),
                        T[k] < T_f ? S[k] : zero(FT),
                        λ_src, Δz[k], ρ_env[k], ρa_env[k], T_env[k],
                        pr[k] / pr[1], q_vap[k], Φ[k],
                    )
                    state = CA.precipitation_evaporation_step(
                        pe, thp, p1.dt, state, inp,
                    )
                    ref[k, :] .= state[5:7]
                    F[k, :] .= state[1:4]
                end
                @test col(ᶜprecip_evap.ρ_evap_rai) ≈ ref[:, 1] rtol = 1e-12
                @test col(ᶜprecip_evap.ρ_evap_sno) ≈ ref[:, 2] rtol = 1e-12
                @test col(ᶜprecip_evap.ρe_evap) ≈ ref[:, 3] rtol = 1e-12
                @test all(>=(0), F)
                @test all(F[:, 3] .<= F[:, 1]) && all(F[:, 4] .<= F[:, 2])
                # The flux at the surface is the (net) surface precipitation.
                (rain1, snow1, _) = sfc(p1)
                @test F[1, 1] ≈ -rain1 rtol = 1e-8
                @test F[1, 2] ≈ -snow1 rtol = 1e-8 atol = 1e-15
            end

            @testset "allocations" begin
                tm = p1.atmos.turbconv_model
                CA.set_precipitation_evaporation_cache!(Y1, p1, tm)
                CA.set_precipitation_surface_fluxes!(Y1, p1, p1.atmos.microphysics_model)
                Yₜ = zero(Y1)
                CA.precipitation_evaporation_tendency!(Yₜ, p1)
                @test (@allocated CA.set_precipitation_evaporation_cache!(Y1, p1, tm)) == 0
                # The CPU column integrals of the 0M surface fluxes allocate a
                # little already with evaporation off; no more with it on.
                mm = p1.atmos.microphysics_model
                CA.set_precipitation_surface_fluxes!(Y0, p0, mm)
                a_off = @allocated CA.set_precipitation_surface_fluxes!(Y0, p0, mm)
                a_on = @allocated CA.set_precipitation_surface_fluxes!(Y1, p1, mm)
                @info "set_precipitation_surface_fluxes! allocations" edmf a_off a_on
                @test a_on <= a_off
                @test (@allocated CA.precipitation_evaporation_tendency!(Yₜ, p1)) == 0
            end

            @testset "implicit refresh (frozen evaporation)" begin
                # Newton iterate: ρ drifts; `update_implicit_microphysics_cache!`
                # refreshes the sink and surface fluxes, `ᶜprecip_evap` is frozen.
                mm, tm = p1.atmos.microphysics_model, p1.atmos.turbconv_model
                evap_before = copy(parent(p1.precomputed.ᶜprecip_evap))
                Y2 = copy(Y1)
                @. Y2.c.ρ *= 1 + 1e-4
                @. Y2.c.ρq_tot *= 1 + 1e-4
                @. Y2.c.ρe_tot *= 1 + 1e-4
                CA.update_implicit_microphysics_cache!(Y2, p1, mm, tm)
                @test parent(p1.precomputed.ᶜprecip_evap) == evap_before
                (rain2, snow2, energy2) = sfc(p1)
                Yₜ2 = mp_tendency(Y2, p1)
                @test scalar(colint(Y2, Yₜ2.c.ρq_tot)) ≈ rain2 + snow2 rtol = 1e-10
                @test scalar(colint(Y2, Yₜ2.c.ρe_tot)) ≈ energy2 rtol = 1e-10
                # Even when the frozen evaporation exceeds the refreshed
                # production (large drift), the unclamped surface fluxes keep
                # the column water and energy budgets closed.
                @. Y2.c.ρ *= 1e-3
                @. Y2.c.ρq_tot *= 1e-3
                @. Y2.c.ρe_tot *= 1e-3
                if edmf
                    sgs₁ = Y2.c.sgsʲs.:(1)
                    @. sgs₁.ρa *= 1e-3
                end
                CA.update_implicit_microphysics_cache!(Y2, p1, mm, tm)
                (rain3, snow3, energy3) = sfc(p1)
                Yₜ3 = mp_tendency(Y2, p1)
                @test scalar(colint(Y2, Yₜ3.c.ρq_tot)) ≈ rain3 + snow3 rtol = 1e-10
                @test scalar(colint(Y2, Yₜ3.c.ρe_tot)) ≈ energy3 rtol = 1e-10
                (; ᶜprecip_evap) = p1.precomputed
                evap_r = scalar(colint(Y1, ᶜprecip_evap.ρ_evap_rai))
                evap_s = scalar(colint(Y1, ᶜprecip_evap.ρ_evap_sno))
                @test evap_r + evap_s > 0
            end
        end
    end
end
