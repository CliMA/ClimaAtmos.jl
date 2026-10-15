#=
Single-column test of the below-cloud evaporation and sublimation of the
0-moment precipitation flux, without EDMF and with PrognosticEDMFX: the water,
mass and energy column budgets close with the surface fluxes, which stay
non-negative (downward) when the frozen evaporation exceeds the production
refreshed from a Newton iterate.
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
            (Y1, p1) = build(edmf; overrides = EVAP_ON)
            @test parent(Y0) == parent(Y1)
            sfc(p) = (
                scalar(p.precomputed.surface_rain_flux),
                scalar(p.precomputed.surface_snow_flux),
                scalar(p.precomputed.col_integrated_precip_energy_tendency),
            )

            @testset "evaporation and column budgets" begin
                @test all(iszero, parent(p0.precomputed.ᶜprecip_evap))
                Yₜ0 = mp_tendency(Y0, p0)
                Yₜ1 = mp_tendency(Y1, p1)
                # Same production; only evaporation differs.
                @test parent(p1.precomputed.ᶜρ_dq_tot_dt) ==
                      parent(p0.precomputed.ᶜρ_dq_tot_dt)
                (; ᶜprecip_evap) = p1.precomputed
                ᶜρ_evap = @. ᶜprecip_evap.ρ_evap_rai + ᶜprecip_evap.ρ_evap_sno
                ᶜρe_evap = @. ᶜprecip_evap.ρe_evap_rai + ᶜprecip_evap.ρe_evap_sno
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

                (rain1, snow1, energy1) = sfc(p1)
                (rain0, snow0, energy0) = sfc(p0)
                @test rain1 <= 0 && snow1 <= 0
                # Less reaches the surface, by exactly what evaporates.
                @test (rain1 + snow1) - (rain0 + snow0) ≈ prevap rtol = 1e-10
                # Column budgets close with the surface fluxes.
                @test scalar(colint(Y1, Yₜ1.c.ρq_tot)) ≈ rain1 + snow1 rtol = 1e-10
                @test scalar(colint(Y1, Yₜ1.c.ρ)) ≈ rain1 + snow1 rtol = 1e-10
                @test scalar(colint(Y1, Yₜ1.c.ρe_tot)) ≈ energy1 rtol = 1e-10
                @test col(@. Yₜ1.c.ρq_tot - Yₜ0.c.ρq_tot) ≈ col(ᶜρ_evap)
                @test col(@. Yₜ1.c.ρe_tot - Yₜ0.c.ρe_tot) ≈ col(ᶜρe_evap)
                if edmf
                    # Updraft prognostics untouched: the environment gets it.
                    @test parent(Yₜ1.c.sgsʲs) == parent(Yₜ0.c.sgsʲs)
                end

                # Newton iterate with much less production than the frozen
                # evaporation: the evaporation is rescaled per column and
                # phase, so the surface fluxes stay non-negative (downward)
                # and the column budgets stay closed.
                mm, tm = p1.atmos.microphysics_model, p1.atmos.turbconv_model
                evap_before = copy(parent(ᶜprecip_evap))
                Y2 = copy(Y1)
                @. Y2.c.ρ *= 1e-3
                @. Y2.c.ρq_tot *= 1e-3
                @. Y2.c.ρe_tot *= 1e-3
                if edmf
                    sgs₁ = Y2.c.sgsʲs.:(1)
                    @. sgs₁.ρa *= 1e-3
                end
                CA.update_implicit_microphysics_cache!(Y2, p1, mm, tm)
                @test parent(ᶜprecip_evap) == evap_before
                (; precip_evap_scale_rai, precip_evap_scale_sno) = p1.precomputed
                r_rai = scalar(precip_evap_scale_rai)
                r_sno = scalar(precip_evap_scale_sno)
                @test 0 <= r_rai <= 1 && 0 <= r_sno <= 1
                @test min(r_rai, r_sno) < 1
                (rain2, snow2, energy2) = sfc(p1)
                @test rain2 <= 0 && snow2 <= 0
                Yₜ2 = mp_tendency(Y2, p1)
                prod2 = -scalar(colint(Y2, p1.precomputed.ᶜρ_dq_tot_dt))
                eprod2 = abs(scalar(colint(Y2, p1.precomputed.ᶜρ_de_tot_dt)))
                @test scalar(getd("prevap", Y2, p1)) <= prod2 * (1 + 1e-12)
                tol, etol = 1e-10 * prod2, 1e-10 * eprod2
                @test scalar(colint(Y2, Yₜ2.c.ρq_tot)) ≈ rain2 + snow2 atol = tol
                @test scalar(colint(Y2, Yₜ2.c.ρ)) ≈ rain2 + snow2 atol = tol
                @test scalar(colint(Y2, Yₜ2.c.ρe_tot)) ≈ energy2 atol = etol
            end
        end
    end
end
