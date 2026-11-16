#=
Single-column test of the below-cloud evaporation and sublimation of the
0-moment precipitation flux in `step_coupled_microphysics_sedimentation!`,
without EDMF and with PrognosticEDMFX: the water, mass, and energy column
budgets close with the surface fluxes, which stay non-positive. When all
precipitation evaporates (virga), no energy reaches the surface.
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
const TD = CA.TD

const FT = Float64
const grid = CA.ColumnGrid(FT; z_elem = 30, z_max = 15e3, z_stretch = false)
const dt = FT(120)

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

evap_coefficient(k_E) = Dict(
    "precipitation_evaporation_coefficient" =>
        Dict("value" => k_E, "type" => "float"),
)

"""
Build a 0M column (optionally with PrognosticEDMFX) with a cloudy layer at
3-6 km (q_tot = 1.5 q_sat) over dry air (q_tot = 0.4 q_sat), and set all
precomputed quantities. The TOML always sets the positive evaporation
coefficient `k_E`; `evaporation` turns the `precipitation_evaporation_0m`
option on.
"""
function build(edmf::Bool; evaporation = false, k_E = 5.44e-4)
    params = CA.ClimaAtmosParameters(
        CP.create_toml_dict(FT; override_file = evap_coefficient(k_E));
        precipitation_evaporation_0m = evaporation,
    )
    model = CA.AtmosModel(
        grid;
        params,
        setup = CA.Setups.DecayingProfile(; params),
        microphysics_model = CA.EquilibriumMicrophysics0M(),
        (edmf ? tm_pedmfx(params) : (;))...,
    )
    Y = CA.initial_state(model)
    p = CA.build_cache(Y, model, params, dt, DateTime(2010, 1, 1), nothing)
    CA.set_precomputed_quantities!(Y, p, FT(0))
    thp = CAP.thermodynamics_params(params)
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜq_sat = @. TD.q_vap_saturation(thp, p.precomputed.ᶜT, Y.c.ρ)
    @. Y.c.ρq_tot =
        Y.c.ρ * ifelse((ᶜz > 3e3) & (ᶜz < 6e3), 1.5 * ᶜq_sat, 0.4 * ᶜq_sat)
    CA.set_precomputed_quantities!(Y, p, FT(0))
    return Y, p
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
sfc(p) = (
    scalar(p.precomputed.surface_rain_flux),
    scalar(p.precomputed.surface_snow_flux),
    scalar(p.precomputed.col_integrated_precip_energy_tendency),
)

# Column integrals of dry air, water, and energy, and of |ρe_tot|.
budgets(Y) = (
    dry = scalar(colint(Y, @. Y.c.ρ - Y.c.ρq_tot)),
    water = scalar(colint(Y, Y.c.ρq_tot)),
    energy = scalar(colint(Y, Y.c.ρe_tot)),
    abs_energy = scalar(colint(Y, @. abs(Y.c.ρe_tot))),
)

# One sweep over `dt` on a copy of `Y`; returns the new state and the column
# budgets before and after.
function sweep(Y, p; kwargs...)
    Y1 = copy(Y)
    b0 = budgets(Y1)
    CA.step_coupled_microphysics_sedimentation!(Y1, p, dt; kwargs...)
    return Y1, b0, budgets(Y1)
end

# Dry air is unchanged, and the water and energy changes equal `dt` times the
# surface fluxes (upward-positive, so negative when it precipitates).
function test_column_budgets(b0, b1, p)
    (rain, snow, energy) = sfc(p)
    tol = 1e-12
    @test b1.dry ≈ b0.dry rtol = tol
    @test b1.water - b0.water ≈ dt * (rain + snow) atol = tol * b0.water
    @test b1.energy - b0.energy ≈ dt * energy atol = tol * b0.abs_energy
end

@testset "0M precipitation evaporation (single column)" begin
    for edmf in (false, true)
        @testset "PrognosticEDMFX = $edmf" begin
            (Y0, p0) = build(edmf)
            (Y1, p1) = build(edmf; evaporation = true)
            @test parent(Y0) == parent(Y1)
            @test isnothing(CAP.microphysics_0m_params(p0.params).evaporation)
            @test !isnothing(CAP.microphysics_0m_params(p1.params).evaporation)

            @testset "evaporation and column budgets" begin
                (Y0n, b00, b01) = sweep(Y0, p0)
                (Y1n, b10, b11) = sweep(Y1, p1)
                test_column_budgets(b00, b01, p0)
                test_column_budgets(b10, b11, p1)
                @test all(iszero, parent(p0.precomputed.ᶜprecip_evap))
                @test scalar(getd("prevap", Y0n, p0)) == 0

                (; ᶜprecip_evap) = p1.precomputed
                ᶜρ_evap = @. ᶜprecip_evap.ρ_evap_rai + ᶜprecip_evap.ρ_evap_sno
                ᶜz = Fields.coordinate_field(Y1.c).z
                (rain0, snow0, energy0) = sfc(p0)
                (rain1, snow1, energy1) = sfc(p1)
                prod = -(rain0 + snow0)
                prevap = scalar(getd("prevap", Y1n, p1))
                @test prod > 0
                @test 0 < prevap < prod
                @test prevap ≈ scalar(colint(Y1, ᶜρ_evap)) rtol = 1e-12
                @test all(>=(0), col(ᶜprecip_evap.ρ_evap_rai))
                @test all(>=(0), col(ᶜprecip_evap.ρ_evap_sno))
                # Nothing evaporates above the cloud (no flux from above);
                # most of it below cloud base.
                @test all(iszero, col(ᶜρ_evap)[col(ᶜz) .> 6e3])
                @test scalar(colint(Y1, @. ᶜρ_evap * (ᶜz < 3e3))) > prevap / 2
                @test rain1 <= 0 && snow1 <= 0
                # The sink is the same; less reaches the surface, by exactly
                # what evaporates, which returns to the air.
                @test (rain1 + snow1) - (rain0 + snow0) ≈ prevap rtol = 1e-10
                @test col(@. Y1n.c.ρq_tot - Y0n.c.ρq_tot) ≈ col(@. dt * ᶜρ_evap)
                @test col(@. Y1n.c.ρ - Y0n.c.ρ) ≈ col(@. dt * ᶜρ_evap)
                ᶜtnhusevp = getd("tnhusevp", Y1n, p1)
                @test all(>=(0), col(ᶜtnhusevp))
                @test scalar(colint(Y1n, @. Y1n.c.ρ * ᶜtnhusevp)) > 0
                if edmf
                    # Updraft prognostics untouched: the environment gets it.
                    @test parent(Y1n.c.sgsʲs) == parent(Y0n.c.sgsʲs)
                    @test scalar(getd("prup", Y1n, p1)) ==
                          scalar(getd("prup", Y0n, p0))
                end
            end

            @testset "step means over substeps" begin
                # Two half-weight substeps from the same state average to the
                # diagnostics of a single step.
                pre = p1.precomputed
                col_evap() =
                    scalar(pre.col_integrated_precip_evap_rai) +
                    scalar(pre.col_integrated_precip_evap_sno)
                ᶜevap() =
                    col(pre.ᶜprecip_evap.ρ_evap_rai) .+
                    col(pre.ᶜprecip_evap.ρ_evap_sno)
                sweep(Y1, p1)
                (evap_one, ᶜevap_one) = (col_evap(), ᶜevap())
                sweep(Y1, p1; diag_weight = 0.5, diag_reset = true)
                sweep(Y1, p1; diag_weight = 0.5, diag_reset = false)
                @test evap_one > 0
                @test col_evap() ≈ evap_one rtol = 1e-14
                @test ᶜevap() ≈ ᶜevap_one
            end

            @testset "virga: no energy reaches the surface without water" begin
                # A large coefficient evaporates all precipitation in the dry
                # layer below the cloud.
                (Y3, p3) = build(edmf; evaporation = true, k_E = 10.0)
                (Y3n, b30, b31) = sweep(Y3, p3)
                test_column_budgets(b30, b31, p3)
                (Y0n, _, _) = sweep(Y0, p0)
                (rain0, snow0, energy0) = sfc(p0)
                prod = -(rain0 + snow0)
                @test prod > 0
                @test scalar(getd("prevap", Y3n, p3)) ≈ prod rtol = 1e-10
                (rain3, snow3, energy3) = sfc(p3)
                @test rain3 ≈ 0 atol = 1e-10 * prod
                @test snow3 ≈ 0 atol = 1e-10 * prod
                @test energy3 ≈ 0 atol = 1e-10 * abs(energy0)
            end
        end
    end
end
