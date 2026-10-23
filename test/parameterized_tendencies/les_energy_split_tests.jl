#=
The LES closures diffuse energy through the split enthalpy flux
`-ρ D [∇s_d + (h_eff + Φ) ∇q_tot_eff]`: for the horizontal and vertical
Smagorinsky-Lilly tendencies and the constant horizontal diffusion, the ρe_tot
tendency equals the two parts assembled independently and differs from
diffusing h_tot. The AMD tendencies are finite and nonzero on a moist state
whose wind varies in all three directions. AMD is built from products of the
velocity gradient, so its scalar diffusivity vanishes identically for a wind
that depends on height alone, and its vertical part vanishes again wherever
the vertical velocity is uniform in height, whatever the scalar gradients are.
=#
using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, Operators

include("../test_helpers.jl")

function les_box_config(job_id; extra...)
    dict = Dict{String, Any}(
        "config" => "box",
        "initial_condition" => "Bomex",
        "FLOAT_TYPE" => "Float64",
        "microphysics_model" => "1M",
        "hyperdiff" => nothing,
        "x_max" => 6400.0,
        "x_elem" => 2,
        "y_max" => 6400.0,
        "y_elem" => 2,
        "z_max" => 3000.0,
        "z_elem" => 10,
        "z_stretch" => false,
        "dt" => "1secs",
        "t_end" => "10secs",
        "output_default_diagnostics" => false,
    )
    for (key, value) in extra
        dict[String(key)] = value
    end
    return CA.AtmosConfig(dict; job_id)
end

# Perturb the state horizontally and vertically so the fluxes are nonzero, then
# refresh the cache. `velocity` also tilts the horizontal wind and adds a
# vertical velocity, which AMD needs: its diffusivity is a product of velocity
# gradients, and the vertical part scales with ∂w/∂z.
function perturb!(Y, p, t; velocity = false)
    FT = eltype(Y)
    ᶜx = Fields.coordinate_field(Y.c).x
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜpert = @. 1 + FT(0.1) * sin(FT(2π) * ᶜx / FT(6400)) * cos(FT(π) * ᶜz / FT(3000))
    @. Y.c.ρq_tot *= ᶜpert
    @. Y.c.ρq_rai = FT(1e-5) * Y.c.ρ * ᶜpert
    @. Y.c.ρq_lcl = FT(2e-5) * Y.c.ρ * ᶜpert
    @. Y.c.ρe_tot *= 1 + FT(1e-3) * (ᶜpert - 1)
    if velocity
        ᶠx = Fields.coordinate_field(Y.f).x
        ᶠz = Fields.coordinate_field(Y.f).z
        @. Y.c.uₕ *= ᶜpert
        # Vanishes at both ends, so the no-flux boundaries are untouched.
        @. Y.f.u₃ += CA.C3(
            FT(0.1) * sin(FT(2π) * ᶠx / FT(6400)) * sin(FT(π) * ᶠz / FT(3000)),
        )
    end
    CA.set_precomputed_quantities!(Y, p, t)
    return nothing
end

# The two parts of the split enthalpy flux and the lumped h_tot flux, for a
# horizontal (`gradₕ`) or a vertical diffusivity, assembled without the source's
# helpers.
function reference_fluxes(Y, p)
    FT = eltype(Y)
    thermo_params = CA.CAP.thermodynamics_params(p.params)
    (; ᶜΦ) = p.core
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    ᶜq_vap = @. CA.TD.vapor_specific_humidity(ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice)
    ᶜq_lcl = @. Y.c.ρq_lcl / Y.c.ρ
    ᶜq_icl = @. Y.c.ρq_icl / Y.c.ρ
    ᶜh_eff_plus_Φ = @. (
        CA.TD.enthalpy_vapor(thermo_params, ᶜT) * max(FT(0), ᶜq_vap) +
        CA.TD.enthalpy_liquid(thermo_params, ᶜT) * max(FT(0), ᶜq_lcl) +
        CA.TD.enthalpy_ice(thermo_params, ᶜT) * max(FT(0), ᶜq_icl)
    ) / max(max(FT(0), ᶜq_vap) + max(FT(0), ᶜq_lcl) + max(FT(0), ᶜq_icl), eps(FT)) +
       ᶜΦ
    ᶜs_d = @. CA.TD.dry_static_energy(thermo_params, ᶜT, ᶜΦ)
    ᶜq_tot_eff = @. (Y.c.ρq_tot - Y.c.ρq_rai - Y.c.ρq_sno) / Y.c.ρ
    return (; ᶜs_d, ᶜh_eff_plus_Φ, ᶜq_tot_eff)
end

function check_split(ᶜρe_totₜ, ᶜflux_dse, ᶜflux_water, ᶜflux_h_tot)
    FT = eltype(ᶜρe_totₜ)
    @test maximum(abs, parent(ᶜflux_dse)) > 0
    @test maximum(abs, parent(ᶜflux_water)) > 0
    ᶜflux_total = @. ᶜflux_dse + ᶜflux_water
    @test parent(ᶜρe_totₜ) ≈ parent(ᶜflux_total) rtol = FT(1e-10)
    ᶜdiff = @. ᶜflux_h_tot - ᶜflux_total
    @test maximum(abs, parent(ᶜdiff)) > FT(1e-6) * maximum(abs, parent(ᶜflux_total))
end

@testset "Smagorinsky-Lilly energy flux is split" begin
    config = les_box_config("les_energy_split_smag"; smagorinsky_lilly = "UVW")
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    perturb!(Y, p, t)
    (; ᶜs_d, ᶜh_eff_plus_Φ, ᶜq_tot_eff) = reference_fluxes(Y, p)
    ᶜh_tot = p.precomputed.ᶜh_tot
    model = p.atmos.smagorinsky_lilly

    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.horizontal_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, model)
    (; ᶜD_h) = p.precomputed
    ᶜflux_dse = @. CA.wdivₕ(Y.c.ρ * ᶜD_h * CA.gradₕ(ᶜs_d))
    ᶜflux_water = @. CA.wdivₕ(Y.c.ρ * ᶜD_h * ᶜh_eff_plus_Φ * CA.gradₕ(ᶜq_tot_eff))
    ᶜflux_h_tot = @. CA.wdivₕ(Y.c.ρ * ᶜD_h * CA.gradₕ(ᶜh_tot))
    check_split(Yₜ.c.ρe_tot, ᶜflux_dse, ᶜflux_water, ᶜflux_h_tot)

    Yₜ .= zero(eltype(Yₜ))
    CA.vertical_smagorinsky_lilly_tendency!(Yₜ, Y, p, t, model)
    Pr_t = CA.CAP.Prandtl_number_0(CA.CAP.turbconv_params(p.params))
    ᶠρD = @. CA.ᶠinterp(Y.c.ρ) * CA.ᶠinterp(p.precomputed.ᶜνₜ_v) / Pr_t
    ᶜflux_dse = @. -CA.ᶜdiffdivᵥ(-(ᶠρD * CA.ᶠgradᵥ(ᶜs_d)))
    ᶜflux_water =
        @. -CA.ᶜdiffdivᵥ(-(ᶠρD * CA.ᶠinterp(ᶜh_eff_plus_Φ) * CA.ᶠgradᵥ(ᶜq_tot_eff)))
    ᶜflux_h_tot = @. -CA.ᶜdiffdivᵥ(-(ᶠρD * CA.ᶠgradᵥ(ᶜh_tot)))
    check_split(Yₜ.c.ρe_tot, ᶜflux_dse, ᶜflux_water, ᶜflux_h_tot)
end

@testset "Constant horizontal diffusion energy flux is split" begin
    config = les_box_config("les_energy_split_chd"; constant_horizontal_diffusion = true)
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    perturb!(Y, p, t)
    (; ᶜs_d, ᶜh_eff_plus_Φ, ᶜq_tot_eff) = reference_fluxes(Y, p)
    ᶜh_tot = p.precomputed.ᶜh_tot
    D = p.atmos.constant_horizontal_diffusion.D

    Yₜ = similar(Y)
    Yₜ .= zero(eltype(Yₜ))
    CA.horizontal_constant_diffusion_tendency!(
        Yₜ, Y, p, t, p.atmos.constant_horizontal_diffusion,
    )
    ᶜflux_dse = @. CA.wdivₕ(Y.c.ρ * D * CA.gradₕ(ᶜs_d))
    ᶜflux_water = @. CA.wdivₕ(Y.c.ρ * D * ᶜh_eff_plus_Φ * CA.gradₕ(ᶜq_tot_eff))
    ᶜflux_h_tot = @. CA.wdivₕ(Y.c.ρ * D * CA.gradₕ(ᶜh_tot))
    check_split(Yₜ.c.ρe_tot, ᶜflux_dse, ᶜflux_water, ᶜflux_h_tot)
end

@testset "AMD energy tendency is finite and nonzero" begin
    config = les_box_config("les_energy_split_amd"; amd_les = true)
    (; Y, p, simulation) = generate_test_simulation(config)
    t = simulation.integrator.t
    perturb!(Y, p, t; velocity = true)
    les = p.atmos.amd_les
    for tendency! in (CA.horizontal_amd_tendency!, CA.vertical_amd_tendency!)
        Yₜ = similar(Y)
        Yₜ .= zero(eltype(Yₜ))
        tendency!(Yₜ, Y, p, t, les)
        @test all(isfinite, parent(Yₜ.c.ρe_tot))
        @test all(isfinite, parent(Yₜ.c.ρq_tot))
        @test maximum(abs, parent(Yₜ.c.ρe_tot)) > 0
        @test maximum(abs, parent(Yₜ.c.ρq_tot)) > 0
    end
end
