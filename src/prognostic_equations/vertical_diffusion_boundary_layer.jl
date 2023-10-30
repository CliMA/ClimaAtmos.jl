#####
##### Vertical diffusion boundary layer parameterization
#####

import StaticArrays
import ClimaCore.Geometry: ⊗
import ClimaCore.Utilities: half
import LinearAlgebra: norm
import Thermodynamics as TD
import SurfaceFluxes as SF
import ClimaCore.Spaces as Spaces
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators

# Apply on potential temperature and moisture
# 1) turn the liquid_theta into theta version
# 2) have a total energy version (primary goal)

function eddy_diffusivity_coefficient(C_E::FT, norm_v_a, z_a, p) where {FT}
    p_pbl = FT(85000)
    p_strato = FT(10000)
    K_E = C_E * norm_v_a * z_a
    return p > p_pbl ? K_E : K_E * exp(-((p_pbl - p) / p_strato)^2)
end

function surface_thermo_state(
    ::GCMSurfaceThermoState,
    thermo_params,
    T_sfc,
    ts_int,
    t,
)
    ρ_sfc =
        TD.air_density(thermo_params, ts_int) *
        (
            T_sfc / TD.air_temperature(thermo_params, ts_int)
        )^(
            TD.cv_m(thermo_params, ts_int) /
            TD.gas_constant_air(thermo_params, ts_int)
        )
    q_sfc =
        TD.q_vap_saturation_generic(thermo_params, T_sfc, ρ_sfc, TD.Liquid())
    if ts_int isa TD.PhaseDry
        return TD.PhaseDry_ρT(thermo_params, ρ_sfc, T_sfc)
    elseif ts_int isa TD.PhaseEquil
        return TD.PhaseEquil_ρTq(thermo_params, ρ_sfc, T_sfc, q_sfc)
    else
        error("Unsupported thermo option")
    end
end

function vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t)
    Fields.bycolumn(axes(Y.c.uₕ)) do colidx
        (; vert_diff) = p.atmos
        vertical_diffusion_boundary_layer_tendency!(
            Yₜ,
            Y,
            p,
            t,
            colidx,
            vert_diff,
        )
    end
end

vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) =
    nothing

function vertical_diffusion_boundary_layer_tendency!(
    Yₜ,
    Y,
    p,
    t,
    colidx,
    ::VerticalDiffusion,
)
    ᶜρ = Y.c.ρ
    FT = Spaces.undertype(axes(ᶜρ))
    (; ᶜp, ᶜspecific, sfc_conditions) = p.precomputed # assume ᶜts and ᶜp have been updated
    (; C_E) = p.atmos.vert_diff

    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ

    FT = eltype(Y)
    interior_uₕ = Fields.level(Y.c.uₕ, 1)
    ᶠp = ᶠρK_E = p.scratch.ᶠtemp_scalar
    @. ᶠp[colidx] = ᶠinterp(ᶜp[colidx])
    ᶜΔz_surface = Fields.Δz_field(interior_uₕ)
    @. ᶠρK_E[colidx] =
        ᶠinterp(Y.c.ρ[colidx]) * eddy_diffusivity_coefficient(
            C_E,
            norm(interior_uₕ[colidx]),
            ᶜΔz_surface[colidx] / 2,
            ᶠp[colidx],
        )

    if diffuse_momentum(p.atmos.vert_diff)
        ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_uₕ[colidx]),
        )
        @. Yₜ.c.uₕ[colidx] -=
            ᶜdivᵥ_uₕ(-(ᶠρK_E[colidx] * ᶠgradᵥ(Y.c.uₕ[colidx]))) / Y.c.ρ[colidx]
    end

    if :ρe_tot in propertynames(Y.c)
        (; ᶜh_tot) = p.precomputed

        ᶜdivᵥ_ρe_tot = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(sfc_conditions.ρ_flux_h_tot[colidx]),
        )
        @. Yₜ.c.ρe_tot[colidx] -=
            ᶜdivᵥ_ρe_tot(-(ᶠρK_E[colidx] * ᶠgradᵥ(ᶜh_tot[colidx])))
    end
    ᶜρχₜ_diffusion = p.scratch.ᶜtemp_scalar
    ρ_flux_χ = p.scratch.sfc_temp_C3
    for (ᶜρχₜ, ᶜχ, χ_name) in matching_subfields(Yₜ.c, ᶜspecific)
        χ_name == :e_tot && continue
        if χ_name == :q_tot
            @. ρ_flux_χ[colidx] = sfc_conditions.ρ_flux_q_tot[colidx]
        elseif χ_name == :θ
            @. ρ_flux_χ[colidx] = sfc_conditions.ρ_flux_θ[colidx]
        else
            @. ρ_flux_χ[colidx] = C3(FT(0))
        end
        ᶜdivᵥ_ρχ = Operators.DivergenceF2C(
            top = Operators.SetValue(C3(FT(0))),
            bottom = Operators.SetValue(ρ_flux_χ[colidx]),
        )
        @. ᶜρχₜ_diffusion[colidx] =
            ᶜdivᵥ_ρχ(-(ᶠρK_E[colidx] * ᶠgradᵥ(ᶜχ[colidx])))
        @. ᶜρχₜ[colidx] -= ᶜρχₜ_diffusion[colidx]
        @. Yₜ.c.ρ[colidx] -= ᶜρχₜ_diffusion[colidx]
    end
end
