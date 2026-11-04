#####
##### AMD Diffusion
#####

import ClimaCore.Fields as Fields
import ClimaCore.Operators as Operators
import ClimaCore: DataLayouts, Geometry
import LinearAlgebra: norm_sqr

amd_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

"""
    amd_scalar_diffusivity(c_amd, ∂̂u_uvw, Δ, ∇χ)

Return the AMD diffusivity for a scalar at a point,

    Dₜ = max(0, -c (∂̂ₖuᵢ)(∂̂ₖχ) ∂ᵢχ / (∂ₗχ)(∂ₗχ)),

from the filter-scaled velocity gradient `∂̂u_uvw`, the filter width `Δ` of the
direction the gradient `∇χ` (a `UVW` vector) is taken along, so that
`∂̂χ = Δ ∇χ`, and the coefficient `c = c_amd`. The denominator is the squared
physical norm of the gradient.
"""
function amd_scalar_diffusivity(c_amd, ∂̂u_uvw, Δ, ∇χ)
    FT = eltype(∇χ)
    M = ∂̂u_uvw * (Δ * ∇χ) ⊗ ∇χ
    tr_M = M.components.data.:1 + M.components.data.:5 + M.components.data.:9
    return max(FT(0), -c_amd * tr_M / max(eps(FT), norm_sqr(∇χ)))
end

"""
    amd_tendency!(Yₜ, Y, p, t, les::AnisotropicMinimumDissipation)

Add the horizontal and vertical Anisotropic Minimum Dissipation (AMD) subgrid-scale flux
divergences to `Yₜ` in place; return `nothing`.

The AMD closure of [Akbar2016](@cite) builds the eddy viscosity and diffusivity from
gradients scaled by the anisotropic filter widths, `∂̂ᵢ = Δᵢ ∂ᵢ` with `Δᵢ` the filter width
in direction `i` (horizontal node spacing and vertical layer thickness here):

```math
νₜ = \\max(0, -c \\, (∂̂ₖuᵢ)(∂̂ₖuⱼ) Sᵢⱼ / (∂ₗuₘ)(∂ₗuₘ)), \\quad
Dₜ = \\max(0, -c \\, (∂̂ₖuᵢ)(∂̂ₖχ) ∂ᵢχ / (∂ₗχ)(∂ₗχ)),
```

with `c = les.c_amd` [-], computed in the slab loop. The velocity gradients in `νₜ` are
the full three-dimensional ones; each diffused scalar gets a horizontal `Dₜ` at cell centers
from its horizontal gradient and a vertical `Dₜ` at cell faces from its vertical gradient
(`amd_scalar_diffusivity`). Momentum receives `-∇·(ρ τ)/ρ` with the SGS momentum flux
tensor `τ = -2 νₜ S`, as a horizontal and a vertical divergence. Total energy receives the
divergence of the enthalpy flux
`-ρ [Dₜ(T) (c_p ∇T + ∇Φ) + Dₜ(q_tot) ((h_v + Φ) ∇q_tot + (h_l - h_v) ∇q_liq + (h_i - h_v) ∇q_ice)]`
along each direction, with `c_p` the heat capacity of the diffusing mixture
(`add_sensible_heat_diffusionₕ!`, `add_sensible_heat_diffusionᵥ!`), where `q_liq` and
`q_ice` include rain and snow; its water terms are absent in dry configurations. Total
water diffuses with `Dₜ(q_tot)` (mirrored to `Yₜ.c.ρ` for mass conservation), each cloud and
precipitation species diffuses down its own gradient with the same diffusivity, except that
cloud liquid and ice take the blended vertical flux of `add_cloud_species_diffusionᵥ!` when
`blended_cloud_species_flux` holds, and each passive grid-scale tracer `χ` diffuses with its
own `Dₜ(χ)`. Reads `ᶜu`, `ᶠu³`, `ᶜT`, `ᶜq_liq`, and `ᶜq_ice` from `p.precomputed`.

The velocity gradients are evaluated once for both directions, in a sequence of fused
loops whose intermediate values are register temporaries. Slab loops cannot apply vertical
operators and column loops cannot apply horizontal ones, so the gradients that cross from
one kind of loop to the other are stored in `p.scratch`:

 1. a column loop stores the face velocity (`ᶠtemp_UVW`), the vertical velocity
    differences at centers (`ᶜtemp_UVW`, 3 values per point rather than the 9 of their
    gradient), and the vertical velocity gradient at faces (`ᶠtemp_UVWxUVW`);
 2. a center slab loop applies the horizontal tendencies at cell centers and stores the
    strain rate (`ᶜtemp_UVWxUVW`) and `νₜ` (`ᶜtemp_scalar`);
 3. a column loop stores the face values of `ρ νₜ` (`ᶠtemp_scalar`);
 4. a face slab loop replaces the vertical face gradient with the horizontal one and
    `ρ νₜ` with the horizontal stress divergence of `u₃`;
 5. a column loop divides that divergence by the face density and applies the vertical
    momentum tendencies; and
 6. a column loop applies the vertical scalar tendencies and stores the face weight of the
    total water flux (`ᶠtemp_scalar_2`) for the blended cloud-species flux.

The vertical momentum and scalar tendencies are in separate column loops, since a loop with
both needs more registers than keep two blocks of 256 threads resident on a GPU
multiprocessor.

This tendency is always applied explicitly, from `remaining_tendency!`; it is not part of
the implicit solver. To remove it in debugging workflows, comment out the call in
`remaining_tendency.jl`.
"""
function amd_tendency!(Yₜ, Y, p, t, les::AnisotropicMinimumDissipation)
    (; microphysics_model) = p.atmos
    (; ᶜu, ᶠu³, ᶜT, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; ᶜtemp_UVW, ᶠtemp_UVW, ᶜtemp_UVWxUVW, ᶠtemp_UVWxUVW) = p.scratch
    (; ᶜtemp_scalar, ᶠtemp_scalar, ᶠtemp_scalar_2) = p.scratch
    ᶜΦ = p.core.ᶜΦ
    ᶜΔ_z = Fields.Δz_field(Y.c)
    ᶠΔ_z = Fields.Δz_field(Y.f)
    blend = Val(blended_cloud_species_flux(Y, p))
    thermo_params = CAP.thermodynamics_params(p.params)
    c_amd = les.c_amd
    Δ_h = Spaces.node_horizontal_length_scale(Spaces.horizontal_space(axes(Y.c)))
    water_names = amd_water_species_names(Y, false)
    passive_names = passive_gs_tracer_names(Y)
    self_water_names = amd_water_species_names(Y, blend isa Val{true})
    axis_uvw = (Geometry.UVWAxis(),)
    ∇ᵥuvw_boundary = Geometry.outer(Geometry.WVector(0), UVW(0, 0, 0))
    ᶠgradᵥ_uvw = Operators.GradientC2F(
        bottom = Operators.SetGradient(∇ᵥuvw_boundary),
        top = Operators.SetGradient(∇ᵥuvw_boundary),
    )
    ᶠgradᵥ_scalar = Operators.GradientC2F(
        bottom = Operators.SetGradient(UVW(0, 0, 0)),
        top = Operators.SetGradient(UVW(0, 0, 0)),
    )

    DataLayouts.foreach_column(
        ᶠtemp_UVW, ᶜtemp_UVW, ᶠtemp_UVWxUVW, Y.c.uₕ, ᶠu³, ᶜu,
    ) do ᶠu_uvw, ᶜΔu, ᶠ∇ᵥu, ᶜuₕ, ᶠu³, ᶜu
        @inline
        ᶠu_uvw_column = @. UVW(ᶠinterp(ᶜuₕ)) + UVW(ᶠu³)
        @. ᶠu_uvw = ᶠu_uvw_column
        @. ᶜΔu = vertical_difference(ᶜgradᵥ(ᶠu_uvw_column))
        ᶜu_uvw = @. UVW(ᶜu)
        @. ᶠ∇ᵥu = Geometry.project(axis_uvw, ᶠgradᵥ_uvw(ᶜu_uvw))
    end

    DataLayouts.foreach_slab(
        Yₜ.c, Y.c, ᶜu, ᶜtemp_UVW, ᶜtemp_UVWxUVW, ᶜtemp_scalar, ᶜΔ_z, ᶜT, ᶜq_liq,
        ᶜq_ice, ᶜΦ,
    ) do Yₜc, Yc, ᶜu, ᶜΔu, ᶜS_out, ᶜνₜ_out, ᶜΔ_z, ᶜT, ᶜq_liq, ᶜq_ice, ᶜΦ
        @inline
        ᶜu_uvw = @. UVW(ᶜu)
        ᶜ∇ₕu = @. Geometry.project(axis_uvw, gradₕ(ᶜu_uvw))
        ᶜ∇ᵥu = @. Geometry.project(axis_uvw, vertical_gradient(ᶜΔu))
        ᶜ∇u = @. ᶜ∇ᵥu + ᶜ∇ₕu
        ᶜS = @. symmetric_part(ᶜ∇u)
        ᶜ∂̂u = @. Δ_h * ᶜ∇ₕu + ᶜΔ_z * ᶜ∇ᵥu
        # The AMD eddy viscosity at a point,
        #     νₜ = max(0, -c (∂̂ₖuᵢ)(∂̂ₖuⱼ) Sᵢⱼ / (∂ₗuₘ)(∂ₗuₘ)),
        # from the filter-scaled velocity gradient ∂̂u, the strain rate S, and
        # the velocity gradient ∇u (`UVW × UVW` tensors), with c = c_amd.
        amd_viscosity =
            (∂̂u, S, ∇u) -> begin
                M = (∂̂u * adjoint(∂̂u)) * S
                tr_M =
                    M.components.data.:1 + M.components.data.:5 + M.components.data.:9
                max(zero(tr_M), -c_amd * tr_M / max(eps(tr_M), norm_sqr(∇u)))
            end
        ᶜνₜ = @. amd_viscosity(ᶜ∂̂u, ᶜS, ᶜ∇u)
        ᶜuₕₜ = @. C12(wdivₕ(Yc.ρ * (-2 * ᶜνₜ * ᶜS)) / Yc.ρ)
        @. ᶜS_out = ᶜS
        @. ᶜνₜ_out = ᶜνₜ
        @. Yₜc.uₕ -= ᶜuₕₜ
        # The horizontal AMD diffusion of every scalar, with the scaled velocity gradient
        # `ᶜ∂̂u`. As in the slab form of `add_scalar_diffusionₕ!`, every tendency stays a
        # register temporary until all of them are computed, and every diffused scalar is a
        # register temporary of one type.
        ᶜT = as_register(ᶜT)
        ᶜΦ = as_register(ᶜΦ)
        ᶜq_water = diffusing_water_phases(Yc, ᶜq_liq, ᶜq_ice, microphysics_model, Val(true))
        ᶜq_water = isnothing(ᶜq_water) ? nothing : unrolled_map(as_register, ᶜq_water)
        ᶜq_tot = isnothing(ᶜq_water) ? nothing : ᶜq_water[1]
        ᶜρD_T = @. Yc.ρ *
           amd_scalar_diffusivity(c_amd, ᶜ∂̂u, Δ_h, Geometry.project(axis_uvw, gradₕ(ᶜT)))
        ᶜρD_q =
            isnothing(ᶜq_tot) ? nothing :
            (@. Yc.ρ * amd_scalar_diffusivity(
                c_amd, ᶜ∂̂u, Δ_h, Geometry.project(axis_uvw, gradₕ(ᶜq_tot)),
            ))
        ᶜρe_totₜ = as_register(Yₜc.ρe_tot)
        add_sensible_heat_diffusionₕ!(ᶜρe_totₜ, ᶜρD_T, ᶜT, ᶜq_water, thermo_params, ᶜΦ)
        isnothing(ᶜq_water) ||
            add_water_enthalpy_diffusionₕ!(ᶜρe_totₜ, ᶜρD_q, ᶜT, ᶜq_water, thermo_params, ᶜΦ)
        ᶜρq_totₜ = isnothing(ᶜq_tot) ? nothing : diffusion_divₕ(ᶜρD_q, ᶜq_tot)
        ᶜρχₜs_water = unrolled_map(water_names) do ρχ_name
            @inline
            ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
            ᶜχ = @. specific(ᶜρχ, Yc.ρ)
            diffusion_divₕ(ᶜρD_q, ᶜχ)
        end
        ᶜρχₜs_passive = unrolled_map(passive_names) do ρχ_name
            @inline
            ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
            ᶜχ = @. specific(ᶜρχ, Yc.ρ)
            ᶜρD_χ = @. Yc.ρ *
               amd_scalar_diffusivity(
                c_amd,
                ᶜ∂̂u,
                Δ_h,
                Geometry.project(axis_uvw, gradₕ(ᶜχ)),
            )
            diffusion_divₕ(ᶜρD_χ, ᶜχ)
        end
        @. Yₜc.ρe_tot = ᶜρe_totₜ
        if !isnothing(ᶜρq_totₜ)
            @. Yₜc.ρq_tot += ᶜρq_totₜ
            @. Yₜc.ρ += ᶜρq_totₜ
        end
        unrolled_foreach(
            (water_names..., passive_names...),
            (ᶜρχₜs_water..., ᶜρχₜs_passive...),
        ) do ρχ_name, ᶜρχₜ_diffusion
            @inline
            ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
            @. ᶜρχₜ += ᶜρχₜ_diffusion
        end
    end

    DataLayouts.foreach_column(ᶠtemp_scalar, Y.c.ρ, ᶜtemp_scalar) do ᶠρνₜ, ᶜρ, ᶜνₜ
        @inline
        @. ᶠρνₜ = ᶠinterp(ᶜρ) * ᶠinterp(ᶜνₜ)
    end

    # The slab loop stores the horizontal gradient over the vertical one and the
    # divergence over `ρ νₜ`, after reading them (slab loops cannot interpolate the
    # density that divides the divergence).
    DataLayouts.foreach_slab(
        ᶠtemp_UVW, ᶠtemp_UVWxUVW, ᶠtemp_scalar,
    ) do ᶠu_uvw, ᶠ∇u, ᶠρνₜ
        @inline
        ᶠ∇ᵥu = as_register(ᶠ∇u)
        ᶠ∇ₕu = @. Geometry.project(axis_uvw, gradₕ(ᶠu_uvw))
        @. ᶠ∇u = ᶠ∇ₕu
        ᶠρu₃ₜ = @. first(parent(C3(wdivₕ(-2 * ᶠρνₜ * symmetric_part(ᶠ∇ᵥu + ᶠ∇ₕu)))))
        @. ᶠρνₜ = ᶠρu₃ₜ
    end

    DataLayouts.foreach_column(
        Yₜ.c, Yₜ.f, Y.c, ᶜu, ᶜtemp_UVWxUVW, ᶜtemp_scalar, ᶠtemp_UVWxUVW, ᶠtemp_scalar,
    ) do Yₜc, Yₜf, Yc, ᶜu, ᶜS, ᶜνₜ, ᶠ∇ₕu, ᶠρu₃ₜ
        @inline
        ᶠρ = @. ᶠinterp(Yc.ρ)
        ᶠwₜ = Yₜf.u₃.components.data.:1
        @. ᶠwₜ -= ᶠρu₃ₜ / ᶠρ
        @. Yₜf.u₃ -= C3(ᶠdiffdivᵥ_u₃(Yc.ρ * (-2 * ᶜνₜ * ᶜS)) / ᶠρ)
        ᶠνₜ = @. ᶠinterp(ᶜνₜ)
        ᶜu_uvw = @. UVW(ᶜu)
        ᶠS = @. symmetric_part(Geometry.project(axis_uvw, ᶠgradᵥ_uvw(ᶜu_uvw)) + ᶠ∇ₕu)
        @. Yₜc.uₕ -= C12(ᶜdivᵥ(ᶠρ * (-2 * ᶠνₜ * ᶠS)) / Yc.ρ)
    end

    DataLayouts.foreach_column(
        Yₜ.c, Y.c, ᶜu, ᶠtemp_UVWxUVW, ᶠtemp_scalar_2, ᶠΔ_z, ᶜT, ᶜq_liq, ᶜq_ice, ᶜΦ,
    ) do Yₜc, Yc, ᶜu, ᶠ∇ₕu, ᶠρD_out, ᶠΔ_z, ᶜT, ᶜq_liq, ᶜq_ice, ᶜΦ
        @inline
        ᶠρ = @. ᶠinterp(Yc.ρ)
        ᶜu_uvw = @. UVW(ᶜu)
        ᶠ∂̂u = @. Δ_h * ᶠ∇ₕu + ᶠΔ_z * Geometry.project(axis_uvw, ᶠgradᵥ_uvw(ᶜu_uvw))
        ᶜT = as_register(ᶜT)
        ᶜq_water = diffusing_water_phases(Yc, ᶜq_liq, ᶜq_ice, microphysics_model, Val(true))
        ᶜq_water = isnothing(ᶜq_water) ? nothing : unrolled_map(as_register, ᶜq_water)
        ᶜq_tot = isnothing(ᶜq_water) ? nothing : ᶜq_water[1]
        ᶠρD_T = @. ᶠρ * amd_scalar_diffusivity(
            c_amd, ᶠ∂̂u, ᶠΔ_z, Geometry.project(axis_uvw, ᶠgradᵥ_scalar(ᶜT)),
        )
        add_sensible_heat_diffusionᵥ!(Yₜc.ρe_tot, ᶠρD_T, ᶜT, ᶜq_water, thermo_params, ᶜΦ)
        if !isnothing(ᶜq_tot)
            ᶠρD_q = @. ᶠρ * amd_scalar_diffusivity(
                c_amd, ᶠ∂̂u, ᶠΔ_z, Geometry.project(axis_uvw, ᶠgradᵥ_scalar(ᶜq_tot)),
            )
            add_water_diffusionᵥ!(Yₜc, nothing, ᶠρD_q, ᶜT, ᶜq_water, thermo_params, ᶜΦ)
            unrolled_foreach(self_water_names) do ρχ_name
                @inline
                ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
                ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
                @. ᶜρχₜ -= ᶜdiffdivᵥ(-(ᶠρD_q * ᶠgradᵥ(specific(ᶜρχ, Yc.ρ))))
            end
            blend isa Val{true} && @. ᶠρD_out = ᶠρD_q
        end
        unrolled_foreach(passive_names) do ρχ_name
            @inline
            ᶜρχ = MatrixFields.get_field(Yc, ρχ_name)
            ᶜρχₜ = MatrixFields.get_field(Yₜc, ρχ_name)
            ᶜχ = @. specific(ᶜρχ, Yc.ρ)
            ᶠρD_χ = @. ᶠρ * amd_scalar_diffusivity(
                c_amd, ᶠ∂̂u, ᶠΔ_z, Geometry.project(axis_uvw, ᶠgradᵥ_scalar(ᶜχ)),
            )
            @. ᶜρχₜ -= ᶜdiffdivᵥ(-(ᶠρD_χ * ᶠgradᵥ(ᶜχ)))
        end
    end
    blend isa Val{true} || return nothing
    add_blended_cloud_species_diffusionᵥ!(Yₜ, Y, p, ᶠtemp_scalar_2; precipitation = true)
    return nothing
end

# The names of the water species that diffuse down their own gradients with the
# diffusivity of total water (`diffused_water_species` with precipitation), without
# cloud liquid and ice if `blend`.
amd_water_species_names(Y, blend) = unrolled_map(
    last,
    unrolled_filter(diffused_water_species(true)) do (χ_name, ρχ_name)
        MatrixFields.has_field(Y.c, ρχ_name) &&
            !(blend && !isnothing(cloud_species_phase(χ_name)))
    end,
)
