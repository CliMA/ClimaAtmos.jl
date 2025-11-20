#####
##### Apply surface fluxes as boundary conditions and translate them 
##### into tendencies for the relevant prognostic variables.
#####

import ClimaCore.Geometry: ⊗
import ClimaCore.Operators as Operators

"""
    boundary_tendency_momentum(ᶜρ, ᶜuₕ, ρ_flux_uₕ_surface)

Calculates the tendency contribution for horizontal momentum (`uₕ`) due to a
specified vertical flux of horizontal momentum at the bottom boundary.

This function constructs a divergence term that is non-zero only in the grid cell
adjacent to the bottom boundary. The divergence effectively introduces the
`ρ_flux_uₕ_surface` as a source/sink. The result is divided by density `ᶜρ`
to yield a tendency for specific horizontal momentum `uₕ`.

Arguments:
- `ᶜρ`: Cell-center air density field.
- `ᶜuₕ`: Cell-center horizontal velocity field (used for type/structure, not value in flux calc).
- `ρ_flux_uₕ_surface`: The vertical flux of horizontal momentum through the bottom
  boundary. This is a `ClimaCore.Geometry.AxisTensor` of type
  `C3{FT} ⊗ C12{FT}` (e.g., representing surface stress `τ` as
  `e_3 ⊗ τ` if defined as flux into the domain, or simply
  the stress vector `τ` if the `SetValue` operator handles the normal).
   Conventionally, a positive flux represents momentum transfer from the
  surface to the atmosphere.

Returns:
- A `ClimaCore.Field` representing the tendency `∂uₕ/∂t` due to the surface flux.
"""
function boundary_tendency_momentum(ᶜρ, ᶜuₕ, ρ_flux_uₕ_surface)
    FT = eltype(ᶜρ)
    ᶠgradᵥ = Operators.GradientC2F()
    ᶜdivᵥ_uₕ = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
        bottom = Operators.SetValue(ρ_flux_uₕ_surface),
    )
    return @. lazy(ᶜdivᵥ_uₕ((0 * ᶠgradᵥ(ᶜuₕ))) / ᶜρ)
end

"""
    boundary_tendency_scalar(ᶜχ, ρ_flux_χ_surface)

Calculates the tendency contribution for a scalar quantity `χ` (for the prognostic 
variable `ρχ`) due to a specified vertical flux of that scalar at the bottom boundary.

This function constructs a divergence term that is non-zero only in the grid cell
adjacent to the bottom boundary, effectively introducing `ρ_flux_χ_surface` as a
source/sink. When positive, the flux `ρ_flux_χ_surface` is directed from the surface to 
the atmosphere, i.e., represents an atmospheric source.  

Arguments:
- `ᶜχ`: cell-center scalar field (used for eltype and spatial structure,
  not its values in the flux calculation).
- `ρ_flux_χ_surface`: The vertical flux of the scalar quantity `χ` (density-weighted,
  i.e., flux of `ρχ`) through the bottom boundary. This is a
  `ClimaCore.Geometry.C3{FT}` vector representing the scalar value of the flux.

Returns:
- A `ClimaCore.Field` representing the tendency (e.g., `∂(ρχ)/∂t` or `∂χ/∂t`
  depending on how the caller uses it) due to the surface flux.
"""
function boundary_tendency_scalar(ᶜχ, ρ_flux_χ_surface)
    FT = eltype(ᶜχ)
    ᶠgradᵥ = Operators.GradientC2F()
    ᶜdivᵥ_χ = Operators.DivergenceF2C(
        top = Operators.SetValue(C3(FT(0))),
        bottom = Operators.SetValue(ρ_flux_χ_surface),
    )
    return @. lazy(ᶜdivᵥ_χ(0 * ᶠgradᵥ(ᶜχ)))
end

"""
    surface_flux_tendency!(Yₜ, Y, p, t)

Applies tendencies to prognostic variables due to surface fluxes.

This function computes and adds contributions from surface fluxes of momentum,
total energy, and total specific humidity (`q_tot`) to their respective tendency
terms in `Yₜ`. Other specific tracers currently have zero surface flux applied by
this function.

The actual flux values are obtained from `p.precomputed.sfc_conditions`.
The tendency contributions are localized to the grid cells adjacent to the
surface using the helper functions `boundary_tendency_momentum` and
`boundary_tendency_scalar`.

The application of these tendencies can be globally disabled via
`p.atmos.disable_surface_flux_tendency`, and momentum flux tendency can be
disabled if vertical diffusion for momentum is inactive.

Arguments:
- `Yₜ`: The tendency state vector, modified in place.
- `Y`: The current state vector.
- `p`: Cache containing parameters, precomputed fields (including `sfc_conditions`),
       and atmospheric model configurations.
- `t`: Current simulation time.
"""
function surface_flux_tendency!(Yₜ, Y, p, t)

    p.atmos.disable_surface_flux_tendency && return

    FT = eltype(Y)
    (; params) = p
    (; sfc_conditions, ᶜts) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)

    if !disable_momentum_vertical_diffusion(p.atmos.vertical_diffusion)
        btt =
            boundary_tendency_momentum(Y.c.ρ, Y.c.uₕ, sfc_conditions.ρ_flux_uₕ)
        @. Yₜ.c.uₕ -= btt
    end

    ᶜh_tot = @. lazy(
        TD.total_specific_enthalpy(
            thermo_params,
            ᶜts,
            specific(Y.c.ρe_tot, Y.c.ρ),
        ),
    )
    btt = boundary_tendency_scalar(ᶜh_tot, sfc_conditions.ρ_flux_h_tot)
    @. Yₜ.c.ρe_tot -= btt

    ρ_flux_χ = p.scratch.sfc_temp_C3
    foreach_gs_tracer(Yₜ, Y) do ᶜρχₜ, ᶜρχ, ρχ_name
        ᶜχ = @. lazy(specific(ᶜρχ, Y.c.ρ))
        if ρχ_name == @name(ρq_tot)
            @. ρ_flux_χ = sfc_conditions.ρ_flux_q_tot
        else
            @. ρ_flux_χ = C3(FT(0))
        end
        btt = boundary_tendency_scalar(ᶜχ, ρ_flux_χ)
        @. ᶜρχₜ -= btt
        if ρχ_name == @name(ρq_tot)
            @. Yₜ.c.ρ -= btt
        end
    end
end

"""
    edmfx_surface_flux_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

Applies surface–flux tendencies to EDMF updraft prognostic variables when the
turbulent convection scheme is the `PrognosticEDMFX` model.

This function computes and adds contributions from surface fluxes of moist
static energy (`mse`) and total specific humidity (`q_tot`) to the corresponding
updraft subdomain tendencies in `Yₜ`. For each EDMF subdomain, it evaluates
subgrid scalar fluxes using `sgs_scalar_flux_bc` and converts these fluxes into
tendencies localized to the surface-adjacent grid cell via
`boundary_tendency_scalar`.

# Arguments
- `Yₜ`: Tendency state vector, modified in place.
- `Y`: Current state vector.
- `p`: Cache containing parameters, thermodynamic settings, precomputed fields,
       and EDMF configuration information.
- `t`: Current simulation time.
- `turbconv_model::PrognosticEDMFX`: Dispatch selector specifying the EDMF scheme.
"""
edmfx_surface_flux_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing
function edmfx_surface_flux_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)
    p.atmos.disable_surface_flux_tendency && return

    (; params) = p
    (; ᶜρʲs, ᶜK, ᶜts) = p.precomputed
    thermo_params = CAP.thermodynamics_params(params)
    n = n_mass_flux_subdomains(p.atmos.turbconv_model)

    (; ρ_flux_h_tot, ts) =
        p.precomputed.sfc_conditions

    ᶜaʲ = p.scratch.ᶜtemp_scalar
    ρ_flux_χʲ = p.scratch.sfc_temp_C3
    # We need field_values everywhere because we are mixing
    # information from surface and first interior inside the
    # sgs_scalar_flux_bc call.
    ρ_flux_χʲ_val = Fields.field_values(ρ_flux_χʲ)

    # Based on boundary conditions for updrafts we compute
    # the tendency due to the surface flux for EDMFX ᶜmseʲ...
    ρ_flux_h_tot_val = Fields.field_values(ρ_flux_h_tot)
    h_tot_sfc = @. lazy(TD.specific_enthalpy(thermo_params, ts))
    h_tot_sfc_val = Fields.field_values(h_tot_sfc)
    ᶜh_tot = @. lazy(
        TD.total_specific_enthalpy(
            thermo_params,
            ᶜts,
            specific(Y.c.ρe_tot, Y.c.ρ),
        ),
    )
    ᶜh_tot_int_val = Fields.field_values(Fields.level(ᶜh_tot, 1))
    ᶜK_int_val = Fields.field_values(Fields.level(ᶜK, 1))

    for j in 1:n
        @. ᶜaʲ = draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
        ᶜaʲ_int_val = Fields.field_values(Fields.level(ᶜaʲ, 1))
        ᶜmseʲ_int_val = Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).mse, 1))

        @. ρ_flux_χʲ_val = sgs_scalar_flux_bc(
            h_tot_sfc_val,
            ᶜh_tot_int_val - ᶜK_int_val,
            ᶜmseʲ_int_val,
            ᶜaʲ_int_val,
            ρ_flux_h_tot_val,
        )
        btt = boundary_tendency_scalar(Y.c.sgsʲs.:(1).mse, ρ_flux_χʲ)
        @. Yₜ.c.sgsʲs.:($$j).mse -= btt / p.precomputed.ᶜρʲs.:($$j)
    end

    # ... and the tendency for EDMFX ᶜq_totʲ.
    if !(p.atmos.moisture_model isa DryModel)
        ρ_flux_q_tot_val = Fields.field_values(p.precomputed.sfc_conditions.ρ_flux_q_tot)
        q_tot_sfc = @. lazy(TD.total_specific_humidity(thermo_params, ts))
        q_tot_sfc_val = Fields.field_values(q_tot_sfc)
        ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
        ᶜq_tot_int_val = Fields.field_values(Fields.level(ᶜq_tot, 1))

        for j in 1:n
            @. ᶜaʲ = draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))
            ᶜaʲ_int_val = Fields.field_values(Fields.level(ᶜaʲ, 1))
            ᶜq_totʲ_int_val = Fields.field_values(Fields.level(Y.c.sgsʲs.:($j).q_tot, 1))

            @. ρ_flux_χʲ_val = sgs_scalar_flux_bc(
                q_tot_sfc_val,
                ᶜq_tot_int_val,
                ᶜq_totʲ_int_val,
                ᶜaʲ_int_val,
                ρ_flux_q_tot_val,
            )
            btt = boundary_tendency_scalar(Y.c.sgsʲs.:(1).q_tot, ρ_flux_χʲ)
            @. Yₜ.c.sgsʲs.:($$j).q_tot -= btt / p.precomputed.ᶜρʲs.:($$j)
        end
    end

end
