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

function vertical_diffusion_boundary_layer_cache(
    Y,
    atmos,
    ::Type{FT};
    C_E::FT = FT(0),
    diffuse_momentum = true,
) where {FT}
    (; surface_scheme, coupling) = atmos
    z_bottom = Spaces.level(Fields.coordinate_field(Y.c).z, 1)

    dif_flux_uₕ =
        Geometry.Contravariant3Vector.(zeros(axes(z_bottom))) .⊗
        Geometry.Covariant12Vector.(
            zeros(axes(z_bottom)),
            zeros(axes(z_bottom)),
        )
    dif_flux_energy = similar(z_bottom, Geometry.WVector{FT})
    dif_flux_ρq_tot = if :ρq_tot in propertynames(Y.c)
        similar(z_bottom, Geometry.WVector{FT})
    else
        Ref(Geometry.WVector(FT(0)))
    end

    cond_type = NamedTuple{(:shf, :lhf, :E, :ρτxz, :ρτyz), NTuple{5, FT}}
    surface_normal = Geometry.WVector.(ones(axes(Fields.level(Y.c, 1))))

    ts_type = thermo_state_type(atmos.moisture_model, FT)
    ts_inst = zero(ts_type)

    # TODO: replace with (TODO add support for) Base.zero / similar
    sfc_input_kwargs = if surface_scheme isa BulkSurfaceScheme

        sfc_inputs_type = typeof(
            SF.Coefficients{FT}(;
                state_in = SF.InteriorValues(
                    FT(0),
                    StaticArrays.SVector(FT(0), FT(0)),
                    ts_inst,
                ),
                state_sfc = SF.SurfaceValues(
                    FT(0),
                    StaticArrays.SVector(FT(0), FT(0)),
                    ts_inst,
                ),
                z0m = FT(0),
                z0b = FT(0),
                Cd = FT(0),
                Ch = FT(0),
            ),
        )
        sfc_inputs = similar(Fields.level(Y.f, half), sfc_inputs_type)
        fill!(sfc_inputs.Cd, FT(0.0044)) #FT(0.001)
        fill!(sfc_inputs.Ch, FT(0.0044)) #FT(0.0001)
        (; sfc_inputs)
    elseif surface_scheme isa MoninObukhovSurface

        sfc_inputs_type = typeof(
            SF.ValuesOnly{FT}(;
                state_in = SF.InteriorValues(
                    FT(0),
                    StaticArrays.SVector(FT(0), FT(0)),
                    ts_inst,
                ),
                state_sfc = SF.SurfaceValues(
                    FT(0),
                    StaticArrays.SVector(FT(0), FT(0)),
                    ts_inst,
                ),
                z0m = FT(0),
                z0b = FT(0),
            ),
        )
        sfc_inputs = similar(Fields.level(Y.f, half), sfc_inputs_type)
        fill!(sfc_inputs.z0m, FT(1e-5))
        fill!(sfc_inputs.z0b, FT(1e-5))
        (; sfc_inputs)
    else
        NamedTuple()
    end

    return (;
        surface_scheme,
        sfc_input_kwargs...,
        C_E,
        ᶠp = similar(Y.f, FT),
        ᶠK_E = similar(Y.f, FT),
        uₕ_int_phys = similar(
            Spaces.level(Y.c.uₕ, 1),
            typeof(Geometry.UVVector(FT(0), FT(0))),
        ),
        uₕ_int_phys_vec = similar(
            Spaces.level(Y.c.uₕ, 1),
            StaticArrays.SVector{2, FT},
        ),
        dif_flux_uₕ,
        dif_flux_uₕ_bc = similar(dif_flux_uₕ),
        dif_flux_energy,
        dif_flux_ρq_tot,
        dif_flux_energy_bc = similar(dif_flux_energy),
        dif_flux_ρq_tot_bc = similar(dif_flux_ρq_tot),
        diffuse_momentum,
        coupling,
        surface_normal,
        z_bottom,
    )
end

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
    return TD.PhaseEquil_ρTq(thermo_params, ρ_sfc, T_sfc, q_sfc)
end

set_surface_thermo_state!(::Coupled, args...) = nothing

function set_surface_thermo_state!(
    ::Decoupled,
    sfc_thermo_state_type,
    ts_sfc,
    T_sfc,
    ts_int,
    thermo_params,
    t,
)
    @. ts_sfc = surface_thermo_state(
        sfc_thermo_state_type,
        thermo_params,
        T_sfc,
        ts_int,
        t,
    )
    return nothing
end

set_surface_inputs!(sfc_inputs, ::Nothing, args...) = nothing

function set_surface_inputs!(
    sfc_inputs,
    ::BulkSurfaceScheme,
    ts_sfc,
    ts_int,
    uₕ_int_phys_vec,
    z_int,
    z_sfc,
)
    FT = Spaces.undertype(axes(z_sfc))

    # wrap state values
    @. sfc_inputs = SF.Coefficients(
        SF.InteriorValues(z_int, uₕ_int_phys_vec, ts_int), # state_in
        SF.SurfaceValues(                                  # state_sfc
            z_sfc,
            StaticArrays.SVector(FT(0), FT(0)),
            ts_sfc,
        ),
        sfc_inputs.Cd,                                     # Cd
        sfc_inputs.Ch,                                     # Ch
        FT(0),                                             # z0m
        FT(0),                                             # z0b
        FT(1),                                             # gustiness
    )
    return nothing
end

function set_surface_inputs!(
    sfc_inputs,
    ::MoninObukhovSurface,
    ts_sfc,
    ts_int,
    uₕ_int_phys_vec,
    z_int,
    z_sfc,
)
    FT = Spaces.undertype(axes(z_sfc))

    # wrap state values
    @. sfc_inputs = SF.ValuesOnly(
        SF.InteriorValues(z_int, uₕ_int_phys_vec, ts_int), # state_in
        SF.SurfaceValues(                                  # state_sfc
            z_sfc,
            StaticArrays.SVector(FT(0), FT(0)),
            ts_sfc,
        ),
        sfc_inputs.z0m,                                    # z0m
        sfc_inputs.z0b,                                    # z0b
        FT(-1),                                            # L_MO_init
        FT(1),                                             # gustiness
    )

end

function vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t)
    Fields.bycolumn(axes(Y.c.uₕ)) do colidx
        get_surface_fluxes!(Y, p, t, colidx, p.atmos.coupling)
        vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, colidx)
    end
end

get_surface_fluxes!(Y, p, t, colidx, ::Coupled) = nothing

function get_surface_fluxes!(Y, p, t, colidx, ::Decoupled)
    (; z_sfc, ᶜts, T_sfc) = p
    (;
        sfc_conditions,
        sfc_inputs,
        dif_flux_uₕ,
        dif_flux_uₕ_bc,
        dif_flux_energy,
        dif_flux_energy_bc,
        dif_flux_ρq_tot,
        dif_flux_ρq_tot_bc,
        diffuse_momentum,
        z_bottom,
        ts_sfc,
        uₕ_int_phys_vec,
        uₕ_int_phys,
        params,
        coupling,
    ) = p
    (; surface_normal) = p

    # parameters
    thermo_params = CAP.thermodynamics_params(params)

    if !(p.surface_scheme isa Nothing)
        uₕ_int = Spaces.level(Y.c.uₕ[colidx], 1)

        # TODO: Remove use of parent
        @. uₕ_int_phys[colidx] = Geometry.UVVector(uₕ_int)
        @. uₕ_int_phys_vec[colidx] = StaticArrays.SVector(
            uₕ_int_phys[colidx].components.data.:1,
            uₕ_int_phys[colidx].components.data.:2,
        )

        (; sfc_thermo_state_type) = p.surface_scheme
        # get the near-surface thermal state
        set_surface_thermo_state!(
            coupling,
            sfc_thermo_state_type,
            p.ts_sfc[colidx],
            T_sfc[colidx],
            Spaces.level(ᶜts[colidx], 1),
            thermo_params,
            t,
        )

        set_surface_inputs!(
            p.sfc_inputs[colidx],
            p.surface_scheme,
            p.ts_sfc[colidx],
            Spaces.level(ᶜts[colidx], 1),
            uₕ_int_phys_vec[colidx],
            z_bottom[colidx],
            z_sfc[colidx],
        )

        # calculate all fluxes (saturated surface conditions)
        sf_params = CAP.surface_fluxes_params(params)
        @. sfc_conditions[colidx] =
            SF.surface_conditions(sf_params, p.sfc_inputs[colidx])
    end

    if diffuse_momentum
        ρτxz = sfc_conditions[colidx].ρτxz
        ρτyz = sfc_conditions[colidx].ρτyz
        ρ_1 = Fields.level(Y.c.ρ[colidx], 1)
        @. dif_flux_uₕ[colidx] =
            Geometry.Contravariant3Vector(surface_normal[colidx]) ⊗
            Geometry.Covariant12Vector(
                Geometry.UVVector(ρτxz / ρ_1, ρτyz / ρ_1),
            )
        @. dif_flux_uₕ_bc[colidx] = -dif_flux_uₕ[colidx]
    end
    if :ρe_tot in propertynames(Y.c)
        if isnothing(p.surface_scheme)
            @. dif_flux_energy[colidx] *= 0
        else
            @. dif_flux_energy[colidx] = Geometry.WVector(
                sfc_conditions[colidx].shf + sfc_conditions[colidx].lhf,
            )
        end
        @. dif_flux_energy_bc[colidx] = -dif_flux_energy[colidx]
    end

    if :ρq_tot in propertynames(Y.c)
        if isnothing(p.surface_scheme)
            @. dif_flux_ρq_tot[colidx] *= 0
        else
            @. dif_flux_ρq_tot[colidx] = Geometry.WVector(
                SF.evaporation(
                    sf_params,
                    sfc_inputs[colidx],
                    sfc_conditions[colidx].Ch,
                ),
            )
        end
        @. dif_flux_ρq_tot_bc[colidx] = -dif_flux_ρq_tot[colidx]
    end
end

function vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t, colidx)
    ᶜρ = Y.c.ρ
    (; ᶠinterp) = p.operators
    FT = Spaces.undertype(axes(ᶜρ))
    (; ᶜp, ᶠK_E, C_E) = p # assume ᶜts and ᶜp have been updated
    (;
        dif_flux_uₕ_bc,
        dif_flux_energy_bc,
        dif_flux_ρq_tot_bc,
        diffuse_momentum,
        ᶠp,
        z_bottom,
    ) = p

    ᶠgradᵥ = Operators.GradientC2F() # apply BCs to ᶜdivᵥ, which wraps ᶠgradᵥ

    uₕ_int = Spaces.level(Y.c.uₕ[colidx], 1)
    @. ᶠp[colidx] = ᶠinterp(ᶜp[colidx])
    @. ᶠK_E[colidx] = eddy_diffusivity_coefficient(
        C_E,
        norm(uₕ_int),
        z_bottom[colidx],
        ᶠp[colidx],
    )

    if diffuse_momentum
        ᶜdivᵥ = Operators.DivergenceF2C(
            top = Operators.SetValue(
                Geometry.Contravariant3Vector(FT(0)) ⊗
                Geometry.Covariant12Vector(FT(0), FT(0)),
            ),
            bottom = Operators.SetValue(dif_flux_uₕ_bc[colidx]),
        )
        @. Yₜ.c.uₕ[colidx] += ᶜdivᵥ(ᶠK_E[colidx] * ᶠgradᵥ(Y.c.uₕ[colidx]))
    end

    if :ρe_tot in propertynames(Y.c)
        ᶜdivᵥ = Operators.DivergenceF2C(
            top = Operators.SetValue(Geometry.WVector(FT(0))),
            bottom = Operators.SetValue(dif_flux_energy_bc[colidx]),
        )
        @. Yₜ.c.ρe_tot[colidx] += ᶜdivᵥ(
            ᶠK_E[colidx] *
            ᶠinterp(ᶜρ[colidx]) *
            ᶠgradᵥ((Y.c.ρe_tot[colidx] + ᶜp[colidx]) / ᶜρ[colidx]),
        )
    end

    if :ρq_tot in propertynames(Y.c)
        ᶜdivᵥ = Operators.DivergenceF2C(
            top = Operators.SetValue(Geometry.WVector(FT(0))),
            bottom = Operators.SetValue(dif_flux_ρq_tot_bc[colidx]),
        )
        @. Yₜ.c.ρq_tot[colidx] += ᶜdivᵥ(
            ᶠK_E[colidx] *
            ᶠinterp(ᶜρ[colidx]) *
            ᶠgradᵥ(Y.c.ρq_tot[colidx] / ᶜρ[colidx]),
        )
        @. Yₜ.c.ρ[colidx] += ᶜdivᵥ(
            ᶠK_E[colidx] *
            ᶠinterp(ᶜρ[colidx]) *
            ᶠgradᵥ(Y.c.ρq_tot[colidx] / ᶜρ[colidx]),
        )
    end
end
