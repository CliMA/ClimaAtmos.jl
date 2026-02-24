"""
    update_surface_conditions!(Y, p, t)

Updates the value of `p.precomputed.sfc_conditions` based on the current state `Y` and time
`t`. This function will only update the surface conditions if the surface_setup
is not a PrescribedSurface.
"""
function update_surface_conditions!(Y, p, t)
    # Need to extract the field values so that we can do
    # a DataLayout broadcast rather than a Field broadcast
    # because we are mixing surface and interior fields
    sfc_local_geometry_values = Fields.field_values(
        Fields.level(Fields.local_geometry_field(Y.f), Fields.half),
    )
    int_local_geometry_values =
        Fields.field_values(Fields.level(Fields.local_geometry_field(Y.c), 1))
    (; ᶜT, ᶜq_tot_safe, ᶜq_liq_rai, ᶜq_ice_sno, ᶜu, sfc_conditions) = p.precomputed
    (; params, sfc_setup, atmos) = p
    thermo_params = CAP.thermodynamics_params(params)
    surface_fluxes_params = CAP.surface_fluxes_params(params)
    surface_temp_params = CAP.surface_temp_params(params)
    int_T_values = Fields.field_values(Fields.level(ᶜT, 1))
    int_ρ_values = Fields.field_values(Fields.level(Y.c.ρ, 1))
    int_q_tot_values = Fields.field_values(Fields.level(ᶜq_tot_safe, 1))
    int_q_liq_values = Fields.field_values(Fields.level(ᶜq_liq_rai, 1))
    int_q_ice_values = Fields.field_values(Fields.level(ᶜq_ice_sno, 1))
    int_u_values = Fields.field_values(Fields.level(ᶜu, 1))
    int_z_values = Fields.field_values(Fields.level(Fields.coordinate_field(Y.c).z, 1))
    sfc_conditions_values = Fields.field_values(sfc_conditions)
    wrapped_sfc_setup = sfc_setup_wrapper(sfc_setup)
    if p.atmos.sfc_temperature isa ExternalTVColumnSST
        (; surface_inputs, surface_timevaryinginputs) = p.external_forcing
        evaluate!(surface_inputs.ts, surface_timevaryinginputs.ts, t)
        sfc_temp_var = Fields.field_values(surface_inputs.ts)
    elseif p.atmos.surface_model isa SlabOceanSST
        sfc_temp_var = Fields.field_values(Y.sfc.T)
    else
        sfc_temp_var = nothing
    end

    @. sfc_conditions_values = surface_state_to_conditions(
        wrapped_sfc_setup,
        sfc_local_geometry_values,
        int_T_values,
        int_ρ_values,
        int_q_tot_values,
        int_q_liq_values,
        int_q_ice_values,
        projected_vector_data(CT1, int_u_values, int_local_geometry_values),
        projected_vector_data(CT2, int_u_values, int_local_geometry_values),
        int_z_values,
        thermo_params,
        surface_fluxes_params,
        surface_temp_params,
        atmos,
        sfc_temp_var,
        t,
    )
    return nothing
end


# default case
sfc_setup_wrapper(sfc_setup::SurfaceState) = (sfc_setup,)

# case when surface setup is func
sfc_setup_wrapper(sfc_setup::Function) = (sfc_setup,)

#this is the case for the coupler
function sfc_setup_wrapper(sfc_setup::Fields.Field)
    @assert eltype(sfc_setup) <: SurfaceState
    return Fields.field_values(sfc_setup)
end

surface_state(sfc_setup_wrapper::SurfaceState, _, _, _) = sfc_setup_wrapper

surface_state(
    wrapped_sfc_setup::F, sfc_local_geometry_values, int_z_values, t,
) where {F <: Function} =
    wrapped_sfc_setup(sfc_local_geometry_values.coordinates, int_z_values, t)

# This is a hack for meeting the August 7th deadline. It is to ensure that the
# coupler will be able to construct an integrator before overwriting its surface
# conditions, but without throwing an error during the computation of
# precomputed quantities for diagnostic EDMF due to uninitialized surface
# conditions.
# TODO: Refactor the surface conditions API to avoid needing to do this.
function set_dummy_surface_conditions!(p)
    (; params, atmos) = p
    (; sfc_conditions) = p.precomputed
    FT = eltype(params)
    @. sfc_conditions.T_sfc = FT(300)
    @. sfc_conditions.q_vap_sfc = FT(0)
    @. sfc_conditions.ustar = FT(0.2)
    @. sfc_conditions.obukhov_length = FT(1e-4)
    @. sfc_conditions.buoyancy_flux = FT(0)
    if !(atmos.microphysics_model isa DryModel)
        @. sfc_conditions.ρ_flux_q_tot = C3(FT(0))
    end
    @. sfc_conditions.ρ_flux_h_tot = C3(FT(0))

    # Zero out the surface momentum flux
    c = p.scratch.ᶠtemp_scalar
    # elsewhere known as 𝒢
    sfc_local_geometry = Fields.level(Fields.local_geometry_field(c), half)
    @. sfc_conditions.ρ_flux_uₕ = tensor_from_components(0, 0, sfc_local_geometry)
end

ifelsenothing(x, default) = x
ifelsenothing(x::Nothing, default) = default

"""
    surface_state_to_conditions(
        wrapped_sfc_setup,
        surface_local_geometry,
        T_int,
        ρ_int,
        q_tot_int,
        q_liq_int,
        q_ice_int,
        u_int,
        v_int,
        z_int,
        thermo_params,
        surface_fluxes_params,
        surface_temp_params,
        atmos,
        sfc_prognostic_temp,
        t,
    )

Computes the surface conditions, given information about the surface and the
first interior point. Implements the assumptions listed for `SurfaceState`.
"""
function surface_state_to_conditions(
    wrapped_sfc_setup::WSS,
    surface_local_geometry,
    T_int,
    ρ_int,
    q_tot_int,
    q_liq_int,
    q_ice_int,
    u_int,
    v_int,
    z_int,
    thermo_params,
    surface_fluxes_params,
    surface_temp_params,
    atmos,
    sfc_temp_var,
    t,
) where {WSS}
    surf_state = surface_state(wrapped_sfc_setup, surface_local_geometry, z_int, t)
    parameterization = surf_state.parameterization
    (; coordinates) = surface_local_geometry
    Φ_sfc = geopotential(SFP.grav(surface_fluxes_params), coordinates.z)
    Δz = z_int - coordinates.z

    FT = eltype(thermo_params)
    (!isnothing(surf_state.q_vap) && atmos.microphysics_model isa DryModel) &&
        error("surface q_vap cannot be specified when using a DryModel")

    T_sfc = if isnothing(sfc_temp_var)
        if isnothing(surf_state.T)
            surface_temperature(atmos.sfc_temperature, coordinates, surface_temp_params)
        else
            surf_state.T
        end
    else
        sfc_temp_var
    end
    u = ifelsenothing(surf_state.u, FT(0))
    v = ifelsenothing(surf_state.v, FT(0))

    uv_int = SA.SVector(u_int, v_int)
    uv_sfc = SA.SVector(u, v)

    ρ_sfc = SF.surface_density(
        surface_fluxes_params,
        T_int,
        ρ_int,
        T_sfc,
        Δz,
        q_tot_int,
        q_liq_int,
        q_ice_int,
    )
    if atmos.microphysics_model isa DryModel
        q_vap = 0
    else
        # Assume that the surface is water with saturated air directly
        # above it.
        q_vap_sat = TD.q_vap_saturation(thermo_params, T_sfc, ρ_sfc, TD.Liquid())
        q_vap = ifelsenothing(surf_state.q_vap, q_vap_sat)
    end

    gustiness = ifelsenothing(surf_state.gustiness, FT(1))

    if parameterization isa ExchangeCoefficients
        flux_specs = SF.FluxSpecs(Cd = parameterization.Cd, Ch = parameterization.Ch)
        config = SF.default_surface_flux_config(FT)
    elseif parameterization isa MoninObukhov
        if isnothing(parameterization.fluxes)
            config = SF.SurfaceFluxConfig(
                SF.ConstantRoughnessParams(parameterization.z0m, parameterization.z0b),
                SF.ConstantGustinessSpec(gustiness),
            )
            flux_specs = nothing
        else
            if parameterization.fluxes isa HeatFluxes
                (; shf, lhf) = parameterization.fluxes
                if isnothing(lhf)
                    lhf = FT(0)
                else
                    atmos.microphysics_model isa DryModel &&
                        error("lhf cannot be specified when using a DryModel")
                end
            elseif parameterization.fluxes isa θAndQFluxes
                (; θ_flux, q_flux) = parameterization.fluxes
                if isnothing(q_flux)
                    q_flux = FT(0)
                else
                    atmos.microphysics_model isa DryModel && error(
                        "q_flux cannot be specified when using a DryModel",
                    )
                end
                shf = θ_flux * ρ_sfc * TD.cp_m(thermo_params, q_vap)
                lhf = q_flux * ρ_sfc * TD.latent_heat_vapor(thermo_params, T_sfc)
            end
            flux_specs = SF.FluxSpecs(ustar = parameterization.ustar, shf = shf, lhf = lhf)
            config = SF.default_surface_flux_config(FT)
        end
    end

    return atmos_surface_conditions(
        surface_fluxes_params,
        SF.surface_fluxes(surface_fluxes_params, T_int, q_tot_int, q_liq_int, q_ice_int,
            ρ_int, T_sfc, q_vap, Φ_sfc, Δz, 0, uv_int, uv_sfc, nothing, config,
            UF.PointValueScheme(), nothing, flux_specs),
        ρ_sfc,
        surface_local_geometry,
    )
end

#Sphere SST distribution from Wing et al. (2023) https://gmd.copernicus.org/preprints/gmd-2023-235/
function surface_temperature(
    ::RCEMIPIISST,
    coordinates::Union{Geometry.LatLongZPoint, Geometry.LatLongPoint},
    surface_temp_params,
)
    (; lat) = coordinates
    (; SST_mean, SST_delta, SST_wavelength_latitude) = surface_temp_params
    T = SST_mean + SST_delta / 2 * cosd(360 * lat / SST_wavelength_latitude)
    return T
end

#Box SST distribution from Wing et al. (2023) https://gmd.copernicus.org/preprints/gmd-2023-235/
function surface_temperature(
    ::RCEMIPIISST,
    coordinates::Union{Geometry.XZPoint, Geometry.XYZPoint},
    surface_temp_params,
)
    (; x) = coordinates
    (; SST_mean, SST_delta, SST_wavelength) = surface_temp_params
    T = SST_mean - SST_delta / 2 * cospi(2 * x / SST_wavelength)
    return T
end

#For non-RCEMIPII box models with prescribed surface temp, assume that the latitude is 0.
function surface_temperature(
    ::ZonallySymmetricSST,
    coordinates,
    surface_temp_params,
)
    (; z) = coordinates
    FT = eltype(z)
    return FT(300)
end

function surface_temperature(
    ::ZonallySymmetricSST,
    coordinates::Geometry.LatLongZPoint,
    surface_temp_params,
)
    (; lat, z) = coordinates
    FT = eltype(lat)
    T = FT(271) + FT(29) * exp(-coordinates.lat^2 / (2 * 26^2)) - FT(6.5e-3) * z
    return T
end

"""
    atmos_surface_conditions(surface_conditions, ρ_sfc, surface_local_geometry)

Adds local geometry information to the `SurfaceFluxes.SurfaceFluxConditions` struct.
The resulting values are the ones actually used by ClimaAtmos operator boundary conditions.
"""
function atmos_surface_conditions(
    surface_fluxes_params,
    surface_conditions,
    ρ_sfc,
    surface_local_geometry,
)
    (; ustar, L_MO, ρτxz, ρτyz, shf, lhf, evaporation, T_sfc, q_vap_sfc) =
        surface_conditions

    # surface normal
    z = surface_normal(surface_local_geometry)

    buoy_flux = SF.buoyancy_flux(surface_fluxes_params, shf, lhf, T_sfc, ρ_sfc, q_vap_sfc)

    energy_flux = (; ρ_flux_h_tot = vector_from_component(shf + lhf, z))

    # NOTE: Technically, ρ_flux_q_tot is not needed when the model is Dry ...
    moisture_flux = (; ρ_flux_q_tot = vector_from_component(evaporation, z))

    return (;
        T_sfc,
        q_vap_sfc,
        ustar,
        obukhov_length = L_MO,
        buoyancy_flux = buoy_flux,
        # This drops the C3 component of ρ_flux_u, need to add ρ_flux_u₃
        ρ_flux_uₕ = tensor_from_components(ρτxz, ρτyz, surface_local_geometry, z),
        energy_flux...,
        moisture_flux...,
    )
end

surface_normal(L::Geometry.LocalGeometry) = C3(unit_basis_vector_data(C3, L))

vector_from_component(f₁, n₁) = f₁ * n₁
vector_from_component(f₁, L::Geometry.LocalGeometry) =
    vector_from_component(f₁, surface_normal(L))

function tensor_from_components(f₁₃, f₂₃, L, n₃ = surface_normal(L))
    xz = CT12(CT1(unit_basis_vector_data(CT1, L)), L)
    yz = CT12(CT2(unit_basis_vector_data(CT2, L)), L)
    f = C12(f₁₃ * xz + f₂₃ * yz, L)
    return n₃ ⊗ f
end

"""
    surface_conditions_type(atmos_model, FT)

Gets the return type of `surface_conditions` without evaluating the function.
"""
function surface_conditions_type(atmos, ::Type{FT}) where {FT}
    energy_flux_names = (:ρ_flux_h_tot,)
    # NOTE: Technically ρ_flux_q_tot is not really needed for a dry model, but
    # SF always has evaporation
    moisture_flux_names = (:ρ_flux_q_tot,)
    names = (:T_sfc, :q_vap_sfc, :ustar, :obukhov_length, :buoyancy_flux, :ρ_flux_uₕ,
        energy_flux_names..., moisture_flux_names...,
    )
    type_tuple = Tuple{
        FT, FT, FT, FT, FT,
        typeof(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
        ntuple(_ -> C3{FT}, Val(length(energy_flux_names)))...,
        ntuple(_ -> C3{FT}, Val(length(moisture_flux_names)))...,
    }
    return NamedTuple{names, type_tuple}
end
