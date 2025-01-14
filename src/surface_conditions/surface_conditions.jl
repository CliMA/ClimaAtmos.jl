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
    (; ᶜts, ᶜu, sfc_conditions) = p.precomputed
    (; params, sfc_setup, atmos) = p
    thermo_params = CAP.thermodynamics_params(params)
    surface_fluxes_params = CAP.surface_fluxes_params(params)
    surface_temp_params = CAP.surface_temp_params(params)
    int_ts_values = Fields.field_values(Fields.level(ᶜts, 1))
    int_u_values = Fields.field_values(Fields.level(ᶜu, 1))
    int_z_values =
        Fields.field_values(Fields.level(Fields.coordinate_field(Y.c).z, 1))
    sfc_conditions_values = Fields.field_values(sfc_conditions)
    wrapped_sfc_setup = sfc_setup_wrapper(sfc_setup)
    sfc_temp_var =
        p.atmos.surface_model isa PrognosticSurfaceTemperature ?
        Fields.field_values(Y.sfc.T) : nothing
    @. sfc_conditions_values = surface_state_to_conditions(
        wrapped_sfc_setup,
        sfc_local_geometry_values,
        int_ts_values,
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
    wrapped_sfc_setup::F,
    sfc_local_geometry_values,
    int_z_values,
    t,
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
    thermo_params = CAP.thermodynamics_params(params)
    @. sfc_conditions.ustar = FT(0.2)
    @. sfc_conditions.obukhov_length = FT(1e-4)
    @. sfc_conditions.buoyancy_flux = FT(0)
    if atmos.moisture_model isa DryModel
        @. sfc_conditions.ts = TD.PhaseDry_ρT(thermo_params, FT(1), FT(300))
    else
        @. sfc_conditions.ts = TD.PhaseNonEquil_ρTq(
            thermo_params,
            FT(1),
            FT(300),
            TD.PhasePartition(FT(0)),
        )
        @. sfc_conditions.ρ_flux_q_tot = C3(FT(0))
    end
    @. sfc_conditions.ρ_flux_h_tot = C3(FT(0))
end

"""
    set_surface_conditions!(p, surface_conditions, surface_ts)

Sets `p.precomputed.sfc_conditions` according to `surface_conditions` and `surface_ts`,
which are `Field`s of `SurfaceFluxes.SurfaceFluxConditions` and `Thermodynamics.ThermodynamicState`s
This functions needs to be called by the coupler whenever either field changes
to ensure that the simulation is properly updated.
"""
function set_surface_conditions!(p, surface_conditions, surface_ts)
    (; params, atmos) = p
    (; sfc_conditions,) = p.precomputed
    (; ᶠtemp_scalar) = p.scratch

    FT = eltype(params)
    FT′ = eltype(parent(surface_conditions))
    FT′′ = eltype(parent(surface_ts))
    @assert FT === FT′ === FT′′
    @assert eltype(surface_conditions) <: SF.SurfaceFluxConditions
    @assert eltype(surface_ts) <: TD.ThermodynamicState

    sfc_local_geometry =
        Fields.level(Fields.local_geometry_field(ᶠtemp_scalar), Fields.half)
    @. sfc_conditions = atmos_surface_conditions(
        surface_conditions,
        surface_ts,
        sfc_local_geometry,
    )
end

ifelsenothing(x, default) = x
ifelsenothing(x::Nothing, default) = default

"""
    surface_state_to_conditions(
        wrapped_sfc_setup,
        surface_local_geometry,
        interior_ts,
        interior_u,
        interior_v,
        interior_z,
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
    interior_ts,
    interior_u,
    interior_v,
    interior_z,
    thermo_params,
    surface_fluxes_params,
    surface_temp_params,
    atmos,
    sfc_prognostic_temp,
    t,
) where {WSS}
    surf_state =
        surface_state(wrapped_sfc_setup, surface_local_geometry, interior_z, t)
    parameterization = surf_state.parameterization
    (; coordinates) = surface_local_geometry
    FT = eltype(thermo_params)

    (!isnothing(surf_state.q_vap) && atmos.moisture_model isa DryModel) &&
        error("surface q_vap cannot be specified when using a DryModel")

    T = if isnothing(sfc_prognostic_temp)
        if isnothing(surf_state.T)
            surface_temperature(
                atmos.sfc_temperature,
                coordinates,
                surface_temp_params,
            )
        else
            surf_state.T
        end
    else
        sfc_prognostic_temp
    end
    u = ifelsenothing(surf_state.u, FT(0))
    v = ifelsenothing(surf_state.v, FT(0))

    if isnothing(surf_state.p)
        # Assume an adiabatic profile with constant cv and R above the surface.
        cv = TD.cv_m(thermo_params, interior_ts)
        R = TD.gas_constant_air(thermo_params, interior_ts)
        interior_ρ = TD.air_density(thermo_params, interior_ts)
        interior_T = TD.air_temperature(thermo_params, interior_ts)
        ρ = interior_ρ * (T / interior_T)^(cv / R)
        if atmos.moisture_model isa DryModel
            ts = TD.PhaseDry_ρT(thermo_params, ρ, T)
        else
            # Assume that the surface is water with saturated air directly
            # above it.
            q_vap_sat =
                TD.q_vap_saturation_generic(thermo_params, T, ρ, TD.Liquid())
            q_vap = ifelsenothing(surf_state.q_vap, q_vap_sat)
            q = TD.PhasePartition(q_vap)
            ts = TD.PhaseNonEquil_ρTq(thermo_params, ρ, T, q)
        end
    else
        p = surf_state.p
        if atmos.moisture_model isa DryModel
            ts = TD.PhaseDry_pT(thermo_params, p, T)
        else
            q_vap = if isnothing(surf_state.q_vap)
                # Assume that the surface is water with saturated air directly
                # above it.
                phase = TD.Liquid()
                p_sat =
                    TD.saturation_vapor_pressure(thermo_params, T, phase)
                ϵ_v =
                    TD.Parameters.R_d(thermo_params) /
                    TD.Parameters.R_v(thermo_params)
                ϵ_v * p_sat / (p - p_sat * (1 - ϵ_v))
            else
                surf_state.q_vap
            end
            q = TD.PhasePartition(q_vap)
            ts = TD.PhaseNonEquil_pTq(thermo_params, p, T, q)
        end
    end

    surface_values = SF.StateValues(coordinates.z, SA.SVector(u, v), ts)
    interior_values = SF.StateValues(
        interior_z,
        SA.SVector(interior_u, interior_v),
        interior_ts,
    )

    if parameterization isa ExchangeCoefficients
        gustiness = ifelsenothing(surf_state.gustiness, FT(1))
        beta = ifelsenothing(surf_state.beta, FT(1))
        inputs = SF.Coefficients(
            interior_values,
            surface_values,
            parameterization.Cd,
            parameterization.Ch,
            gustiness,
            beta,
        )
    elseif parameterization isa MoninObukhov
        if isnothing(parameterization.fluxes)
            gustiness = ifelsenothing(surf_state.gustiness, FT(1))
            beta = ifelsenothing(surf_state.beta, FT(1))
            isnothing(parameterization.ustar) || error(
                "ustar cannot be specified when surface fluxes are prescribed",
            )
            inputs = SF.ValuesOnly(
                interior_values,
                surface_values,
                parameterization.z0m,
                parameterization.z0b,
                gustiness,
                beta,
            )
        else
            if parameterization.fluxes isa HeatFluxes
                (; shf, lhf) = parameterization.fluxes
                if isnothing(lhf)
                    lhf = FT(0)
                else
                    atmos.moisture_model isa DryModel &&
                        error("lhf cannot be specified when using a DryModel")
                end
            elseif parameterization.fluxes isa θAndQFluxes
                (; θ_flux, q_flux) = parameterization.fluxes
                if isnothing(q_flux)
                    q_flux = FT(0)
                else
                    atmos.moisture_model isa DryModel && error(
                        "q_flux cannot be specified when using a DryModel",
                    )
                end
                ρ = TD.air_density(thermo_params, ts)
                shf = θ_flux * ρ * TD.cp_m(thermo_params, ts)
                lhf = q_flux * ρ * TD.latent_heat_vapor(thermo_params, ts)
            end
            if isnothing(surf_state.gustiness)
                buoyancy_flux = SF.compute_buoyancy_flux(
                    surface_fluxes_params,
                    shf,
                    lhf,
                    interior_ts,
                    ts,
                    SF.PointValueScheme(),
                )
                gustiness = get_wstar(buoyancy_flux)
                # TODO: We are assuming that the average mixed layer depth is
                # always 1000 meters. This needs to be adjusted for deep
                # convective cases like TRMM.
            else
                gustiness = surf_state.gustiness
            end
            isnothing(surf_state.beta) || error(
                "beta cannot be specified when surface fluxes are prescribed",
            )
            if isnothing(parameterization.ustar)
                inputs = SF.Fluxes(
                    interior_values,
                    surface_values,
                    shf,
                    lhf,
                    parameterization.z0m,
                    parameterization.z0b,
                    gustiness,
                )
            else
                inputs = SF.FluxesAndFrictionVelocity(
                    interior_values,
                    surface_values,
                    shf,
                    lhf,
                    parameterization.ustar,
                    parameterization.z0m,
                    parameterization.z0m,
                    gustiness,
                )
            end
        end
    end

    return atmos_surface_conditions(
        SF.surface_conditions(surface_fluxes_params, inputs),
        ts,
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
    FT = eltype(lat)
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
    FT = eltype(x)
    T = SST_mean - SST_delta / 2 * cos(2 * FT(pi) * x / SST_wavelength)
    return T
end

function surface_temperature(
    ::RCEMIPIISST,
    coordinates::Geometry.ZPoint,
    surface_temp_params,
)
    (; z) = coordinates
    (; SST_mean) = surface_temp_params
    FT = eltype(z)
    return FT(SST_mean)
end

#For non-RCEMIPII box models with prescribed surface temp, assume that the latitude is 0.
function surface_temperature(
    ::Union{ZonallySymmetricSST, ZonallyAsymmetricSST},
    coordinates::Union{Geometry.XZPoint, Geometry.XYZPoint},
    surface_temp_params,
)
    (; x) = coordinates
    FT = eltype(x)
    return FT(300)
end

function surface_temperature(
    ::Union{ZonallySymmetricSST, ZonallyAsymmetricSST},
    coordinates::Geometry.ZPoint,
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

function surface_temperature(
    ::ZonallyAsymmetricSST,
    coordinates::Geometry.LatLongZPoint,
    surface_temp_params,
)
    (; lat, long, z) = coordinates
    FT = eltype(lat)
    #Assume a surface temperature that varies with both longitude and latitude, Neale and Hoskins, 2021
    T =
        (
            (-60 < lat < 60) ?
            (FT(27) * (FT(1) - sind((FT(3) * lat) / FT(2))^2) + FT(273.16)) :
            FT(273.16)
        ) + (
            (-180 < long < 180 && -30 < lat < 30) ?
            (
                FT(3) * cosd(long + FT(90)) * cospi(FT(0.5) * lat / FT(30))^2 +
                FT(0)
            ) : FT(0)
        ) - FT(6.5e-3) * z
    return T
end

"""
    atmos_surface_conditions(
        surface_conditions,
        ts,
        surface_local_geometry
    )

Adds local geometry information to the `SurfaceFluxes.SurfaceFluxConditions` struct
along with information about the thermodynamic state. The resulting values are the
ones actually used by ClimaAtmos operator boundary conditions.
"""
function atmos_surface_conditions(
    surface_conditions,
    ts,
    surface_local_geometry,
)
    (; ustar, L_MO, buoy_flux, ρτxz, ρτyz, shf, lhf, evaporation) =
        surface_conditions

    # surface normal
    z = surface_normal(surface_local_geometry)

    energy_flux = (; ρ_flux_h_tot = vector_from_component(shf + lhf, z))

    # NOTE: Technically, ρ_flux_q_tot is not needed when the model is Dry ...
    moisture_flux = (; ρ_flux_q_tot = vector_from_component(evaporation, z))

    return (;
        ts,
        ustar,
        obukhov_length = L_MO,
        buoyancy_flux = buoy_flux,
        # This drops the C3 component of ρ_flux_u, need to add ρ_flux_u₃
        ρ_flux_uₕ = tensor_from_components(
            ρτxz,
            ρτyz,
            surface_local_geometry,
            z,
        ),
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
    surface_conditions_type(moisture_model, FT)

Gets the return type of `surface_conditions` without evaluating the function.
"""
function surface_conditions_type(atmos, ::Type{FT}) where {FT}
    energy_flux_names = (:ρ_flux_h_tot,)
    # NOTE: Technically ρ_flux_q_tot is not really needed for a dry model, but
    # SF always has evaporation
    moisture_flux_names = (:ρ_flux_q_tot,)
    names = (
        :ts,
        :ustar,
        :obukhov_length,
        :buoyancy_flux,
        :ρ_flux_uₕ,
        energy_flux_names...,
        moisture_flux_names...,
    )
    type_tuple = Tuple{
        atmos.moisture_model isa DryModel ? TD.PhaseDry{FT} :
        TD.PhaseNonEquil{FT},
        FT,
        FT,
        FT,
        typeof(C3(FT(0)) ⊗ C12(FT(0), FT(0))),
        ntuple(_ -> C3{FT}, Val(length(energy_flux_names)))...,
        ntuple(_ -> C3{FT}, Val(length(moisture_flux_names)))...,
    }
    return NamedTuple{names, type_tuple}
end
