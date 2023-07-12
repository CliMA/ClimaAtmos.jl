# TODO: Remove this type piracy.
Base.Broadcast.BroadcastStyle(
    ::Base.Broadcast.Style{Tuple},
    ds::DataLayouts.DataStyle,
) = ds

"""
    unit_basis_vector_data(type, local_geometry)

The component of the vector of the specified type with length 1 in physical units.
The type should correspond to a vector with only one component, i.e., a basis vector.
"""
function unit_basis_vector_data(::Type{V}, local_geometry) where {V}
    FT = Geometry.undertype(typeof(local_geometry))
    return FT(1) / Geometry._norm(V(FT(1)), local_geometry)
end

"""
    projected_vector_data(::Type{V}, vector, local_geometry)

Projects the given vector onto the axis of V, then extracts the component data and rescales it to physical units.
The type should correspond to a vector with only one component, i.e., a basis vector.
"""
projected_vector_data(::Type{V}, vector, local_geometry) where {V} =
    V(vector, local_geometry)[1] / unit_basis_vector_data(V, local_geometry)

"""
    update_surface_conditions!(Y, p, t)

Updates the value of `p.sfc_conditions` based on the current state `Y` and time
`t`. This function will only update the surface conditions if the surface_setup
is not a PrescribedSurface.
"""
function update_surface_conditions!(Y, p, t)
    isnothing(p.sfc_setup) && return
    # Need to extract the field values so that we can do
    # a DataLayout broadcast rather than a Field broadcast
    # because we are mixing surface and interior fields
    sfc_local_geometry_values = Fields.field_values(
        Fields.level(Fields.local_geometry_field(Y.f), Fields.half),
    )
    int_local_geometry_values =
        Fields.field_values(Fields.level(Fields.local_geometry_field(Y.c), 1))

    (; ᶜts, ᶜu, sfc_conditions, sfc_setup, params, atmos) = p
    int_ts_values = Fields.field_values(Fields.level(ᶜts, 1))
    int_u_values = Fields.field_values(Fields.level(ᶜu, 1))
    int_z_values =
        Fields.field_values(Fields.level(Fields.coordinate_field(Y.c).z, 1))
    sfc_conditions_values = Fields.field_values(sfc_conditions)

    if sfc_setup isa Function
        @. sfc_conditions_values = surface_state_to_conditions(
            sfc_setup(sfc_local_geometry_values.coordinates, int_z_values, t),
            sfc_local_geometry_values,
            int_ts_values,
            projected_vector_data(CT1, int_u_values, int_local_geometry_values),
            projected_vector_data(CT2, int_u_values, int_local_geometry_values),
            int_z_values,
            params,
            atmos,
        )
    elseif sfc_setup isa Fields.Field # This case is needed for the Coupler
        @assert eltype(sfc_setup) <: SurfaceState
        sfc_setup_values = Fields.field_values(sfc_setup)
        @. sfc_conditions_values = surface_state_to_conditions(
            sfc_setup_values,
            sfc_local_geometry_values,
            int_ts_values,
            projected_vector_data(CT1, int_u_values, int_local_geometry_values),
            projected_vector_data(CT2, int_u_values, int_local_geometry_values),
            int_z_values,
            params,
            atmos,
        )
    else # SurfaceSetup is a SurfaceState
        tup_sfc_setup = tuple(sfc_setup)
        @. sfc_conditions_values = surface_state_to_conditions(
            tup_sfc_setup,
            sfc_local_geometry_values,
            int_ts_values,
            projected_vector_data(CT1, int_u_values, int_local_geometry_values),
            projected_vector_data(CT2, int_u_values, int_local_geometry_values),
            int_z_values,
            params,
            atmos,
        )
    end
    return nothing
end

"""
    set_surface_conditions!(p, surface_conditions, surface_ts)

Sets `p.sfc_conditions` according to `surface_conditions` and `surface_ts`,
which are `Field`s of `SurfaceFluxes.SurfaceFluxConditions` and `Thermodynamics.ThermodynamicState`s
This functions needs to be called by the coupler whenever either field changes
to ensure that the simulation is properly updated.
"""
function set_surface_conditions!(p, surface_conditions, surface_ts)
    (; sfc_conditions, params, atmos, ᶠtemp_scalar) = p

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
        atmos,
        params,
    )
end

"""
    surface_state_to_conditions(
        surface_state,
        surface_local_geometry,
        interior_ts,
        interior_u,
        interior_v,
        interior_z,
        params,
        atmos,
    )

Computes the surface conditions, given information about the surface and the
first interior point. Implements the assumptions listed for `SurfaceState`.
"""
function surface_state_to_conditions(
    surface_state,
    surface_local_geometry,
    interior_ts,
    interior_u,
    interior_v,
    interior_z,
    params,
    atmos,
)
    (; parameterization, T, p, q_vap, u, v, gustiness, beta) = surface_state
    (; coordinates) = surface_local_geometry
    FT = eltype(params)
    thermo_params = CAP.thermodynamics_params(params)
    surface_params = CAP.surface_fluxes_params(params)

    (!isnothing(q_vap) && atmos.moisture_model isa DryModel) &&
        error("surface q_vap cannot be specified when using a DryModel")

    if isnothing(T)
        # Assume an idealized latitude-dependent surface temperature.
        if coordinates isa Geometry.LatLongZPoint ||
           coordinates isa Geometry.LatLongPoint
    
            T = ((-60 < coordinates.lat < 60) ? 
            (FT(27) * (FT(1) - sinpi((FT(3) * coordinates.lat)/(FT(2) * FT(180)))^2) + FT(273.16)) : FT(273.16)) + 
            ((-180 < coordinates.long < 180 && -30 < coordinates.lat < 30) ? 
            (FT(3) * cospi((coordinates.long + FT(90))/(FT(180))) * cospi(FT(0.5) * coordinates.lat / FT(30))^2 + FT(0)) : FT(0))

            #Zonal SST = (-60 < coordinates.lat < 60) ? FT(27) * (FT(1) - sinpi((FT(3) * coordinates.lat)/(FT(2) * FT(180)))^2) + FT(273.16) : FT(273.16)
            #Perturbation = (-180 < coordinates.long < 180 && -30 < coordinates.lat < 30) ? FT(3) * cospi((coordinates.long - FT(90))/(FT(180))) * cospi(FT(0.5) * coordinates.lat / FT(30))^2 + FT(273) : FT(273)

            #Original
            #T = FT(320) + FT(29) * exp(-coordinates.lat^2 / (2 * 26^2))

            #T + ΔT, following TC approach of Zhang et al. 2020
            #T = FT(273.16) + 28 * (1 - sind(FT(0.85) * (coordinates.lat - 10))^2)
            #ΔT = 2 * cosd(coordinates.long) * cospi((coordinates.lat - 10)/(2 * 30))^2
            #T = FT(273.16) + 28 * (1 - sind(FT(0.85) * (coordinates.lat - 10))^2) + 2 * cosd(coordinates.long) * cospi((coordinates.lat - 10)/(2 * 30))^2

        else
            # Assume that the latitude is 0.
            T = FT(273.16)
        end
    end
    if isnothing(u)
        u = FT(0)
    end
    if isnothing(v)
        v = FT(0)
    end
    if isnothing(p)
        # Assume an adiabatic profile with constant cv and R above the surface.
        cv = TD.cv_m(thermo_params, interior_ts)
        R = TD.gas_constant_air(thermo_params, interior_ts)
        interior_ρ = TD.air_density(thermo_params, interior_ts)
        interior_T = TD.air_temperature(thermo_params, interior_ts)
        ρ = interior_ρ * (T / interior_T)^(cv / R)
        if atmos.moisture_model isa DryModel
            ts = TD.PhaseDry_ρT(thermo_params, ρ, T)
        else
            if isnothing(q_vap)
                # Assume that the surface is water with saturated air directly
                # above it.
                phase = TD.Liquid()
                q_vap = TD.q_vap_saturation_generic(thermo_params, T, ρ, phase)
            end
            q = TD.PhasePartition(q_vap)
            ts = TD.PhaseNonEquil_ρTq(thermo_params, ρ, T, q)
        end
    else
        if atmos.moisture_model isa DryModel
            ts = TD.PhaseDry_pT(thermo_params, p, T)
        else
            if isnothing(q_vap)
                # Assume that the surface is water with saturated air directly
                # above it.
                phase = TD.Liquid()
                p_sat = TD.saturation_vapor_pressure(thermo_params, T, phase)
                ϵ_v =
                    TD.Parameters.R_d(thermo_params) /
                    TD.Parameters.R_v(thermo_params)
                q_vap = ϵ_v * p_sat / (p - p_sat * (1 - ϵ_v))
            end
            q = TD.PhasePartition(q_vap)
            ts = TD.PhaseNonEquil_pTq(thermo_params, p, T, q)
        end
    end

    surface_values = SF.SurfaceValues(coordinates.z, SA.SVector(u, v), ts)
    interior_values = SF.InteriorValues(
        interior_z,
        SA.SVector(interior_u, interior_v),
        interior_ts,
    )

    if parameterization isa ExchangeCoefficients
        if isnothing(gustiness)
            gustiness = FT(1)
        end
        if isnothing(beta)
            beta = FT(1)
        end
        inputs = SF.Coefficients(
            interior_values,
            surface_values,
            parameterization.Cd,
            parameterization.Ch,
            FT(NaN), # TODO: Remove z0m from SF.Coefficients
            FT(NaN), # TODO: Remove z0b from SF.Coefficients
            gustiness,
            beta,
        )
    elseif parameterization isa MoninObukhov
        if isnothing(parameterization.fluxes)
            if isnothing(gustiness)
                gustiness = FT(1)
            end
            if isnothing(beta)
                beta = FT(1)
            end
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
            if isnothing(gustiness)
                buoyancy_flux = SF.compute_buoyancy_flux(
                    surface_params,
                    shf,
                    lhf,
                    interior_ts,
                    ts,
                    SF.FVScheme(),
                )
                gustiness = get_wstar(buoyancy_flux)
                # TODO: We are assuming that the average mixed layer depth is
                # always 1000 meters. This needs to be adjusted for deep
                # convective cases like TRMM.
            end
            isnothing(beta) || error(
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
                    FT(NaN),
                    FT(NaN),
                    gustiness,
                )
            end
        end
    end

    return atmos_surface_conditions(
        SF.surface_conditions(surface_params, inputs),
        ts,
        surface_local_geometry,
        atmos,
        params,
    )
end

"""
    atmos_surface_conditions(
        surface_conditions,
        ts,
        surface_local_geometry,
        atmos,
        params,
    )

Adds local geometry information to the `SurfaceFluxes.SurfaceFluxConditions` struct
along with information about the thermodynamic state. The resulting values are the
ones actually used by ClimaAtmos operator boundary conditions.
"""
function atmos_surface_conditions(
    surface_conditions,
    ts,
    surface_local_geometry,
    atmos,
    params,
)
    (; ustar, L_MO, buoy_flux, ρτxz, ρτyz, shf, lhf, evaporation) =
        surface_conditions

    thermo_params = CAP.thermodynamics_params(params)

    surface_normal = C3(unit_basis_vector_data(C3, surface_local_geometry))
    energy_flux = if atmos.energy_form isa PotentialTemperature
        (; ρ_flux_θ = shf / TD.cp_m(thermo_params, ts) * surface_normal)
    elseif atmos.energy_form isa TotalEnergy
        if atmos.turbconv_model isa TC.EDMFModel
            (;
                ρ_flux_h_tot = (shf + lhf) * surface_normal,
                ρ_flux_θ = shf / TD.cp_m(thermo_params, ts) * surface_normal,
            )
        else
            (; ρ_flux_h_tot = (shf + lhf) * surface_normal)
        end
    end
    moisture_flux =
        atmos.moisture_model isa DryModel &&
        !(atmos.turbconv_model isa TC.EDMFModel) ? (;) :
        (; ρ_flux_q_tot = evaporation * surface_normal)
    return (;
        ts,
        ustar,
        obukhov_length = L_MO,
        buoyancy_flux = buoy_flux,
        # This drops the C3 component of ρ_flux_u, need to add ρ_flux_u₃
        ρ_flux_uₕ = surface_normal ⊗ C12(
            ρτxz * CT12(
                CT1(unit_basis_vector_data(CT1, surface_local_geometry)),
                surface_local_geometry,
            ) +
            ρτyz * CT12(
                CT2(unit_basis_vector_data(CT2, surface_local_geometry)),
                surface_local_geometry,
            ),
            surface_local_geometry,
        ),
        energy_flux...,
        moisture_flux...,
    )
end

"""
    surface_conditions_type(moisture_model, energy_form, FT)

Gets the return type of `surface_conditions` without evaluating the function.
"""
function surface_conditions_type(atmos, ::Type{FT}) where {FT}
    energy_flux_names = if atmos.energy_form isa PotentialTemperature
        (:ρ_flux_θ,)
    elseif atmos.energy_form isa TotalEnergy
        if atmos.turbconv_model isa TC.EDMFModel
            (:ρ_flux_h_tot, :ρ_flux_θ)
        else
            (:ρ_flux_h_tot,)
        end
    end
    moisture_flux_names =
        atmos.moisture_model isa DryModel &&
        !(atmos.turbconv_model isa TC.EDMFModel) ? () : (:ρ_flux_q_tot,)
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
