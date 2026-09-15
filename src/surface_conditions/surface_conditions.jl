"""
    update_surface_conditions!(Y, p, t)

Fill `p.precomputed.sfc_conditions` from the current state `Y` and time `t`.

Called once per explicit precomputed-quantity update, from
`set_explicit_precomputed_quantities!`. Returns `nothing` early, leaving
`sfc_conditions` untouched, when `atmos.surface.flux_scheme` is `nothing` (the
coupler-handoff case, in which an external driver writes the surface fields).

The surface temperature and the flux scheme are resolved once here — not per
cell — and the boundary overrides are wrapped so that both a scalar
[`SurfaceBoundaryOverrides`](@ref) and a coupler-provided
`Fields.Field{<:SurfaceBoundaryOverrides}` broadcast correctly. The per-point
work is done by [`surface_state_to_conditions`](@ref), broadcast over
`DataLayout`s (rather than `Field`s) because it mixes surface and first-interior
values.

See the [Surface Conditions](@ref "Surface Conditions") page for the user-facing
guide and the Surface Conditions Internals page for the data flow.
"""
function update_surface_conditions!(Y, p, t)
    atmos = p.atmos
    isnothing(atmos.surface.flux_scheme) && return nothing

    # Need to extract the field values so that we can do
    # a DataLayout broadcast rather than a Field broadcast
    # because we are mixing surface and interior fields
    sfc_local_geometry_values = Fields.field_values(
        Fields.level(Fields.local_geometry_field(Y.f), Fields.half),
    )
    int_local_geometry_values =
        Fields.field_values(Fields.level(Fields.local_geometry_field(Y.c), 1))
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice, ᶜu, sfc_conditions) = p.precomputed
    (; params, sfc_setup) = p
    thermo_params = CAP.thermodynamics_params(params)
    surface_fluxes_params = CAP.surface_fluxes_params(params)
    surface_temp_params = CAP.surface_temp_params(params)
    int_T_values = Fields.field_values(Fields.level(ᶜT, 1))
    int_ρ_values = Fields.field_values(Fields.level(Y.c.ρ, 1))
    int_q_tot_values = Fields.field_values(Fields.level(ᶜq_tot_nonneg, 1))
    int_q_liq_values = Fields.field_values(Fields.level(ᶜq_liq, 1))
    int_q_ice_values = Fields.field_values(Fields.level(ᶜq_ice, 1))
    int_u_values = Fields.field_values(Fields.level(ᶜu, 1))
    int_z_values = Fields.field_values(Fields.level(Fields.coordinate_field(Y.c).z, 1))
    sfc_conditions_values = Fields.field_values(sfc_conditions)

    overrides = boundary_overrides_wrapper(sfc_setup)
    T_sfc_values = surface_temperature(atmos.surface.temperature, Y, p, t)
    flux_scheme = resolve_flux_scheme(atmos.surface.flux_scheme, t, eltype(params))

    @. sfc_conditions_values = surface_state_to_conditions(
        overrides,
        flux_scheme,
        T_sfc_values,
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
        atmos.microphysics_model,
        t,
    )
    return nothing
end

"""
    resolve_flux_scheme(flux_scheme, t, ::Type{FT})

Evaluate a time-varying prescribed-flux closure at time `t`.

A [`MoninObukhov`](@ref) whose `fluxes` field is a callable
`(t, FT) -> PrescribedFluxes` is replaced by one holding the evaluated fluxes;
every other flux scheme passes through unchanged. Called once per surface
update from [`update_surface_conditions!`](@ref), before the per-cell
broadcast, so the callable is never invoked inside the broadcast kernel.
"""
function resolve_flux_scheme(p::MoninObukhov, t, ::Type{FT}) where {FT}
    p.fluxes isa Function || return p
    return MoninObukhov(p.z0m, p.z0b, p.fluxes(t, FT), p.ustar)
end
resolve_flux_scheme(p, t, ::Type{FT}) where {FT} = p

"""
    boundary_overrides_wrapper(overrides)

Wrap `p.sfc_setup` so that it broadcasts as a single value per surface point.

A scalar [`SurfaceBoundaryOverrides`](@ref) is wrapped in a tuple; a
`Fields.Field{<:SurfaceBoundaryOverrides}` (the coupler case, one override per
cell) is unwrapped to its `DataLayout`. Called from
[`update_surface_conditions!`](@ref).
"""
boundary_overrides_wrapper(o::SurfaceBoundaryOverrides) = tuple(o)
function boundary_overrides_wrapper(o::Fields.Field)
    @assert eltype(o) <: SurfaceBoundaryOverrides
    return Fields.field_values(o)
end

"""
    resolve_T_sfc(T_sfc_in, coords, surface_temp_params, t_time)

Return the surface temperature at one point [K].

An [`AnalyticTemperature`](@ref) is evaluated at `coords` and `t_time`; a scalar
or an already-resolved per-cell value passes through unchanged. Called from
[`surface_state_to_conditions`](@ref), inside the per-cell broadcast.
"""
resolve_T_sfc(t::AnalyticTemperature, coords, surface_temp_params, t_time) =
    t.f(coords, surface_temp_params, t_time)
resolve_T_sfc(t, coords, surface_temp_params, t_time) = t

ifelsenothing(x, default) = x
ifelsenothing(::Nothing, default) = default

"""
    init_sfc_conditions_zero!(p)

Fill `p.precomputed.sfc_conditions` with placeholder values and return
`nothing`.

All fluxes are set to zero, with nonzero placeholders for the quantities that
downstream consumers divide by or take logs of (`T_sfc = 300` K,
`ustar = 0.2` m/s, `obukhov_length = 1e-4` m). Used when the flux scheme is
`nothing`, so that the first `set_precomputed_quantities!` call does not expose
uninitialized memory to consumers such as RRTMGP and diagnostic EDMF before the
external driver writes the real surface fields.
"""
function init_sfc_conditions_zero!(p)
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
    c = p.scratch.ᶠtemp_scalar
    sfc_local_geometry = Fields.level(Fields.local_geometry_field(c), half)
    @. sfc_conditions.ρ_flux_uₕ = tensor_from_components(0, 0, sfc_local_geometry)
    return nothing
end

"""
    surface_state_to_conditions(
        overrides, parameterization, T_sfc_in, surface_local_geometry,
        T_int, ρ_int, q_tot_int, q_liq_int, q_ice_int, u_int, v_int, z_int,
        thermo_params, surface_fluxes_params, surface_temp_params,
        microphysics_model, t_time,
    )

Compute the surface conditions at one surface point.

Broadcast over the surface by [`update_surface_conditions!`](@ref). The surface
density comes from `SurfaceFluxes.surface_density` (extrapolated from the first
interior level), and, for a moist model, the surface air is assumed saturated
over liquid water unless `overrides.q_vap` says otherwise. The
`parameterization` selects how `SurfaceFluxes.surface_fluxes` is configured:
[`ExchangeCoefficients`](@ref) supplies fixed `Cd`/`Ch`, while
[`MoninObukhov`](@ref) supplies roughness lengths (and gustiness) when no fluxes
are prescribed, or the prescribed `shf`/`lhf` and `ustar` when they are. A
[`θAndQFluxes`](@ref) closure is converted here to `shf`/`lhf` using the local
`ρ_sfc`, `cp_m`, and latent heat of vaporization.

# Arguments

  - `overrides`: Per-point [`SurfaceBoundaryOverrides`](@ref); only `q_vap`, `u`,
    `v`, and `gustiness` are consumed.
  - `parameterization`: The [`SurfaceParameterization`](@ref) flux closure, with
    any time-varying fluxes already resolved by `resolve_flux_scheme`.
  - `T_sfc_in`: A scalar or per-cell surface temperature [K], or an
    [`AnalyticTemperature`](@ref) to evaluate at this point (see `resolve_T_sfc`).
  - `surface_local_geometry`: Local geometry at the surface, supplying the
    coordinates and the surface normal.
  - `T_int`, `ρ_int`, `q_tot_int`, `q_liq_int`, `q_ice_int`, `u_int`, `v_int`,
    `z_int`: First-interior-level temperature [K], density [kg/m³], specific
    humidities [kg/kg], horizontal velocity components [m/s], and height [m].
  - `thermo_params`, `surface_fluxes_params`, `surface_temp_params`: Parameter
    sets for thermodynamics, `SurfaceFluxes`, and the analytic surface
    temperature.
  - `microphysics_model`: The model's microphysics, used here to detect a
    `DryModel`. This takes the microphysics rather than the whole `AtmosModel`
    because this runs inside a broadcast: every argument reaches the GPU kernel
    and so must be a bitstype.
  - `t_time`: Simulation time, passed to an `AnalyticTemperature` [s].

# Returns

The NamedTuple built by [`atmos_surface_conditions`](@ref), whose type is given
by `surface_conditions_type`.

Errors when `overrides.q_vap`, `lhf`, or `q_flux` is specified for a `DryModel`.
"""
function surface_state_to_conditions(
    overrides::SurfaceBoundaryOverrides,
    parameterization::SurfaceParameterization,
    T_sfc_in,
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
    microphysics_model,
    t_time,
)
    (; coordinates) = surface_local_geometry
    Φ_sfc = geopotential(SFP.grav(surface_fluxes_params), coordinates.z)
    Δz = z_int - coordinates.z

    FT = eltype(thermo_params)
    (!isnothing(overrides.q_vap) && microphysics_model isa DryModel) &&
        error("surface q_vap cannot be specified when using a DryModel")

    T_sfc = resolve_T_sfc(T_sfc_in, coordinates, surface_temp_params, t_time)
    u = ifelsenothing(overrides.u, FT(0))
    v = ifelsenothing(overrides.v, FT(0))

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
    if microphysics_model isa DryModel
        q_vap = 0
    else
        # Assume that the surface is water with saturated air directly
        # above it.
        q_vap_sat = TD.q_vap_saturation(thermo_params, T_sfc, ρ_sfc, TD.Liquid())
        q_vap = ifelsenothing(overrides.q_vap, q_vap_sat)
    end

    gustiness = ifelsenothing(overrides.gustiness, FT(1))

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
                    microphysics_model isa DryModel &&
                        error("lhf cannot be specified when using a DryModel")
                end
            elseif parameterization.fluxes isa θAndQFluxes
                (; θ_flux, q_flux) = parameterization.fluxes
                if isnothing(q_flux)
                    q_flux = FT(0)
                else
                    microphysics_model isa DryModel && error(
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

"""
    atmos_surface_conditions(
        surface_fluxes_params, surface_conditions, ρ_sfc, surface_local_geometry,
    )

Convert a `SurfaceFluxes.SurfaceFluxConditions` struct into the NamedTuple of
surface values and covariant flux vectors used by ClimaAtmos.

The scalar fluxes returned by `SurfaceFluxes` are given a direction here: the
energy and moisture fluxes are projected onto the surface normal, and the
momentum fluxes `ρτxz`, `ρτyz` are assembled into a tensor. Only the horizontal
part of the momentum flux is kept (`ρ_flux_uₕ`). The buoyancy flux is computed
from `shf`, `lhf`, and `ρ_sfc`.

# Returns

`(; T_sfc, q_vap_sfc, ustar, obukhov_length, buoyancy_flux, ρ_flux_uₕ, ρ_flux_h_tot, ρ_flux_q_tot)`, with temperature [K], specific humidity [kg/kg],
friction velocity [m/s], Obukhov length [m], buoyancy flux [m²/s³], and
the energy [W/m²] and moisture [kg/m²/s] fluxes as `C3` vectors, positive
upward. `ρ_flux_q_tot` is always present, even for a `DryModel`.
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

"""
    tensor_from_components(f₁₃, f₂₃, L, n₃ = surface_normal(L))

Assemble the vertical fluxes `f₁₃`, `f₂₃` of the two horizontal momentum
components into the surface momentum-flux tensor `n₃ ⊗ f` at local geometry
`L`.
"""
function tensor_from_components(f₁₃, f₂₃, L, n₃ = surface_normal(L))
    xz = CT12(CT1(unit_basis_vector_data(CT1, L)), L)
    yz = CT12(CT2(unit_basis_vector_data(CT2, L)), L)
    f = C12(f₁₃ * xz + f₂₃ * yz, L)
    return n₃ ⊗ f
end

"""
    surface_conditions_type(atmos, ::Type{FT})

Return the `NamedTuple` type produced by [`atmos_surface_conditions`](@ref),
without evaluating it.

Used to allocate `p.precomputed.sfc_conditions` before any surface update has
run. `ρ_flux_q_tot` is included even for a `DryModel`, because `SurfaceFluxes`
always returns an evaporation rate.
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
