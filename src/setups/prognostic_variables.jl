"""
    Prognostic variable assembly layer.

Converts a `physical_state` NamedTuple provided by
[`center_initial_condition`](@ref) into the prognostic NamedTuple required by a
given `AtmosModel` configuration (Layer 2, model-aware). This mirrors the
dispatch logic in `src/initial_conditions/atmos_state.jl` but operates on plain
NamedTuples instead of `LocalState` structs.
"""

# ============================================================================
# Center prognostic variables
# ============================================================================

"""
    center_prognostic_variables(physical_state, local_geometry, params, atmos_model)

Convert a physical-state NamedTuple `physical_state` (from
`center_initial_condition`) into the center prognostic NamedTuple required by
`atmos_model`. Dispatches on moisture, microphysics, and turbconv model types.
"""
function center_prognostic_variables(physical_state, local_geometry, params, atmos_model)
    gs = grid_scale_center_variables(physical_state, local_geometry, params, atmos_model)
    sgs = turbconv_center_variables(
        physical_state,
        local_geometry,
        params,
        atmos_model.turbconv_model,
        atmos_model.moisture_model,
        atmos_model.microphysics_model,
    )
    return (; gs..., sgs...)
end

"""
    grid_scale_center_variables(physical_state, local_geometry, params, atmos_model)

Build the grid-scale prognostic variables (ρ, uₕ, ρe_tot, moisture, precip)
from a physical-state NamedTuple.
"""
function grid_scale_center_variables(physical_state, local_geometry, params, atmos_model)
    (; T, u, v, q_tot, q_liq, q_ice) = physical_state
    ρ = get_density(physical_state, params)
    uₕ = C12(Geometry.UVVector(u, v), local_geometry)
    ρe_tot =
        ρ * total_specific_energy(
            params, T, local_geometry;
            u, v, q_tot, q_liq, q_ice,
        )
    return (;
        ρ,
        uₕ,
        ρe_tot,
        moisture_variables(ρ, physical_state, atmos_model.moisture_model)...,
        precip_variables(ρ, physical_state, atmos_model.microphysics_model)...,
    )
end

# ============================================================================
# Moisture dispatch
# ============================================================================

moisture_variables(ρ, physical_state, ::DryModel) = (;)
moisture_variables(ρ, ps, ::EquilMoistModel) = (; ρq_tot = ρ * ps.q_tot)
moisture_variables(ρ, physical_state, ::NonEquilMoistModel) = (;
    ρq_tot = ρ * physical_state.q_tot,
    ρq_liq = ρ * physical_state.q_liq,
    ρq_ice = ρ * physical_state.q_ice,
)

# ============================================================================
# Precipitation dispatch
# ============================================================================

precip_variables(ρ, physical_state, ::NoPrecipitation) = (;)
precip_variables(ρ, physical_state, ::Microphysics0Moment) = (;)
precip_variables(ρ, physical_state, ::Microphysics1Moment) = (;
    ρq_rai = ρ * physical_state.q_rai,
    ρq_sno = ρ * physical_state.q_sno,
)
precip_variables(ρ, physical_state, ::Microphysics2Moment) = (;
    ρn_liq = ρ * physical_state.n_liq,
    ρn_rai = ρ * physical_state.n_rai,
    ρq_rai = ρ * physical_state.q_rai,
    ρq_sno = ρ * physical_state.q_sno,
)
function precip_variables(ρ, physical_state, ::Microphysics2MomentP3)
    warm_state = (;
        ρn_liq = ρ * physical_state.n_liq,
        ρn_rai = ρ * physical_state.n_rai,
        ρq_rai = ρ * physical_state.q_rai,
        ρq_sno = ρ * physical_state.q_sno,
    )
    cold_state = (;
        ρq_ice = ρ * physical_state.q_ice,
        ρn_ice = ρ * physical_state.n_ice,
        ρq_rim = ρ * physical_state.q_rim,
        ρb_rim = ρ * physical_state.b_rim,
    )
    return (; warm_state..., cold_state...)
end

# ============================================================================
# Turbconv center dispatch
# ============================================================================

"""
    uniform_subdomains(nt::NamedTuple, turbconv_model)

Create `n` identical subdomain copies of `nt`, where `n` is the number of
mass-flux subdomains in `turbconv_model`.
"""
uniform_subdomains(nt, turbconv_model) =
    ntuple(_ -> nt, Val(n_mass_flux_subdomains(turbconv_model)))

turbconv_center_variables(physical_state, local_geometry, params, ::Nothing, _, _) = (;)

function turbconv_center_variables(
    physical_state,
    local_geometry,
    params,
    turbconv_model::PrognosticEDMFX,
    moisture_model,
    microphysics_model,
)
    ρ = get_density(physical_state, params)
    (; tke, draft_area, T, q_tot, q_liq, q_ice) = physical_state
    n = n_mass_flux_subdomains(turbconv_model)
    ρtke = ρ * tke
    ρa = ρ * draft_area / n
    mse = moist_static_energy(
        params, T, local_geometry;
        q_tot, q_liq, q_ice,
    )
    sgsʲs = uniform_subdomains((; ρa, mse, q_tot), turbconv_model)
    return (; ρtke, sgsʲs)
end

function turbconv_center_variables(
    physical_state,
    local_geometry,
    params,
    turbconv_model::PrognosticEDMFX,
    moisture_model::NonEquilMoistModel,
    microphysics_model::Union{Microphysics1Moment, Microphysics2Moment},
)
    (; T, q_tot, q_liq, q_ice, q_rai, q_sno, tke, draft_area) = physical_state
    ρ = get_density(physical_state, params)
    n = n_mass_flux_subdomains(turbconv_model)
    ρtke = ρ * tke
    ρa = ρ * draft_area / n
    mse = moist_static_energy(params, T, local_geometry; q_tot, q_liq, q_ice)
    if microphysics_model isa Microphysics1Moment
        sgsʲs = uniform_subdomains(
            (; ρa, mse, q_tot, q_liq, q_ice, q_rai, q_sno),
            turbconv_model,
        )
    else  # Microphysics2Moment
        sgsʲs = uniform_subdomains(
            (; ρa, mse, q_tot, q_liq, q_ice, q_rai, q_sno, n_liq, n_rai),
            turbconv_model,
        )
    end
    return (; ρtke, sgsʲs)
end

function turbconv_center_variables(
    physical_state,
    local_geometry,
    params,
    turbconv_model::Union{EDOnlyEDMFX, DiagnosticEDMFX},
    _,
    _,
)
    ρ = get_density(physical_state, params)
    ρtke = ρ * physical_state.tke
    return (; ρtke)
end

# ============================================================================
# Face prognostic variables
# ============================================================================

"""
    face_prognostic_variables(face_state, local_geometry, atmos_model)

Convert a face-state NamedTuple (from `face_initial_condition`) into the face
prognostic NamedTuple required by `atmos_model`.
"""
function face_prognostic_variables(face_state, local_geometry, atmos_model)
    FT = typeof(face_state.w)
    u₃ = C3(Geometry.WVector(face_state.w), local_geometry)
    w_draft = Geometry.WVector(get(face_state, :w_draft, zero(FT)))
    return (;
        u₃,
        turbconv_face_variables(u₃, w_draft, local_geometry, atmos_model.turbconv_model)...,
    )
end

# ============================================================================
# Turbconv face dispatch
# ============================================================================

turbconv_face_variables(u₃, w_draft, local_geometry, ::Nothing) = (;)
turbconv_face_variables(u₃, w_draft, local_geometry, ::DiagnosticEDMFX) = (;)
turbconv_face_variables(u₃, w_draft, local_geometry, ::EDOnlyEDMFX) = (;)
function turbconv_face_variables(u₃, w_draft, lg, turbconv_model::PrognosticEDMFX)
    return (; sgsʲs = uniform_subdomains((; u₃ = C3(w_draft, lg)), turbconv_model))
end

# ============================================================================
# Surface field
# ============================================================================

"""
    atmos_surface_field(surface_space, surface_model)

Initialize surface fields based on the surface model type. This is called during
initial state construction to populate the `sfc` component of the prognostic state.

## Arguments
- `surface_space`: The surface (half-level) finite-difference space
- `surface_model`: The surface model type (e.g., `PrescribedSST`, `SlabOceanSST`)

## Returns
A NamedTuple containing surface prognostic fields:
- For `PrescribedSST`: empty NamedTuple `(;)` (surface state is externally prescribed)
- For `SlabOceanSST`: `(; sfc)` where `sfc` is a field containing `(; T, water)` at each point
  - `T`: Initial surface temperature (K) — latitude-dependent if lat/long geometry, constant 300K otherwise
  - `water`: Initial water depth (m) — set to zero
"""
atmos_surface_field(surface_space, ::PrescribedSST) = (;)
function atmos_surface_field(surface_space, ::SlabOceanSST)
    if :lat in propertynames(Fields.coordinate_field(surface_space))
        return (;
            sfc = map(
                coord -> (;
                    T = Geometry.float_type(coord)(
                        271 + 29 * exp(-coord.lat^2 / (2 * 26^2)),
                    ),
                    water = Geometry.float_type(coord)(0),
                ),
                Fields.coordinate_field(surface_space),
            )
        )
    else
        return (;
            sfc = map(
                coord -> (;
                    T = Geometry.float_type(coord)(300),
                    water = Geometry.float_type(coord)(0),
                ),
                Fields.coordinate_field(surface_space),
            )
        )
    end
end
