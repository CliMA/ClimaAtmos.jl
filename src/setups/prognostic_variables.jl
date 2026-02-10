"""
    Prognostic variable assembly layer.

Converts a `physical_state` NamedTuple (Layer 1, setup-specific) into the
prognostic NamedTuple required by a given `AtmosModel` configuration
(Layer 2, model-aware). This mirrors the dispatch logic in
`src/initial_conditions/atmos_state.jl` but operates on plain NamedTuples
instead of `LocalState` structs.
"""

# ============================================================================
# Center prognostic variables
# ============================================================================

"""
    center_prognostic_variables(ps, local_geometry, params, atmos_model)

Convert a physical-state NamedTuple `ps` (from `center_initial_condition`) into
the center prognostic NamedTuple required by `atmos_model`. Dispatches on
moisture, microphysics, and turbconv model types.
"""
function center_prognostic_variables(ps, local_geometry, params, atmos_model)
    gs = grid_scale_center_variables(ps, local_geometry, params, atmos_model)
    sgs = turbconv_center_variables(
        ps,
        local_geometry,
        params,
        atmos_model.turbconv_model,
        atmos_model.moisture_model,
        atmos_model.microphysics_model,
    )
    return (; gs..., sgs...)
end

"""
    grid_scale_center_variables(ps, local_geometry, params, atmos_model)

Build the grid-scale prognostic variables (ρ, uₕ, ρe_tot, moisture, precip)
from a physical-state NamedTuple.
"""
function grid_scale_center_variables(ps, local_geometry, params, atmos_model)
    FT = typeof(ps.T)
    ρ = air_density(
        params,
        ps.T,
        ps.p;
        q_tot = ps.q_tot,
        q_liq = ps.q_liq,
        q_ice = ps.q_ice,
    )
    velocity = isnothing(ps.velocity) ? Geometry.UVVector(zero(FT), zero(FT)) : ps.velocity
    uₕ = C12(velocity, local_geometry)
    ρe_tot =
        ρ * total_specific_energy(
            params, ps.T, local_geometry;
            velocity, q_tot = ps.q_tot, q_liq = ps.q_liq, q_ice = ps.q_ice,
        )
    return (;
        ρ,
        uₕ,
        ρe_tot,
        moisture_variables(ρ, ps, atmos_model.moisture_model)...,
        precip_variables(ρ, ps, atmos_model.microphysics_model)...,
    )
end

# ============================================================================
# Moisture dispatch
# ============================================================================

moisture_variables(ρ, ps, ::DryModel) = (;)
moisture_variables(ρ, ps, ::EquilMoistModel) = (; ρq_tot = ρ * ps.q_tot)
moisture_variables(ρ, ps, ::NonEquilMoistModel) = (;
    ρq_tot = ρ * ps.q_tot,
    ρq_liq = ρ * ps.q_liq,
    ρq_ice = ρ * ps.q_ice,
)

# ============================================================================
# Precipitation dispatch
# ============================================================================

precip_variables(ρ, ps, ::NoPrecipitation) = (;)
precip_variables(ρ, ps, ::Microphysics0Moment) = (;)
precip_variables(ρ, ps, ::Microphysics1Moment) = (;
    ρq_rai = ρ * ps.q_rai,
    ρq_sno = ρ * ps.q_sno,
)
precip_variables(ρ, ps, ::Microphysics2Moment) = (;
    ρn_liq = ρ * ps.n_liq,
    ρn_rai = ρ * ps.n_rai,
    ρq_rai = ρ * ps.q_rai,
    ρq_sno = ρ * ps.q_sno,
)
function precip_variables(ρ, ps, ::Microphysics2MomentP3)
    warm_state = (;
        ρn_liq = ρ * ps.n_liq,
        ρn_rai = ρ * ps.n_rai,
        ρq_rai = ρ * ps.q_rai,
        ρq_sno = ρ * ps.q_sno,
    )
    cold_state = (;
        ρq_ice = ρ * ps.q_ice,
        ρn_ice = ρ * ps.n_ice,
        ρq_rim = ρ * ps.q_rim,
        ρb_rim = ρ * ps.b_rim,
    )
    return (; warm_state..., cold_state...)
end

# ============================================================================
# Turbconv center dispatch
# ============================================================================

turbconv_center_variables(ps, lg, params, ::Nothing, _, _) = (;)

function turbconv_center_variables(
    ps,
    local_geometry,
    params,
    turbconv_model::PrognosticEDMFX,
    moisture_model,
    microphysics_model,
)
    n = n_mass_flux_subdomains(turbconv_model)
    ρ = air_density(
        params,
        ps.T,
        ps.p;
        q_tot = ps.q_tot,
        q_liq = ps.q_liq,
        q_ice = ps.q_ice,
    )
    ρtke = ρ * ps.tke
    ρa = ρ * ps.draft_area / n
    mse = moist_static_energy(
        params, ps.T, local_geometry;
        q_tot = ps.q_tot, q_liq = ps.q_liq, q_ice = ps.q_ice,
    )
    sgsʲs = ntuple(_ -> (; ρa, mse, q_tot = ps.q_tot), Val(n))
    return (; ρtke, sgsʲs)
end

function turbconv_center_variables(
    ps,
    local_geometry,
    params,
    turbconv_model::PrognosticEDMFX,
    moisture_model::NonEquilMoistModel,
    microphysics_model::Union{Microphysics1Moment, Microphysics2Moment},
)
    n = n_mass_flux_subdomains(turbconv_model)
    ρ = air_density(
        params,
        ps.T,
        ps.p;
        q_tot = ps.q_tot,
        q_liq = ps.q_liq,
        q_ice = ps.q_ice,
    )
    ρtke = ρ * ps.tke
    ρa = ρ * ps.draft_area / n
    mse = moist_static_energy(
        params, ps.T, local_geometry;
        q_tot = ps.q_tot, q_liq = ps.q_liq, q_ice = ps.q_ice,
    )
    if microphysics_model isa Microphysics1Moment
        sgsʲs = ntuple(
            _ -> (;
                ρa,
                mse,
                q_tot = ps.q_tot,
                q_liq = ps.q_liq,
                q_ice = ps.q_ice,
                q_rai = ps.q_rai,
                q_sno = ps.q_sno,
            ),
            Val(n),
        )
    else  # Microphysics2Moment
        sgsʲs = ntuple(
            _ -> (;
                ρa,
                mse,
                q_tot = ps.q_tot,
                q_liq = ps.q_liq,
                q_ice = ps.q_ice,
                q_rai = ps.q_rai,
                q_sno = ps.q_sno,
                n_liq = ps.n_liq,
                n_rai = ps.n_rai,
            ),
            Val(n),
        )
    end
    return (; ρtke, sgsʲs)
end

function turbconv_center_variables(
    ps,
    local_geometry,
    params,
    turbconv_model::Union{EDOnlyEDMFX, DiagnosticEDMFX},
    _,
    _,
)
    ρ = air_density(
        params,
        ps.T,
        ps.p;
        q_tot = ps.q_tot,
        q_liq = ps.q_liq,
        q_ice = ps.q_ice,
    )
    ρtke = ρ * ps.tke
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

turbconv_face_variables(u₃, w_draft, lg, ::Nothing) = (;)
turbconv_face_variables(u₃, w_draft, lg, ::DiagnosticEDMFX) = (;)
turbconv_face_variables(u₃, w_draft, lg, ::EDOnlyEDMFX) = (;)
function turbconv_face_variables(u₃, w_draft, lg, turbconv_model::PrognosticEDMFX)
    n = n_mass_flux_subdomains(turbconv_model)
    return (; sgsʲs = ntuple(_ -> (; u₃ = C3(w_draft, lg)), Val(n)))
end

# ============================================================================
# Surface field
# ============================================================================

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
