"""
    Prognostic variable assembly layer.

Converts a `physical_state` NamedTuple provided by
[`center_initial_condition`](@ref) into the prognostic NamedTuple required by a
given `AtmosModel` configuration.
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
        physical_state, local_geometry, params,
        atmos_model.turbconv_model, atmos_model.microphysics_model,
    )
    return (; gs..., sgs...)
end

"""
    grid_scale_center_variables(physical_state, local_geometry, params, atmos_model)

Build the grid-scale prognostic variables (ρ, uₕ, ρe_tot, moisture, precip, aerosols)
from a physical-state NamedTuple.
"""
function grid_scale_center_variables(physical_state, local_geometry, params, atmos_model)
    (; T, u, v, q_tot, q_liq, q_ice) = physical_state
    ρ = air_density(physical_state, params)
    uₕ = C12(Geometry.UVVector(u, v), local_geometry)
    e_kin = (u^2 + v^2) / 2
    thermo_params = CAP.thermodynamics_params(params)
    grav = CAP.grav(params)
    z = local_geometry.coordinates.z
    e_pot = geopotential(grav, z)
    ρe_tot = ρ * TD.total_energy(thermo_params, e_kin, e_pot, T, q_tot, q_liq, q_ice)
    return (;
        ρ,
        uₕ,
        ρe_tot,
        moisture_variables(ρ, physical_state, atmos_model.microphysics_model)...,
        precip_variables(ρ, physical_state, atmos_model.microphysics_model)...,
        prognostic_aerosol_variables(ρ, atmos_model.prognostic_aerosols)...,
    )
end

function prognostic_aerosol_variables(ρ, names::NTuple{N, Symbol}) where {N}
    ρ_names = ntuple(i -> Symbol(:ρ, names[i]), Val(N))
    vals = ntuple(_ -> zero(ρ), Val(N))
    return NamedTuple{ρ_names}(vals)
end
prognostic_aerosol_variables(ρ, ::Tuple{}) = (;)

# ============================================================================
# Moisture dispatch
# ============================================================================

moisture_variables(ρ, physical_state, ::DryModel) = (;)
moisture_variables(ρ, ps, ::EquilibriumMicrophysics0M) = (; ρq_tot = ρ * ps.q_tot)
moisture_variables(ρ, physical_state, ::NonEquilibriumMicrophysics) = (;
    ρq_tot = ρ * physical_state.q_tot,
    ρq_lcl = ρ * physical_state.q_liq,
    ρq_icl = ρ * physical_state.q_ice,
)

# ============================================================================
# Precipitation dispatch
# ============================================================================

precip_variables(ρ, physical_state, ::DryModel) = (;)
precip_variables(ρ, physical_state, ::EquilibriumMicrophysics0M) = (;)
precip_variables(ρ, physical_state, ::NonEquilibriumMicrophysics1M) = (;
    ρq_rai = ρ * physical_state.q_rai,
    ρq_sno = ρ * physical_state.q_sno,
)
precip_variables(ρ, physical_state, ::NonEquilibriumMicrophysics2M) = (;
    ρn_lcl = ρ * physical_state.n_liq,
    ρn_rai = ρ * physical_state.n_rai,
    ρq_rai = ρ * physical_state.q_rai,
    ρq_sno = ρ * physical_state.q_sno,
)
function precip_variables(ρ, physical_state, ::NonEquilibriumMicrophysics2MP3)
    warm_state = (;
        ρn_lcl = ρ * physical_state.n_liq,
        ρn_rai = ρ * physical_state.n_rai,
        ρq_rai = ρ * physical_state.q_rai,
        ρq_sno = ρ * physical_state.q_sno,
    )
    cold_state = (;
        ρq_icl = ρ * physical_state.q_ice,
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
    ntuple(Returns(nt), Val(n_mass_flux_subdomains(turbconv_model)))

turbconv_center_variables(physical_state, local_geometry, params, ::Nothing, _) = (;)

function turbconv_center_variables(
    physical_state,
    local_geometry,
    params,
    turbconv_model::PrognosticEDMFX,
    microphysics_model,
)
    ρ = air_density(physical_state, params)
    (; tke, draft_area, T, q_tot, q_liq, q_ice) = physical_state
    n = n_mass_flux_subdomains(turbconv_model)
    ρtke = ρ * tke
    ρa = ρ * draft_area / n
    thermo_params = CAP.thermodynamics_params(params)
    e_pot = geopotential(CAP.grav(params), local_geometry.coordinates.z)
    mse = TD.moist_static_energy(thermo_params, T, e_pot, q_tot, q_liq, q_ice)
    sgsʲs = uniform_subdomains((; ρa, mse, q_tot), turbconv_model)
    return (; ρtke, sgsʲs)
end

function turbconv_center_variables(
    physical_state,
    local_geometry,
    params,
    turbconv_model::PrognosticEDMFX,
    microphysics_model::NonEquilibriumMicrophysics,
)
    (; T, q_tot, q_liq, q_ice, q_rai, q_sno, n_liq, n_rai, tke, draft_area) = physical_state
    ρ = air_density(physical_state, params)
    n = n_mass_flux_subdomains(turbconv_model)
    ρtke = ρ * tke
    ρa = ρ * draft_area / n
    thermo_params = CAP.thermodynamics_params(params)
    e_pot = geopotential(CAP.grav(params), local_geometry.coordinates.z)
    mse = TD.moist_static_energy(thermo_params, T, e_pot, q_tot, q_liq, q_ice)
    if microphysics_model isa NonEquilibriumMicrophysics1M
        sgsʲs = uniform_subdomains(
            (; ρa, mse, q_tot, q_lcl = q_liq, q_icl = q_ice, q_rai, q_sno),
            turbconv_model,
        )
    else  # NonEquilibriumMicrophysics2M
        sgsʲs = uniform_subdomains(
            (; ρa, mse, q_tot,
                q_lcl = q_liq, q_icl = q_ice, q_rai, q_sno,
                n_lcl = n_liq, n_rai,
            ),
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
)
    ρ = air_density(physical_state, params)
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
    u₃ = C3(Geometry.WVector(face_state.w), local_geometry)
    w_draft = Geometry.WVector(face_state.w_draft)
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
# Surface prognostic variables
# ============================================================================

"""
    surface_prognostic_variables(local_geometry, surface_model)

Pointwise function returning a NamedTuple of surface prognostic variables.
Follows the same broadcast-over-local-geometry pattern as
`center_prognostic_variables` and `face_prognostic_variables`.

- `PrescribedSST`: returns empty NamedTuple (no surface prognostic state)
- `SlabOceanSST`: returns `(; T, water)` — latitude-dependent SST if available,
  otherwise constant 300K; water depth initialized to zero.
"""
surface_prognostic_variables(local_geometry, ::PrescribedSST) = (;)
function surface_prognostic_variables(local_geometry, ::SlabOceanSST)
    FT = Geometry.float_type(local_geometry.coordinates)
    coord = local_geometry.coordinates
    T = if :lat in propertynames(coord)
        FT(271 + 29 * exp(-coord.lat^2 / (2 * 26^2)))
    else
        FT(300)
    end
    return (; T, water = FT(0))
end

# PrescribedSST has no prognostic surface state, so omit sfc from FieldVector
surface_kwargs(surface_space, ::PrescribedSST) = (;)
function surface_kwargs(surface_space, sm)
    sfc_ic(lg) = surface_prognostic_variables(lg, sm)
    return (; sfc = sfc_ic.(Fields.local_geometry_field(surface_space)))
end

"""
    air_density(physical_state, params)

Extract or compute air density from a `physical_state` NamedTuple.

If `physical_state.ρ` is provided, returns it directly. Otherwise computes
density from `physical_state.p` via `Thermodynamics.air_density`.
"""
function air_density(physical_state, params)
    (; T, p, ρ, q_tot, q_liq, q_ice) = physical_state
    thermo_params = CAP.thermodynamics_params(params)
    # Use ifelse (not if) to keep the return type branch-free, which is
    # required for GPU compatibility and ClimaCore broadcast inference.
    return ifelse(isnan(ρ), TD.air_density(thermo_params, T, p, q_tot, q_liq, q_ice), ρ)
end
