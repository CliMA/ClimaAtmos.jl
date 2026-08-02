# Prognostic variable assembly layer: converts the `physical_state` NamedTuple
# returned by a setup's `center_initial_condition` into the prognostic
# NamedTuple required by a given `AtmosModel` configuration. Each group of
# variables is selected by dispatch on the corresponding component model, so a
# new model type adds a method here rather than a branch.

# ============================================================================
# Center prognostic variables
# ============================================================================

"""
    center_prognostic_variables(physical_state, local_geometry, params, atmos_model)

Convert the physical state at one center point into the center prognostic
variables required by `atmos_model`.

The grid-scale variables come from `grid_scale_center_variables` and the
subgrid-scale ones from `turbconv_center_variables`, which dispatch on the
microphysics, chemistry, and turbulence-convection models.
"""
function center_prognostic_variables(physical_state, local_geometry, params, atmos_model)
    gs = grid_scale_center_variables(physical_state, local_geometry, params, atmos_model)
    sgs = turbconv_center_variables(
        physical_state, local_geometry, params,
        atmos_model.turbconv_model, atmos_model.microphysics_model,
        atmos_model.chemistry_model,
    )
    return (; gs..., sgs...)
end

"""
    grid_scale_center_variables(physical_state, local_geometry, params, atmos_model)

Build the grid-scale center prognostic variables from the physical state.

Always returns the density `ρ` [kg/m³], the horizontal momentum-carrying
velocity `uₕ` [m/s], and the total energy density `ρe_tot` [J/m³], which
includes kinetic, potential, and internal contributions. The moisture,
precipitation, and chemistry variables are appended by `moisture_variables`,
`precip_variables`, and `chemistry_variables`, which may each contribute
nothing.
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
        chemistry_variables(ρ, physical_state, atmos_model.chemistry_model)...,
    )
end

# ============================================================================
# Moisture dispatch
# ============================================================================

"""
    moisture_variables(ρ, physical_state, microphysics_model)

Return the grid-scale moisture prognostic variables [kg/m³] required by
`microphysics_model`.

Empty for a `DryModel`; `ρq_tot` alone for `EquilibriumMicrophysics0M`, whose
condensate is diagnosed; `ρq_tot`, `ρq_lcl`, and `ρq_icl` for every
non-equilibrium model, which carries cloud liquid and cloud ice.
"""
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

"""
    precip_variables(ρ, physical_state, microphysics_model)

Return the grid-scale precipitation prognostic variables required by
`microphysics_model`.

Empty for `DryModel` and `EquilibriumMicrophysics0M`, whose precipitation is
diagnostic. One-moment adds the rain and snow contents `ρq_rai`, `ρq_sno`
[kg/m³]; two-moment adds the droplet and raindrop number concentrations
`ρn_lcl`, `ρn_rai` [1/m³]; the P3 variant further adds the ice variables
`ρq_icl`, `ρn_ice`, `ρq_rim`, and `ρb_rim`.

The P3 method also re-emits `ρq_icl`, which `moisture_variables` has already
supplied with the same value; the duplicate key is harmless because the
NamedTuple keeps the last one.
"""
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
# Chemistry dispatch
# ============================================================================

"""
    chemistry_variables(ρ, physical_state, chemistry_model)

Return the grid-scale chemistry tracers: `ρq_gas_A` [kg/m³] with a chemistry
model, and nothing without one.
"""
chemistry_variables(ρ, physical_state, ::Nothing) = (;)
chemistry_variables(ρ, physical_state, ::AbstractChemistryModel) =
    (; ρq_gas_A = ρ * physical_state.q_gas_A)

"""
    chemistry_sgs_variables(physical_state, chemistry_model)

Return the chemistry tracers to include in each updraft subdomain: the specific
concentration `q_gas_A` [kg/kg] with a chemistry model, and nothing without
one.
"""
chemistry_sgs_variables(physical_state, ::Nothing) = (;)
chemistry_sgs_variables(physical_state, ::AbstractChemistryModel) =
    (; q_gas_A = physical_state.q_gas_A)

# ============================================================================
# Turbconv center dispatch
# ============================================================================

"""
    uniform_subdomains(nt::NamedTuple, turbconv_model)

Replicate the subdomain state `nt` across the `n` mass-flux subdomains of
`turbconv_model`, returning an `n`-tuple.

The updrafts are initialized identically, so the PROPHET (`EDMFX` in code)
subdomains start indistinguishable and differentiate only once the entrainment
and detrainment closures act.
"""
uniform_subdomains(nt, turbconv_model) =
    ntuple(Returns(nt), Val(n_mass_flux_subdomains(turbconv_model)))

"""
    turbconv_center_variables(
        physical_state, local_geometry, params,
        turbconv_model, microphysics_model, chemistry_model,
    )

Return the subgrid-scale center prognostic variables required by
`turbconv_model`.

Empty without a turbulence-convection model. `EDOnlyEDMFX` adds only the TKE
density `ρtke` [J/m³]. `PrognosticEDMFX` additionally adds `sgsʲs`, one
identical state per updraft (see `uniform_subdomains`) holding the area
density `ρa` [kg/m³] — the setup's total `draft_area` split evenly across the
subdomains — the moist static energy `mse` [J/kg], `q_tot` [kg/kg], and, for a
non-equilibrium microphysics model, the subdomain condensate, precipitation,
and number concentrations.
"""
turbconv_center_variables(physical_state, local_geometry, params, ::Nothing, _, _) = (;)

function turbconv_center_variables(
    physical_state,
    local_geometry,
    params,
    turbconv_model::PrognosticEDMFX,
    microphysics_model,
    chemistry_model,
)
    ρ = air_density(physical_state, params)
    (; tke, draft_area, T, q_tot, q_liq, q_ice) = physical_state
    n = n_mass_flux_subdomains(turbconv_model)
    ρtke = ρ * tke
    ρa = ρ * draft_area / n
    thermo_params = CAP.thermodynamics_params(params)
    e_pot = geopotential(CAP.grav(params), local_geometry.coordinates.z)
    mse = TD.moist_static_energy(thermo_params, T, e_pot, q_tot, q_liq, q_ice)
    sgsʲs = uniform_subdomains(
        (; ρa, mse, q_tot, chemistry_sgs_variables(physical_state, chemistry_model)...),
        turbconv_model,
    )
    return (; ρtke, sgsʲs)
end

function turbconv_center_variables(
    physical_state,
    local_geometry,
    params,
    turbconv_model::PrognosticEDMFX,
    microphysics_model::NonEquilibriumMicrophysics,
    chemistry_model,
)
    (; T, q_tot, q_liq, q_ice, q_rai, q_sno, n_liq, n_rai, tke, draft_area) = physical_state
    ρ = air_density(physical_state, params)
    n = n_mass_flux_subdomains(turbconv_model)
    ρtke = ρ * tke
    ρa = ρ * draft_area / n
    thermo_params = CAP.thermodynamics_params(params)
    e_pot = geopotential(CAP.grav(params), local_geometry.coordinates.z)
    mse = TD.moist_static_energy(thermo_params, T, e_pot, q_tot, q_liq, q_ice)
    chem_sgs = chemistry_sgs_variables(physical_state, chemistry_model)
    if microphysics_model isa NonEquilibriumMicrophysics1M
        sgsʲs = uniform_subdomains(
            (; ρa, mse, q_tot, q_lcl = q_liq, q_icl = q_ice, q_rai, q_sno,
                chem_sgs...),
            turbconv_model,
        )
    else  # NonEquilibriumMicrophysics2M
        sgsʲs = uniform_subdomains(
            (; ρa, mse, q_tot,
                q_lcl = q_liq, q_icl = q_ice, q_rai, q_sno,
                n_lcl = n_liq, n_rai,
                chem_sgs...,
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
    turbconv_model::EDOnlyEDMFX,
    _,
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

Convert the face state at one point into the face prognostic variables required
by `atmos_model`.

Always returns the covariant vertical velocity `u₃` [m/s]; a `PrognosticEDMFX`
model adds one `sgsʲs` entry per updraft, each holding the draft vertical
velocity.
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

"""
    turbconv_face_variables(u₃, w_draft, local_geometry, turbconv_model)

Return the subgrid-scale face prognostic variables: one `sgsʲs` entry per
updraft, each holding the covariant draft vertical velocity `u₃` [m/s], for a
`PrognosticEDMFX`; nothing for the other turbulence-convection models.
"""
turbconv_face_variables(u₃, w_draft, local_geometry, ::Nothing) = (;)
turbconv_face_variables(u₃, w_draft, local_geometry, ::EDOnlyEDMFX) = (;)
function turbconv_face_variables(u₃, w_draft, lg, turbconv_model::PrognosticEDMFX)
    return (; sgsʲs = uniform_subdomains((; u₃ = C3(w_draft, lg)), turbconv_model))
end

# ============================================================================
# Surface prognostic variables
# ============================================================================

"""
    surface_prognostic_variables(local_geometry, temperature)

Return the surface prognostic variables at one surface point.

Defined only for a `SurfaceConditions.SlabOceanTemperature`, the only surface
temperature that carries prognostic state: the slab temperature `T` [K],
initialized to the zonally symmetric aquaplanet SST on the sphere and to 300 K
otherwise, and the surface water content `water` [kg/m²], initialized to zero.
"""
function surface_prognostic_variables(
    local_geometry, ::SurfaceConditions.SlabOceanTemperature,
)
    FT = Geometry.float_type(local_geometry.coordinates)
    coord = local_geometry.coordinates
    T = if :lat in propertynames(coord)
        FT(271 + 29 * exp(-coord.lat^2 / (2 * 26^2)))
    else
        FT(300)
    end
    return (; T, water = FT(0))
end

"""
    surface_kwargs(surface_space, temperature)

Return the `sfc` entry of the prognostic `FieldVector`, or nothing.

Only a `SurfaceConditions.SlabOceanTemperature` carries prognostic surface
state; for every other surface temperature, `Y` has no `sfc` field at all.
"""
surface_kwargs(surface_space, ::SurfaceConditions.SurfaceTemperature) = (;)
function surface_kwargs(
    surface_space, t::SurfaceConditions.SlabOceanTemperature,
)
    sfc_ic(lg) = surface_prognostic_variables(lg, t)
    return (; sfc = sfc_ic.(Fields.local_geometry_field(surface_space)))
end

"""
    air_density(physical_state, params)

Return the air density of a physical state [kg/m³].

`physical_state.ρ` is used when the setup supplied it, and otherwise the
density is computed from `p`, `T`, and the specific humidities. Both branches
are always evaluated, because a branch-free `ifelse` is required for GPU
compatibility and ClimaCore broadcast inference; the unused branch may contain
`NaN`s.
"""
function air_density(physical_state, params)
    (; T, p, ρ, q_tot, q_liq, q_ice) = physical_state
    thermo_params = CAP.thermodynamics_params(params)
    # Use ifelse (not if) to keep the return type branch-free, which is
    # required for GPU compatibility and ClimaCore broadcast inference.
    return ifelse(isnan(ρ), TD.air_density(thermo_params, T, p, q_tot, q_liq, q_ice), ρ)
end
