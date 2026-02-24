"""
    atmos_state(local_state, atmos_model, center_space, face_space)

Allocates and sets the state of an `AtmosModel`, given a pointwise function of
the form `local_state(local_geometry)::LocalState`. Converts the generic state
at every point into a state that is specific to the given `AtmosModel` by
calling the `atmos_center_variables` and `atmos_face_variables` functions.

The result of the `local_state` function is never actually allocated, so the
`LocalState` struct does not necessarily need to be limited in size.
"""
atmos_state(local_state, atmos_model, center_space, face_space) =
    Fields.FieldVector(;
        c = atmos_center_variables.(
            local_state.(Fields.local_geometry_field(center_space)),
            atmos_model,
        ),
        f = atmos_face_variables.(
            local_state.(Fields.local_geometry_field(face_space)),
            atmos_model,
        ),
        atmos_surface_field(
            Fields.level(face_space, Fields.half),
            atmos_model.surface_model,
        )...,
    )

"""
    atmos_center_variables(ls, atmos_model)

Generates all of the variables the given `AtmosModel` requires at a cell center,
based on the `LocalState` at that cell center. Ensures that all of the nonzero
values in the `LocalState` can be handled by the `AtmosModel`, throwing an error
if this is not the case.

In order to extend this function to new sub-models, add more functions of the
form `variables(ls, model, args...)`, where `args` can be used for dispatch or
for avoiding duplicate computations.
"""
function atmos_center_variables(ls, atmos_model)
    gs_vars = grid_scale_center_variables(ls, atmos_model)
    sgs_vars = turbconv_center_variables(
        ls,
        atmos_model.turbconv_model,
        atmos_model.microphysics_model,
        gs_vars,
    )
    return (; gs_vars..., sgs_vars...)
end

"""
    atmos_face_variables(ls, atmos_model)

Like `atmos_center_variables`, but for cell faces.
"""
atmos_face_variables(ls, atmos_model) = (;
    u₃ = C3(ls.velocity, ls.geometry),
    turbconv_face_variables(ls, atmos_model.turbconv_model)...,
)

grid_scale_center_variables(ls, atmos_model) = (;
    ρ = ls.ρ,
    uₕ = C12(ls.velocity, ls.geometry),
    energy_variables(ls)...,
    moisture_variables(ls, atmos_model.microphysics_model)...,
    precip_variables(ls, atmos_model.microphysics_model)...,
)

energy_variables(ls) = (;
    ρe_tot = ls.ρ * (
        TD.internal_energy(ls.thermo_params, ls.T, ls.q_tot, ls.q_liq, ls.q_ice) +
        norm_sqr(ls.velocity) / 2 +
        geopotential(CAP.grav(ls.params), ls.geometry.coordinates.z)
    )
)

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

function moisture_variables(ls, ::DryModel)
    @assert ls.q_tot == 0 && ls.q_liq == 0 && ls.q_ice == 0
    return (;)
end
function moisture_variables(ls, ::EquilibriumMicrophysics0M)
    @assert ls.q_liq == 0 && ls.q_ice == 0 # Equilibrium model: condensate is diagnosed
    return (; ρq_tot = ls.ρ * ls.q_tot)
end
moisture_variables(ls, ::NonEquilibriumMicrophysics) = (;
    ρq_tot = ls.ρ * ls.q_tot,
    ρq_liq = ls.ρ * ls.q_liq,
    ρq_ice = ls.ρ * ls.q_ice,
)
precip_variables(ls, ::DryModel) = (;)
precip_variables(ls, ::EquilibriumMicrophysics0M) = (;)
precip_variables(ls, ::NonEquilibriumMicrophysics1M) = (;
    ρq_rai = ls.ρ * ls.precip_state.q_rai,
    ρq_sno = ls.ρ * ls.precip_state.q_sno,
)
precip_variables(ls, ::NonEquilibriumMicrophysics2M) = (;
    ρn_liq = ls.ρ * ls.precip_state.n_liq,
    ρn_rai = ls.ρ * ls.precip_state.n_rai,
    ρq_rai = ls.ρ * ls.precip_state.q_rai,
    ρq_sno = ls.ρ * ls.precip_state.q_sno,
)
function precip_variables(ls, ::NonEquilibriumMicrophysics2MP3)
    (; ρ) = ls
    (; n_liq, n_rai, q_rai) = ls.precip_state.warm
    (; n_ice, q_ice, q_rim, b_rim) = ls.precip_state.cold
    ls_warm = (; ρ, precip_state = ls.precip_state.warm)
    warm_state = precip_variables(ls_warm, NonEquilibriumMicrophysics2M())
    cold_state = (;
        ρq_ice = ρ * q_ice, ρn_ice = ρ * n_ice,
        ρq_rim = ρ * q_rim, ρb_rim = ρ * b_rim,
    )
    return (; warm_state..., cold_state...)
end

# Geometry requirement traits - mark simple arithmetic operations as minimal
import ClimaCore.Geometry: geometry_requirement, NeedsMinimal

# These functions only do arithmetic (ρ * q) - no coordinate transformations
geometry_requirement(::typeof(precip_variables)) = NeedsMinimal()
geometry_requirement(::typeof(moisture_variables)) = NeedsMinimal()
geometry_requirement(::typeof(energy_variables)) = NeedsMinimal()

# We can use paper-based cases for LES type configurations (no TKE)
# or SGS type configurations (initial TKE needed), so we do not need to assert
# that there is no TKE when there is no turbconv_model.
turbconv_center_variables(ls, ::Nothing, _, gs_vars) = (;)

function turbconv_center_variables(
    ls,
    turbconv_model::PrognosticEDMFX,
    microphysics_model,
    gs_vars,
)
    n = n_mass_flux_subdomains(turbconv_model)
    a_draft = ls.turbconv_state.draft_area
    ρtke = ls.ρ * ls.turbconv_state.tke
    ρa = ls.ρ * a_draft / n
    mse =
        TD.enthalpy(ls.thermo_params, ls.T, ls.q_tot, ls.q_liq, ls.q_ice) +
        geopotential(CAP.grav(ls.params), ls.geometry.coordinates.z)
    sgsʲs = ntuple(_ -> (; ρa = ρa, mse = mse, q_tot = ls.q_tot), Val(n))
    return (; ρtke = ρtke, sgsʲs)
end
function turbconv_center_variables(
    ls,
    turbconv_model::PrognosticEDMFX,
    microphysics_model::Union{NonEquilibriumMicrophysics1M, NonEquilibriumMicrophysics2M},
    gs_vars,
)
    # TODO - Instead of dispatching, should we unify this with the above function?
    n = n_mass_flux_subdomains(turbconv_model)
    a_draft = ls.turbconv_state.draft_area
    ρtke = ls.ρ * ls.turbconv_state.tke
    ρa = ls.ρ * a_draft / n
    mse =
        TD.enthalpy(ls.thermo_params, ls.T, ls.q_tot, ls.q_liq, ls.q_ice) +
        geopotential(CAP.grav(ls.params), ls.geometry.coordinates.z)
    q_rai = ls.precip_state.q_rai
    q_sno = ls.precip_state.q_sno
    n_liq = ls.precip_state.n_liq
    n_rai = ls.precip_state.n_rai
    if microphysics_model isa NonEquilibriumMicrophysics1M
        sgsʲs = ntuple(
            _ -> (;
                ρa = ρa,
                mse = mse,
                q_tot = ls.q_tot,
                q_liq = ls.q_liq,
                q_ice = ls.q_ice,
                q_rai = q_rai,
                q_sno = q_sno,
            ),
            Val(n),
        )
    elseif microphysics_model isa NonEquilibriumMicrophysics2M
        sgsʲs = ntuple(
            _ -> (;
                ρa = ρa,
                mse = mse,
                q_tot = ls.q_tot,
                q_liq = ls.q_liq,
                q_ice = ls.q_ice,
                q_rai = q_rai,
                q_sno = q_sno,
                n_liq = n_liq,
                n_rai = n_rai,
            ),
            Val(n),
        )
    end
    return (; ρtke = ρtke, sgsʲs)
end

function turbconv_center_variables(
    ls,
    turbconv_model::Union{EDOnlyEDMFX, DiagnosticEDMFX},
    _,
    gs_vars,
)
    ρtke = ls.ρ * ls.turbconv_state.tke
    return (; ρtke = ρtke)
end

turbconv_face_variables(ls, ::Nothing) = (;)
turbconv_face_variables(ls, turbconv_model::PrognosticEDMFX) = (;
    sgsʲs = ntuple(
        _ -> (; u₃ = C3(ls.turbconv_state.velocity, ls.geometry)),
        Val(n_mass_flux_subdomains(turbconv_model)),
    )
)
turbconv_face_variables(ls, turbconv_model::DiagnosticEDMFX) = (;)

turbconv_face_variables(ls, turbconv_model::EDOnlyEDMFX) = (;)
