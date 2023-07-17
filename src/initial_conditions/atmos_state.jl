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
    sgs_vars =
        turbconv_center_variables(ls, atmos_model.turbconv_model, gs_vars)
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
    energy_variables(ls, atmos_model.energy_form)...,
    moisture_variables(ls, atmos_model.moisture_model)...,
    precip_variables(ls, atmos_model.precip_model, atmos_model.perf_mode)...,
)

# TODO: Rename ρθ to ρθ_liq_ice.
energy_variables(ls, ::PotentialTemperature) =
    (; ρθ = ls.ρ * TD.liquid_ice_pottemp(ls.thermo_params, ls.thermo_state))
energy_variables(ls, ::TotalEnergy) = (;
    ρe_tot = ls.ρ * (
        TD.internal_energy(ls.thermo_params, ls.thermo_state) +
        norm_sqr(ls.velocity) / 2 +
        CAP.grav(ls.params) * ls.geometry.coordinates.z
    )
)

function moisture_variables(ls, ::DryModel)
    @assert ls.thermo_state isa TD.AbstractPhaseDry
    return (;)
end
function moisture_variables(ls, ::EquilMoistModel)
    @assert !(ls.thermo_state isa TD.AbstractPhaseNonEquil)
    return (;
        ρq_tot = ls.ρ *
                 TD.total_specific_humidity(ls.thermo_params, ls.thermo_state)
    )
end
moisture_variables(ls, ::NonEquilMoistModel) = (;
    ρq_tot = ls.ρ *
             TD.total_specific_humidity(ls.thermo_params, ls.thermo_state),
    ρq_liq = ls.ρ *
             TD.liquid_specific_humidity(ls.thermo_params, ls.thermo_state),
    ρq_ice = ls.ρ * TD.ice_specific_humidity(ls.thermo_params, ls.thermo_state),
)

# TODO: Remove perf_mode. Currently, adding tracers hurts performance.
precip_variables(ls, ::NoPrecipitation, ::PerfStandard) = (;)
precip_variables(ls, ::Microphysics0Moment, ::PerfStandard) = (;)
precip_variables(ls, ::Microphysics1Moment, ::PerfStandard) =
    (; ρq_rai = zero(eltype(ls)), ρq_sno = zero(eltype(ls)))
precip_variables(ls, _, ::PerfExperimental) =
    (; ρq_rai = zero(eltype(ls)), ρq_sno = zero(eltype(ls)))

# We can use paper-based cases for LES type configurations (no TKE)
# or SGS type configurations (initial TKE needed), so we do not need to assert
# that there is no TKE when there is no turbconv_model.
turbconv_center_variables(ls, ::Nothing, gs_vars) = (;)
function turbconv_center_variables(ls, turbconv_model::TC.EDMFModel, gs_vars)
    n = TC.n_updrafts(turbconv_model)
    a_draft = max(ls.turbconv_state.draft_area, n * turbconv_model.minimum_area)
    ρa = ls.ρ * a_draft / n
    ρae_tot =
        ρa * (
            TD.internal_energy(ls.thermo_params, ls.thermo_state) +
            norm_sqr(ls.velocity) / 2 +
            CAP.grav(ls.params) * ls.geometry.coordinates.z
        )
    ρaq_tot = ρa * TD.total_specific_humidity(ls.thermo_params, ls.thermo_state)
    en = (; ρatke = (ls.ρ - n * ρa) * ls.turbconv_state.tke)
    up = ntuple(_ -> (; ρarea = ρa, ρae_tot, ρaq_tot), Val(n))
    return (; turbconv = (; en, up))
end
function turbconv_center_variables(ls, turbconv_model::EDMFX, gs_vars)
    n = n_mass_flux_subdomains(turbconv_model)
    a_draft = ls.turbconv_state.draft_area
    sgs⁰ = (; ρatke = ls.ρ * (1 - a_draft) * ls.turbconv_state.tke)
    sgsʲs = ntuple(_ -> gs_to_sgs(gs_vars, a_draft / n), Val(n))
    return (; sgs⁰, sgsʲs)
end

function turbconv_center_variables(ls, turbconv_model::DiagnosticEDMFX, gs_vars)
    sgs⁰ = (; ρatke = ls.ρ * ls.turbconv_state.tke)
    return (; sgs⁰)
end

turbconv_face_variables(ls, ::Nothing) = (;)
turbconv_face_variables(ls, turbconv_model::TC.EDMFModel) = (;
    turbconv = (;
        up = ntuple(
            _ -> (; w = C3(zero(eltype(ls)))),
            Val(TC.n_updrafts(turbconv_model)),
        )
    )
)
turbconv_face_variables(ls, turbconv_model::EDMFX) = (;
    sgsʲs = ntuple(
        _ -> (; u₃ = C3(ls.turbconv_state.velocity, ls.geometry)),
        Val(n_mass_flux_subdomains(turbconv_model)),
    )
)
turbconv_face_variables(ls, turbconv_model::DiagnosticEDMFX) = (;)
