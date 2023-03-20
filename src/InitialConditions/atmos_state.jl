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
atmos_center_variables(ls, atmos_model::AtmosModel) = (;
    ρ = ls.ρ,
    uₕ = Geometry.project(Geometry.Covariant12Axis(), ls.velocity, ls.geometry),
    energy_variables(ls, atmos_model.energy_form)...,
    moisture_variables(ls, atmos_model.moisture_model)...,
    precip_variables(ls, atmos_model.precip_model, atmos_model.perf_mode)...,
    turbconv_center_variables(ls, atmos_model.turbconv_model)...,
)

"""
    atmos_face_variables(ls, atmos_model)

Like `atmos_center_variables`, but for cell faces.
"""
atmos_face_variables(ls, atmos_model::AtmosModel) = (;
    w = Geometry.project(Geometry.Covariant3Axis(), ls.velocity, ls.geometry),
    turbconv_face_variables(ls, atmos_model.turbconv_model)...,
)

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
# or SGS type configurations (initial TKE needed)
turbconv_center_variables(ls, ::Nothing) = (;)
function turbconv_center_variables(ls, turbconv_model::TC.EDMFModel)
    ρa = ls.ρ * turbconv_model.minimum_area
    ρaθ_liq_ice = ρa * TD.liquid_ice_pottemp(ls.thermo_params, ls.thermo_state)
    ρaq_tot = ρa * TD.total_specific_humidity(ls.thermo_params, ls.thermo_state)
    n_up = TC.n_updrafts(turbconv_model)
    ρa_en_tke = (ls.ρ - n_up * ρa) * ls.turbconv_state.tke
    return (;
        turbconv = (;
            en = (; ρatke = ρa_en_tke),
            up = ntuple(_ -> (; ρarea = ρa, ρaθ_liq_ice, ρaq_tot), Val(n_up)),
        )
    )
end

turbconv_face_variables(ls, ::Nothing) = (;)
turbconv_face_variables(ls, turbconv_model::TC.EDMFModel) = (;
    turbconv = (;
        up = ntuple(
            _ -> (; w = Geometry.Covariant3Vector(zero(eltype(ls)))),
            Val(TC.n_updrafts(turbconv_model)),
        )
    )
)
