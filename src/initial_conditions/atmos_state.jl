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
    energy_variables(ls)...,
    moisture_variables(ls, atmos_model.moisture_model)...,
    precip_variables(ls, atmos_model.precip_model)...,
)

energy_variables(ls) = (;
    ρe_tot = ls.ρ * (
        TD.internal_energy(ls.thermo_params, ls.thermo_state) +
        norm_sqr(ls.velocity) / 2 +
        CAP.grav(ls.params) * ls.geometry.coordinates.z
    )
)

atmos_surface_field(surface_space, ::PrescribedSurfaceTemperature) = (;)
function atmos_surface_field(surface_space, ::PrognosticSurfaceTemperature)
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

precip_variables(ls, ::NoPrecipitation) = (;)
precip_variables(ls, ::Microphysics0Moment) = (;)
precip_variables(ls, ::Microphysics1Moment) = (;
    ρq_rai = ls.ρ * ls.precip_state.q_rai,
    ρq_sno = ls.ρ * ls.precip_state.q_sno,
)

# We can use paper-based cases for LES type configurations (no TKE)
# or SGS type configurations (initial TKE needed), so we do not need to assert
# that there is no TKE when there is no turbconv_model.
turbconv_center_variables(ls, ::Nothing, gs_vars) = (;)

function turbconv_center_variables(ls, turbconv_model::PrognosticEDMFX, gs_vars)
    n = n_mass_flux_subdomains(turbconv_model)
    a_draft = ls.turbconv_state.draft_area
    sgs⁰ = (; ρatke = ls.ρ * (1 - a_draft) * ls.turbconv_state.tke)
    ρa = ls.ρ * a_draft / n
    mse =
        TD.specific_enthalpy(ls.thermo_params, ls.thermo_state) +
        CAP.grav(ls.params) * ls.geometry.coordinates.z
    q_tot = TD.total_specific_humidity(ls.thermo_params, ls.thermo_state)
    sgsʲs = ntuple(_ -> (; ρa = ρa, mse = mse, q_tot = q_tot), Val(n))
    return (; sgs⁰, sgsʲs)
end

function turbconv_center_variables(
    ls,
    turbconv_model::Union{EDOnlyEDMFX, DiagnosticEDMFX},
    gs_vars,
)
    sgs⁰ = (; ρatke = ls.ρ * ls.turbconv_state.tke)
    return (; sgs⁰)
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
