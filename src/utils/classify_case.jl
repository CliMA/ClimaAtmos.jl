#TODO - do we want to change anything here now?
is_solid_body(atmos, perturb_initstate) = all((
    atmos.model_config isa SphericalModel,
    atmos.forcing_type isa Nothing,
    atmos.radiation_mode isa nothing,
    !perturb_initstate,
))

is_column_without_edmf(atmos) = all((
    atmos.model_config isa SingleColumnModel,
    atmos.turbconv_model isa Nothing,
    atmos.forcing_type isa Nothing,
    atmos.turbconv_model isa Nothing,
))

is_column_edmf(atmos) = all((
    atmos.model_config isa SingleColumnModel,
    atmos.energy_form isa TotalEnergy,
    atmos.forcing_type isa Nothing,
    atmos.turbconv_model isa TC.EDMFModel,
    atmos.radiation_mode isa RadiationDYCOMS_RF01 ||
    atmos.radiation_mode isa RadiationTRMM_LBA ||
    atmos.radiation_mode isa nothing,
))
