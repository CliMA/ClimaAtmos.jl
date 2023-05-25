geopotential(thermo_params::TD.Parameters.ThermodynamicsParameters, z::Real) =
    z * TD.Parameters.grav(thermo_params)

function enthalpy(h_tot::FT, e_kin::FT, e_pot::FT) where {FT}
    return h_tot - e_kin - e_pot
end

function enthalpy(mse::FT, e_pot::FT) where {FT}
    return mse - e_pot
end
