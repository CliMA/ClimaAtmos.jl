"""
    Shared infrastructure for file-based initial conditions.

These utilities support setups that initialize prognostic state from NetCDF files
(e.g., MoistFromFile, WeatherModel, AMIPFromERA5). The main entry point is
`overwrite_from_file!`, which regrids file data onto the model grid and populates
all prognostic variables.
"""

# ============================================================================
# Topographic pressure correction
# ============================================================================

"""
    correct_surface_pressure_for_topography!(
        p_sfc, file_path, face_space, Y, ᶜT, ᶜq_tot,
        thermo_params, regridder_kwargs;
        surface_altitude_var = "z_sfc",
    )

Adjust the surface pressure field `p_sfc` to account for mismatches between
ERA5 (file) surface altitude and the model orography:

    Δz = z_model_surface - z_sfc
    p_sfc .= p_sfc .* exp.(-Δz * g ./ (R_m_sfc .* T_sfc))

Returns `true` if the correction is applied; `false` if the surface altitude
field cannot be loaded.
"""
function correct_surface_pressure_for_topography!(
    p_sfc,
    file_path,
    face_space,
    Y,
    ᶜT,
    ᶜq_tot,
    thermo_params,
    regridder_kwargs;
    surface_altitude_var = "z_sfc",
)
    regridder_type = :InterpolationsRegridder
    ᶠz_surface = Fields.level(
        SpaceVaryingInputs.SpaceVaryingInput(
            file_path,
            surface_altitude_var,
            face_space;
            regridder_type,
            regridder_kwargs = regridder_kwargs,
        ),
        Fields.half,
    )

    if ᶠz_surface === nothing
        return false
    end

    grav = thermo_params.grav

    ᶠz_model_surface = Fields.level(Fields.coordinate_field(Y.f).z, Fields.half)
    ᶠΔz = @. ᶠz_model_surface - ᶠz_surface

    ᶠR_m = ᶠinterp.(TD.gas_constant_air.(thermo_params, ᶜq_tot))
    ᶠR_m_sfc = Fields.level(ᶠR_m, Fields.half)

    ᶠT = ᶠinterp.(ᶜT)
    ᶠT_sfc = Fields.level(ᶠT, Fields.half)

    @. p_sfc = p_sfc * exp(-(ᶠΔz) * grav / (ᶠR_m_sfc * ᶠT_sfc))

    @info "Adjusted surface pressure to account for ERA5/model surface-height differences."
    return true
end

# ============================================================================
# Internal helpers (shared by overwrite_from_file! and WeatherModel)
# ============================================================================

"""
    _hydrostatic_pressure(p_sfc, ᶜT, ᶜq_tot, face_space, thermo_params)

Compute face pressure by hydrostatic integration from surface pressure.
Solves ∂(ln p)/∂z = -g/(Rₘ(q)T) using `column_integral_indefinite!`.
"""
function _hydrostatic_pressure(p_sfc, ᶜT, ᶜq_tot, face_space, thermo_params)
    ᶜ∂lnp∂z = @. -thermo_params.grav /
                 (TD.gas_constant_air(thermo_params, ᶜq_tot) * ᶜT)
    ᶠlnp_over_psfc = zeros(face_space)
    Operators.column_integral_indefinite!(ᶠlnp_over_psfc, ᶜ∂lnp∂z)
    return p_sfc .* exp.(ᶠlnp_over_psfc)
end

"""
    _assign_velocity_energy!(Y, ᶜT, ᶜq_tot, ᶠp, thermo_params, file_path, svi_kwargs)

Regrid velocity from file, compute kinetic and total energy, and assign to Y.
"""
function _assign_velocity_energy!(
    Y, ᶜT, ᶜq_tot, ᶠp, thermo_params, file_path, svi_kwargs,
)
    center_space = Fields.axes(Y.c)
    vel =
        Geometry.UVWVector.(
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path, "u", center_space; svi_kwargs...,
            ),
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path, "v", center_space; svi_kwargs...,
            ),
            SpaceVaryingInputs.SpaceVaryingInput(
                file_path, "w", center_space; svi_kwargs...,
            ),
        )
    Y.c.uₕ .= C12.(Geometry.UVVector.(vel))
    Y.f.u₃ .= ᶠinterp.(C3.(Geometry.WVector.(vel)))
    e_kin = similar(ᶜT)
    e_kin .= compute_kinetic(Y.c.uₕ, Y.f.u₃)
    e_pot = geopotential.(thermo_params.grav, Fields.coordinate_field(Y.c).z)
    Y.c.ρe_tot .=
        TD.total_energy.(thermo_params, e_kin, e_pot, ᶜT, ᶜq_tot) .* Y.c.ρ
    return e_pot
end

"""
    _assign_moisture_edmf!(Y, ᶜT, ᶜq_tot, e_pot, thermo_params, file_path, svi_kwargs)

Assign moisture variables, cloud water from file (if available), and
EDMF subdomain initialization.
"""
function _assign_moisture_edmf!(
    Y, ᶜT, ᶜq_tot, e_pot, thermo_params, file_path, svi_kwargs,
)
    center_space = Fields.axes(Y.c)

    if hasproperty(Y.c, :ρq_tot)
        Y.c.ρq_tot .= ᶜq_tot .* Y.c.ρ
    else
        error(
            "`dry` configurations are incompatible with the interpolated initial conditions.",
        )
    end

    if hasproperty(Y.c, :ρq_liq)
        fill!(Y.c.ρq_liq, 0)
    end
    if hasproperty(Y.c, :ρq_ice)
        fill!(Y.c.ρq_ice, 0)
    end
    if hasproperty(Y.c, :ρq_sno) && hasproperty(Y.c, :ρq_rai)
        has_microphysics_vars = NC.NCDataset(file_path) do ds
            haskey(ds, "cswc") && haskey(ds, "crwc")
        end
        if has_microphysics_vars
            Y.c.ρq_sno .=
                SpaceVaryingInputs.SpaceVaryingInput(
                    file_path, "cswc", center_space; svi_kwargs...,
                ) .* Y.c.ρ
            Y.c.ρq_rai .=
                SpaceVaryingInputs.SpaceVaryingInput(
                    file_path, "crwc", center_space; svi_kwargs...,
                ) .* Y.c.ρ
        else
            fill!(Y.c.ρq_sno, 0)
            fill!(Y.c.ρq_rai, 0)
        end
    end

    # Initialize prognostic EDMF subdomains if present
    if hasproperty(Y.c, :sgsʲs)
        ᶜmse = TD.enthalpy.(thermo_params, ᶜT, ᶜq_tot) .+ e_pot
        for name in propertynames(Y.c.sgsʲs)
            s = getproperty(Y.c.sgsʲs, name)
            hasproperty(s, :ρa) && fill!(s.ρa, 0)
            hasproperty(s, :mse) && (s.mse .= ᶜmse)
            hasproperty(s, :q_tot) && (s.q_tot .= ᶜq_tot)
        end
    end

    if hasproperty(Y.c, :ρtke)
        fill!(Y.c.ρtke, 0)
    end

    return nothing
end

# ============================================================================
# Main shared overwrite function
# ============================================================================

"""
    overwrite_from_file!(file_path, extrapolation_bc, Y, thermo_params;
                         regridder_type=nothing, interpolation_method=nothing)

Overwrite the prognostic state `Y` with data regridded from a NetCDF file.
Recomputes vertical pressure levels assuming hydrostatic balance from
surface pressure.

Expected variables in the file:
- `p`: pressure (2D surface, broadcast in z)
- `t`: temperature (3D)
- `q`: specific humidity (3D)
- `u, v, w`: velocity (3D)
- `cswc, crwc`: snow and rain water content (optional, for 1-moment microphysics)
- `z_sfc`: surface altitude (optional, for topographic pressure correction)
"""
function overwrite_from_file!(
    file_path::String,
    extrapolation_bc,
    Y,
    thermo_params;
    regridder_type = nothing,
    interpolation_method = nothing,
)
    regridder_kwargs =
        if isnothing(extrapolation_bc) && isnothing(interpolation_method)
            ()
        elseif isnothing(interpolation_method)
            (; extrapolation_bc)
        elseif isnothing(extrapolation_bc)
            (; interpolation_method)
        else
            (; extrapolation_bc, interpolation_method)
        end
    svi_kwargs =
        isnothing(regridder_type) ? (; regridder_kwargs) :
        (; regridder_type, regridder_kwargs)

    isfile(file_path) || error("$(file_path) is not a file")
    @info "Overwriting initial conditions with data from file $(file_path)"

    center_space = Fields.axes(Y.c)
    face_space = Fields.axes(Y.f)

    # Regrid temperature and humidity from file
    ᶜT = SpaceVaryingInputs.SpaceVaryingInput(
        file_path, "t", center_space; svi_kwargs...,
    )
    ᶜq_tot = SpaceVaryingInputs.SpaceVaryingInput(
        file_path, "q", center_space; svi_kwargs...,
    )

    # Surface pressure with optional topographic correction
    p_sfc = Fields.level(
        SpaceVaryingInputs.SpaceVaryingInput(
            file_path, "p", face_space; svi_kwargs...,
        ),
        Fields.half,
    )

    surface_altitude_var = "z_sfc"
    has_surface_altitude = NC.NCDataset(file_path) do ds
        haskey(ds, surface_altitude_var)
    end
    if has_surface_altitude
        correct_surface_pressure_for_topography!(
            p_sfc, file_path, face_space, Y, ᶜT, ᶜq_tot,
            thermo_params, regridder_kwargs;
            surface_altitude_var,
        )
    end

    # Hydrostatic pressure integration
    ᶠp = _hydrostatic_pressure(p_sfc, ᶜT, ᶜq_tot, face_space, thermo_params)

    # Density
    Y.c.ρ .= TD.air_density.(thermo_params, ᶜT, ᶜinterp.(ᶠp), ᶜq_tot)

    # Velocity and energy
    e_pot = _assign_velocity_energy!(
        Y, ᶜT, ᶜq_tot, ᶠp, thermo_params, file_path, svi_kwargs,
    )

    # Moisture and EDMF
    _assign_moisture_edmf!(
        Y, ᶜT, ᶜq_tot, e_pot, thermo_params, file_path, svi_kwargs,
    )

    return nothing
end
