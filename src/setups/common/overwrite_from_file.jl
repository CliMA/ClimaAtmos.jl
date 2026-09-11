# Shared infrastructure for file-based initial conditions, used by the setups
# that initialize the prognostic state from NetCDF files (MoistFromFile,
# WeatherModel, AMIPFromERA5). The entry point is `overwrite_from_file!`, which
# regrids file data onto the model grid and populates every prognostic
# variable.

# ============================================================================
# Topographic pressure correction
# ============================================================================

"""
    correct_surface_pressure_for_topography!(
        p_sfc, file_path, face_space, Y, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice,
        thermo_params, regridder_kwargs;
        surface_altitude_var = "z_sfc",
    )

Adjust the surface pressure field `p_sfc` in place for the mismatch between the
file's surface altitude and the model orography, and return `true`.

The correction is hydrostatic over the altitude difference
`Δz = z_model_surface - z_sfc`:

```math
p_{sfc} ← p_{sfc} \\exp(-Δz \\, g / (R_m T_{sfc}))
```

`R_m` uses the full moisture partition `(ᶜq_tot, ᶜq_liq, ᶜq_ice)`; see
`thermodynamic_partition`.

The caller is responsible for checking that the file carries
`surface_altitude_var`; reading a variable the file lacks throws.
"""
function correct_surface_pressure_for_topography!(
    p_sfc,
    file_path,
    face_space,
    Y,
    ᶜT,
    ᶜq_tot,
    ᶜq_liq,
    ᶜq_ice,
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

    ᶠR_m =
        ᶠinterp.(TD.gas_constant_air.(thermo_params, ᶜq_tot, ᶜq_liq, ᶜq_ice))
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
    hydrostatic_pressure(p_sfc, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice, face_space, thermo_params)

Compute face pressure by hydrostatic integration from surface pressure.
Solves ∂(ln p)/∂z = -g/(Rₘ(q)T) using `column_integral_indefinite!`, where
`Rₘ` uses the full moisture partition `(ᶜq_tot, ᶜq_liq, ᶜq_ice)`.
"""
function hydrostatic_pressure(
    p_sfc, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice, face_space, thermo_params,
)
    ᶜ∂lnp∂z = @. -thermo_params.grav / (
        TD.gas_constant_air(thermo_params, ᶜq_tot, ᶜq_liq, ᶜq_ice) * ᶜT
    )
    ᶠlnp_over_psfc = zeros(face_space)
    Operators.column_integral_indefinite!(ᶠlnp_over_psfc, ᶜ∂lnp∂z)
    return p_sfc .* exp.(ᶠlnp_over_psfc)
end

"""
    assign_velocity_energy!(
        Y, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice, ᶠp, thermo_params, file_path, svi_kwargs,
    )

Regrid velocity from file, compute kinetic and total energy, and assign to Y.
`ρe_tot` uses the full moisture partition `(ᶜq_tot, ᶜq_liq, ᶜq_ice)`.
"""
function assign_velocity_energy!(
    Y, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice, ᶠp, thermo_params, file_path, svi_kwargs,
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
    e_kin = compute_kinetic(Y.c.uₕ, Y.f.u₃)
    e_pot = geopotential.(thermo_params.grav, Fields.coordinate_field(Y.c).z)
    Y.c.ρe_tot .=
        TD.total_energy.(
            thermo_params, e_kin, e_pot, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice,
        ) .* Y.c.ρ
    return e_pot
end

"""
    read_microphysics_from_file(file_path, center_space, svi_kwargs)

Regrid the optional condensate fields from `file_path`, returning the named
tuple `(; ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno)` of specific humidities. Cloud
liquid/ice come from `clwc`/`ciwc` and rain/snow from `crwc`/`cswc`. Any species
whose file variables are absent is returned as `nothing`, which downstream code
treats as "initialize to zero".
"""
function read_microphysics_from_file(file_path, center_space, svi_kwargs)
    # Cloud liquid/ice water content (clwc/ciwc) for 1M/2M cloud condensate.
    has_cloud_vars = NC.NCDataset(file_path) do ds
        haskey(ds, "clwc") && haskey(ds, "ciwc")
    end
    if has_cloud_vars
        @info "Initializing cloud condensate from file (clwc, ciwc)."
        ᶜq_lcl = SpaceVaryingInputs.SpaceVaryingInput(
            file_path, "clwc", center_space; svi_kwargs...,
        )
        ᶜq_icl = SpaceVaryingInputs.SpaceVaryingInput(
            file_path, "ciwc", center_space; svi_kwargs...,
        )
    else
        ᶜq_lcl = nothing
        ᶜq_icl = nothing
    end

    # Rain/snow water content (crwc/cswc) for 1M/2M precipitation.
    has_precip_vars = NC.NCDataset(file_path) do ds
        haskey(ds, "cswc") && haskey(ds, "crwc")
    end
    if has_precip_vars
        @info "Initializing precipitation from file (crwc, cswc)."
        ᶜq_rai = SpaceVaryingInputs.SpaceVaryingInput(
            file_path, "crwc", center_space; svi_kwargs...,
        )
        ᶜq_sno = SpaceVaryingInputs.SpaceVaryingInput(
            file_path, "cswc", center_space; svi_kwargs...,
        )
    else
        ᶜq_rai = nothing
        ᶜq_sno = nothing
    end

    return (; ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno)
end

"""
    thermodynamic_partition(Y, ᶜq_vap, ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno)

Assemble the moisture partition `(; ᶜq_tot, ᶜq_liq, ᶜq_ice)` used to build the
thermodynamic part of the initial state (density, pressure, and energy). Here
`ᶜq_liq`/`ᶜq_ice` are the thermodynamic liquid/ice partitions (cloud grouped
with precipitation), while `ᶜq_lcl`/`ᶜq_icl` are the file cloud-liquid/ice
specific humidities.

`ᶜq_vap` is the file water-vapor specific humidity (ERA5 `q`). ClimaAtmos carries
*total* water `ρq_tot` prognostically and diagnoses vapor as a residual
(`q_vap = q_tot - q_liq - q_ice`, with rain/snow grouped into the liquid/ice
partitions). To keep the diagnosed vapor equal to the file value, each condensate
species is added to `ᶜq_tot` only when it is both present in the file and carried
as a prognostic variable in `Y` — exactly the species that
`assign_moisture_edmf!` stores into `ρq_tot`. Following the runtime
convention in the precomputed thermodynamic state, rain is grouped with the
liquid partition and snow with the ice partition.

The equilibrium (0M) scheme has no prognostic condensate fields, but any cloud
condensate present in the file is still folded into `ᶜq_tot` (and the
`ᶜq_liq`/`ᶜq_ice` partition) so that total water is conserved; saturation
adjustment repartitions it at runtime. For a 0M run this includes cloud
liquid/ice but not rain/snow, since 0M carries no precipitation. When the file
has no condensate this reduces to `ᶜq_tot = ᶜq_vap` and `ᶜq_liq = ᶜq_ice = 0`,
i.e. the previous vapor-only behavior.
"""
function thermodynamic_partition(Y, ᶜq_vap, ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno)
    ᶜq_tot = copy(ᶜq_vap)
    ᶜq_liq = zeros(axes(ᶜq_vap))
    ᶜq_ice = zeros(axes(ᶜq_vap))
    # In the equilibrium (0M) scheme there are no prognostic condensate tracers,
    # so cloud condensate from the file is added to `ᶜq_tot` for total-water
    # conservation (saturation adjustment repartitions it). In non-equilibrium
    # schemes it is added only when the matching prognostic tracer exists, which
    # keeps the diagnosed vapor equal to the file value.
    equilibrium = !hasproperty(Y.c, :ρq_lcl) && !hasproperty(Y.c, :ρq_icl)
    if !isnothing(ᶜq_lcl) && (hasproperty(Y.c, :ρq_lcl) || equilibrium)
        @. ᶜq_tot += ᶜq_lcl
        @. ᶜq_liq += ᶜq_lcl
    end
    if !isnothing(ᶜq_icl) && (hasproperty(Y.c, :ρq_icl) || equilibrium)
        @. ᶜq_tot += ᶜq_icl
        @. ᶜq_ice += ᶜq_icl
    end
    if hasproperty(Y.c, :ρq_rai) &&
       hasproperty(Y.c, :ρq_sno) &&
       !isnothing(ᶜq_rai)
        @. ᶜq_tot += ᶜq_rai + ᶜq_sno
        @. ᶜq_liq += ᶜq_rai
        @. ᶜq_ice += ᶜq_sno
    end
    return (; ᶜq_tot, ᶜq_liq, ᶜq_ice)
end

"""
    assign_moisture_edmf!(
        Y, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice, e_pot, thermo_params,
        ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno,
    )

Assign moisture variables and EDMF subdomain initialization.

`ᶜq_tot`, `ᶜq_liq`, and `ᶜq_ice` are the thermodynamic moisture partition from
`thermodynamic_partition`; `ρq_tot` is stored directly as `ᶜq_tot * ρ` so that it
is consistent with the density and energy already computed from the same
partition.

`ᶜq_lcl`, `ᶜq_icl`, `ᶜq_rai`, and `ᶜq_sno` are the cloud-liquid, cloud-ice,
rain, and snow specific humidities regridded from file (see
`read_microphysics_from_file`), or `nothing` when the file does not
contain the corresponding field. They are written into the individual prognostic
condensate tracers when those tracers are carried by `Y`.

For the equilibrium (0M) scheme there are no prognostic condensate tracers, so
only `ρq_tot` is set; any file cloud condensate has already been folded into
`ᶜq_tot` by `thermodynamic_partition`, and saturation adjustment repartitions it
at runtime.
"""
function assign_moisture_edmf!(
    Y,
    ᶜT,
    ᶜq_tot,
    ᶜq_liq,
    ᶜq_ice,
    e_pot,
    thermo_params,
    ᶜq_lcl,
    ᶜq_icl,
    ᶜq_rai,
    ᶜq_sno,
)

    if hasproperty(Y.c, :ρq_tot)
        # Total water = file vapor + every prognostic condensate species, exactly
        # the partition used for ρ and ρe_tot (see `thermodynamic_partition`).
        Y.c.ρq_tot .= ᶜq_tot .* Y.c.ρ
    else
        error(
            "`dry` configurations are incompatible with the interpolated initial conditions.",
        )
    end

    if hasproperty(Y.c, :ρq_lcl)
        if !isnothing(ᶜq_lcl)
            Y.c.ρq_lcl .= ᶜq_lcl .* Y.c.ρ
        else
            fill!(Y.c.ρq_lcl, 0)
        end
    end
    if hasproperty(Y.c, :ρq_icl)
        if !isnothing(ᶜq_icl)
            Y.c.ρq_icl .= ᶜq_icl .* Y.c.ρ
        else
            fill!(Y.c.ρq_icl, 0)
        end
    end
    if hasproperty(Y.c, :ρq_rai) && hasproperty(Y.c, :ρq_sno)
        if !isnothing(ᶜq_rai)
            Y.c.ρq_rai .= ᶜq_rai .* Y.c.ρ
            Y.c.ρq_sno .= ᶜq_sno .* Y.c.ρ
        else
            fill!(Y.c.ρq_rai, 0)
            fill!(Y.c.ρq_sno, 0)
        end
    end

    # Initialize prognostic EDMF subdomains if present
    if hasproperty(Y.c, :sgsʲs)
        ᶜmse =
            TD.enthalpy.(
                thermo_params, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice,
            ) .+ e_pot
        for name in propertynames(Y.c.sgsʲs)
            s = getproperty(Y.c.sgsʲs, name)
            hasproperty(s, :ρa) && fill!(s.ρa, 0)
            hasproperty(s, :mse) && (s.mse .= ᶜmse)
            hasproperty(s, :q_tot) && (s.q_tot .= ᶜq_tot)
            # SGS 1M microphysics tracers
            hasproperty(s, :q_lcl) && fill!(s.q_lcl, 0)
            hasproperty(s, :q_icl) && fill!(s.q_icl, 0)
            if hasproperty(s, :q_rai) && hasproperty(s, :q_sno)
                if !isnothing(ᶜq_rai)
                    s.q_rai .= ᶜq_rai
                    s.q_sno .= ᶜq_sno
                else
                    fill!(s.q_rai, 0)
                    fill!(s.q_sno, 0)
                end
            end
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
  - `q`: water vapor specific humidity (3D; ERA5 `q`)
  - `u, v, w`: velocity (3D)
  - `clwc, ciwc`: cloud liquid and ice water content (optional, initializes
    condensate for the 1-moment / 2-moment schemes)
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
    regridder_kwargs = filter(!isnothing, (; extrapolation_bc, interpolation_method))
    svi_kwargs =
        isnothing(regridder_type) ? (; regridder_kwargs) :
        (; regridder_type, regridder_kwargs)

    isfile(file_path) || error("$(file_path) is not a file")
    @info "Overwriting initial conditions with data from file $(file_path)"

    center_space = Fields.axes(Y.c)
    face_space = Fields.axes(Y.f)

    # Regrid temperature and vapor specific humidity from file
    ᶜT = SpaceVaryingInputs.SpaceVaryingInput(
        file_path, "t", center_space; svi_kwargs...,
    )
    ᶜq_vap = SpaceVaryingInputs.SpaceVaryingInput(
        file_path, "q", center_space; svi_kwargs...,
    )

    # Optional condensate species. These are read before ρ / p / ρe_tot so that
    # the initial thermodynamic state is built from the same total water that is
    # stored in ρq_tot.
    (; ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno) =
        read_microphysics_from_file(file_path, center_space, svi_kwargs)
    (; ᶜq_tot, ᶜq_liq, ᶜq_ice) =
        thermodynamic_partition(Y, ᶜq_vap, ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno)

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
            p_sfc, file_path, face_space, Y, ᶜT,
            ᶜq_tot, ᶜq_liq, ᶜq_ice,
            thermo_params, regridder_kwargs;
            surface_altitude_var,
        )
    end

    # Hydrostatic pressure integration
    ᶠp = hydrostatic_pressure(
        p_sfc, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice, face_space,
        thermo_params,
    )

    # Density
    Y.c.ρ .= TD.air_density.(
        thermo_params, ᶜT, ᶜinterp.(ᶠp), ᶜq_tot, ᶜq_liq, ᶜq_ice,
    )

    # Velocity and energy
    e_pot = assign_velocity_energy!(
        Y, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice, ᶠp, thermo_params,
        file_path, svi_kwargs,
    )

    # Moisture and EDMF
    assign_moisture_edmf!(
        Y, ᶜT, ᶜq_tot, ᶜq_liq, ᶜq_ice, e_pot, thermo_params,
        ᶜq_lcl, ᶜq_icl, ᶜq_rai, ᶜq_sno,
    )

    return nothing
end
