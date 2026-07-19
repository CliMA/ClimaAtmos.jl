#####
##### Orographic drag parameterization
#####

# This orographic gravity wave drag follows the paper by Garner 2005:
# https://journals.ametsoc.org/view/journals/atsc/62/7/jas3496.1.xml?tab_body=pdf
# and the GFDL implementation:
# https://github.com/NOAA-GFDL/atmos_phys/blob/main/atmos_param/topo_drag/topo_drag.F90

using ClimaUtilities.ClimaArtifacts
using ClimaCore: InputOutput
import .AtmosArtifacts as AA

orographic_gravity_wave_cache(Y, atmos::AtmosModel) =
    orographic_gravity_wave_cache(Y, atmos.orographic_gravity_wave)

orographic_gravity_wave_cache(Y, ::Nothing) = (;)

function get_topo_info(Y, ogw::OrographicGravityWave)
    # For now, the initialisation of the cache is the same for all types of
    # orographic gravity wave drag parameterizations

    if ogw.topo_info == Val(:gfdl_restart)
        topo_path = @clima_artifact("topo_drag", ClimaComms.context(Y.c))
        orographic_info_rll = joinpath(topo_path, "topo_drag.res.nc")
        topo_info = regrid_OGW_info(Y, orographic_info_rll)
    elseif ogw.topo_info == Val(:raw_topo)
        earth_radius =
            Spaces.topology(
                Spaces.horizontal_space(axes(Y.c)),
            ).mesh.domain.radius
        topo_info = compute_ogw_drag(
            Y,
            earth_radius,
            ogw.topography,
            ogw.h_frac,
        )
    elseif ogw.topo_info == Val(:linear)
        # For user-defined analytical tests
        topo_info = initialize_drag_input_as_fields(Y, ogw.drag_input)
    else
        error("topo_info must be a symbol of type gfdl_restart, raw_topo, or linear")
    end

    return topo_info

end

function orographic_gravity_wave_cache(Y, ogw::OrographicGravityWave, topo_info = nothing)
    # For now, the initialisation of the cache is the same for all types of
    # orographic gravity wave drag parameterizations
    @assert Spaces.topology(Spaces.horizontal_space(axes(Y.c))).mesh.domain isa
            Domains.SphereDomain

    FT = Spaces.undertype(axes(Y.c))
    (; γ, ϵ, β, ρscale, L0, a0, a1, Fr_crit) = ogw

    if topo_info === nothing
        topo_info = get_topo_info(Y, ogw)
    end

    center_space, face_space = axes(Y.c), axes(Y.f)

    # Prepare cache
    return (;
        ogw_params = (;
            Fr_crit = Fr_crit,
            topo_ρscale = ρscale,
            topo_L0 = L0,
            topo_a0 = a0,
            topo_a1 = a1,
            topo_γ = γ,
            topo_β = β,
            topo_ϵ = ϵ,
        ),
        topo_ᶜτ_sat = Fields.Field(FT, axes(Y.c)),
        topo_ᶠτ_sat = Fields.Field(FT, axes(Y.f.u₃)),
        topo_ᶠVτ = Fields.Field(FT, axes(Y.f.u₃)),
        topo_τ_x = similar(Fields.level(Y.c.ρ, 1)),
        topo_τ_y = similar(Fields.level(Y.c.ρ, 1)),
        topo_τ_l = similar(Fields.level(Y.c.ρ, 1)),
        topo_τ_p = similar(Fields.level(Y.c.ρ, 1)),
        topo_τ_np = similar(Fields.level(Y.c.ρ, 1)),
        topo_U_sat = similar(Fields.level(Y.c.ρ, 1)),
        topo_FrU_sat = similar(Fields.level(Y.c.ρ, 1)),
        topo_FrU_max = similar(Fields.level(Y.c.ρ, 1)),
        topo_FrU_min = similar(Fields.level(Y.c.ρ, 1)),
        topo_FrU_clp = similar(Fields.level(Y.c.ρ, 1)),
        topo_ᶜz_pbl = similar(Fields.level(Y.c.ρ, 1)),
        topo_ᶠz_pbl = similar(Fields.level(Y.f.u₃, half)),
        values_at_z_pbl = similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT, FT, FT}),
        topo_info = topo_info,
        ᶜbuoyancy_frequency = Fields.Field(FT, center_space),
        ᶠbuoyancy_frequency = Fields.Field(FT, face_space),
        ᶜuforcing = zero(Y.c.ρ),
        ᶜvforcing = zero(Y.c.ρ),
        ᶜdTdz = similar(Y.c.ρ),
        ᶠp_m1 = Fields.Field(FT, face_space),
        ᶠp_ref = similar(Fields.level(Y.f.u₃, half), FT),
        ᶜmask = Fields.Field(Bool, center_space),
    )

end

orographic_gravity_wave_compute_tendency!(Y, p, ::Nothing) = nothing

function orographic_gravity_wave_compute_tendency!(Y, p, ::FullOrographicGravityWave)
    @debug begin
        if !hasfield(typeof(p), :ogwd_call_counter)
            @info "OGWD tendency function called for the first time"
        end

        # DEBUG: Check if Y has NaNs at entry
        if any(isnan, parent(Y.c.ρ))
            @error "OGWD: Input Y.c.ρ already has NaNs at function entry!"
            error("Cannot compute OGWD tendency with NaN inputs")
        end
    end

    # unpack cache
    (; ᶜp, ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed
    (; params) = p
    (; ᶜuforcing, ᶜvforcing) = p.orographic_gravity_wave
    (; ᶜdTdz) = p.orographic_gravity_wave
    (; ᶠp_m1) = p.orographic_gravity_wave
    (; ᶜbuoyancy_frequency, ᶠbuoyancy_frequency) = p.orographic_gravity_wave

    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z
    ᶠdz = Fields.Δz_field(axes(Y.f))
    FT = Spaces.undertype(axes(Y.c))
    ᶜρ = Y.c.ρ

    # parameters
    cp_d = CAP.cp_d(params)
    grav = CAP.grav(params)
    thermo_params = CAP.thermodynamics_params(params)

    # compute buoyancy frequency
    ᶜdTdz .= Geometry.WVector.(ᶜgradᵥ.(ᶠinterp.(ᶜT))).components.data.:1
    @. ᶜbuoyancy_frequency =
        (grav / ᶜT) *
        (ᶜdTdz + grav / TD.cp_m(thermo_params, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice))
    @. ᶜbuoyancy_frequency =
        ifelse(ᶜbuoyancy_frequency < eps(FT), sqrt(eps(FT)), sqrt(abs(ᶜbuoyancy_frequency))) # to avoid small numbers
    @. ᶠbuoyancy_frequency = ᶠinterp(ᶜbuoyancy_frequency)

    # compute ᶠp and ᶠp_m1
    # load array from scratch
    ᶠp = p.scratch.ᶠtemp_scalar
    @. ᶠp = ᶠinterp(ᶜp)
    scale_height_values = p.scratch.ᶠtemp_field_level
    z_extrapolated_values = p.scratch.temp_data_face_level

    # explicit scale height approach for pressure extrapolation
    # Fields.level returns by reference
    ᶠz_bottom = Fields.level(ᶠz, half)
    ᶠz_second = Fields.level(ᶠz, 1 + half)
    ᶠp_bottom = Fields.level(ᶠp, half)
    ᶠp_second = Fields.level(ᶠp, 1 + half)

    # Calculate scale height from the two levels
    Fields.field_values(scale_height_values) .=
        (Fields.field_values(ᶠz_second) .- Fields.field_values(ᶠz_bottom)) ./
        log.(Fields.field_values(ᶠp_bottom) ./ Fields.field_values(ᶠp_second))

    # Calculate the extrapolated height (one level below bottom)
    z_extrapolated_values .=
        Fields.field_values(ᶠz_bottom) .-
        (Fields.field_values(ᶠz_second) .- Fields.field_values(ᶠz_bottom))

    # Extrapolate pressure using barometric formula: p = p₀ * exp(-z/H)
    Boundary_value = Fields.Field(
        Fields.field_values(ᶠp_bottom) .*
        exp.(
            (z_extrapolated_values .- Fields.field_values(ᶠz_bottom)) ./
            Fields.field_values(scale_height_values),
        ),
        axes(ᶠp_bottom),
    )

    field_shiftface_down!(ᶠp, ᶠp_m1, Boundary_value)

    # prepare physical uv input variables for gravity_wave_forcing()
    ᶜu = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    ᶜv = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    @. ᶜuforcing = 0
    @. ᶜvforcing = 0

    orographic_gravity_wave_forcing!(
        ᶜu,
        ᶜv,
        ᶜbuoyancy_frequency,
        ᶠbuoyancy_frequency,
        ᶜz,
        ᶠz,
        ᶠdz,
        ᶜuforcing,
        ᶜvforcing,
        ᶜρ,
        ᶜp,
        ᶠp,
        ᶠp_m1,
        ᶜT,
        grav,
        cp_d,
        p,
    )

    @debug begin
        # Checkpoint 2b: Check computed forcing magnitude
        @info "  Computed forcing: u_max=$(maximum(abs, ᶜuforcing)) m/s², v_max=$(maximum(abs, ᶜvforcing)) m/s²"

        # Warn if forcing is very large (>0.01 m/s² = 1 cm/s² is already strong)
        max_forcing = max(maximum(abs, ᶜuforcing), maximum(abs, ᶜvforcing))
        if max_forcing > 0.01
            @warn "OGWD forcing very large! This may cause instability."
            @warn "  max_accel=$(max_forcing) m/s² (threshold: 0.01 m/s²)"
        end
    end
end

orographic_gravity_wave_apply_tendency!(Yₜ, p, ::Nothing) = nothing

function orographic_gravity_wave_apply_tendency!(
    Yₜ,
    p,
    ::OrographicGravityWave,
)
    (; ᶜuforcing, ᶜvforcing) = p.orographic_gravity_wave

    @. Yₜ.c.uₕ +=
        C12.(Geometry.UVVector.(ᶜuforcing, ᶜvforcing))

end


function orographic_gravity_wave_forcing!(
    u_phy,
    v_phy,
    ᶜbuoyancy_frequency,
    ᶠbuoyancy_frequency,
    ᶜz,
    ᶠz,
    ᶠdz,
    ᶜuforcing,
    ᶜvforcing,
    ᶜρ,
    ᶜp,
    ᶠp,
    ᶠp_m1,
    ᶜT,
    grav,
    cp_d,
    p,
)

    FT = eltype(ᶠbuoyancy_frequency)
    Δz_bot = Fields.level(ᶠdz, half)

    (; topo_ᶜz_pbl, topo_ᶠz_pbl, topo_τ_x, topo_τ_y, topo_τ_l, topo_τ_p, topo_τ_np) =
        p.orographic_gravity_wave
    (; topo_ᶜτ_sat, topo_ᶠτ_sat) = p.orographic_gravity_wave
    (; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) =
        p.orographic_gravity_wave
    (; topo_ᶠVτ, values_at_z_pbl, topo_info) = p.orographic_gravity_wave
    (; ᶜmask, ᶠp_ref) = p.orographic_gravity_wave

    # Extract parameters
    ogw_params = p.orographic_gravity_wave.ogw_params

    # we copy the z_pbl from a cell-centered to face array.
    # the z-values don't change, but this is necessary for
    # calc_nonpropagating_forcing! to work on the GPU
    get_pbl_z!(topo_ᶜz_pbl, ᶜp, ᶜT, ᶜz, grav, cp_d)
    parent(topo_ᶠz_pbl) .= parent(topo_ᶜz_pbl) .- FT(1 / 2) .* parent(Δz_bot)
    topo_ᶠz_pbl = topo_ᶠz_pbl.components.data.:1

    # compute base flux at the planetary boundary layer height
    calc_base_flux!(
        topo_τ_x,
        topo_τ_y,
        topo_τ_l,
        topo_τ_p,
        topo_τ_np,
        #
        topo_U_sat,
        topo_FrU_sat,
        topo_FrU_clp,
        topo_FrU_max,
        topo_FrU_min,
        topo_ᶜz_pbl,
        #
        values_at_z_pbl,
        #
        ogw_params,
        topo_info,
        #
        ᶜρ,
        u_phy,
        v_phy,
        ᶜz,
        ᶜbuoyancy_frequency,
    )

    calc_saturation_profile!(
        topo_ᶠτ_sat,
        topo_ᶠVτ,
        #
        topo_U_sat,
        topo_FrU_sat,
        topo_FrU_clp,
        topo_FrU_max,
        topo_FrU_min,
        topo_ᶜτ_sat,
        topo_τ_x,
        topo_τ_y,
        topo_τ_p,
        topo_ᶜz_pbl,
        #
        ogw_params,
        #
        ᶜρ,
        u_phy,
        v_phy,
        ᶜp,
        ᶜbuoyancy_frequency,
        ᶜz,
    )

    # compute drag tendencies due to propagating part
    ᶜdτ_sat_dz = p.scratch.ᶜtemp_scalar
    calc_propagate_forcing!(
        ᶜuforcing,
        ᶜvforcing,
        topo_τ_x,
        topo_τ_y,
        topo_τ_l,
        topo_ᶠτ_sat,
        ᶜdτ_sat_dz,
        ᶜρ,
    )

    ᶜweights = p.scratch.ᶜtemp_scalar
    ᶜdiff = p.scratch.ᶜtemp_scalar_2
    ᶜwtsum = p.scratch.temp_field_level
    ᶠz_ref = p.scratch.ᶠtemp_field_level
    calc_nonpropagating_forcing!(
        ᶜuforcing,
        ᶜvforcing,
        #
        topo_τ_x,
        topo_τ_y,
        topo_τ_l,
        topo_τ_np,
        topo_ᶠVτ,
        topo_ᶠz_pbl,
        #
        ᶠz_ref,
        ᶠp_ref,
        ᶜmask,
        ᶜweights,
        ᶜdiff,
        ᶜwtsum,
        #
        ᶠp,
        ᶠp_m1,
        ᶠbuoyancy_frequency,
        ᶠz,
        ᶠdz,
        grav,
    )

    # constrain forcing
    @. ᶜuforcing = max(FT(-3e-3), min(FT(3e-3), ᶜuforcing))
    @. ᶜvforcing = max(FT(-3e-3), min(FT(3e-3), ᶜvforcing))

    @debug begin
        # DEBUG: Check for NaNs in OGWD forcing
        if any(isnan, parent(ᶜuforcing)) || any(isnan, parent(ᶜvforcing))
            @error "NaN detected in OGWD forcing!"
            @error "  ᶜuforcing: has_nan=$(any(isnan, parent(ᶜuforcing))), min=$(minimum(parent(ᶜuforcing))), max=$(maximum(parent(ᶜuforcing)))"
            @error "  ᶜvforcing: has_nan=$(any(isnan, parent(ᶜvforcing))), min=$(minimum(parent(ᶜvforcing))), max=$(maximum(parent(ᶜvforcing)))"
            error("OGWD produced NaN forcing - aborting")
        end
    end
end

function calc_nonpropagating_forcing!(
    ᶜuforcing,
    ᶜvforcing,
    #
    τ_x,
    τ_y,
    τ_l,
    τ_np,
    ᶠVτ,
    ᶠz_pbl,
    #
    ᶠz_ref,
    ᶠp_ref,
    ᶜmask,
    ᶜweights,
    ᶜdiff,
    ᶜwtsum,
    #
    ᶠp,
    ᶠp_m1,
    ᶠN,
    ᶠz,
    ᶠdz,
    grav,
)
    FT = eltype(ᶠN)

    # Convert type parameters to values before using in closure
    pi_val = FT(π)
    min_n_val = FT(0.7e-2)
    max_n_val = FT(1.7e-2)
    min_Vτ_val = FT(1.0)

    # Compute z_ref using column_reduce
    input = @. lazy(
        tuple(ᶠz_pbl, ᶠz, ᶠN, ᶠVτ, pi_val, min_n_val, max_n_val, min_Vτ_val),
    )

    Operators.column_reduce!(
        ᶠz_ref,
        input;
        init = (FT(0.0), FT(0.0), FT(0.0), false),
        transform = first,
    ) do (z_ref_acc, ᶠz_pbl_acc, phase_acc, done),
    (
        ᶠz_pbl_itr,
        z_face,
        N_face,
        Vτ_face,
        pi_val,
        min_n_val,
        max_n_val,
        min_Vτ_val,
    )
        if done
            # If already done, return the accumulated values
            return (z_ref_acc, ᶠz_pbl_acc, phase_acc, true)
        end
        if (z_face > ᶠz_pbl_itr)
            # Only accumulate phase above z_pbl
            phase_acc +=
                (z_face - ᶠz_pbl_itr) * max(min_n_val, min(max_n_val, N_face)) /
                max(min_Vτ_val, Vτ_face)

            # If phase exceeds π, stop and return current z_col as z_ref
            if phase_acc > pi_val
                return (z_face, ᶠz_pbl_itr, phase_acc, true)
            end
            # Update z_ref only when above z_pbl
            # If phase never exceeds π, z_ref will end up at model top
            return (z_face, ᶠz_pbl_itr, phase_acc, false)
        end
        # Below z_pbl, keep previous z_ref_acc unchanged
        return (z_ref_acc, ᶠz_pbl_acc, phase_acc, false)
    end

    eps_val = eps(FT)
    half_val = FT(0.5)
    nan_val = FT(NaN)

    input = @. lazy(tuple(ᶠz_ref, ᶠp, ᶠz, ᶠdz, eps_val, half_val))

    Operators.column_reduce!(
        ᶠp_ref,
        input;
        init = nan_val,
    ) do ᶠp_ref, (z_ref, ᶠp, ᶠz, ᶠdz, eps_val, half_val)
        if abs(ᶠz - z_ref) < (half_val * ᶠdz + eps_val)
            if isnan(ᶠp_ref)
                ᶠp_ref = ᶠp
            end
        end
        return ᶠp_ref
    end

    # Include cells that overlap with [z_pbl, z_ref):
    # - ᶜright_bias checks upper face > z_pbl (cell extends above z_pbl)
    # - ᶜleft_bias checks lower face < z_ref (cell starts below z_ref)
    # This ensures at least one cell is included when z_ref > z_pbl
    @. ᶜmask = ᶜright_bias.((ᶠz .> ᶠz_pbl)) .&& ᶜleft_bias.((ᶠz .< ᶠz_ref))
    @. ᶜweights = ᶜinterp.(ᶠp .- ᶠp_ref)
    @. ᶜdiff = ᶜinterp.(ᶠp_m1 .- ᶠp)

    # Exclude cells with zero weights from the mask to avoid division by zero.
    # Zero weight means p == p_ref at that cell, so it contributes nothing
    # to the pressure-weighted average.
    @. ᶜmask = ᶜmask && (!iszero(ᶜweights))

    parent(ᶜweights) .= parent(ᶜweights .* ᶜmask)

    input = @. lazy(ifelse(ᶜmask == true, ᶜdiff / ᶜweights, FT(0)))

    Operators.column_reduce!(ᶜwtsum, input; init = FT(0)) do acc, wtsum_field
        return acc + wtsum_field
    end

    if any(isnan.(parent(ᶜwtsum)))
        @warn "NaN encountered in weight sum calculation of orographic gravity wave drag"
    end

    # Compute drag, handling empty mask case (wtsum=0) gracefully
    # When wtsum=0, the mask is empty (no cells between z_pbl and z_ref),
    # so we set forcing to 0 for those columns
    @. ᶜuforcing += ifelse(
        iszero(ᶜwtsum),
        FT(0),
        grav * τ_x * τ_np / τ_l / ᶜwtsum * ᶜweights,
    )
    @. ᶜvforcing += ifelse(
        iszero(ᶜwtsum),
        FT(0),
        grav * τ_y * τ_np / τ_l / ᶜwtsum * ᶜweights,
    )

end

function calc_propagate_forcing!(
    ᶜuforcing,
    ᶜvforcing,
    τ_x,
    τ_y,
    τ_l,
    τ_sat,
    dτ_sat_dz,
    ᶜρ,
)
    parent(dτ_sat_dz) .=
        parent(Geometry.WVector.(ᶜgradᵥ.(τ_sat)).components.data.:1)

    @. ᶜuforcing -= τ_x / τ_l / ᶜρ * dτ_sat_dz
    @. ᶜvforcing -= τ_y / τ_l / ᶜρ * dτ_sat_dz
    return nothing
end

"""
    get_pbl_z!(result, ᶜp, ᶜT, ᶜz, grav, cp_d)

Calculate the planetary boundary layer (PBL) height for each atmospheric column.

The PBL height is determined by finding the highest level where both pressure and
temperature lapse rate criteria are satisfied. This uses a thermodynamic approach
to identify the transition from the well-mixed boundary layer to the stratified
free atmosphere above.

# Arguments

  - `result`: Output field to store the computed PBL heights (modified in-place)
  - `ᶜp`: Cell-centered pressure field [Pa]
  - `ᶜT`: Cell-centered temperature field [K]
  - `ᶜz`: Cell-centered geometric height field [m]
  - `grav`: Gravitational acceleration [m/s²]
  - `cp_d`: Specific heat capacity at constant pressure for dry air [J/(kg·K)]

# Algorithm

The function uses a column reduction operation that iterates upward through each
atmospheric column. At each level, it checks:

 1. **Pressure criterion**: p ≥ 0.5 × p_surface (limits search to lower atmosphere)
 2. **Temperature lapse rate criterion**: (T_sfc + 1.5 - T) > (g/cp_d) × (z - z_sfc)

The PBL height is set to the highest level where both conditions are met.

# Physical interpretation

The temperature criterion compares the actual temperature profile against a dry
adiabatic lapse rate (g/cp_d) with a 1.5 K offset. This effectively detects where
the atmosphere transitions from the convectively mixed boundary layer to the more
stable free atmosphere above.

# Implementation notes

  - Uses `Operators.column_reduce!` for GPU compatibility
  - Initializes with surface height if no levels satisfy the criteria    # Get surface values (first level values)
"""
function get_pbl_z!(result, ᶜp, ᶜT, ᶜz, grav, cp_d)
    FT = eltype(ᶜp)

    # Get surface values (first level values)
    p_sfc = Fields.level(ᶜp, 1)
    T_sfc = Fields.level(ᶜT, 1)
    z_sfc = Fields.level(ᶜz, 1)

    half_val = FT(0.5)
    temp_offset = FT(1.5)
    grav_val = FT(grav)
    cp_d_val = FT(cp_d)
    zero_val = FT(0)

    # Create a lazy tuple of inputs for column_reduce
    input = @. lazy(
        tuple(
            ᶜp,
            ᶜT,
            ᶜz,
            p_sfc,
            T_sfc,
            z_sfc,
            grav_val,
            cp_d_val,
            half_val,
            temp_offset,
            zero_val,
        ),
    )

    # Perform the column reduction
    Operators.column_reduce!(
        result,
        input;
        init = FT(0),
        transform = first, # Extract just the z_pbl value
    ) do z_pbl,
    (
        p_col,
        T_col,
        z_col,
        p_sfc,
        T_sfc,
        z_sfc,
        grav_val,
        cp_d_val,
        half_val,
        temp_offset,
        zero_val,
    )

        if z_pbl == zero_val
            z_pbl = z_sfc
        end
        # Check conditions
        p_threshold = p_col >= (half_val * p_sfc)
        T_threshold =
            (T_sfc + temp_offset - T_col) >
            (grav_val / cp_d_val * (z_col - z_sfc))

        # If both conditions are met, update z_pbl to current height
        if p_threshold && T_threshold
            z_pbl = z_col
        end

        # Move to next level
        return z_pbl
    end
end



"""
    field_shiftface_down!(source_field, shifted_field, boundary_value)

Shift a face-centered field downward by one level, storing the result in `shifted_field`.

This is needed to access face values at level `k-1` from within a level-`k` computation
(e.g., computing `ᶠp[k-1]` for pressure differences across cell layers). ClimaCore `column_reduce` and `column_accumulate` do not support direct `field[k-1]` indexing in broadcast expressions, so we
construct the shifted view via a round-trip through the cell-center grid:

 1. `LeftBiasedF2C` interpolates faces → cell centers using the value from below.
 2. `LeftBiasedC2F` interpolates cell centers → faces using the value from below,
    with `boundary_value` prescribed at the bottom face.

The net effect is `shifted_field[k] = source_field[k-1]` for interior faces,
and `shifted_field[bottom] = boundary_value` at the lowest face.
"""
function field_shiftface_down!(source_field, shifted_field, boundary_value)
    L1 = Operators.LeftBiasedC2F(; bottom = Operators.SetValue(boundary_value))
    shifted_field .= L1.(ᶜleft_bias.(source_field))
end

function calc_base_flux!(
    τ_x,
    τ_y,
    τ_l,
    τ_p,
    τ_np,
    #
    U_sat,
    FrU_sat,
    FrU_clp,
    FrU_max,
    FrU_min,
    z_pbl,
    #
    values_at_z_pbl,
    #
    ogw_params,
    topo_info,
    #
    ᶜρ,
    u_phy,
    v_phy,
    ᶜz,
    ᶜN,
)
    (;
        Fr_crit,
        topo_ρscale,
        topo_L0,
        topo_a0,
        topo_a1,
        topo_γ,
        topo_β,
        topo_ϵ,
    ) = ogw_params
    (; hmax, hmin, t11, t12, t21, t22) = topo_info

    FT = eltype(Fr_crit)
    γ = topo_γ
    β = topo_β
    ϵ = topo_ϵ

    input = @. lazy(tuple(ᶜρ, u_phy, v_phy, ᶜN, ᶜz, z_pbl))

    Operators.column_reduce!(
        values_at_z_pbl,
        input;
        init = (FT(0.0), FT(0.0), FT(0.0), FT(0.0)),
    ) do (ρ_acc, u_acc, v_acc, N_acc), (ρ, u, v, N, z_col, z_target)

        # Check if current level height is at or above z_pbl
        # Use the last valid level that satisfies z_col <= z_target
        if z_col <= z_target
            return (ρ, u, v, N)
        else
            return (ρ_acc, u_acc, v_acc, N_acc)
        end
    end

    # These are views
    ρ_pbl = values_at_z_pbl.:1
    u_pbl = values_at_z_pbl.:2
    v_pbl = values_at_z_pbl.:3
    N_pbl = values_at_z_pbl.:4

    # Calculate τ components
    @. τ_x = ρ_pbl * N_pbl * (t11 * u_pbl + t21 * v_pbl)
    @. τ_y = ρ_pbl * N_pbl * (t12 * u_pbl + t22 * v_pbl)

    # Calculate Vτ using field operations
    Vτ = @. lazy(
        max(
            eps(FT),
            -(u_pbl * τ_x + v_pbl * τ_y) / max(eps(FT), sqrt(τ_x^2 + τ_y^2)),
        ),
    )

    # Calculate Froude numbers
    Fr_max = @. lazy(max(FT(0), hmax) * N_pbl / Vτ)
    Fr_min = @. lazy(max(FT(0), hmin) * N_pbl / Vτ)

    # Calculate U_sat
    @. U_sat = sqrt(ρ_pbl / topo_ρscale * Vτ^3 / N_pbl / topo_L0)

    # Calculate FrU values
    @. FrU_sat = Fr_crit * U_sat
    @. FrU_min = Fr_min * U_sat
    @. FrU_max = max(Fr_max * U_sat, FrU_min + eps(FT))
    @. FrU_clp = min(FrU_max, max(FrU_min, FrU_sat))

    # Calculate drag components
    @. τ_l = ((FrU_max)^(2 + γ - ϵ) - (FrU_min)^(2 + γ - ϵ)) / (2 + γ - ϵ)

    # Calculate propagating drag
    @. τ_p =
        topo_a0 * (
            (FrU_clp^(2 + γ - ϵ) - FrU_min^(2 + γ - ϵ)) / (2 + γ - ϵ) +
            FrU_sat^(β + 2) * (FrU_max^(γ - ϵ - β) - FrU_clp^(γ - ϵ - β)) /
            (γ - ϵ - β)
        )

    # Calculate non-propagating drag
    @. τ_np =
        topo_a1 * U_sat / (1 + β) * (
            (FrU_max^(1 + γ - ϵ) - FrU_clp^(1 + γ - ϵ)) / (1 + γ - ϵ) -
            FrU_sat^(β + 1) * (FrU_max^(γ - ϵ - β) - FrU_clp^(γ - ϵ - β)) /
            (γ - ϵ - β)
        )

    # Apply scaling
    @. τ_np = τ_np / max(Fr_crit, Fr_max)

    return nothing
end

function calc_saturation_profile!(
    ᶠτ_sat,
    ᶠVτ,
    #
    U_sat,
    FrU_sat,
    FrU_clp,
    FrU_max,
    FrU_min,
    ᶜτ_sat,
    τ_x,
    τ_y,
    τ_p,
    z_pbl,
    #
    ogw_params,
    #
    ᶜρ,
    u_phy,
    v_phy,
    ᶜp,
    ᶜN,
    ᶜz,
)
    # Extract parameters from tuple
    (; Fr_crit, topo_ρscale, topo_L0, topo_a0, topo_γ, topo_β, topo_ϵ) =
        ogw_params

    FT = eltype(Fr_crit)
    γ = topo_γ
    β = topo_β
    ϵ = topo_ϵ

    # Calculate Vτ at cell faces using field operations
    ᶜVτ = @. lazy(
        max(
            eps(FT),
            (-(u_phy * τ_x + v_phy * τ_y) / max(eps(FT), sqrt(τ_x^2 + τ_y^2))),
        ),
    )

    # Calculate derivatives for ᶠd2Vτdz
    d2udz = lazy.(ᶜd2dz2(u_phy))
    d2vdz = lazy.(ᶜd2dz2(v_phy))
    # Calculate derivative for L1; tmp_field_2 == d2Vτdz
    d2Vτdz = @. lazy(
        max(
            eps(FT),
            -(d2udz * τ_x + d2vdz * τ_y) / max(eps(FT), sqrt(τ_x^2 + τ_y^2)),
        ),
    )

    # Calculate tmp_field_1 == L1
    # Here on the RHS, tmp_field_2 == d2Vτdz
    L1 = @. lazy(
        topo_L0 *
        max(FT(0.5), min(FT(2.0), FT(1.0) - FT(2.0) * ᶜVτ * d2Vτdz / ᶜN^2)),
    )

    # Create field for U_k calculation
    # Here, U_k == tmp_field_1
    U_k_field = @. lazy(sqrt(ᶜρ / topo_ρscale * ᶜVτ^3 / ᶜN / L1))

    z_surf = Fields.level(ᶜz, 1)
    # Create combined input for column_accumulate
    input = @. lazy(
        tuple(
            FrU_clp,
            FrU_sat,
            U_k_field,
            FrU_max,
            FrU_min,
            Fr_crit,
            z_surf,
            ᶜz,
            z_pbl,
            topo_a0,
            τ_p,
            U_sat,
        ),
    )

    # Initialize the result field with τ_p at the lowest face
    fill!(ᶜτ_sat, 0.0)

    Operators.column_accumulate!(
        ᶜτ_sat,
        input;
        init = (FT(0.0), FT(0.0), FT(0.0), FT(0.0)),
        transform = first,
    ) do (tau_sat_val, U_sat_val, local_FrU_sat, local_FrU_clp),
    (
        FrU_clp0,
        FrU_sat0,
        U,
        FrU_max,
        FrU_min,
        Fr_crit_val,
        z_surf,
        z_col,
        z_target,
        topo_a0,
        τ_p,
        U_sat,
    )

        if z_col == z_surf
            U_sat_val = U_sat
        end

        U_sat_val = min(U_sat_val, U)
        local_FrU_sat = Fr_crit_val * U_sat_val  # Use local variable instead
        local_FrU_clp = min(FrU_max, max(FrU_min, local_FrU_sat))  # Use local variable instead

        if z_col <= z_target
            tau_sat_val = τ_p
        else
            tau_sat_val =
                topo_a0 * (
                    (local_FrU_clp^(2 + γ - ϵ) - FrU_min^(2 + γ - ϵ)) /
                    (2 + γ - ϵ) +
                    local_FrU_sat^2 *
                    FrU_sat0^β *
                    (FrU_max^(γ - ϵ - β) - FrU_clp0^(γ - ϵ - β)) / (γ - ϵ - β) +
                    local_FrU_sat^2 *
                    (FrU_clp0^(γ - ϵ) - local_FrU_clp^(γ - ϵ)) / (γ - ϵ)
                )
        end

        return (tau_sat_val, U_sat_val, local_FrU_sat, local_FrU_clp)
    end

    top_values = Fields.level(ᶜτ_sat, Spaces.nlevels(axes(ᶜτ_sat)))
    p_surf = Fields.level(ᶜp, 1)
    p_top = Fields.level(ᶜp, Spaces.nlevels(axes(ᶜp)))

    zero_val = FT(0.0)

    input = @. lazy(tuple(top_values, ᶜτ_sat, p_surf, p_top, ᶜp, zero_val))

    Operators.column_accumulate!(
        ᶜτ_sat,
        input;
        init = FT(0.0),
        transform = identity,
    ) do τ_sat_val, (top_values, ᶜτ_sat, p_surf, p_top, ᶜp, zero_val)

        τ_sat_val = ᶜτ_sat

        if top_values > zero_val
            τ_sat_val -= (top_values * (p_surf - ᶜp) / (p_surf - p_top))
        end

        return τ_sat_val
    end

    @. ᶠτ_sat = ᶠinterp(ᶜτ_sat)
    @. ᶠVτ = ᶠinterp(ᶜVτ)

    return nothing
end


function compute_ogw_drag(
    Y,
    earth_radius,
    topography,
    h_frac,
)
    FT = eltype(Y)
    center_space = axes(Y.c)
    h_elem = Spaces.n_elements_per_panel_direction(center_space)
    face_space = axes(Y.f)
    ᶜsurface_space = Fields.level(center_space, 1)
    J_bot = Fields.level(Fields.local_geometry_field(face_space).J, half)
    Δz_bot = Fields.level(Fields.Δz_field(face_space), half)
    cell_area_bot = @. J_bot / Δz_bot

    z_surface = Fields.level(Fields.coordinate_field(Y.f).z, half)

    cg_lat = Fields.level(Fields.coordinate_field(Y.f).lat, half)

    if topography == Val(:Earth) || topography == Val(:NoWarp)
        # Try local file first (for development when preprocessing has been run)
        local_filename = "computed_drag_Earth_false_1_$(h_elem)"
        local_path = joinpath(pkgdir(@__MODULE__), "$(local_filename).hdf5")

        if isfile(local_path)
            @info "Loading computed drag from local file: $(local_path)"
            topo_info = load_preprocessed_topography(local_filename)

            @debug begin
                # Checkpoint 1: Validate loaded drag tensor
                @info "OGWD drag tensor loaded from LOCAL FILE (h_elem=$(h_elem)):"
                @info "  hmax: min=$(minimum(parent(topo_info.hmax))), max=$(maximum(parent(topo_info.hmax))), mean=$(sum(parent(topo_info.hmax))/length(parent(topo_info.hmax)))"
                @info "  t11: min=$(minimum(parent(topo_info.t11))), max=$(maximum(parent(topo_info.t11)))"
                @info "  NaN/Inf: hmax_nan=$(any(isnan, parent(topo_info.hmax))), t11_inf=$(any(isinf, parent(topo_info.t11)))"
            end
        else
            # Fall back to ClimaArtifacts
            @info "Local file not found, loading from ClimaArtifacts (h_elem=$(h_elem))..."
            artifact_path =
                AA.ogw_computed_drag_file_path(; h_elem, context = ClimaComms.context(Y.c))
            @info "Loading from: $(artifact_path)"
            reader = InputOutput.HDF5Reader(artifact_path, ClimaComms.context(Y.c))
            topo_info = InputOutput.read_field(reader, "computed_drag")
            Base.close(reader)

            @debug begin
                # Checkpoint 1: Validate loaded drag tensor
                @info "OGWD drag tensor loaded from ARTIFACT (h_elem=$(h_elem)):"
                @info "  hmax: min=$(minimum(topo_info.hmax)), max=$(maximum(topo_info.hmax)), mean=$(sum(parent(topo_info.hmax))/length(parent(topo_info.hmax)))"
                @info "  t11: min=$(minimum(topo_info.t11)), max=$(maximum(topo_info.t11))"
                @info "  NaN/Inf: hmax_nan=$(any(isnan, topo_info.hmax)), t11_inf=$(any(isinf, topo_info.t11))"
            end
        end

        return set_topo_info_target_space(topo_info, ᶜsurface_space)

        ### Handle analytical test cases
        # NOTE: OGW for analytical topography cases is not yet tested.
    elseif topography == Val(:DCMIP200)
        topography_function = topography_dcmip200
    elseif topography == Val(:Hughes2023)
        topography_function = topography_hughes2023
    elseif topography == Val(:Agnesi)
        topography_function = topography_agnesi
    elseif topography == Val(:Schar)
        topography_function = topography_schar
    elseif topography == Val(:Cosine2d)
        topography_function = topography_cosine_2d
    elseif topography == Val(:Cosine3d)
        topography_function = topography_cosine_3d
    else
        error("Topography required for orographic gravity wave drag: $topography")
    end

    real_elev = SpaceVaryingInput(topography_function, face_space)
    real_elev = Fields.level(real_elev, half)
    @. real_elev = max(0, real_elev)

    hmax = @. real_elev - z_surface
    hmin = @. h_frac * hmax

    χ = @. hmax * cell_area_bot * earth_radius / (FT(2) * FT(pi))

    ∇ₕχ = @. Geometry.UVVector(gradₕ(χ))
    ∇ₕhmax = @. Geometry.UVVector(gradₕ(hmax))

    # Negate the velocity-potential gradient so the drag opposes the low-level
    # flow, matching the offline pipeline (calc_orographic_tensor uses
    # `.-calc_∇A(χ, …)`) and Garner (2005) Eq. 6/8. Without this the analytical
    # topography tensor would carry the wrong sign and accelerate the flow.
    dχdx = @. -∇ₕχ.components.data.:1
    dχdy = @. -∇ₕχ.components.data.:2

    dhdx = ∇ₕhmax.components.data.:1
    dhdy = ∇ₕhmax.components.data.:2

    # Handle drag vector elements at the antarctic region
    @. dχdx = ifelse(cg_lat < FT(-88), 0, dχdx)
    @. dχdy = ifelse(cg_lat < FT(-88), 0, dχdy)

    # We convert the face-centered drag vector elements to cell-centered
    # quantities as these are used to compute the physics associated with the
    # orographic gravity wave drag in the cell.
    hmax = Fields.Field(Fields.field_values(hmax), ᶜsurface_space)
    hmin = Fields.Field(Fields.field_values(hmin), ᶜsurface_space)
    t11 = Fields.Field(Fields.field_values(dχdx .* dhdx), ᶜsurface_space)
    t21 = Fields.Field(Fields.field_values(dχdx .* dhdy), ᶜsurface_space)
    t12 = Fields.Field(Fields.field_values(dχdy .* dhdx), ᶜsurface_space)
    t22 = Fields.Field(Fields.field_values(dχdy .* dhdy), ᶜsurface_space)

    return (; hmax, hmin, t11, t21, t12, t22)

end


ᶜd2dz2(ᶜscalar) =
    lazy.(Geometry.WVector.(ᶜgradᵥ.(ᶠddz(ᶜscalar))).components.data.:1)

ᶜddz(ᶠscalar) = lazy.(Geometry.WVector.(ᶜgradᵥ.(ᶠscalar)).components.data.:1)

ᶠddz(ᶜscalar) = lazy.(Geometry.WVector.(ᶠgradᵥ.(ᶜscalar)).components.data.:1)
