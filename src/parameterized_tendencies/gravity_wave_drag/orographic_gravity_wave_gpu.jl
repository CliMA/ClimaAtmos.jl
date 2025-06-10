#####
##### Orographic drag parameterization
#####

# This orographic gravity wave drag follows the paper by Garner 2005:
# https://journals.ametsoc.org/view/journals/atsc/62/7/jas3496.1.xml?tab_body=pdf
# and the GFDL implementation:
# https://github.com/NOAA-GFDL/atmos_phys/blob/main/atmos_param/topo_drag/topo_drag.F90

#=
The following are default false in GFDL code but usually run with true in nml setups:
use_pbl_from_lock = .true.
use_uref_4stable = .true.
Not yet included in our codebase
=#
using ClimaUtilities.ClimaArtifacts
using ClimaCore: InputOutput

orographic_gravity_wave_cache(Y, atmos::AtmosModel) =
    orographic_gravity_wave_cache(Y, atmos.orographic_gravity_wave)

orographic_gravity_wave_cache(Y, ::Nothing) = (;)

orographic_gravity_wave_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function get_topo_info(Y, ogw::OrographicGravityWave)
    # For now, the initialisation of the cache is the same for all types of
    # orographic gravity wave drag parameterizations

    if ogw.topo_info == "gfdl_restart"
        topo_path = @clima_artifact("topo_drag", ClimaComms.context(Y.c))
        orographic_info_rll = joinpath(topo_path, "topo_drag.res.nc")
        topo_info = regrid_OGW_info(Y, orographic_info_rll)
    elseif ogw.topo_info == "raw_topo"
        # TODO: right now this option may easily crash
        # because we did not incorporate any smoothing when interpolate back to model grid
        elevation_rll =
            AA.earth_orography_file_path(; context = ClimaComms.context(Y.c))
        radius =
            Spaces.topology(
                Spaces.horizontal_space(axes(Y.c)),
            ).mesh.domain.radius
        topo_info = compute_OGW_info(Y, elevation_rll, radius, γ, h_frac)
    elseif ogw.topo_info == "linear"
        # For user-defined analytical tests
        topo_info = initialize_drag_input_as_fields(Y, ogw.drag_input)
    else
        error("topo_info must be one of gfdl_restart, raw_topo, or linear")
    end

    return topo_info

end

function orographic_gravity_wave_cache(Y, ogw::OrographicGravityWave, topo_info)
    # For now, the initialisation of the cache is the same for all types of
    # orographic gravity wave drag parameterizations
    @assert Spaces.topology(Spaces.horizontal_space(axes(Y.c))).mesh.domain isa
            Domains.SphereDomain

    FT = Spaces.undertype(axes(Y.c))
    (; γ, ϵ, β, h_frac, ρscale, L0, a0, a1, Fr_crit) = ogw

    topo_level_idx = similar(Y.c.ρ, FT)

    # topo_info = fill(
    #     (;
    #         t11 = FT(Fields.field_values(loaded_topo_info.t11)),
    #         t12 = FT(Fields.field_values(loaded_topo_info.t12)),
    #         t21 = FT(Fields.field_values(loaded_topo_info.t21)),
    #         t22 = FT(Fields.field_values(loaded_topo_info.t22)),
    #         hmin = FT(Fields.field_values(loaded_topo_info.hmin)),
    #         hmax = FT(Fields.field_values(loaded_topo_info.hmax)),
    #     ),
    #     axes(Fields.level(Y.c.ρ, 1)),
    # )

    # topo_info = (; 

    # Prepare cache
    # QN: Is there a limit to how big the cache can be?
    # Limit is the GPU memory -- since the cache is stored anywhere on the device.
    return (;
        Fr_crit = Fr_crit,
        topo_γ = γ,
        topo_β = β,
        topo_ϵ = ϵ,
        topo_ρscale = ρscale,
        topo_L0 = L0,
        topo_a0 = a0,
        topo_a1 = a1,
        topo_ᶜτ_sat = Fields.Field(FT, axes(Y.c)),
        topo_ᶠτ_sat = Fields.Field(FT, axes(Y.f.u₃)),
        topo_ᶜVτ = Fields.Field(FT, axes(Y.c)),
        topo_ᶠVτ = Fields.Field(FT, axes(Y.f.u₃)),
        topo_τ_x = similar(Fields.level(Y.c.ρ, 1)),
        topo_τ_y = similar(Fields.level(Y.c.ρ, 1)),
        topo_τ_l = similar(Fields.level(Y.c.ρ, 1)),
        topo_τ_p = similar(Fields.level(Y.c.ρ, 1)),
        topo_τ_np = similar(Fields.level(Y.c.ρ, 1)),
        # topo_τ_x = Fields.Field(FT, axes(Y.f.u₃)),
        # topo_τ_y = Fields.Field(FT, axes(Y.f.u₃)),
        # topo_τ_l = Fields.Field(FT, axes(Y.f.u₃)),
        # topo_τ_p = Fields.Field(FT, axes(Y.f.u₃)),
        # topo_τ_np = Fields.Field(FT, axes(Y.f.u₃)),
        topo_U_sat = similar(Fields.level(Y.c.ρ, 1)),
        topo_FrU_sat = similar(Fields.level(Y.c.ρ, 1)),
        topo_FrU_max = similar(Fields.level(Y.c.ρ, 1)),
        topo_FrU_min = similar(Fields.level(Y.c.ρ, 1)),
        topo_FrU_clp = similar(Fields.level(Y.c.ρ, 1)),
        topo_tmp_1 = similar(Fields.level(Y.c.ρ, 1)),
        topo_tmp_2 = similar(Fields.level(Y.c.ρ, 1)),

        topo_d2Vτdz = Fields.Field(FT, axes(Y.c)),
        topo_L1 = Fields.Field(FT, axes(Y.c)),
        topo_U_k_field = Fields.Field(FT, axes(Y.c)),
        topo_level_idx = topo_level_idx,

        topo_base_Vτ = similar(Fields.level(Y.c.ρ, 1)),
        topo_k_pbl = similar(Fields.level(Y.c.ρ, 1)),
        topo_ᶜz_pbl = similar(Fields.level(Y.c.ρ, 1)),
        topo_ᶠz_pbl = similar(Fields.level(Y.f.u₃, half)),
        topo_k_pbl_values = similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT, FT, FT}),
        topo_info = topo_info,
        ᶜN = similar(Fields.level(Y.c.ρ, 1)),
        uforcing = similar(Y.c.u_phy),
        vforcing = similar(Y.c.v_phy),
        ᶜweights = similar(Y.c.ρ),
        ᶜdTdz = similar(Y.c.ρ),
        ᶜdτ_sat_dz = similar(Y.c.ρ)
    )

end

function orographic_gravity_wave_tendency!(Yₜ, Y, p, t, ::FullOrographicGravityWave)
    ᶜT = p.scratch.ᶜtemp_scalar
    (; params) = p
    (; ᶜts, ᶜp) = p.precomputed
    (; ᶜdTdz) = p.orographic_gravity_wave
    (;
        topo_k_pbl,
        topo_τ_x,
        topo_τ_y,
        topo_τ_l,
        topo_τ_p,
        topo_τ_np,
        topo_ᶠτ_sat,
        topo_ᶠVτ,
    ) = p.orographic_gravity_wave

    (; topo_U_sat, topo_FrU_sat, topo_FrU_max, topo_FrU_min, topo_FrU_clp) =
        p.orographic_gravity_wave
    (; hmax, hmin, t11, t12, t21, t22) = p.orographic_gravity_wave.topo_info
    FT = Spaces.undertype(axes(Y.c))

    # parameters
    thermo_params = CAP.thermodynamics_params(params)
    grav = FT(CAP.grav(params))
    cp_d = FT(CAP.cp_d(params))

    # z
    ᶜz = Fields.coordinate_field(Y.c).z
    ᶠz = Fields.coordinate_field(Y.f).z

    # get PBL info
    @. ᶜT = TD.air_temperature(thermo_params, ᶜts)
    Fields.bycolumn(axes(Y.c.ρ)) do colidx
        parent(topo_k_pbl[colidx]) .=
            get_pbl(ᶜp[colidx], ᶜT[colidx], ᶜz[colidx], grav, cp_d)
    end

    # buoyancy frequency at cell centers
    ᶜdTdz .= Geometry.WVector.(ᶜgradᵥ.(ᶠinterp.(ᶜT))).components.data.:1
    ᶜN = @. (grav / ᶜT) * (ᶜdTdz + grav / TD.cp_m(thermo_params, ᶜts)) # this is actually ᶜN^2
    @. ᶜN = ifelse(ᶜN < eps(FT), sqrt(eps(FT)), sqrt(abs(ᶜN))) # to avoid small numbers

    # prepare physical uv input variables for gravity_wave_forcing()
    u_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:1
    v_phy = Geometry.UVVector.(Y.c.uₕ).components.data.:2

    # compute base flux at k_pbl
    Fields.bycolumn(axes(Y.c.ρ)) do colidx
        calc_base_flux!(
            topo_τ_x[colidx],
            topo_τ_y[colidx],
            topo_τ_l[colidx],
            topo_τ_p[colidx],
            topo_τ_np[colidx],
            topo_U_sat[colidx],
            topo_FrU_sat[colidx],
            topo_FrU_max[colidx],
            topo_FrU_min[colidx],
            topo_FrU_clp[colidx],
            p,
            max(0, parent(hmax[colidx])[1]),
            max(0, parent(hmin[colidx])[1]),
            parent(t11[colidx])[1],
            parent(t12[colidx])[1],
            parent(t21[colidx])[1],
            parent(t22[colidx])[1],
            parent(Y.c.ρ[colidx]),
            parent(u_phy[colidx]),
            parent(v_phy[colidx]),
            parent(ᶜN[colidx]),
            Int(parent(topo_k_pbl[colidx])[1]),
        )
    end

    # buoyancy frequency at cell faces
    ᶠN = ᶠinterp.(ᶜN) # alternatively, can be computed from ᶠT and ᶠdTdz

    # compute saturation profile
    Fields.bycolumn(axes(Y.c.ρ)) do colidx
        calc_saturation_profile!(
            topo_ᶠτ_sat[colidx],
            topo_U_sat[colidx],
            topo_FrU_sat[colidx],
            topo_FrU_clp[colidx],
            topo_ᶠVτ[colidx],
            p,
            topo_FrU_max[colidx],
            topo_FrU_min[colidx],
            ᶠN[colidx],
            topo_τ_x[colidx],
            topo_τ_y[colidx],
            topo_τ_p[colidx],
            u_phy[colidx],
            v_phy[colidx],
            Y.c.ρ[colidx],
            ᶜp[colidx],
            Int(parent(topo_k_pbl[colidx])[1]),
        )
    end

    # a place holder to store physical forcing on uv
    uforcing = zeros(axes(u_phy))
    vforcing = zeros(axes(v_phy))

    # compute drag tendencies due to propagating part
    Fields.bycolumn(axes(Y.c.ρ)) do colidx
        calc_propagate_forcing!(
            uforcing[colidx],
            vforcing[colidx],
            topo_τ_x[colidx],
            topo_τ_y[colidx],
            topo_τ_l[colidx],
            topo_ᶠτ_sat[colidx],
            Y.c.ρ[colidx],
        )
    end

    # compute drag tendencies due to non-propagating part
    Fields.bycolumn(axes(Y.c.ρ)) do colidx
        calc_nonpropagating_forcing!(
            uforcing[colidx],
            vforcing[colidx],
            ᶠN[colidx],
            topo_ᶠVτ[colidx],
            ᶜp[colidx],
            topo_τ_x[colidx],
            topo_τ_y[colidx],
            topo_τ_l[colidx],
            topo_τ_np[colidx],
            ᶠz[colidx],
            ᶜz[colidx],
            Int(parent(topo_k_pbl[colidx])[1]),
            grav,
        )
    end

    # constrain forcing
    @. uforcing = max(FT(-3e-3), min(FT(3e-3), uforcing))
    @. vforcing = max(FT(-3e-3), min(FT(3e-3), vforcing))

    # convert to covariant vector and add to tendency
    @. Yₜ.c.uₕ +=
        Geometry.Covariant12Vector.(Geometry.UVVector.(uforcing, vforcing))
end

function calc_nonpropagating_forcing!(
    ᶜuforcing,
    ᶜvforcing,
    ᶠN,
    ᶠVτ,
    ᶠp,
    ᶜp,
    ᶠp_m1,
    τ_x,
    τ_y,
    τ_l,
    τ_np,
    ᶠz,
    ᶜz,
    z_pbl,
    ᶠdz,
    grav,
    ᶜweights
)
    FT = eltype(grav)

    # Initialize fields for z_ref and phase computation
    z_ref = similar(Fields.level(ᶠz, half), FT)

    # Convert type parameters to values before using in closure
    zero_val = FT(0)
    pi_val = FT(π)
    min_n_val = FT(0.7e-2)
    max_n_val = FT(1.7e-2)
    min_Vτ_val = FT(1.0)

    # Compute z_ref using column_reduce
    input = @. lazy(tuple(z_pbl, ᶠz, ᶠN, ᶠVτ, zero_val, pi_val, min_n_val, max_n_val, min_Vτ_val))

    Operators.column_reduce!(
        z_ref,
        input;
        init = (FT(0.0), FT(0.0), FT(0.0), false),
        transform = first
    ) do (z_ref_acc, ᶠz_pbl_acc, phase_acc, done), (z_pbl_itr, z_face, N_face, Vτ_face, zero_val, pi_val, min_n_val, max_n_val, min_Vτ_val)
        if done
            # If already done, return the accumulated values
            return (z_ref_acc, ᶠz_pbl_acc, phase_acc, true)
        end
        if (z_face > z_pbl_itr)
        # Only accumulate phase above z_pbl
            phase_acc += (z_face - z_pbl_itr) * 
                max(min_n_val, min(max_n_val, N_face)) / 
                max(min_Vτ_val, Vτ_face)
            
            # If phase exceeds π, stop and return current z_col as z_ref
            if phase_acc > pi_val
                return (z_face, z_pbl_itr, phase_acc, true)
            end
        end
        # Always return the accumulator tuple
        return (z_ref_acc, ᶠz_pbl_acc, phase_acc, false)
    end

    ᶠp_ref = similar(Fields.level(ᶠz, half), FT)

    eps_val = eps(FT)
    half_val = FT(0.5)
    nan_val = FT(NaN)
    
    input = @. lazy(tuple(z_ref, ᶠp, ᶠz, ᶠdz, eps_val, half_val))

    Operators.column_reduce!(
        ᶠp_ref,
        input;
        init = nan_val
    ) do ᶠp_ref, (z_ref, ᶠp, ᶠz, ᶠdz, eps_val, half_val)
        if abs(ᶠz - z_ref) < (half_val * ᶠdz + eps_val)
            if isnan(ᶠp_ref)
                ᶠp_ref = ᶠp
            end
        end
        return ᶠp_ref
    end
    
    mask = Fields.Field(Bool, axes(ᶠz))
    @. mask = (ᶠz .> z_pbl) .&& (ᶠz .<= z_ref)
    L2 = Operators.LeftBiasedF2C(;)
    mask = L2.(mask)


    ᶠweights = ᶠp .- ᶠp_ref
    weights = ᶜinterp.(ᶠweights)
    f_diff = ᶠp_m1 .- ᶠp
    f_diff = ᶜinterp.(f_diff)

    wtsum_field = @. ifelse(mask != 0, f_diff / weights, 0.0)

    parent(ᶜweights) .= parent(weights .* mask)

    wtsum = similar(Fields.level(ᶜuforcing, 1), FT)

    Operators.column_reduce!(
        wtsum, 
        wtsum_field;
        init = FT(0)
    ) do acc, wtsum_field
        return acc + wtsum_field
    end

    if any(isnan, parent(wtsum)) || any(x -> x == 0, parent(wtsum))
    @warn "wtsum contains invalid values!"
    end

    # compute drag
    @. ᶜuforcing += grav * τ_x * τ_np / τ_l / wtsum * ᶜweights
    @. ᶜvforcing += grav * τ_y * τ_np / τ_l / wtsum * ᶜweights

end

function calc_propagate_forcing!(ᶜuforcing, ᶜvforcing, τ_x, τ_y, τ_l, τ_sat, ᶜρ, dτ_sat_dz)
    # QN: Again, I can't inline this, right?
    # Adding the dollar sign tells @. to stop before ᶜddz(...)
    # This is necessary as we are lazily evaluating the expression
    # dτ_sat_dz_lazy = lazy.(ᶜddz(τ_sat))
    # @. dτ_sat_dz = dτ_sat_dz_lazy

    parent(dτ_sat_dz) .= parent(Geometry.WVector.(ᶜgradᵥ.(τ_sat)).components.data.:1)

    @. ᶜuforcing -= τ_x / τ_l / ᶜρ * dτ_sat_dz
    @. ᶜvforcing -= τ_y / τ_l / ᶜρ * dτ_sat_dz
    return nothing
end

function get_pbl(ᶜp, ᶜT, ᶜz, grav, cp_d)
    FT = eltype(cp_d)
    
    # Initialize result field to hold k_pbl values
    result = similar(Fields.level(ᶜp, 1), FT)
    
    # Get surface values (first level values)
    p_sfc = Fields.level(ᶜp, 1)
    T_sfc = Fields.level(ᶜT, 1)
    z_sfc = Fields.level(ᶜz, 1)

    # Convert constants to the appropriate type beforehand
    half_val = FT(0.5)
    temp_offset = FT(1.5)
    grav_val = FT(grav)
    cp_d_val = FT(cp_d)

    # Create a lazy tuple of inputs for column_reduce
    input = @. lazy(tuple(ᶜp, ᶜT, ᶜz, p_sfc, T_sfc, z_sfc, half_val, temp_offset, grav_val, cp_d_val))

    # Perform the column reduction
    Operators.column_reduce!(
        result,
        input;
        init = (1, 2),  # (k_pbl, current_level)
        transform = first # Extract just the k_pbl value
    ) do (k_pbl, level_idx), (p_col, T_col, z_col, p_sfc, T_sfc, z_sfc, half_val, temp_offset, grav_val, cp_d_val)
        
        # Check conditions
        p_threshold = p_col >= (half_val * p_sfc)
        T_threshold = (T_sfc + temp_offset - T_col) > (grav_val / cp_d_val * (z_col - z_sfc))
        
        # If both conditions are met, update k_pbl
        if p_threshold && T_threshold
            k_pbl = level_idx
        end
        
        # Move to next level
        return (k_pbl, level_idx + 1)
    end

    return result
end

function get_pbl_z(ᶜp, ᶜT, ᶜz, grav, cp_d)
    FT = eltype(cp_d)
    
    # Initialize result field to hold z_pbl values
    result = similar(Fields.level(ᶜp, 1), FT)
    
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
    input = @. lazy(tuple(ᶜp, ᶜT, ᶜz, p_sfc, T_sfc, z_sfc, grav_val, cp_d_val, half_val, temp_offset, zero_val))

    # Perform the column reduction
    Operators.column_reduce!(
        result,
        input;
        init = FT(0),
        transform = first # Extract just the z_pbl value
    ) do z_pbl, (p_col, T_col, z_col, p_sfc, T_sfc, z_sfc, grav_val, cp_d_val, half_val, temp_offset, zero_val)
        
        if z_pbl == zero_val
            z_pbl = z_sfc
        end
        # Check conditions
        p_threshold = p_col >= (half_val * p_sfc)
        T_threshold = (T_sfc + temp_offset - T_col) > (grav_val / cp_d_val * (z_col - z_sfc))
        
        # If both conditions are met, update z_pbl to current height
        if p_threshold && T_threshold
            z_pbl = z_col
        end
        
        # Move to next level
        return z_pbl
    end

    return result
end

function field_shiftface_down!(ᶠexample_field, ᶠshifted_field, Boundary_value)
    L1 = Operators.LeftBiasedC2F(; bottom = Operators.SetValue(Boundary_value))
    L2 = Operators.LeftBiasedF2C(;)
    ᶠshifted_field .= L1.(L2.(ᶠexample_field))
end

function calc_base_flux!(
    τ_x,
    τ_y,
    τ_l,
    τ_p,
    τ_np,
    U_sat,
    FrU_sat,
    FrU_max,
    FrU_min,
    FrU_clp,
    Vτ,
    Fr_max,
    Fr_min,
    ogw_params,
    hmax,
    hmin,
    t11,
    t12,
    t21,
    t22,
    ᶜρ,
    u_phy,
    v_phy,
    ᶜN,
    z_pbl,
    k_pbl_values
)
    # Extract parameters
    # QN: When should I pass an array as argument, and when should I extract them from cache?
    # Extract parameters from tuple
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
    
    FT = eltype(Fr_crit)
    γ = topo_γ
    β = topo_β
    ϵ = topo_ϵ
    
    # Create an input tuple for column_reduce to extract k_pbl level data
    # input = @. lazy(tuple(ᶜρ, u_phy, v_phy, ᶜN, k_pbl))
    
    # Use column_reduce to extract values at k_pbl level
    # k_pbl_values = similar(hmax, Tuple{FT, FT, FT, FT})
    # Operators.column_reduce!(
    #     k_pbl_values,
    #     input;
    #     init = (1, nothing, nothing, nothing, nothing),  # Start with level index 1
    #     transform = x -> (x[2], x[3], x[4], x[5])        # Extract just the values of interest
    # ) do (level_idx, ρ_acc, u_acc, v_acc, N_acc), (ρ, u, v, N, k_level)
               
    #     # If we're at the target level, extract values
    #     if level_idx == k_level
    #         return (level_idx + 1, ρ, u, v, N)
    #     # Otherwise, just increment the level counter
    #     else
    #         return (level_idx + 1, ρ_acc, u_acc, v_acc, N_acc)
    #     end
    # end
    ᶜz = Fields.coordinate_field(ᶜρ).z
    input = @. lazy(tuple(ᶜρ, u_phy, v_phy, ᶜN, ᶜz, z_pbl))

    Operators.column_reduce!(
        k_pbl_values,
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
    
    # Extract values from the tuple
    # QN: Is this a view or a copy?
    # These are views
    ρ_pbl = k_pbl_values.:1
    u_pbl = k_pbl_values.:2
    v_pbl = k_pbl_values.:3
    N_pbl = k_pbl_values.:4
    
    # Calculate τ components
    @. τ_x = ρ_pbl * N_pbl * (t11 * u_pbl + t21 * v_pbl)
    @. τ_y = ρ_pbl * N_pbl * (t12 * u_pbl + t22 * v_pbl)
    
    # Calculate Vτ using field operations
    @. Vτ = max(
        eps(FT),
        -(u_pbl * τ_x + v_pbl * τ_y) / max(eps(FT), sqrt(τ_x^2 + τ_y^2))
    )
    
    # Calculate Froude numbers
    @. Fr_max = max(FT(0), hmax) * N_pbl / Vτ
    @. Fr_min = max(FT(0), hmin) * N_pbl / Vτ
    
    # Calculate U_sat
    @. U_sat = sqrt.(ρ_pbl / topo_ρscale * @. Vτ^3 / N_pbl / topo_L0)
    
    # Calculate FrU values
    @. FrU_sat = Fr_crit * U_sat
    @. FrU_min = Fr_min * U_sat
    @. FrU_max = max(Fr_max * U_sat, FrU_min + eps(FT))
    @. FrU_clp = min(FrU_max, max(FrU_min, FrU_sat))
    
    # Calculate drag components
    @. τ_l = ((FrU_max)^(2 + γ - ϵ) - (FrU_min)^(2 + γ - ϵ)) / (2 + γ - ϵ)
    
    # Calculate propagating drag
    @. τ_p = topo_a0 * (
        (FrU_clp^(2 + γ - ϵ) - FrU_min^(2 + γ - ϵ)) / (2 + γ - ϵ) +
        FrU_sat^(β + 2) * (FrU_max^(γ - ϵ - β) - FrU_clp^(γ - ϵ - β)) / (γ - ϵ - β)
    )
    
    # Calculate non-propagating drag
    @. τ_np = topo_a1 * U_sat / (1 + β) * (
        (FrU_max^(1 + γ - ϵ) - FrU_clp^(1 + γ - ϵ)) / (1 + γ - ϵ) -
        FrU_sat^(β + 1) * (FrU_max^(γ - ϵ - β) - FrU_clp^(γ - ϵ - β)) / (γ - ϵ - β)
    )
    
    # Apply scaling
    @. τ_np = τ_np / max(Fr_crit, Fr_max)
    
    return nothing
end

function calc_saturation_profile!(
    ᶜτ_sat,
    ᶠτ_sat,
    U_sat, 
    FrU_sat,
    FrU_clp,
    ᶜVτ,
    ᶠVτ,
    ogw_params,
    FrU_max,
    FrU_min,
    ᶜN,
    τ_x,
    τ_y,
    τ_p,
    u_phy,
    v_phy,
    ᶜρ,
    ᶜp,
    z_pbl,
    d2Vτdz,
    L1,
    U_k_field,
    level_idx,
)
    # Extract parameters from tuple
    (; Fr_crit, topo_ρscale, topo_L0, topo_a0, topo_γ, topo_β, topo_ϵ) = ogw_params

    FT = eltype(Fr_crit)
    γ = topo_γ
    β = topo_β
    ϵ = topo_ϵ
    
    # Calculate Vτ at cell faces using field operations
    @. ᶜVτ = max(
        eps(FT),
        (
            -(u_phy * τ_x + v_phy * τ_y) / max(eps(FT), sqrt(τ_x^2 + τ_y^2))
        )
    )
    
    # Calculate derivatives for ᶠd2Vτdz
    # QN: Is the Julia compiler smart enough to inline these?
    # Lazy this (done)
    d2udz = lazy.(ᶜd2dz2(u_phy))
    d2vdz = lazy.(ᶜd2dz2(v_phy))
    # Calculate derivative for L1; tmp_field_2 == d2Vτdz
    @. d2Vτdz = max(
        eps(FT),
        -(d2udz * τ_x + d2vdz * τ_y) / max(eps(FT), sqrt(τ_x^2 + τ_y^2))
    )
    
    # Calculate tmp_field_1 == L1
    # Here on the RHS, tmp_field_2 == d2Vτdz
    @. L1 = topo_L0 * max(FT(0.5), min(FT(2.0), FT(1.0) - FT(2.0) * ᶜVτ * d2Vτdz / ᶜN^2))
    
    # Store original values for later use
    # To remove
    FrU_clp0 = copy(FrU_clp)
    FrU_sat0 = copy(FrU_sat)
    
    # Create field for U_k calculation
    # Here, U_k == tmp_field_1
    @. U_k_field = sqrt(ᶜρ / topo_ρscale * ᶜVτ^3 / ᶜN / L1)
    
    # Prepare a level index field to help with operations at specific levels
    for i in 1:Spaces.nlevels(axes(ᶜρ))
        fill!(Fields.level(level_idx, i), i)
    end

    # Get height coordinate for comparison
    ᶜz = Fields.coordinate_field(ᶜρ).z
    
    z_surf = Fields.level(ᶜz, 1)
    # Create combined input for column_accumulate
    input = @. lazy(tuple(
        FrU_clp0,
        FrU_sat0,
        U_k_field,
        FrU_max,
        FrU_min,
        z_surf,
        ᶜz, 
        z_pbl,
        topo_a0,
        τ_p, 
        U_sat
    ))
    
    # Initialize the result field with τ_p at the lowest face
    fill!(ᶜτ_sat, 0.0)
    # Fields.level(τ_sat, half) .= parent(τ_p )
    # L2 = Operators.LeftBiasedF2C(;)

    # Fields.level(ᶜτ_sat, 1) .= τ_p
    Operators.column_accumulate!(
        ᶜτ_sat,
        input;
        init = (FT(0.0), FT(0.0)),
        transform = first,
    ) do (tau_sat_val, U_sat_val),
        (FrU_clp0, FrU_sat0, U, FrU_max, FrU_min, z_surf, z_col, z_target, topo_a0, τ_p, U_sat)

        if z_col == z_surf
            U_sat_val = U_sat
        end

        U_sat_val = min(U_sat_val, U)
        local_FrU_sat = Fr_crit * U_sat_val  # Use local variable instead
        local_FrU_clp = min(FrU_max, max(FrU_min, local_FrU_sat))  # Use local variable instead

        if z_col <= z_target
            tau_sat_val = τ_p
        else
            tau_sat_val = topo_a0 * (
            (local_FrU_clp^(2 + γ - ϵ) - FrU_min^(2 + γ - ϵ)) / (2 + γ - ϵ) +
            local_FrU_sat^2 * FrU_sat0^β *
                (FrU_max^(γ - ϵ - β) - FrU_clp0^(γ - ϵ - β)) / (γ - ϵ - β) +
            local_FrU_sat^2 *
                (FrU_clp0^(γ - ϵ) - local_FrU_clp^(γ - ϵ)) / (γ - ϵ)
            )
        end

        return (tau_sat_val, U_sat_val)
    end
    
    top_values = Fields.level(ᶜτ_sat, Spaces.nlevels(axes(ᶜτ_sat)))
    p_surf = Fields.level(ᶜp, 1)
    p_top = Fields.level(ᶜp, Spaces.nlevels(axes(ᶜp)))

    zero_val = FT(0.0)

    input = @. lazy(tuple(
        top_values,
        ᶜτ_sat,
        p_surf,
        p_top,
        ᶜp,
        zero_val,
    ))

    Operators.column_accumulate!(
        ᶜτ_sat,
        input;
        init = FT(0.0),
        transform = identity,
    ) do τ_sat_val,
        (top_values, ᶜτ_sat, p_surf, p_top, ᶜp, zero_val)

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



ᶜd2dz2(ᶜscalar) =
    lazy.(Geometry.WVector.(ᶜgradᵥ.(ᶠddz(ᶜscalar))).components.data.:1)

ᶜddz(ᶠscalar) = lazy.(Geometry.WVector.(ᶜgradᵥ.(ᶠscalar)).components.data.:1)

ᶠddz(ᶜscalar) = lazy.(Geometry.WVector.(ᶠgradᵥ.(ᶜscalar)).components.data.:1)