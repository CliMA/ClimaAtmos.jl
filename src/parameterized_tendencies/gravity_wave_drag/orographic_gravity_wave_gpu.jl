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

function orographic_gravity_wave_cache(Y, ogw::OrographicGravityWave)
    # For now, the initialisation of the cache is the same for all types of
    # orographic gravity wave drag parameterizations

    @assert Spaces.topology(Spaces.horizontal_space(axes(Y.c))).mesh.domain isa
            Domains.SphereDomain

    FT = Spaces.undertype(axes(Y.c))
    (; γ, ϵ, β, h_frac, ρscale, L0, a0, a1, Fr_crit) = ogw

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

    # Prepare cache
    # QN: Is there a limit to how big the cache can be?
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
        topo_tmp_field_1 = Fields.Field(FT, axes(Y.c)),
        topo_tmp_field_2 = Fields.Field(FT, axes(Y.c)),
        # QN: What is the difference between:
        # topo_tmp_f = similar(Fields.level(Y.f, half)),
        # and the line below?
        topo_tmp_f = similar(Fields.level(Y.f, half)),
        topo_base_Vτ = similar(Fields.level(Y.c.ρ, 1)),
        topo_k_pbl = similar(Fields.level(Y.c.ρ, 1)),
        topo_k_pbl_values = similar(Fields.level(Y.c.ρ, 1), Tuple{FT, FT, FT, FT}),
        topo_info = topo_info,
        ᶜN = similar(Fields.level(Y.c.ρ, 1)),
        ᶜdTdz = similar(Y.c.ρ),
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
    parent(ᶜdTdz) .= parent(Geometry.WVector.(ᶜgradᵥ.(ᶠinterp.(ᶜT))))
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
    ᶜp,
    τ_x,
    τ_y,
    τ_l,
    τ_np,
    ᶠz,
    ᶜz,
    k_pbl,
    grav,
)
    FT = eltype(grav)
    ᶠp = ᶠinterp.(ᶜp)
    # compute k_ref: the upper bound for nonpropagating drag to function
    phase = FT(0)
    zlast = parent(Fields.level(ᶠz, k_pbl + half))[1]
    k_ref = k_pbl
    for k in (k_pbl + 1):length(parent(ᶜz))
        phase +=
            (parent(ᶜz)[k + 1] - zlast) *
            max(FT(0.7e-2), min(FT(1.7e-2), parent(ᶠN)[k])) /
            max(FT(1.0), parent(ᶠVτ)[k])
        if phase > π
            k_ref = k
            break
        end
    end

    # compute weights
    weights = FT(0) .* ᶜz
    wtsum = FT(0)
    for k in k_pbl:k_ref
        tmp = Fields.level(weights, k)
        parent(tmp) .= parent(ᶜp)[k] - parent(ᶠp)[k_ref]
        wtsum += (parent(ᶠp)[k - 1] - parent(ᶠp)[k]) / parent(weights)[k]
    end

    # compute drag
    @. ᶜuforcing += grav * τ_x * τ_np / τ_l / wtsum * weights
    @. ᶜvforcing += grav * τ_y * τ_np / τ_l / wtsum * weights

end

function calc_propagate_forcing!(ᶜuforcing, ᶜvforcing, τ_x, τ_y, τ_l, τ_sat, ᶜρ)
    dτdz = ᶜddz(τ_sat)
    @. ᶜuforcing -= τ_x / τ_l / ᶜρ * dτdz
    @. ᶜvforcing -= τ_y / τ_l / ᶜρ * dτdz
    return nothing
end

function get_pbl(ᶜp, ᶜT, ᶜz, grav, cp_d)
    FT = eltype(cp_d)
    idx =
        (parent(ᶜp) .>= (FT(0.5) * parent(ᶜp)[1])) .& (
            (parent(ᶜT)[1] + FT(1.5) .- parent(ᶜT)) .>
            (grav / cp_d * (parent(ᶜz) .- parent(ᶜz)[1]))
        )
    # parent(ᶜp) .>= (FT(0.5) * parent(ᶜp)[1]) follows the criterion in GFDL codes
    # that the lowest layer that is geq to half of pressure at first face level; while
    # in our code, when interpolate from center to face, the first face level inherits
    # values at the first center level
    ci = findlast(idx)::CartesianIndex
    return 1 + ci[1]
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
    p,
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
    k_pbl,
    k_pbl_values
)
    # Extract parameters
    # QN: When should I pass an array as argument, and when should I extract them from cache?
    (;
        Fr_crit,
        topo_ρscale,
        topo_L0,
        topo_a0,
        topo_a1,
        topo_γ,
        topo_β,
        topo_ϵ,
    ) = p.orographic_gravity_wave
    
    FT = eltype(Fr_crit)
    γ = topo_γ
    β = topo_β
    ϵ = topo_ϵ
    
    # Create an input tuple for column_reduce to extract k_pbl level data
    input = @. lazy(tuple(ᶜρ, u_phy, v_phy, ᶜN, k_pbl))
    
    # Use column_reduce to extract values at k_pbl level
    # k_pbl_values = similar(hmax, Tuple{FT, FT, FT, FT})
    Operators.column_reduce!(
        k_pbl_values,
        input;
        init = (1, nothing, nothing, nothing, nothing),  # Start with level index 1
        transform = x -> (x[2], x[3], x[4], x[5])        # Extract just the values of interest
    ) do (level_idx, ρ_acc, u_acc, v_acc, N_acc), (ρ, u, v, N, k_level)
               
        # If we're at the target level, extract values
        if level_idx == k_level
            return (level_idx + 1, ρ, u, v, N)
        # Otherwise, just increment the level counter
        else
            return (level_idx + 1, ρ_acc, u_acc, v_acc, N_acc)
        end
    end
    
    # Extract values from the tuple
    # QN: Is this a view or a copy?
    ρ_pbl = @. k_pbl_values.:1
    u_pbl = @. k_pbl_values.:2
    v_pbl = @. k_pbl_values.:3
    N_pbl = @. k_pbl_values.:4
    
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
    tmp_field_1,
    tmp_field_2,
    ᶜVτ,
    ᶠVτ,
    p,
    FrU_max,
    FrU_min,
    ᶜN,
    τ_x,
    τ_y,
    τ_p,
    u_phy,
    v_phy,
    ᶜp,
    ᶠp,
    k_pbl
)
    # Extract parameters
    (; Fr_crit, topo_ρscale, topo_L0, topo_a0, topo_γ, topo_β, topo_ϵ) = p.orographic_gravity_wave
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
    d2udz = (ᶜd2dz2(u_phy, p))
    d2vdz = (ᶜd2dz2(v_phy, p))
    
    # Calculate derivative for L1; tmp_field_2 == d2Vτdz
    @. tmp_field_2 = max(
        eps(FT),
        -(d2udz * τ_x + d2vdz * τ_y) / max(eps(FT), sqrt(τ_x^2 + τ_y^2))
    )
    # @Main.infiltrate
    
    # Calculate tmp_field_1 == L1
    # Here on the RHS, tmp_field_1 == Vτ, tmp_field_2 == d2Vτdz
    @. tmp_field_1 = topo_L0 * max(FT(0.5), min(FT(2.0), FT(1.0) - FT(2.0) * tmp_field_1 * tmp_field_2 / ᶜN^2))
    
    # Store original values for later use
    # To remove
    FrU_clp0 = copy(FrU_clp)
    FrU_sat0 = copy(FrU_sat)
    
    # Create field for U_k calculation
    # Here, U_k == tmp_field_1
    @. tmp_field_1 = sqrt(ᶜp / topo_ρscale * ᶜVτ^3 / ᶜN / tmp_field_1)
    
    # Prepare a level index field to help with operations at specific levels
    # level_idx = similar(Vτ, FT)
    # Here, tmp_field_2 == level_idx
    for i in 1:Spaces.nlevels(axes(ᶜVτ))
        fill!(Fields.level(tmp_field_2, i), i)
    end
    
    # Create combined input for column_accumulate
    input = @. lazy(tuple(
        FrU_clp0,
        FrU_sat0,
        tmp_field_1,
        FrU_max,
        FrU_min,
        τ_p,
        tmp_field_2,
        k_pbl,
        topo_a0
    ))
    
    # Initialize the result field with τ_p at the lowest face
    fill!(ᶜτ_sat, 0.0)
    # @Main.infiltrate
    # Fields.level(τ_sat, half) .= parent(τ_p )
    # L2 = Operators.LeftBiasedF2C(;)

    Fields.level(ᶜτ_sat, 1) .= τ_p

    Operators.column_accumulate!(
        ᶜτ_sat,
        input;
        init = (FT(0.0), U_sat),
        transform = first,
    ) do (ᶜτ_sat, U_sat),
        (FrU_clp0, FrU_sat0, U, FrU_max, FrU_min, τ_p, level_idx, k_pbl, topo_a0)

        if level_idx == 1
            return (U, Fr_crit * U, FrU_sat0, τ_p)
        end

        U_sat = min(U_sat, U)
        FrU_sat = Fr_crit * U_sat
        FrU_clp = min(FrU_max, max(FrU_min, FrU_sat))

        if level_idx <= k_pbl
            ᶜτ_sat = τ_p
        else
            term1 = (FrU_clp^(2 + γ - ϵ) - FrU_min^(2 + γ - ϵ)) / (2 + γ - ϵ)
            term2 = FrU_sat^2 * FrU_sat0^β *
                (FrU_max^(γ - ϵ - β) - FrU_clp0^(γ - ϵ - β)) / (γ - ϵ - β)
            term3 = FrU_sat^2 * (FrU_clp0^(γ - ϵ) - FrU_clp^(γ - ϵ)) / (γ - ϵ)
            ᶜτ_sat = topo_a0 * (term1 + term2 + term3)
        end

        return (ᶜτ_sat, U_sat)
    end
    
    # Apply top correction using field operations
    # QN: How can I GPUify this? Tried for hours!
    ᶠτ_sat .= ᶠinterp.(ᶜτ_sat)
    ᶠVτ .= ᶠinterp.(ᶜVτ)
    
    if parent(ᶠτ_sat)[end] > FT(0)
        # QN: Why can't I use ᶠp .= ᶠinterp.(ᶜp) here?
        # QN: Is this a copy operation then?
        ᶠp = ᶠinterp.(ᶜp)
        ᶠτ_sat .-=
            parent(ᶠτ_sat)[end] .* (parent(ᶠp)[1] .- ᶠp) ./
            (parent(ᶠp)[1] .- parent(ᶠp)[end])
    end

    return nothing
end



ᶜd2dz2(ᶜscalar, p) =
    Geometry.WVector.(ᶜgradᵥ.(ᶠddz(ᶜscalar, p))).components.data.:1

ᶜddz(ᶠscalar) = Geometry.WVector.(ᶜgradᵥ.(ᶠscalar)).components.data.:1

ᶠddz(ᶜscalar, p) = Geometry.WVector.(ᶠgradᵥ.(ᶜscalar)).components.data.:1
