
function update_cloud_frac(edmf::EDMFModel, grid::Grid, state::State)
    # update grid-mean cloud fraction and cloud cover
    aux_bulk = center_aux_bulk(state)
    aux_gm = center_aux_grid_mean(state)
    aux_en = center_aux_environment(state)
    a_up_bulk = aux_bulk.area
    @inbounds for k in real_center_indices(grid) # update grid-mean cloud fraction and cloud cover
        aux_gm.cloud_fraction[k] = aux_en.area[k] * aux_en.cloud_fraction[k] + a_up_bulk[k] * aux_bulk.cloud_fraction[k]
    end
end

function compute_turbconv_tendencies!(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    param_set::APS,
    surf::SurfaceBase,
    Δt::Real,
)
    compute_up_tendencies!(edmf, grid, state, param_set, surf)
    compute_en_tendencies!(edmf, grid, state, param_set, Val(:tke), Val(:ρatke))

    if edmf.thermo_covariance_model isa PrognosticThermoCovariances
        compute_en_tendencies!(edmf, grid, state, param_set, Val(:Hvar), Val(:ρaHvar))
        compute_en_tendencies!(edmf, grid, state, param_set, Val(:QTvar), Val(:ρaQTvar))
        compute_en_tendencies!(edmf, grid, state, param_set, Val(:HQTcov), Val(:ρaHQTcov))
    end

    return nothing
end
function compute_sgs_flux!(edmf::EDMFModel, grid::Grid, state::State, surf::SurfaceBase, param_set::APS)
    N_up = n_updrafts(edmf)
    tendencies_gm = center_tendencies_grid_mean(state)
    FT = float_type(state)
    prog_gm = center_prog_grid_mean(state)
    aux_gm = center_aux_grid_mean(state)
    prog_gm_f = face_prog_grid_mean(state)
    aux_gm_f = face_aux_grid_mean(state)
    aux_en = center_aux_environment(state)
    aux_en_f = face_aux_environment(state)
    aux_up = center_aux_updrafts(state)
    aux_tc_f = face_aux_turbconv(state)
    aux_up_f = face_aux_updrafts(state)
    ρ_f = aux_gm_f.ρ
    ρ_c = prog_gm.ρ
    p_c = aux_gm.p
    kf_surf = kf_surface(grid)
    kc_surf = kc_surface(grid)
    massflux = aux_tc_f.massflux
    massflux_h = aux_tc_f.massflux_h
    massflux_qt = aux_tc_f.massflux_qt
    aux_tc = center_aux_turbconv(state)

    wvec = CC.Geometry.WVector
    ∇c = CCO.DivergenceF2C()

    # TODO: we shouldn't need to call parent here
    a_en = aux_en.area
    w_en = aux_en_f.w
    w_gm = prog_gm_f.w
    h_tot_gm = aux_gm.h_tot
    q_tot_gm = aux_gm.q_tot
    q_tot_en = aux_en.q_tot
    a_en_bcs = a_en_boundary_conditions(surf, edmf)
    Ifae = CCO.InterpolateC2F(; a_en_bcs...)
    If = CCO.InterpolateC2F(; bottom = CCO.SetValue(FT(0)), top = CCO.SetValue(FT(0)))
    Ic = CCO.InterpolateF2C()

    # compute total enthalpies
    ts_en = center_aux_environment(state).ts
    ts_gm = center_aux_grid_mean(state).ts
    @. h_tot_gm = total_enthalpy(param_set, prog_gm.ρe_tot / ρ_c, ts_gm)
    # Compute the mass flux and associated scalar fluxes
    @. massflux = ρ_f * Ifae(a_en) * (w_en - toscalar(w_gm))
    @. massflux_h = ρ_f * Ifae(a_en) * (w_en - toscalar(w_gm)) * (If(aux_en.h_tot) - If(h_tot_gm))
    @. massflux_qt = ρ_f * Ifae(a_en) * (w_en - toscalar(w_gm)) * (If(q_tot_en) - If(q_tot_gm))
    @inbounds for i in 1:N_up
        aux_up_f_i = aux_up_f[i]
        aux_up_i = aux_up[i]
        a_up_bcs = a_up_boundary_conditions(surf, edmf, i)
        Ifau = CCO.InterpolateC2F(; a_up_bcs...)
        a_up = aux_up[i].area
        w_up_i = aux_up_f[i].w
        q_tot_up = aux_up_i.q_tot
        h_tot_up = aux_up_i.h_tot
        @. aux_up_f[i].massflux = ρ_f * Ifau(a_up) * (w_up_i - toscalar(w_gm))
        @. massflux_h += ρ_f * (Ifau(a_up) * (w_up_i - toscalar(w_gm)) * (If(h_tot_up) - If(h_tot_gm)))
        @. massflux_qt += ρ_f * (Ifau(a_up) * (w_up_i - toscalar(w_gm)) * (If(q_tot_up) - If(q_tot_gm)))
    end

    if edmf.moisture_model isa NonEquilibriumMoisture
        massflux_en = aux_tc_f.massflux_en
        massflux_ql = aux_tc_f.massflux_ql
        massflux_qi = aux_tc_f.massflux_qi
        q_liq_en = aux_en.q_liq
        q_ice_en = aux_en.q_ice
        q_liq_gm = prog_gm.q_liq
        q_ice_gm = prog_gm.q_ice
        @. massflux_en = ρ_f * Ifae(a_en) * (w_en - toscalar(w_gm))
        @. massflux_ql = massflux_en * (If(q_liq_en) - If(q_liq_gm))
        @. massflux_qi = massflux_en * (If(q_ice_en) - If(q_ice_gm))
        @inbounds for i in 1:N_up
            aux_up_f_i = aux_up_f[i]
            aux_up_i = aux_up[i]
            q_liq_up = aux_up_i.q_liq
            q_ice_up = aux_up_i.q_ice
            massflux_up_i = aux_up_f[i].massflux
            @. massflux_ql += massflux_up_i * (If(q_liq_up) - If(q_liq_gm))
            @. massflux_qi += massflux_up_i * (If(q_ice_up) - If(q_ice_gm))
        end
        massflux_ql[kf_surf] = 0
        massflux_qi[kf_surf] = 0
    end

    massflux_h[kf_surf] = 0
    massflux_qt[kf_surf] = 0

    massflux_tendency_h = aux_tc.massflux_tendency_h
    massflux_tendency_qt = aux_tc.massflux_tendency_qt
    # Compute the  mass flux tendencies
    # Adjust the values of the grid mean variables
    # Prepare the output
    @. massflux_tendency_h = -∇c(wvec(massflux_h)) / ρ_c
    @. massflux_tendency_qt = -∇c(wvec(massflux_qt)) / ρ_c

    diffusive_flux_h = aux_tc_f.diffusive_flux_h
    diffusive_flux_qt = aux_tc_f.diffusive_flux_qt
    diffusive_flux_uₕ = aux_tc_f.diffusive_flux_uₕ

    sgs_flux_h_tot = aux_gm_f.sgs_flux_h_tot
    sgs_flux_q_tot = aux_gm_f.sgs_flux_q_tot
    sgs_flux_uₕ = aux_gm_f.sgs_flux_uₕ

    @. sgs_flux_h_tot = diffusive_flux_h + massflux_h
    @. sgs_flux_q_tot = diffusive_flux_qt + massflux_qt
    @. sgs_flux_uₕ = diffusive_flux_uₕ # + massflux_u

    # apply surface BC as SGS flux at lowest level
    lg_surf = CC.Fields.local_geometry_field(axes(ρ_f))[kf_surf]
    sgs_flux_h_tot[kf_surf] = surf.ρe_tot_flux
    sgs_flux_q_tot[kf_surf] = surf.ρq_tot_flux
    sgs_flux_uₕ[kf_surf] =
        CCG.Covariant3Vector(wvec(FT(1)), lg_surf) ⊗
        CCG.Covariant12Vector(CCG.UVVector(surf.ρu_flux, surf.ρv_flux), lg_surf)

    if edmf.moisture_model isa NonEquilibriumMoisture
        massflux_tendency_ql = aux_tc.massflux_tendency_ql
        massflux_tendency_qi = aux_tc.massflux_tendency_qi

        @. massflux_tendency_ql = -∇c(wvec(massflux_ql)) / ρ_c
        @. massflux_tendency_qi = -∇c(wvec(massflux_qi)) / ρ_c

        diffusive_flux_ql = aux_tc_f.diffusive_flux_ql
        diffusive_flux_qi = aux_tc_f.diffusive_flux_qi

        sgs_flux_q_liq = aux_gm_f.sgs_flux_q_liq
        sgs_flux_q_ice = aux_gm_f.sgs_flux_q_ice

        @. sgs_flux_q_liq = diffusive_flux_ql + massflux_ql
        @. sgs_flux_q_ice = diffusive_flux_qi + massflux_qi

        sgs_flux_q_liq[kf_surf] = surf.ρq_liq_flux
        sgs_flux_q_ice[kf_surf] = surf.ρq_ice_flux
    end

    return nothing
end

function compute_diffusive_fluxes(edmf::EDMFModel, grid::Grid, state::State, surf::SurfaceBase, param_set::APS)
    FT = float_type(state)
    aux_bulk = center_aux_bulk(state)
    aux_tc_f = face_aux_turbconv(state)
    aux_en_f = face_aux_environment(state)
    aux_en = center_aux_environment(state)
    aux_gm = center_aux_grid_mean(state)
    aux_gm_f = face_aux_grid_mean(state)
    KM = center_aux_turbconv(state).KM
    KH = center_aux_turbconv(state).KH
    aeKM = center_aux_turbconv(state).ϕ_temporary
    aeKH = center_aux_turbconv(state).ψ_temporary
    prog_gm_uₕ = grid_mean_uₕ(state)

    ρ_f = aux_gm_f.ρ
    a_en = aux_en.area
    @. aeKM = a_en * KM
    @. aeKH = a_en * KH
    kc_surf = kc_surface(grid)
    kc_toa = kc_top_of_atmos(grid)
    kf_surf = kf_surface(grid)
    prog_gm = center_prog_grid_mean(state)
    IfKH = CCO.InterpolateC2F(; bottom = CCO.SetValue(aeKH[kc_surf]), top = CCO.SetValue(aeKH[kc_toa]))
    IfKM = CCO.InterpolateC2F(; bottom = CCO.SetValue(aeKM[kc_surf]), top = CCO.SetValue(aeKM[kc_toa]))
    Ic = CCO.InterpolateF2C()

    @. aux_tc_f.ρ_ae_KH = IfKH(aeKH) * ρ_f
    @. aux_tc_f.ρ_ae_KM = IfKM(aeKM) * ρ_f

    aeKHq_tot_bc = -surf.ρq_tot_flux / a_en[kc_surf] / aux_tc_f.ρ_ae_KH[kf_surf]
    aeKHh_tot_bc = -surf.ρe_tot_flux / a_en[kc_surf] / aux_tc_f.ρ_ae_KH[kf_surf]
    aeKMu_bc = -surf.ρu_flux / a_en[kc_surf] / aux_tc_f.ρ_ae_KM[kf_surf]
    aeKMv_bc = -surf.ρv_flux / a_en[kc_surf] / aux_tc_f.ρ_ae_KM[kf_surf]

    aeKMuₕ_bc = CCG.UVVector(aeKMu_bc, aeKMv_bc)

    ∇q_tot_en = CCO.DivergenceC2F(; bottom = CCO.SetDivergence(aeKHq_tot_bc), top = CCO.SetDivergence(FT(0)))
    ∇h_tot_en = CCO.DivergenceC2F(; bottom = CCO.SetDivergence(aeKHh_tot_bc), top = CCO.SetDivergence(FT(0)))
    # CCG.Covariant3Vector(FT(1)) ⊗ CCG.Covariant12Vector(FT(aeKMu_bc),FT(aeKMv_bc))
    local_geometry_surf = CC.Fields.local_geometry_field(axes(ρ_f))[kf_surf]
    wvec = CC.Geometry.WVector
    ∇uₕ_gm = CCO.GradientC2F(;
        bottom = CCO.SetGradient(
            CCG.Covariant3Vector(wvec(FT(1)), local_geometry_surf) ⊗
            CCG.Covariant12Vector(aeKMuₕ_bc, local_geometry_surf),
        ),
        top = CCO.SetGradient(
            CCG.Covariant3Vector(wvec(FT(0)), local_geometry_surf) ⊗ CCG.Covariant12Vector(FT(0), FT(0)),
        ),
    )

    @. aux_tc_f.diffusive_flux_qt = -aux_tc_f.ρ_ae_KH * ∇q_tot_en(wvec(aux_en.q_tot))
    @. aux_tc_f.diffusive_flux_h = -aux_tc_f.ρ_ae_KH * ∇h_tot_en(wvec(aux_en.h_tot))
    @. aux_tc_f.diffusive_flux_uₕ = -aux_tc_f.ρ_ae_KM * ∇uₕ_gm(prog_gm_uₕ)

    if edmf.moisture_model isa NonEquilibriumMoisture
        aeKHq_liq_bc = FT(0)
        aeKHq_ice_bc = FT(0)

        ∇q_liq_en = CCO.DivergenceC2F(; bottom = CCO.SetDivergence(aeKHq_liq_bc), top = CCO.SetDivergence(FT(0)))
        ∇q_ice_en = CCO.DivergenceC2F(; bottom = CCO.SetDivergence(aeKHq_ice_bc), top = CCO.SetDivergence(FT(0)))

        @. aux_tc_f.diffusive_flux_ql = -aux_tc_f.ρ_ae_KH * ∇q_liq_en(wvec(aux_en.q_liq))
        @. aux_tc_f.diffusive_flux_qi = -aux_tc_f.ρ_ae_KH * ∇q_ice_en(wvec(aux_en.q_ice))
    end

    return nothing
end

function affect_filter!(edmf::EDMFModel, grid::Grid, state::State, param_set::APS, surf::SurfaceBase, t::Real)
    prog_en = center_prog_environment(state)
    aux_en = center_aux_environment(state)
    ###
    ### Filters
    ###
    set_edmf_surface_bc(edmf, grid, state, surf, param_set)
    filter_updraft_vars(edmf, grid, state, surf, param_set)

    @inbounds for k in real_center_indices(grid)
        prog_en.ρatke[k] = max(prog_en.ρatke[k], 0.0)
        if edmf.thermo_covariance_model isa PrognosticThermoCovariances
            prog_en.ρaHvar[k] = max(prog_en.ρaHvar[k], 0.0)
            prog_en.ρaQTvar[k] = max(prog_en.ρaQTvar[k], 0.0)
            prog_en.ρaHQTcov[k] = max(prog_en.ρaHQTcov[k], -sqrt(prog_en.ρaHvar[k] * prog_en.ρaQTvar[k]))
            prog_en.ρaHQTcov[k] = min(prog_en.ρaHQTcov[k], sqrt(prog_en.ρaHvar[k] * prog_en.ρaQTvar[k]))
        end
    end
    return nothing
end

function set_edmf_surface_bc(edmf::EDMFModel, grid::Grid, state::State, surf::SurfaceBase, param_set::APS)
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = float_type(state)
    N_up = n_updrafts(edmf)
    kc_surf = kc_surface(grid)
    kf_surf = kf_surface(grid)
    aux_gm = center_aux_grid_mean(state)
    aux_gm_f = face_aux_grid_mean(state)
    prog_gm = center_prog_grid_mean(state)
    prog_up = center_prog_updrafts(state)
    prog_en = center_prog_environment(state)
    prog_up_f = face_prog_updrafts(state)
    aux_bulk = center_aux_bulk(state)
    ts_gm = aux_gm.ts
    cp = TD.cp_m(thermo_params, ts_gm[kc_surf])
    ρ_c = prog_gm.ρ
    ρ_f = aux_gm_f.ρ
    ae_surf::FT = 1
    @inbounds for i in 1:N_up
        θ_surf = θ_surface_bc(surf, grid, state, edmf, i, param_set)
        q_surf = q_surface_bc(surf, grid, state, edmf, i)
        a_surf = area_surface_bc(surf, edmf, i)
        prog_up[i].ρarea[kc_surf] = ρ_c[kc_surf] * a_surf
        prog_up[i].ρaθ_liq_ice[kc_surf] = prog_up[i].ρarea[kc_surf] * θ_surf
        prog_up[i].ρaq_tot[kc_surf] = prog_up[i].ρarea[kc_surf] * q_surf
        if edmf.moisture_model isa NonEquilibriumMoisture
            q_liq_surf = FT(0)
            q_ice_surf = FT(0)
            prog_up[i].ρaq_liq[kc_surf] = prog_up[i].ρarea[kc_surf] * q_liq_surf
            prog_up[i].ρaq_ice[kc_surf] = prog_up[i].ρarea[kc_surf] * q_ice_surf
        end
        prog_up_f[i].ρaw[kf_surf] = ρ_f[kf_surf] * w_surface_bc(surf)
        ae_surf -= a_surf
    end

    flux1 = surf.shf / cp
    flux2 = surf.ρq_tot_flux
    zLL::FT = grid.zc[kc_surf].z
    ustar = surf.ustar
    oblength = surf.obukhov_length
    ρLL = prog_gm.ρ[kc_surf]
    ρ_ae = ρ_c[kc_surf] * ae_surf
    mix_len_params = mixing_length_params(edmf)
    prog_en.ρatke[kc_surf] = ρ_ae * get_surface_tke(mix_len_params, surf.ustar, zLL, surf.obukhov_length)
    if edmf.thermo_covariance_model isa PrognosticThermoCovariances
        prog_en.ρaHvar[kc_surf] = ρ_ae * get_surface_variance(flux1 / ρLL, flux1 / ρLL, ustar, zLL, oblength)
        prog_en.ρaQTvar[kc_surf] = ρ_ae * get_surface_variance(flux2 / ρLL, flux2 / ρLL, ustar, zLL, oblength)
        prog_en.ρaHQTcov[kc_surf] = ρ_ae * get_surface_variance(flux1 / ρLL, flux2 / ρLL, ustar, zLL, oblength)
    end
    return nothing
end

function surface_helper(surf::SurfaceBase, grid::Grid, state::State)
    FT = float_type(state)
    kc_surf = kc_surface(grid)
    prog_gm = center_prog_grid_mean(state)
    zLL::FT = grid.zc[kc_surf].z
    ustar = surf.ustar
    oblength = surf.obukhov_length
    ρLL = prog_gm.ρ[kc_surf]
    return (; ustar, zLL, oblength, ρLL)
end

function a_up_boundary_conditions(surf::SurfaceBase, edmf::EDMFModel, i::Int)
    a_surf = area_surface_bc(surf, edmf, i)
    return (; bottom = CCO.SetValue(a_surf), top = CCO.Extrapolate())
end

function a_bulk_boundary_conditions(surf::SurfaceBase, edmf::EDMFModel)
    N_up = n_updrafts(edmf)
    a_surf = sum(i -> area_surface_bc(surf, edmf, i), 1:N_up)
    return (; bottom = CCO.SetValue(a_surf), top = CCO.Extrapolate())
end

function a_en_boundary_conditions(surf::SurfaceBase, edmf::EDMFModel)
    N_up = n_updrafts(edmf)
    a_surf = 1 - sum(i -> area_surface_bc(surf, edmf, i), 1:N_up)
    return (; bottom = CCO.SetValue(a_surf), top = CCO.Extrapolate())
end

function area_surface_bc(surf::SurfaceBase{FT}, edmf::EDMFModel, i::Int)::FT where {FT}
    N_up = n_updrafts(edmf)
    return surf.bflux > 0 ? edmf.surface_area / N_up : FT(0)
end

function w_surface_bc(::SurfaceBase{FT})::FT where {FT}
    return FT(0)
end
function uₕ_bcs()
    return CCO.InterpolateC2F(bottom = CCO.Extrapolate(), top = CCO.Extrapolate())
end
function θ_surface_bc(
    surf::SurfaceBase{FT},
    grid::Grid,
    state::State,
    edmf::EDMFModel,
    i::Int,
    param_set::APS,
)::FT where {FT}
    thermo_params = TCP.thermodynamics_params(param_set)
    aux_gm = center_aux_grid_mean(state)
    prog_gm = center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ
    kc_surf = kc_surface(grid)
    ts_gm = aux_gm.ts
    c_p = TD.cp_m(thermo_params, ts_gm[kc_surf])
    UnPack.@unpack ustar, zLL, oblength, ρLL = surface_helper(surf, grid, state)

    surf.bflux > 0 || return FT(0)
    set_src_seed = edmf.set_src_seed
    a_total = edmf.surface_area
    a_ = area_surface_bc(surf, edmf, i)
    ρθ_liq_ice_flux = surf.shf / c_p # assuming no ql,qi flux
    h_var = get_surface_variance(ρθ_liq_ice_flux / ρLL, ρθ_liq_ice_flux / ρLL, ustar, zLL, oblength)
    surface_scalar_coeff =
        percentile_bounds_mean_norm(1 - a_total + (i - 1) * a_, 1 - a_total + i * a_, 1000, set_src_seed)
    return aux_gm.θ_liq_ice[kc_surf] + surface_scalar_coeff * sqrt(h_var)
end
function q_surface_bc(surf::SurfaceBase{FT}, grid::Grid, state::State, edmf::EDMFModel, i::Int)::FT where {FT}
    aux_gm = center_aux_grid_mean(state)
    prog_gm = center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ
    kc_surf = kc_surface(grid)
    surf.bflux > 0 || return aux_gm.q_tot[kc_surf]
    a_total = edmf.surface_area
    set_src_seed = edmf.set_src_seed
    a_ = area_surface_bc(surf, edmf, i)
    UnPack.@unpack ustar, zLL, oblength, ρLL = surface_helper(surf, grid, state)
    ρq_tot_flux = surf.ρq_tot_flux
    qt_var = get_surface_variance(ρq_tot_flux / ρLL, ρq_tot_flux / ρLL, ustar, zLL, oblength)
    surface_scalar_coeff =
        percentile_bounds_mean_norm(1 - a_total + (i - 1) * a_, 1 - a_total + i * a_, 1000, set_src_seed)
    return aux_gm.q_tot[kc_surf] + surface_scalar_coeff * sqrt(qt_var)
end
function ql_surface_bc(surf::SurfaceBase{FT})::FT where {FT}
    return FT(0)
end
function qi_surface_bc(surf::SurfaceBase{FT})::FT where {FT}
    return FT(0)
end

function get_GMV_CoVar(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    ::Val{covar_sym},
    ::Val{ϕ_sym},
    ::Val{ψ_sym},
) where {covar_sym, ϕ_sym, ψ_sym}
    N_up = n_updrafts(edmf)
    is_tke = covar_sym == :tke
    FT = float_type(state)
    tke_factor = is_tke ? FT(0.5) : 1
    aux_gm_c = center_aux_grid_mean(state)
    aux_gm_f = face_aux_grid_mean(state)
    prog_gm_c = center_prog_grid_mean(state)
    prog_gm_f = face_prog_grid_mean(state)
    aux_up_f = face_aux_updrafts(state)
    aux_en_c = center_aux_environment(state)
    aux_en_f = face_aux_environment(state)
    aux_en = is_tke ? aux_en_f : aux_en_c
    aux_up_c = center_aux_updrafts(state)
    aux_up = is_tke ? aux_up_f : aux_up_c
    gmv_covar = getproperty(center_aux_grid_mean(state), covar_sym)
    covar_e = getproperty(center_aux_environment(state), covar_sym)
    gm = is_tke ? prog_gm_f : aux_gm_c
    to_scalar = is_tke ? toscalar : x -> x
    ϕ_gm = getproperty(gm, ϕ_sym)
    ψ_gm = getproperty(gm, ψ_sym)
    ϕ_en = getproperty(aux_en, ϕ_sym)
    ψ_en = getproperty(aux_en, ψ_sym)
    area_en = aux_en_c.area

    Icd = is_tke ? CCO.InterpolateF2C() : x -> x
    @. gmv_covar = tke_factor * area_en * Icd(ϕ_en - to_scalar(ϕ_gm)) * Icd(ψ_en - to_scalar(ψ_gm)) + area_en * covar_e
    @inbounds for i in 1:N_up
        ϕ_up = getproperty(aux_up[i], ϕ_sym)
        ψ_up = getproperty(aux_up[i], ψ_sym)
        @. gmv_covar += tke_factor * aux_up_c[i].area * Icd(ϕ_up - to_scalar(ϕ_gm)) * Icd(ψ_up - to_scalar(ψ_gm))
    end
    return nothing
end

function compute_updraft_top(grid::Grid{FT}, state::State, i::Int)::FT where {FT}
    aux_up = center_aux_updrafts(state)
    return z_findlast_center(k -> aux_up[i].area[k] > 1e-3, grid)
end

function compute_plume_scale_height(grid::Grid, state::State, H_up_min::FT, i::Int)::FT where {FT}
    updraft_top::FT = compute_updraft_top(grid, state, i)
    return max(updraft_top, H_up_min)
end

function compute_up_stoch_tendencies!(edmf::EDMFModel, grid::Grid, state::State, param_set::APS, surf::SurfaceBase)
    N_up = n_updrafts(edmf)

    aux_up = center_aux_updrafts(state)
    tendencies_up = center_tendencies_updrafts(state)

    @inbounds for i in 1:N_up
        # prognostic entr/detr
        tends_ε_nondim = tendencies_up[i].ε_nondim
        tends_δ_nondim = tendencies_up[i].δ_nondim

        c_gen_stoch = edmf.entr_closure.c_gen_stoch
        mean_entr = aux_up[i].ε_nondim
        mean_detr = aux_up[i].δ_nondim
        ε_σ² = c_gen_stoch[1]
        δ_σ² = c_gen_stoch[2]
        ε_λ = c_gen_stoch[3]
        δ_λ = c_gen_stoch[4]
        @. tends_ε_nondim = √(2ε_λ * mean_entr * ε_σ²)
        @. tends_δ_nondim = √(2δ_λ * mean_detr * δ_σ²)
    end
end


function compute_up_tendencies!(edmf::EDMFModel, grid::Grid, state::State, param_set::APS, surf::SurfaceBase)
    N_up = n_updrafts(edmf)
    kc_surf = kc_surface(grid)
    kf_surf = kf_surface(grid)
    FT = float_type(state)

    aux_up = center_aux_updrafts(state)
    aux_en = center_aux_environment(state)
    aux_en_f = face_aux_environment(state)
    aux_up_f = face_aux_updrafts(state)
    prog_up = center_prog_updrafts(state)
    prog_up_f = face_prog_updrafts(state)
    tendencies_up = center_tendencies_updrafts(state)
    tendencies_up_f = face_tendencies_updrafts(state)
    prog_gm = center_prog_grid_mean(state)
    aux_gm_f = face_aux_grid_mean(state)
    ρ_c = prog_gm.ρ
    ρ_f = aux_gm_f.ρ
    au_lim = edmf.max_area

    @inbounds for i in 1:N_up
        aux_up_i = aux_up[i]
        @. aux_up_i.entr_turb_dyn = aux_up_i.entr_sc + aux_up_i.frac_turb_entr
        @. aux_up_i.detr_turb_dyn = aux_up_i.detr_sc + aux_up_i.frac_turb_entr
    end

    UB = CCO.UpwindBiasedProductC2F(bottom = CCO.SetValue(FT(0)), top = CCO.SetValue(FT(0)))
    Ic = CCO.InterpolateF2C()

    wvec = CC.Geometry.WVector
    ∇c = CCO.DivergenceF2C()
    w_bcs = (; bottom = CCO.SetValue(wvec(FT(0))), top = CCO.SetValue(wvec(FT(0))))
    LBF = CCO.LeftBiasedC2F(; bottom = CCO.SetValue(FT(0)))

    # Solve for updraft area fraction
    @inbounds for i in 1:N_up
        aux_up_i = aux_up[i]
        w_up = aux_up_f[i].w
        a_up = aux_up_i.area
        q_tot_up = aux_up_i.q_tot
        q_tot_en = aux_en.q_tot
        θ_liq_ice_en = aux_en.θ_liq_ice
        θ_liq_ice_up = aux_up_i.θ_liq_ice
        entr_turb_dyn = aux_up_i.entr_turb_dyn
        detr_turb_dyn = aux_up_i.detr_turb_dyn
        θ_liq_ice_tendency_precip_formation = aux_up_i.θ_liq_ice_tendency_precip_formation
        qt_tendency_precip_formation = aux_up_i.qt_tendency_precip_formation

        ρarea = prog_up[i].ρarea
        ρaθ_liq_ice = prog_up[i].ρaθ_liq_ice
        ρaq_tot = prog_up[i].ρaq_tot

        tends_ρarea = tendencies_up[i].ρarea
        tends_ρaθ_liq_ice = tendencies_up[i].ρaθ_liq_ice
        tends_ρaq_tot = tendencies_up[i].ρaq_tot

        @. tends_ρarea =
            -∇c(wvec(LBF(Ic(w_up) * ρarea))) + (ρarea * Ic(w_up) * entr_turb_dyn) - (ρarea * Ic(w_up) * detr_turb_dyn)

        @. tends_ρaθ_liq_ice =
            -∇c(wvec(LBF(Ic(w_up) * ρaθ_liq_ice))) + (ρarea * Ic(w_up) * entr_turb_dyn * θ_liq_ice_en) -
            (ρaθ_liq_ice * Ic(w_up) * detr_turb_dyn) + (ρ_c * θ_liq_ice_tendency_precip_formation)

        @. tends_ρaq_tot =
            -∇c(wvec(LBF(Ic(w_up) * ρaq_tot))) + (ρarea * Ic(w_up) * entr_turb_dyn * q_tot_en) -
            (ρaq_tot * Ic(w_up) * detr_turb_dyn) + (ρ_c * qt_tendency_precip_formation)

        if edmf.moisture_model isa NonEquilibriumMoisture

            q_liq_up = aux_up_i.q_liq
            q_ice_up = aux_up_i.q_ice
            q_liq_en = aux_en.q_liq
            q_ice_en = aux_en.q_ice

            ql_tendency_noneq = aux_up_i.ql_tendency_noneq
            qi_tendency_noneq = aux_up_i.qi_tendency_noneq
            ql_tendency_precip_formation = aux_up_i.ql_tendency_precip_formation
            qi_tendency_precip_formation = aux_up_i.qi_tendency_precip_formation

            ρaq_liq = prog_up[i].ρaq_liq
            ρaq_ice = prog_up[i].ρaq_ice

            tends_ρaq_liq = tendencies_up[i].ρaq_liq
            tends_ρaq_ice = tendencies_up[i].ρaq_ice

            @. tends_ρaq_liq =
                -∇c(wvec(LBF(Ic(w_up) * ρaq_liq))) + (ρarea * Ic(w_up) * entr_turb_dyn * q_liq_en) -
                (ρaq_liq * Ic(w_up) * detr_turb_dyn) + (ρ_c * (ql_tendency_precip_formation + ql_tendency_noneq))

            @. tends_ρaq_ice =
                -∇c(wvec(LBF(Ic(w_up) * ρaq_ice))) + (ρarea * Ic(w_up) * entr_turb_dyn * q_ice_en) -
                (ρaq_ice * Ic(w_up) * detr_turb_dyn) + (ρ_c * (qi_tendency_precip_formation + qi_tendency_noneq))

            tends_ρaq_liq[kc_surf] = 0
            tends_ρaq_ice[kc_surf] = 0
        end

        # prognostic entr/detr
        if edmf.entr_closure isa PrognosticNoisyRelaxationProcess
            c_gen_stoch = edmf.entr_closure.c_gen_stoch
            mean_entr = aux_up[i].ε_nondim
            mean_detr = aux_up[i].δ_nondim
            ε_λ = c_gen_stoch[3]
            δ_λ = c_gen_stoch[4]
            tends_ε_nondim = tendencies_up[i].ε_nondim
            tends_δ_nondim = tendencies_up[i].δ_nondim
            ε_nondim = prog_up[i].ε_nondim
            δ_nondim = prog_up[i].δ_nondim
            @. tends_ε_nondim = ε_λ * (mean_entr - ε_nondim)
            @. tends_δ_nondim = δ_λ * (mean_detr - δ_nondim)
        end

        tends_ρarea[kc_surf] = 0
        tends_ρaθ_liq_ice[kc_surf] = 0
        tends_ρaq_tot[kc_surf] = 0
    end

    # Solve for updraft velocity

    # We know that, since W = 0 at z = 0, BCs for entr, detr,
    # and buoyancy should not matter in the end
    zero_bcs = (; bottom = CCO.SetValue(FT(0)), top = CCO.SetValue(FT(0)))
    I0f = CCO.InterpolateC2F(; zero_bcs...)
    adv_bcs = (; bottom = CCO.SetValue(wvec(FT(0))), top = CCO.SetValue(wvec(FT(0))))
    LBC = CCO.LeftBiasedF2C(; bottom = CCO.SetValue(FT(0)))
    ∇f = CCO.DivergenceC2F(; adv_bcs...)

    @inbounds for i in 1:N_up
        a_up_bcs = a_up_boundary_conditions(surf, edmf, i)
        Iaf = CCO.InterpolateC2F(; a_up_bcs...)
        ρaw = prog_up_f[i].ρaw
        tends_ρaw = tendencies_up_f[i].ρaw
        nh_pressure = aux_up_f[i].nh_pressure
        a_up = aux_up[i].area
        w_up = aux_up_f[i].w
        w_en = aux_en_f.w
        entr_w = aux_up[i].entr_turb_dyn
        detr_w = aux_up[i].detr_turb_dyn
        buoy = aux_up[i].buoy

        @. tends_ρaw = -(∇f(wvec(LBC(ρaw * w_up))))
        @. tends_ρaw += (ρaw * (I0f(entr_w) * w_en - I0f(detr_w) * w_up)) + (ρ_f * Iaf(a_up) * I0f(buoy)) + nh_pressure
        tends_ρaw[kf_surf] = 0
    end

    return nothing
end

function filter_updraft_vars(edmf::EDMFModel, grid::Grid, state::State, surf::SurfaceBase, param_set::APS)
    N_up = n_updrafts(edmf)
    kc_surf = kc_surface(grid)
    kf_surf = kf_surface(grid)
    FT = float_type(state)
    N_up = n_updrafts(edmf)

    prog_up = center_prog_updrafts(state)
    prog_gm = center_prog_grid_mean(state)
    aux_gm_f = face_aux_grid_mean(state)
    aux_up = center_aux_updrafts(state)
    aux_up_f = face_aux_updrafts(state)
    prog_up_f = face_prog_updrafts(state)
    ρ_c = prog_gm.ρ
    ρ_f = aux_gm_f.ρ
    a_min = edmf.minimum_area
    a_max = edmf.max_area

    @inbounds for i in 1:N_up
        prog_up[i].ρarea .= max.(prog_up[i].ρarea, 0)
        prog_up[i].ρaθ_liq_ice .= max.(prog_up[i].ρaθ_liq_ice, 0)
        prog_up[i].ρaq_tot .= max.(prog_up[i].ρaq_tot, 0)
        if edmf.entr_closure isa PrognosticNoisyRelaxationProcess
            @. prog_up[i].ε_nondim = max(prog_up[i].ε_nondim, 0)
            @. prog_up[i].δ_nondim = max(prog_up[i].δ_nondim, 0)
        end
        @inbounds for k in real_center_indices(grid)
            prog_up[i].ρarea[k] = min(prog_up[i].ρarea[k], ρ_c[k] * a_max)
        end
    end
    if edmf.moisture_model isa NonEquilibriumMoisture
        @inbounds for i in 1:N_up
            prog_up[i].ρaq_liq .= max.(prog_up[i].ρaq_liq, 0)
            prog_up[i].ρaq_ice .= max.(prog_up[i].ρaq_ice, 0)
        end
    end

    @inbounds for i in 1:N_up
        @. prog_up_f[i].ρaw = max.(prog_up_f[i].ρaw, 0)
        a_up_bcs = a_up_boundary_conditions(surf, edmf, i)
        If = CCO.InterpolateC2F(; a_up_bcs...)
        @. prog_up_f[i].ρaw = Int(If(prog_up[i].ρarea) >= ρ_f * a_min) * prog_up_f[i].ρaw
    end

    @inbounds for k in real_center_indices(grid)
        @inbounds for i in 1:N_up
            is_surface_center(grid, k) && continue
            prog_up[i].ρaq_tot[k] = max(prog_up[i].ρaq_tot[k], 0)
            # this is needed to make sure Rico is unchanged.
            # TODO : look into it further to see why
            # a similar filtering of ρaθ_liq_ice breaks the simulation
            if prog_up[i].ρarea[k] / ρ_c[k] < a_min
                prog_up[i].ρaq_tot[k] = 0
            end
        end
    end
    if edmf.moisture_model isa NonEquilibriumMoisture
        @inbounds for k in real_center_indices(grid)
            @inbounds for i in 1:N_up
                is_surface_center(grid, k) && continue
                prog_up[i].ρaq_tot[k] = max(prog_up[i].ρaq_tot[k], 0)
                # this is needed to make sure Rico is unchanged.
                # TODO : look into it further to see why
                # a similar filtering of ρaθ_liq_ice breaks the simulation
                if prog_up[i].ρarea[k] / ρ_c[k] < a_min
                    prog_up[i].ρaq_liq[k] = 0
                    prog_up[i].ρaq_ice[k] = 0
                end
            end
        end
    end


    Ic = CCO.InterpolateF2C()
    @inbounds for i in 1:N_up
        @. prog_up[i].ρarea = ifelse(Ic(prog_up_f[i].ρaw) <= 0, FT(0), prog_up[i].ρarea)
        @. prog_up[i].ρaθ_liq_ice = ifelse(Ic(prog_up_f[i].ρaw) <= 0, FT(0), prog_up[i].ρaθ_liq_ice)
        @. prog_up[i].ρaq_tot = ifelse(Ic(prog_up_f[i].ρaw) <= 0, FT(0), prog_up[i].ρaq_tot)
        θ_surf = θ_surface_bc(surf, grid, state, edmf, i, param_set)
        q_surf = q_surface_bc(surf, grid, state, edmf, i)
        a_surf = area_surface_bc(surf, edmf, i)
        prog_up[i].ρarea[kc_surf] = ρ_c[kc_surf] * a_surf
        prog_up[i].ρaθ_liq_ice[kc_surf] = prog_up[i].ρarea[kc_surf] * θ_surf
        prog_up[i].ρaq_tot[kc_surf] = prog_up[i].ρarea[kc_surf] * q_surf
    end
    if edmf.moisture_model isa NonEquilibriumMoisture
        @inbounds for i in 1:N_up
            @. prog_up[i].ρaq_liq = ifelse(Ic(prog_up_f[i].ρaw) <= 0, FT(0), prog_up[i].ρaq_liq)
            @. prog_up[i].ρaq_ice = ifelse(Ic(prog_up_f[i].ρaw) <= 0, FT(0), prog_up[i].ρaq_ice)
            ql_surf = ql_surface_bc(surf)
            qi_surf = qi_surface_bc(surf)
            prog_up[i].ρaq_liq[kc_surf] = prog_up[i].ρarea[kc_surf] * ql_surf
            prog_up[i].ρaq_ice[kc_surf] = prog_up[i].ρarea[kc_surf] * qi_surf
        end
    end
    return nothing
end

function compute_covariance_shear(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    ::Val{covar_sym},
    ::Val{ϕ_en_sym},
    ::Val{ψ_en_sym},
) where {covar_sym, ϕ_en_sym, ψ_en_sym}

    aux_tc = center_aux_turbconv(state)
    prog_gm = center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ
    is_tke = covar_sym == :tke
    FT = float_type(state)
    k_eddy = is_tke ? aux_tc.KM : aux_tc.KH
    aux_en_2m = center_aux_environment_2m(state)
    aux_covar = getproperty(aux_en_2m, covar_sym)
    uₕ_gm = grid_mean_uₕ(state)

    aux_en_c = center_aux_environment(state)
    aux_en_f = face_aux_environment(state)
    aux_en = is_tke ? aux_en_f : aux_en_c
    wvec = CC.Geometry.WVector
    ϕ_en = getproperty(aux_en, ϕ_en_sym)
    ψ_en = getproperty(aux_en, ψ_en_sym)

    bcs = (; bottom = CCO.Extrapolate(), top = CCO.SetGradient(wvec(zero(FT))))
    If = CCO.InterpolateC2F(; bcs...)
    area_en = aux_en_c.area
    shear = aux_covar.shear

    C123 = CCG.Covariant123Vector
    local_geometry = CC.Fields.local_geometry_field(axes(ρ_c))
    k̂ = @. CCG.Contravariant3Vector(CCG.WVector(FT(1)), local_geometry)
    Ifuₕ = uₕ_bcs()
    ∇uvw = CCO.GradientF2C()
    # TODO: k_eddy and Shear² should probably be tensors (Contravariant3 tensor),
    #       so that the result (a contraction) is a scalar.
    if is_tke
        uvw = @. C123(Ifuₕ(uₕ_gm)) + C123(wvec(ϕ_en)) # ϕ_en === ψ_en
        Shear² = @. LA.norm_sqr(adjoint(∇uvw(uvw)) * k̂)
        @. shear = ρ_c * area_en * k_eddy * Shear²
    else
        ∇c = CCO.GradientF2C()
        @. shear = 2 * ρ_c * area_en * k_eddy * LA.dot(∇c(If(ϕ_en)), k̂) * LA.dot(∇c(If(ψ_en)), k̂)
    end
    return nothing
end

function compute_covariance_interdomain_src(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    ::Val{covar_sym},
    ::Val{ϕ_sym},
    ::Val{ψ_sym},
) where {covar_sym, ϕ_sym, ψ_sym}
    N_up = n_updrafts(edmf)
    is_tke = covar_sym == :tke
    FT = float_type(state)
    tke_factor = is_tke ? FT(0.5) : 1
    aux_up = center_aux_updrafts(state)
    aux_up_f = face_aux_updrafts(state)
    aux_en_2m = center_aux_environment_2m(state)
    interdomain = getproperty(aux_en_2m, covar_sym).interdomain
    prog_up = is_tke ? aux_up_f : aux_up
    aux_en_c = center_aux_environment(state)
    aux_en_f = face_aux_environment(state)
    aux_en = is_tke ? aux_en_f : aux_en_c
    ϕ_en = getproperty(aux_en, ϕ_sym)
    ψ_en = getproperty(aux_en, ψ_sym)
    Ic = is_tke ? CCO.InterpolateF2C() : x -> x

    parent(interdomain) .= 0
    @inbounds for i in 1:N_up
        ϕ_up = getproperty(prog_up[i], ϕ_sym)
        ψ_up = getproperty(prog_up[i], ψ_sym)
        a_up = aux_up[i].area
        @. interdomain += tke_factor * a_up * (1 - a_up) * (Ic(ϕ_up) - Ic(ϕ_en)) * (Ic(ψ_up) - Ic(ψ_en))
    end
    return nothing
end

function compute_covariance_entr(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    ::Val{covar_sym},
    ::Val{ϕ_sym},
    ::Val{ψ_sym},
) where {covar_sym, ϕ_sym, ψ_sym}

    N_up = n_updrafts(edmf)
    FT = float_type(state)
    is_tke = covar_sym == :tke
    tke_factor = is_tke ? FT(0.5) : 1
    aux_up = center_aux_updrafts(state)
    aux_up_f = face_aux_updrafts(state)
    aux_gm_c = center_aux_grid_mean(state)
    prog_gm_f = face_prog_grid_mean(state)
    prog_gm = center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ
    gm = is_tke ? prog_gm_f : aux_gm_c
    prog_up = is_tke ? aux_up_f : aux_up
    to_scalar = is_tke ? toscalar : x -> x
    ϕ_gm = getproperty(gm, ϕ_sym)
    ψ_gm = getproperty(gm, ψ_sym)
    aux_en_2m = center_aux_environment_2m(state)
    aux_covar = getproperty(aux_en_2m, covar_sym)
    aux_en_c = center_aux_environment(state)
    covar = getproperty(aux_en_c, covar_sym)
    aux_en_f = face_aux_environment(state)
    aux_en = is_tke ? aux_en_f : aux_en_c
    ϕ_en = getproperty(aux_en, ϕ_sym)
    ψ_en = getproperty(aux_en, ψ_sym)
    entr_gain = aux_covar.entr_gain
    detr_loss = aux_covar.detr_loss
    Ic = CCO.InterpolateF2C()
    Idc = is_tke ? Ic : x -> x
    # TODO: we shouldn't need `parent` call here:
    parent(entr_gain) .= 0
    parent(detr_loss) .= 0
    min_area = edmf.minimum_area

    @inbounds for i in 1:N_up
        aux_up_i = aux_up[i]
        frac_turb_entr = aux_up_i.frac_turb_entr
        eps_turb = frac_turb_entr
        detr_sc = aux_up_i.detr_sc
        entr_sc = aux_up_i.entr_sc
        w_up = aux_up_f[i].w
        prog_up_i = prog_up[i]
        ϕ_up = getproperty(prog_up_i, ϕ_sym)
        ψ_up = getproperty(prog_up_i, ψ_sym)

        a_up = aux_up_i.area

        @. entr_gain +=
            Int(a_up > min_area) *
            (tke_factor * ρ_c * a_up * abs(Ic(w_up)) * detr_sc * (Idc(ϕ_up) - Idc(ϕ_en)) * (Idc(ψ_up) - Idc(ψ_en))) + (
                tke_factor *
                ρ_c *
                a_up *
                abs(Ic(w_up)) *
                eps_turb *
                (
                    (Idc(ϕ_en) - Idc(to_scalar(ϕ_gm))) * (Idc(ψ_up) - Idc(ψ_en)) +
                    (Idc(ψ_en) - Idc(to_scalar(ψ_gm))) * (Idc(ϕ_up) - Idc(ϕ_en))
                )
            )

        @. detr_loss += Int(a_up > min_area) * tke_factor * ρ_c * a_up * abs(Ic(w_up)) * (entr_sc + eps_turb) * covar

    end

    return nothing
end

function compute_covariance_dissipation(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    ::Val{covar_sym},
    param_set::APS,
) where {covar_sym}
    FT = float_type(state)
    c_d = mixing_length_params(edmf).c_d
    aux_tc = center_aux_turbconv(state)
    prog_en = center_prog_environment(state)
    prog_gm = center_prog_grid_mean(state)
    aux_en = center_aux_environment(state)
    ρ_c = prog_gm.ρ
    aux_en_2m = center_aux_environment_2m(state)
    aux_covar = getproperty(aux_en_2m, covar_sym)
    covar = getproperty(aux_en, covar_sym)
    dissipation = aux_covar.dissipation
    area_en = aux_en.area
    tke_en = aux_en.tke
    mixing_length = aux_tc.mixing_length

    @. dissipation = ρ_c * area_en * covar * max(tke_en, 0)^FT(0.5) / max(mixing_length, FT(1.0e-3)) * c_d
    return nothing
end

function compute_en_tendencies!(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    param_set::APS,
    ::Val{covar_sym},
    ::Val{prog_sym},
) where {covar_sym, prog_sym}
    N_up = n_updrafts(edmf)
    kc_surf = kc_surface(grid)
    kc_toa = kc_top_of_atmos(grid)
    aux_gm_f = face_aux_grid_mean(state)
    prog_en = center_prog_environment(state)
    prog_gm = center_prog_grid_mean(state)
    aux_en_2m = center_aux_environment_2m(state)
    tendencies_en = center_tendencies_environment(state)
    tend_covar = getproperty(tendencies_en, prog_sym)
    prog_covar = getproperty(prog_en, prog_sym)
    aux_up_f = face_aux_updrafts(state)
    aux_en = center_aux_environment(state)
    covar = getproperty(aux_en, covar_sym)
    aux_covar = getproperty(aux_en_2m, covar_sym)
    aux_up = center_aux_updrafts(state)
    w_en_f = face_aux_environment(state).w
    ρ_c = prog_gm.ρ
    ρ_f = aux_gm_f.ρ
    c_d = mixing_length_params(edmf).c_d
    is_tke = covar_sym == :tke
    FT = float_type(state)

    ρ_ae_K = face_aux_turbconv(state).ρ_ae_K
    KM = center_aux_turbconv(state).KM
    KH = center_aux_turbconv(state).KH
    aux_tc = center_aux_turbconv(state)
    aux_bulk = center_aux_bulk(state)
    D_env = aux_tc.ϕ_temporary
    aeK = aux_tc.ψ_temporary
    a_bulk = aux_bulk.area
    if is_tke
        @. aeK = (1 - a_bulk) * KM
    else
        @. aeK = (1 - a_bulk) * KH
    end

    press = aux_covar.press
    buoy = aux_covar.buoy
    shear = aux_covar.shear
    entr_gain = aux_covar.entr_gain
    rain_src = aux_covar.rain_src

    wvec = CC.Geometry.WVector
    aeK_bcs = (; bottom = CCO.SetValue(aeK[kc_surf]), top = CCO.SetValue(aeK[kc_toa]))
    prog_bcs = (; bottom = CCO.SetGradient(wvec(FT(0))), top = CCO.SetGradient(wvec(FT(0))))

    If = CCO.InterpolateC2F(; aeK_bcs...)
    ∇f = CCO.GradientC2F(; prog_bcs...)
    ∇c = CCO.DivergenceF2C()

    mixing_length = aux_tc.mixing_length
    min_area = edmf.minimum_area

    Ic = CCO.InterpolateF2C()
    area_en = aux_en.area
    tke_en = aux_en.tke

    parent(D_env) .= 0

    @inbounds for i in 1:N_up
        turb_entr = aux_up[i].frac_turb_entr
        entr_sc = aux_up[i].entr_sc
        w_up = aux_up_f[i].w
        a_up = aux_up[i].area
        # TODO: using `Int(bool) *` means that NaNs can propagate
        # into the solution. Could we somehow call `ifelse` instead?
        @. D_env += Int(a_up > min_area) * ρ_c * a_up * Ic(w_up) * (entr_sc + turb_entr)
    end

    RB = CCO.RightBiasedC2F(; top = CCO.SetValue(FT(0)))
    @. tend_covar =
        press + buoy + shear + entr_gain + rain_src - D_env * covar -
        (c_d * sqrt(max(tke_en, 0)) / max(mixing_length, 1)) * prog_covar - ∇c(wvec(RB(prog_covar * Ic(w_en_f)))) +
        ∇c(ρ_f * If(aeK) * ∇f(covar))

    return nothing
end

function update_diagnostic_covariances!(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    param_set::APS,
    ::Val{covar_sym},
) where {covar_sym}
    FT = float_type(state)
    N_up = n_updrafts(edmf)
    kc_surf = kc_surface(grid)
    kc_toa = kc_top_of_atmos(grid)
    prog_gm = center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ
    aux_en_2m = center_aux_environment_2m(state)
    aux_up_f = face_aux_updrafts(state)
    aux_en = center_aux_environment(state)
    covar = getproperty(aux_en, covar_sym)
    aux_covar = getproperty(aux_en_2m, covar_sym)
    aux_up = center_aux_updrafts(state)
    w_en_f = face_aux_environment(state).w
    c_d = mixing_length_params(edmf).c_d
    covar_lim = edmf.thermo_covariance_model.covar_lim

    ρ_ae_K = face_aux_turbconv(state).ρ_ae_K
    KH = center_aux_turbconv(state).KH
    aux_tc = center_aux_turbconv(state)
    aux_bulk = center_aux_bulk(state)
    D_env = aux_tc.ϕ_temporary
    a_bulk = aux_bulk.area
    tke_en = aux_en.tke

    shear = aux_covar.shear
    entr_gain = aux_covar.entr_gain
    rain_src = aux_covar.rain_src
    mixing_length = aux_tc.mixing_length
    min_area = edmf.minimum_area

    Ic = CCO.InterpolateF2C()
    area_en = aux_en.area

    parent(D_env) .= 0
    @inbounds for i in 1:N_up
        turb_entr = aux_up[i].frac_turb_entr
        entr_sc = aux_up[i].entr_sc
        w_up = aux_up_f[i].w
        a_up = aux_up[i].area
        # TODO: using `Int(bool) *` means that NaNs can propagate
        # into the solution. Could we somehow call `ifelse` instead?
        @. D_env += Int(a_up > min_area) * ρ_c * a_up * Ic(w_up) * (entr_sc + turb_entr)
    end

    @. covar =
        (shear + entr_gain + rain_src) /
        max(D_env + ρ_c * area_en * c_d * sqrt(max(tke_en, 0)) / max(mixing_length, 1), covar_lim)
    return nothing
end


function GMV_third_m(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    ::Val{covar_en_sym},
    ::Val{var},
    ::Val{gm_third_m_sym},
) where {covar_en_sym, var, gm_third_m_sym}

    N_up = n_updrafts(edmf)
    gm_third_m = getproperty(center_aux_grid_mean(state), gm_third_m_sym)
    kc_surf = kc_surface(grid)
    FT = float_type(state)

    aux_bulk = center_aux_bulk(state)
    aux_up_f = face_aux_updrafts(state)
    is_tke = covar_en_sym == :tke
    aux_en_c = center_aux_environment(state)
    covar_en = getproperty(aux_en_c, covar_en_sym)
    aux_en_f = face_aux_environment(state)
    aux_en = is_tke ? aux_en_f : aux_en_c
    aux_up_c = center_aux_updrafts(state)
    aux_tc = center_aux_turbconv(state)
    ϕ_gm = aux_tc.ϕ_gm
    ϕ_gm_cov = aux_tc.ϕ_gm_cov
    ϕ_en_cov = aux_tc.ϕ_en_cov
    ϕ_up_cubed = aux_tc.ϕ_up_cubed
    aux_up = is_tke ? aux_up_f : aux_up_c
    var_en = getproperty(aux_en, var)
    area_en = aux_en_c.area
    Ic = is_tke ? CCO.InterpolateF2C() : x -> x
    wvec = CC.Geometry.WVector
    ∇c = CCO.DivergenceF2C()
    w_en = aux_en_f.w

    @. ϕ_gm = area_en * Ic(var_en)
    @inbounds for i in 1:N_up
        a_up = aux_up_c[i].area
        var_up = getproperty(aux_up[i], var)
        @. ϕ_gm += a_up * Ic(var_up)
    end

    # w'w' ≈ 2/3 TKE (isotropic turbulence assumption)
    if is_tke
        @. ϕ_en_cov = FT(2 / 3) * covar_en
    else
        @. ϕ_en_cov = covar_en
    end

    parent(ϕ_up_cubed) .= 0
    @. ϕ_gm_cov = area_en * (ϕ_en_cov + (Ic(var_en) - ϕ_gm)^2)
    @inbounds for i in 1:N_up
        a_up = aux_up_c[i].area
        var_up = getproperty(aux_up[i], var)
        @. ϕ_gm_cov += a_up * (Ic(var_up) - ϕ_gm)^2
        @. ϕ_up_cubed += a_up * Ic(var_up)^3
    end

    @. gm_third_m = ϕ_up_cubed + area_en * (Ic(var_en)^3 + 3 * Ic(var_en) * ϕ_en_cov) - ϕ_gm^3 - 3 * ϕ_gm_cov * ϕ_gm

    gm_third_m[kc_surf] = 0
    return nothing
end
