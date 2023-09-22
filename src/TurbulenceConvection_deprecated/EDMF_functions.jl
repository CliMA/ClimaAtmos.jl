function update_cloud_frac(edmf::EDMFModel, state::State)
    # update grid-mean cloud fraction and cloud cover
    aux_bulk = center_aux_bulk(state)
    aux_gm = center_aux_grid_mean(state)
    aux_en = center_aux_environment(state)
    a_up_bulk = aux_bulk.area
    # update grid-mean cloud fraction and cloud cover
    @. aux_gm.cloud_fraction =
        aux_en.area * aux_en.cloud_fraction +
        a_up_bulk * aux_bulk.cloud_fraction
end

function compute_implicit_turbconv_tendencies!(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
)
    compute_implicit_up_tendencies!(edmf, grid, state)
    return nothing
end

function compute_explicit_turbconv_tendencies!(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
)
    compute_explicit_up_tendencies!(edmf, grid, state)
    return nothing
end

function compute_sgs_flux!(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    param_set::APS,
)
    (; surface_conditions) = state
    thermo_params = TCP.thermodynamics_params(param_set)
    N_up = n_updrafts(edmf)
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
    prog_up_f = face_prog_updrafts(state)
    ρ_f = aux_gm_f.ρ
    ρ_c = prog_gm.ρ
    p_c = center_aux_grid_mean_p(state)
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
    w_gm = prog_gm_f.u₃
    h_tot_gm = aux_gm.h_tot
    q_tot_gm = aux_gm.q_tot
    a_en_bcs = (;
        bottom = CCO.SetValue(
            1 - sum(
                i -> area_surface_bc(surface_conditions, edmf),
                1:n_updrafts(edmf),
            ),
        ),
        top = CCO.Extrapolate(),
    )
    Ifae = CCO.InterpolateC2F(; a_en_bcs...)
    If = CCO.InterpolateC2F(;
        bottom = CCO.SetValue(FT(0)),
        top = CCO.SetValue(FT(0)),
    )
    Ic = CCO.InterpolateF2C()

    # compute total enthalpies
    ts_en = center_aux_environment(state).ts
    ts_gm = center_aux_grid_mean_ts(state)
    @. h_tot_gm =
        TD.total_specific_enthalpy(thermo_params, ts_gm, prog_gm.ρe_tot / ρ_c)
    # Compute the mass flux and associated scalar fluxes
    @. massflux = ρ_f * Ifae(a_en) * (w_en - w_gm)
    @. massflux_h =
        ρ_f * Ifae(a_en) * (w_en - w_gm) * (If(aux_en.h_tot) - If(h_tot_gm))
    @. massflux_qt =
        ρ_f * Ifae(a_en) * (w_en - w_gm) * (If(aux_en.q_tot) - If(q_tot_gm))
    @inbounds for i in 1:N_up
        a_up_bcs = a_up_boundary_conditions(surface_conditions, edmf)
        Ifau = CCO.InterpolateC2F(; a_up_bcs...)
        a_up = aux_up[i].area
        w_up_i = prog_up_f[i].w
        @. aux_up_f[i].massflux = ρ_f * Ifau(a_up) * (w_up_i - w_gm)
        @. massflux_h +=
            ρ_f * (
                Ifau(a_up) *
                (w_up_i - w_gm) *
                (If(aux_up[i].h_tot) - If(h_tot_gm))
            )
        @. massflux_qt +=
            ρ_f * (
                Ifau(a_up) *
                (w_up_i - w_gm) *
                (If(aux_up[i].q_tot) - If(q_tot_gm))
            )
    end

    massflux_h[kf_surf] = zero(eltype(massflux_h))
    massflux_qt[kf_surf] = zero(eltype(massflux_qt))

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
    sgs_flux_h_tot[kf_surf] = surface_conditions.ρ_flux_h_tot
    sgs_flux_q_tot[kf_surf] = surface_conditions.ρ_flux_q_tot
    sgs_flux_uₕ[kf_surf] =
        edmf.zero_uv_fluxes ? zero(surface_conditions.ρ_flux_uₕ) :
        surface_conditions.ρ_flux_uₕ
    return nothing
end

function compute_diffusive_fluxes(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    param_set::APS,
)
    (; surface_conditions) = state
    thermo_params = TCP.thermodynamics_params(param_set)
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
    ts_gm = center_aux_grid_mean_ts(state)
    IfKH = CCO.InterpolateC2F(;
        bottom = CCO.SetValue(aeKH[kc_surf]),
        top = CCO.SetValue(aeKH[kc_toa]),
    )
    IfKM = CCO.InterpolateC2F(;
        bottom = CCO.SetValue(aeKM[kc_surf]),
        top = CCO.SetValue(aeKM[kc_toa]),
    )
    Ic = CCO.InterpolateF2C()

    @. aux_tc_f.ρ_ae_KH = IfKH(aeKH) * ρ_f
    @. aux_tc_f.ρ_ae_KM = IfKM(aeKM) * ρ_f

    aeKHq_tot_bc =
        -surface_conditions.ρ_flux_q_tot / a_en[kc_surf] /
        aux_tc_f.ρ_ae_KH[kf_surf]
    aeKHh_tot_bc =
        -surface_conditions.ρ_flux_h_tot / a_en[kc_surf] /
        aux_tc_f.ρ_ae_KH[kf_surf]
    ∇q_tot_en = CCO.GradientC2F(;
        bottom = CCO.SetGradient(aeKHq_tot_bc),
        top = CCO.SetGradient(zero(aeKHq_tot_bc)),
    )
    ∇h_tot_en = CCO.GradientC2F(;
        bottom = CCO.SetGradient(aeKHh_tot_bc),
        top = CCO.SetGradient(zero(aeKHh_tot_bc)),
    )
    ρ_flux_uₕ =
        edmf.zero_uv_fluxes ? zero(surface_conditions.ρ_flux_uₕ) :
        surface_conditions.ρ_flux_uₕ
    ∇uₕ_gm = CCO.GradientC2F(;
        bottom = CCO.SetGradient(
            -ρ_flux_uₕ / a_en[kc_surf] / aux_tc_f.ρ_ae_KM[kf_surf],
        ),
        top = CCO.SetGradient(zero(ρ_flux_uₕ)),
    )

    @. aux_tc_f.diffusive_flux_qt = -aux_tc_f.ρ_ae_KH * ∇q_tot_en(aux_en.q_tot)
    @. aux_tc_f.diffusive_flux_h = -aux_tc_f.ρ_ae_KH * ∇h_tot_en(aux_en.h_tot)
    @. aux_tc_f.diffusive_flux_uₕ = -aux_tc_f.ρ_ae_KM * C12(∇uₕ_gm(prog_gm_uₕ))

    return nothing
end

function affect_filter!(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    param_set::APS,
    t::Real,
)
    ###
    ### Filters
    ###
    set_edmf_surface_bc(edmf, grid, state, param_set)
    filter_updraft_vars(edmf, grid, state, param_set)
    return nothing
end

function set_edmf_surface_bc(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    param_set::APS,
)
    (; surface_conditions) = state
    thermo_params = TCP.thermodynamics_params(param_set)
    FT = float_type(state)
    N_up = n_updrafts(edmf)
    kc_surf = kc_surface(grid)
    kf_surf = kf_surface(grid)
    prog_gm = center_prog_grid_mean(state)
    prog_gm_uₕ = grid_mean_uₕ(state)
    prog_up = center_prog_updrafts(state)
    prog_up_f = face_prog_updrafts(state)
    ts_gm = center_aux_grid_mean_ts(state)
    ρ_c = prog_gm.ρ
    p_c = TD.air_pressure.(thermo_params, ts_gm)
    C123 = CCG.Covariant123Vector
    Ic = CCO.InterpolateF2C()
    @inbounds for i in 1:N_up
        θ_surf = θ_surface_bc(grid, state, edmf, i, param_set)
        q_surf = q_surface_bc(grid, state, edmf, i, param_set)
        a_surf = area_surface_bc(surface_conditions, edmf)
        e_kin = @. LA.norm_sqr(
            C123(prog_gm_uₕ) + C123(Ic(CCG.WVector(prog_up_f[i].w))),
        ) / 2
        e_pot_surf = geopotential(thermo_params, grid.zc.z[kc_surf])
        ts_up_i_surf =
            TD.PhaseEquil_pθq(thermo_params, p_c[kc_surf], θ_surf, q_surf)
        e_tot_surf = TD.total_energy(
            thermo_params,
            ts_up_i_surf,
            e_kin[kc_surf],
            e_pot_surf,
        )
        prog_up[i].ρarea[kc_surf] = ρ_c[kc_surf] * a_surf
        prog_up[i].ρae_tot[kc_surf] = prog_up[i].ρarea[kc_surf] * e_tot_surf
        prog_up[i].ρaq_tot[kc_surf] = prog_up[i].ρarea[kc_surf] * q_surf
        prog_up_f[i].w[kf_surf] = CCG.Covariant3Vector(
            CCG.WVector(FT(0)),
            CC.Fields.local_geometry_field(axes(prog_up_f))[kf_surf],
        )
    end
    return nothing
end

function surface_helper(grid::Grid, state::State)
    (; surface_conditions) = state
    FT = float_type(state)
    zLL::FT = grid.zc[Cent(1)].z
    ρLL = center_prog_grid_mean(state).ρ[Cent(1)]
    return (;
        ustar = surface_conditions.ustar,
        zLL,
        oblength = surface_conditions.obukhov_length,
        ρLL,
    )
end

function a_up_boundary_conditions(surface_conditions, edmf::EDMFModel)
    a_surf = area_surface_bc(surface_conditions, edmf)
    return (; bottom = CCO.SetValue(a_surf), top = CCO.Extrapolate())
end

function area_surface_bc(surface_conditions, edmf::EDMFModel)
    FT = typeof(edmf.minimum_area)
    return surface_conditions.buoyancy_flux > 0 ?
           edmf.surface_area / n_updrafts(edmf) : FT(0)
end

function θ_surface_bc(
    grid::Grid,
    state::State,
    edmf::EDMFModel,
    i::Int,
    param_set::APS,
)
    FT = eltype(edmf)
    (; surface_conditions) = state
    thermo_params = TCP.thermodynamics_params(param_set)
    aux_gm = center_aux_grid_mean(state)
    kc_surf = kc_surface(grid)
    ts_gm = center_aux_grid_mean_ts(state)
    c_p = TD.cp_m(thermo_params, ts_gm[kc_surf])
    (; ustar, zLL, oblength, ρLL) = surface_helper(grid, state)
    surface_conditions.buoyancy_flux > 0 || return aux_gm.θ_liq_ice[kc_surf]
    a_total = edmf.surface_area
    a_ = area_surface_bc(surface_conditions, edmf)
    ρθ_liq_ice_flux = surface_conditions.ρ_flux_θ
    h_var = get_surface_variance(
        ρθ_liq_ice_flux / ρLL,
        ustar,
        zLL,
        oblength,
        Fields.local_geometry_field(Fields.level(grid.fs, Fields.half))[],
    )
    surface_scalar_coeff = percentile_bounds_mean_norm(
        1 - a_total + (i - 1) * a_,
        1 - a_total + i * a_,
    )
    return aux_gm.θ_liq_ice[kc_surf] + surface_scalar_coeff * sqrt(h_var)
end

function q_surface_bc(
    grid::Grid,
    state::State,
    edmf::EDMFModel,
    i::Int,
    param_set,
)
    (; surface_conditions) = state
    aux_gm = center_aux_grid_mean(state)
    kc_surf = kc_surface(grid)
    surface_conditions.buoyancy_flux > 0 || return aux_gm.q_tot[kc_surf]
    a_total = edmf.surface_area
    a_ = area_surface_bc(surface_conditions, edmf)
    (; ustar, zLL, oblength, ρLL) = surface_helper(grid, state)
    ρq_tot_flux = surface_conditions.ρ_flux_q_tot
    qt_var = get_surface_variance(
        ρq_tot_flux / ρLL,
        ustar,
        zLL,
        oblength,
        Fields.local_geometry_field(Fields.level(grid.fs, Fields.half))[],
    )
    surface_scalar_coeff = percentile_bounds_mean_norm(
        1 - a_total + (i - 1) * a_,
        1 - a_total + i * a_,
    )
    return aux_gm.q_tot[kc_surf] + surface_scalar_coeff * sqrt(qt_var)
end

function compute_implicit_up_tendencies!(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
)
    N_up = n_updrafts(edmf)
    kc_surf = kc_surface(grid)
    kf_surf = kf_surface(grid)
    FT = float_type(state)

    prog_up = center_prog_updrafts(state)
    prog_up_f = face_prog_updrafts(state)
    tendencies_up = center_tendencies_updrafts(state)
    tendencies_up_f = face_tendencies_updrafts(state)
    aux_up = center_aux_updrafts(state)
    ts = center_aux_grid_mean_ts(state)
    p_c = center_aux_grid_mean_p(state)
    prog_gm = center_prog_grid_mean(state)
    ρ_c = prog_gm.ρ

    # Solve for updraft area fraction

    Ic = CCO.InterpolateF2C()
    ∇c = CCO.DivergenceF2C()
    LBF = CCO.LeftBiasedC2F(; bottom = CCO.SetValue(CCG.WVector(FT(0))))

    @inbounds for i in 1:N_up
        w_up = prog_up_f[i].w

        ρarea = prog_up[i].ρarea
        ρae_tot = prog_up[i].ρae_tot
        ρaq_tot = prog_up[i].ρaq_tot

        tends_ρarea = tendencies_up[i].ρarea
        tends_ρae_tot = tendencies_up[i].ρae_tot
        tends_ρaq_tot = tendencies_up[i].ρaq_tot

        volume_term =
            @. -p_c / ρ_c * (-(∇c(LBF(Ic(CCG.WVector(w_up)) * ρarea))))
        @. tends_ρarea += -∇c(LBF(Ic(CCG.WVector(w_up)) * ρarea))
        @. tends_ρae_tot +=
            -∇c(LBF(Ic(CCG.WVector(w_up)) * ρarea * aux_up[i].h_tot)) +
            volume_term
        @. tends_ρaq_tot += -∇c(LBF(Ic(CCG.WVector(w_up)) * ρaq_tot))

        tends_ρarea[kc_surf] = 0
        tends_ρae_tot[kc_surf] = 0
        tends_ρaq_tot[kc_surf] = 0
    end

    # Solve for updraft velocity

    LBC = CCO.LeftBiasedF2C(; bottom = CCO.SetValue(FT(0)))
    prog_bcs = (;
        bottom = CCO.SetGradient(CCG.WVector(FT(0))),
        top = CCO.SetGradient(CCG.WVector(FT(0))),
    )
    grad_f = CCO.GradientC2F(; prog_bcs...)

    @inbounds for i in 1:N_up
        w_up = prog_up_f[i].w
        tends_w = tendencies_up_f[i].w
        @. tends_w += -grad_f(LBC(LA.norm_sqr(CCG.WVector(w_up)) / 2))
        tends_w[kf_surf] = zero(tends_w[kf_surf])
    end

    return nothing
end

function compute_explicit_up_tendencies!(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
)
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
    p_c = center_aux_grid_mean_p(state)
    ρ_c = prog_gm.ρ

    Ic = CCO.InterpolateF2C()
    # We know that, since W = 0 at z = 0, BCs for entr, detr,
    # and buoyancy should not matter in the end
    zero_bcs = (; bottom = CCO.SetValue(FT(0)), top = CCO.SetValue(FT(0)))
    I0f = CCO.InterpolateC2F(; zero_bcs...)

    @inbounds for i in 1:N_up

        w_up = prog_up_f[i].w
        w_en = aux_en_f.w
        # Augment the tendencies of updraft area, tracers and vertical velocity

        # entrainment and detrainment - could be moved to implicit
        volume_term_entr = @. -p_c / ρ_c *
           prog_up[i].ρarea *
           Ic(wcomponent(CCG.WVector(w_up))) *
           (aux_up[i].entr - aux_up[i].detr)
        @. tendencies_up[i].ρarea +=
            prog_up[i].ρarea *
            Ic(wcomponent(CCG.WVector(w_up))) *
            (aux_up[i].entr - aux_up[i].detr)
        @. tendencies_up[i].ρae_tot +=
            prog_up[i].ρarea *
            aux_en.h_tot *
            Ic(wcomponent(CCG.WVector(w_up))) *
            aux_up[i].entr -
            prog_up[i].ρarea *
            aux_up[i].h_tot *
            Ic(wcomponent(CCG.WVector(w_up))) *
            aux_up[i].detr + volume_term_entr
        @. tendencies_up[i].ρaq_tot +=
            prog_up[i].ρarea *
            aux_en.q_tot *
            Ic(wcomponent(CCG.WVector(w_up))) *
            aux_up[i].entr -
            prog_up[i].ρaq_tot *
            Ic(wcomponent(CCG.WVector(w_up))) *
            aux_up[i].detr
        @. tendencies_up_f[i].w +=
            w_up * I0f(aux_up[i].entr) * (wcomponent(CCG.WVector(w_en - w_up)))

        # precipitation formation
        @. tendencies_up[i].ρae_tot +=
            prog_gm.ρ * aux_up[i].e_tot_tendency_precip_formation
        @. tendencies_up[i].ρaq_tot +=
            prog_gm.ρ * aux_up[i].qt_tendency_precip_formation

        # buoyancy and pressure
        @. tendencies_up_f[i].w += CCG.Covariant3Vector(
            CCG.WVector(aux_up_f[i].buoy + aux_up_f[i].nh_pressure),
        )

        # TODO - to be removed?
        tendencies_up[i].ρarea[kc_surf] = 0
        tendencies_up[i].ρae_tot[kc_surf] = 0
        tendencies_up[i].ρaq_tot[kc_surf] = 0
        tendencies_up_f[i].w[kf_surf] = zero(tendencies_up_f[i].w[kf_surf])
    end
    return nothing
end

function filter_updraft_vars(
    edmf::EDMFModel,
    grid::Grid,
    state::State,
    param_set::APS,
)
    (; surface_conditions) = state
    N_up = n_updrafts(edmf)
    kc_surf = kc_surface(grid)
    FT = float_type(state)
    N_up = n_updrafts(edmf)

    thermo_params = TCP.thermodynamics_params(param_set)
    prog_up = center_prog_updrafts(state)
    prog_gm = center_prog_grid_mean(state)
    prog_gm_uₕ = grid_mean_uₕ(state)
    aux_gm_f = face_aux_grid_mean(state)
    aux_bulk = center_aux_bulk(state)
    prog_up_f = face_prog_updrafts(state)
    ρ_c = prog_gm.ρ
    ρ_f = aux_gm_f.ρ
    ts_gm = center_aux_grid_mean_ts(state)
    p_c = TD.air_pressure.(thermo_params, ts_gm)
    If = CCO.InterpolateC2F(;
        bottom = CCO.Extrapolate(),
        top = CCO.Extrapolate(),
    )
    @. ρ_f = If(ρ_c)
    a_min = edmf.minimum_area
    a_max = edmf.max_area

    @inbounds for i in 1:N_up
        @. aux_bulk.filter_flag_1 = ifelse(prog_up[i].ρarea < FT(0), 1, 0)
        @. aux_bulk.filter_flag_3 = ifelse(prog_up[i].ρaq_tot < FT(0), 1, 0)
        @. aux_bulk.filter_flag_4 = ifelse(prog_up[i].ρarea > ρ_c * a_max, 1, 0)

        @. prog_up[i].ρarea = max(prog_up[i].ρarea, 0) #flag_1
        @. prog_up[i].ρaq_tot = max(prog_up[i].ρaq_tot, 0) #flag_3
        @. prog_up[i].ρarea = min(prog_up[i].ρarea, ρ_c * a_max) #flag_4
    end
    @inbounds for i in 1:N_up
        @. prog_up_f[i].w = CCG.Covariant3Vector(
            CCG.WVector(max(wcomponent(CCG.WVector(prog_up_f[i].w)), 0)),
        )
        a_up_bcs = a_up_boundary_conditions(surface_conditions, edmf)
        If = CCO.InterpolateC2F(; a_up_bcs...)
        @. prog_up_f[i].w =
            Int(If(prog_up[i].ρarea) >= ρ_f * a_min) * prog_up_f[i].w
    end

    Δz = Fields.Δz_field(axes(ρ_c))
    z = Fields.coordinate_field(axes(ρ_c)).z
    Δz1 = Spaces.level(Δz, 1)
    @inbounds for i in 1:N_up
        # this is needed to make sure Rico is unchanged.
        # TODO : look into it further to see why
        # a similar filtering of ρaθ_liq_ice breaks the simulation
        @. prog_up[i].ρaq_tot = ifelse(
            z > Δz1 / 2,
            ifelse(
                prog_up[i].ρarea / ρ_c < a_min,
                0,
                max(prog_up[i].ρaq_tot, 0),
            ),
            prog_up[i].ρaq_tot,
        )
        @. prog_up[i].ρarea = ifelse(
            z > Δz1 / 2,
            ifelse(prog_up[i].ρarea / ρ_c < a_min, 0, max(prog_up[i].ρarea, 0)),
            prog_up[i].ρarea,
        )
        @. prog_up[i].ρae_tot = ifelse(
            z > Δz1 / 2,
            ifelse(prog_up[i].ρarea / ρ_c < a_min, 0, prog_up[i].ρae_tot),
            prog_up[i].ρae_tot,
        )
    end

    Ic = CCO.InterpolateF2C()
    C123 = CCG.Covariant123Vector
    @inbounds for i in 1:N_up
        @. prog_up[i].ρarea = ifelse(
            Ic(wcomponent(CCG.WVector(prog_up_f[i].w))) <= 0,
            FT(0),
            prog_up[i].ρarea,
        )
        @. prog_up[i].ρae_tot = ifelse(
            Ic(wcomponent(CCG.WVector(prog_up_f[i].w))) <= 0,
            FT(0),
            prog_up[i].ρae_tot,
        )
        @. prog_up[i].ρaq_tot = ifelse(
            Ic(wcomponent(CCG.WVector(prog_up_f[i].w))) <= 0,
            FT(0),
            prog_up[i].ρaq_tot,
        )

        θ_surf = θ_surface_bc(grid, state, edmf, i, param_set)
        q_surf = q_surface_bc(grid, state, edmf, i, param_set)
        a_surf = area_surface_bc(surface_conditions, edmf)
        e_kin = @. LA.norm_sqr(
            C123(prog_gm_uₕ) + C123(Ic(CCG.WVector(prog_up_f[i].w))),
        ) / 2
        e_pot_surf = geopotential(thermo_params, grid.zc.z[kc_surf])
        ts_up_i_surf =
            TD.PhaseEquil_pθq(thermo_params, p_c[kc_surf], θ_surf, q_surf)
        e_tot_surf = TD.total_energy(
            thermo_params,
            ts_up_i_surf,
            e_kin[kc_surf],
            e_pot_surf,
        )
        prog_up[i].ρarea[kc_surf] = ρ_c[kc_surf] * a_surf
        prog_up[i].ρae_tot[kc_surf] = prog_up[i].ρarea[kc_surf] * e_tot_surf
        prog_up[i].ρaq_tot[kc_surf] = prog_up[i].ρarea[kc_surf] * q_surf
    end
    return nothing
end
