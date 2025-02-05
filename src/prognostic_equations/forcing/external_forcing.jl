#####
##### External forcing (for single column experiments)
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import NCDatasets as NC
import Interpolations as Intp

function interp_vertical_prof(x, xp, fp)
    spl = Intp.extrapolate(
        Intp.interpolate((xp,), fp, Intp.Gridded(Intp.Linear())),
        Intp.Flat(),
    )
    return spl(vec(x))
end

"""
Calculate height-dependent scalar relaxation timescale following from eqn. 11, Shen et al., 2022.
"""
function compute_gcm_driven_scalar_inv_τ(z::FT) where {FT}

    # TODO add to ClimaParameters
    τᵣ = FT(24.0 * 3600.0)
    zᵢ = FT(3000.0)
    zᵣ = FT(3500.0)
    if z < zᵢ
        return FT(0)
    elseif zᵢ <= z <= zᵣ
        cos_arg = pi * ((z - zᵢ) / (zᵣ - zᵢ))
        return (FT(0.5) / τᵣ) * (1 - cos(cos_arg))
    elseif z > zᵣ
        return (1 / τᵣ)
    end
end

# following PyCLES https://github.com/CliMA/pycles/blob/71c1752a1ef1b43bb90e5817de9126468b4eeba9/ForcingGCMFixed.pyx#L260
function eddy_vert_fluctuation!(ᶜρχₜ, ᶜχ, ᶜls_subsidence)
    @. ᶜρχₜ +=
        Geometry.WVector(ᶜgradᵥ(ᶠinterp(ᶜχ))).components.data.:1 *
        ᶜls_subsidence
end

external_forcing_cache(Y, atmos::AtmosModel, params) =
    external_forcing_cache(Y, atmos.external_forcing, params)

external_forcing_cache(Y, external_forcing::Nothing, params) = (;)
function external_forcing_cache(
    Y,
    external_forcing::Union{GCMForcing, ERA5Forcing},
    params,
)
    FT = Spaces.undertype(axes(Y.c))
    ᶜdTdt_fluc = similar(Y.c, FT)
    ᶜdqtdt_fluc = similar(Y.c, FT)
    ᶜdTdt_hadv = similar(Y.c, FT)
    ᶜdqtdt_hadv = similar(Y.c, FT)
    ᶜdTdt_rad = similar(Y.c, FT)
    ᶜT_nudge = similar(Y.c, FT)
    ᶜqt_nudge = similar(Y.c, FT)
    ᶜu_nudge = similar(Y.c, FT)
    ᶜv_nudge = similar(Y.c, FT)
    ᶜinv_τ_wind = similar(Y.c, FT)
    ᶜinv_τ_scalar = similar(Y.c, FT)
    ᶜls_subsidence = similar(Y.c, FT)
    insolation = similar(Fields.level(Y.c.ρ, 1), FT)
    cos_zenith = similar(Fields.level(Y.c.ρ, 1), FT)

    (; external_forcing_file, cfsite_number) = external_forcing
    NC.Dataset(external_forcing_file, "r") do ds

        function setvar!(
            cc_field,
            varname,
            colidx,
            zc_model,
            zc_forcing,
            ::GCMForcing,
        )
            parent(cc_field[colidx]) .= interp_vertical_prof(
                zc_model,
                zc_forcing,
                gcm_driven_profile_tmean(ds.group[cfsite_number], varname),
            )
        end

        function setvar!(
            cc_field,
            varname,
            colidx,
            zc_model,
            zc_forcing,
            ::ERA5Forcing,
        )
            parent(cc_field[colidx]) .= interp_vertical_prof(
                zc_model,
                zc_forcing,
                era5_driven_profile_tmean(ds.group[cfsite_number], varname),
            )
        end

        function setvar_subsidence!(
            cc_field,
            varname,
            colidx,
            zc_gcm,
            zc_forcing,
            params,
            ::GCMForcing,
        )
            parent(cc_field[colidx]) .= interp_vertical_prof(
                zc_gcm,
                zc_forcing,
                gcm_driven_profile_tmean(ds.group[cfsite_number], varname) .* .-(
                    gcm_driven_profile_tmean(ds.group[cfsite_number], "alpha"),
                ) ./ CAP.grav(params),
            )
        end

        function setvar_subsidence!(
            cc_field,
            varname,
            colidx,
            zc_gcm,
            zc_forcing,
            params,
            ::ERA5Forcing,
        )
            parent(cc_field[colidx]) .= interp_vertical_prof(
                zc_gcm,
                zc_forcing,
                era5_driven_profile_tmean(ds.group[cfsite_number], varname) ./ .-(
                    era5_driven_profile_tmean(ds.group[cfsite_number], "rho"),
                ) ./ CAP.grav(params),
            )
        end

        function set_insolation!(cc_field)
            parent(cc_field) .= mean(
                ds.group[cfsite_number]["rsdt"][:] ./
                ds.group[cfsite_number]["coszen"][:],
            )
        end

        function set_cos_zenith!(cc_field)
            parent(cc_field) .= ds.group[cfsite_number]["coszen"][1]
        end

        zc_forcing = nothing
        if external_forcing isa GCMForcing
            zc_forcing = gcm_height(ds.group[cfsite_number])
        elseif external_forcing isa ERA5Forcing
            zc_forcing = era5_height(ds.group[cfsite_number])
        else
            throw(
                ArgumentError(
                    "Unknown external forcing type. Options are GCMForcing or ERA5Forcing.",
                ),
            )
        end

        @info "zc_forcing" zc_forcing
        Fields.bycolumn(axes(Y.c)) do colidx

            zc_model = Fields.coordinate_field(Y.c).z[colidx]

            setvar!(
                ᶜdTdt_hadv,
                "tntha",
                colidx,
                zc_model,
                zc_forcing,
                external_forcing,
            )
            setvar!(
                ᶜdqtdt_hadv,
                "tnhusha",
                colidx,
                zc_model,
                zc_forcing,
                external_forcing,
            )
            #setvar!(ᶜdTdt_rad, "tntr", colidx, zc_model, zc_forcing, external_forcing) # don't need to set this when using RRTMGP
            setvar_subsidence!(
                ᶜls_subsidence,
                "wap",
                colidx,
                zc_model,
                zc_forcing,
                params,
                external_forcing,
            )
            # GCM states, used for nudging + vertical eddy advection
            setvar!(
                ᶜT_nudge,
                "ta",
                colidx,
                zc_model,
                zc_forcing,
                external_forcing,
            )
            setvar!(
                ᶜqt_nudge,
                "hus",
                colidx,
                zc_model,
                zc_forcing,
                external_forcing,
            )
            setvar!(
                ᶜu_nudge,
                "ua",
                colidx,
                zc_model,
                zc_forcing,
                external_forcing,
            )
            setvar!(
                ᶜv_nudge,
                "va",
                colidx,
                zc_model,
                zc_forcing,
                external_forcing,
            )

            # vertical eddy advection (Shen et al., 2022; eqn. 9,10)
            # sum of two terms to give total tendency. First term:
            setvar!(
                ᶜdTdt_fluc,
                "tntva",
                colidx,
                zc_model,
                zc_forcing,
                external_forcing,
            )
            setvar!(
                ᶜdqtdt_fluc,
                "tnhusva",
                colidx,
                zc_model,
                zc_forcing,
                external_forcing,
            )
            # second term:
            eddy_vert_fluctuation!(ᶜdTdt_fluc, ᶜT_nudge, ᶜls_subsidence)
            eddy_vert_fluctuation!(ᶜdqtdt_fluc, ᶜqt_nudge, ᶜls_subsidence)

            set_insolation!(insolation)
            set_cos_zenith!(cos_zenith)

            @. ᶜinv_τ_wind[colidx] = 1 / (6 * 3600)
            @. ᶜinv_τ_scalar[colidx] = compute_gcm_driven_scalar_inv_τ(zc_model)
        end
    end

    return (;
        ᶜdTdt_fluc,
        ᶜdqtdt_fluc,
        ᶜdTdt_hadv,
        ᶜdqtdt_hadv,
        ᶜdTdt_rad,
        ᶜT_nudge,
        ᶜqt_nudge,
        ᶜu_nudge,
        ᶜv_nudge,
        ᶜinv_τ_wind,
        ᶜinv_τ_scalar,
        ᶜls_subsidence,
        insolation,
        cos_zenith,
    )
end

external_forcing_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
function external_forcing_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::Union{GCMForcing, ERA5Forcing},
)
    # horizontal advection, vertical fluctuation, nudging, subsidence (need to add),
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜspecific, ᶜts, ᶜh_tot) = p.precomputed
    (;
        ᶜdTdt_fluc,
        ᶜdqtdt_fluc,
        ᶜdTdt_hadv,
        ᶜdqtdt_hadv,
        ᶜdTdt_rad,
        ᶜT_nudge,
        ᶜqt_nudge,
        ᶜu_nudge,
        ᶜv_nudge,
        ᶜls_subsidence,
        ᶜinv_τ_wind,
        ᶜinv_τ_scalar,
    ) = p.external_forcing

    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜuₕ_nudge = p.scratch.ᶜtemp_C12
    @. ᶜuₕ_nudge = C12(Geometry.UVVector(ᶜu_nudge, ᶜv_nudge), ᶜlg)
    @. Yₜ.c.uₕ -= (Y.c.uₕ - ᶜuₕ_nudge) * ᶜinv_τ_wind

    # nudging tendency
    ᶜdTdt_nudging = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_nudging = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt_nudging =
        -(TD.air_temperature(thermo_params, ᶜts) - ᶜT_nudge) * ᶜinv_τ_scalar
    @. ᶜdqtdt_nudging = -(ᶜspecific.q_tot - ᶜqt_nudge) * ᶜinv_τ_scalar

    ᶜdTdt_sum = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_sum = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt_sum = ᶜdTdt_hadv + ᶜdTdt_nudging + ᶜdTdt_fluc
    @. ᶜdqtdt_sum = ᶜdqtdt_hadv + ᶜdqtdt_nudging + ᶜdqtdt_fluc

    T_0 = TD.Parameters.T_0(thermo_params)
    Lv_0 = TD.Parameters.LH_v0(thermo_params)
    cv_v = TD.Parameters.cv_v(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)
    # total energy
    @. Yₜ.c.ρe_tot +=
        Y.c.ρ * (
            TD.cv_m(thermo_params, ᶜts) * ᶜdTdt_sum +
            (
                cv_v * (TD.air_temperature(thermo_params, ᶜts) - T_0) + Lv_0 -
                R_v * T_0
            ) * ᶜdqtdt_sum
        )
    # total specific humidity
    @. Yₜ.c.ρq_tot += Y.c.ρ * ᶜdqtdt_sum

    ## subsidence -->
    ᶠls_subsidence³ = p.scratch.ᶠtemp_CT3
    @. ᶠls_subsidence³ =
        ᶠinterp(ᶜls_subsidence * CT3(unit_basis_vector_data(CT3, ᶜlg)))
    subsidence!(
        Yₜ.c.ρe_tot,
        Y.c.ρ,
        ᶠls_subsidence³,
        ᶜh_tot,
        Val{:first_order}(),
    )
    subsidence!(
        Yₜ.c.ρq_tot,
        Y.c.ρ,
        ᶠls_subsidence³,
        ᶜspecific.q_tot,
        Val{:first_order}(),
    )

    # needed to address top boundary condition for forcings. Otherwise upper portion of domain is anomalously cold
    ρe_tot_top = Fields.level(Yₜ.c.ρe_tot, Spaces.nlevels(axes(Y.c)))
    @. ρe_tot_top = 0.0

    ρq_tot_top = Fields.level(Yₜ.c.ρq_tot, Spaces.nlevels(axes(Y.c)))
    @. ρq_tot_top = 0.0
    # <-- subsidence

    return nothing
end

# ISDAC external forcing (i.e. nudging)
external_forcing_cache(Y, external_forcing::ISDACForcing, params) = (;)  # Don't need to cache anything
function external_forcing_tendency!(Yₜ, Y, p, t, ::ISDACForcing)
    FT = Spaces.undertype(axes(Y.c))
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜspecific, ᶜts, ᶜh_tot, ᶜp) = p.precomputed

    ᶜinv_τ_scalar = APL.ISDAC_inv_τ_scalar(FT)  # s⁻¹
    ᶜinv_τ_wind = APL.ISDAC_inv_τ_wind(FT)  # s⁻¹
    θ = APL.ISDAC_θ_liq_ice(FT)
    u = APL.ISDAC_u(FT)
    v = APL.ISDAC_v(FT)
    q_tot = APL.ISDAC_q_tot(FT)

    # Convert ISDAC potential temperature to air temperature
    ta_ISDAC =
        (pres, z) -> TD.air_temperature(
            thermo_params,
            TD.PhaseEquil_pθq(thermo_params, pres, θ(z), q_tot(z)),
        )

    ᶜz = Fields.coordinate_field(Y.c).z
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜuₕ_nudge = p.scratch.ᶜtemp_C12
    @. ᶜuₕ_nudge = C12(Geometry.UVVector(u(ᶜz), v(ᶜz)), ᶜlg)
    @. Yₜ.c.uₕ -= (Y.c.uₕ - ᶜuₕ_nudge) * ᶜinv_τ_wind(ᶜz)

    # TODO: May make more sense to use initial ISDAC (hydrostatic) pressure, but would need to add it to cache,
    # so for now just use current pressure.
    ᶜdTdt_nudging = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_nudging = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt_nudging =
        -(TD.air_temperature(thermo_params, ᶜts) - ta_ISDAC(ᶜp, ᶜz)) *
        ᶜinv_τ_scalar(ᶜz)
    @. ᶜdqtdt_nudging = -(ᶜspecific.q_tot - q_tot(ᶜz)) * ᶜinv_τ_scalar(ᶜz)

    T_0 = TD.Parameters.T_0(thermo_params)
    Lv_0 = TD.Parameters.LH_v0(thermo_params)
    cv_v = TD.Parameters.cv_v(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)
    # total energy
    @. Yₜ.c.ρe_tot +=
        Y.c.ρ * (
            TD.cv_m(thermo_params, ᶜts) * ᶜdTdt_nudging +
            (
                cv_v * (TD.air_temperature(thermo_params, ᶜts) - T_0) + Lv_0 -
                R_v * T_0
            ) * ᶜdqtdt_nudging
        )

    # total specific humidity
    @. Yₜ.c.ρq_tot += Y.c.ρ * ᶜdqtdt_nudging
end

# # For ERA5 reanalysis forcing data
# function external_forcing_cache(Y, external_forcing::ERA5Forcing, params)
#     FT = Spaces.undertype(axes(Y.c))
#     ᶜdTdt_fluc = similar(Y.c, FT)
#     ᶜdqtdt_fluc = similar(Y.c, FT)
#     ᶜdTdt_hadv = similar(Y.c, FT)
#     ᶜdqtdt_hadv = similar(Y.c, FT)
#     ᶜdTdt_rad = similar(Y.c, FT)
#     ᶜT_nudge = similar(Y.c, FT)
#     ᶜqt_nudge = similar(Y.c, FT)
#     ᶜu_nudge = similar(Y.c, FT)
#     ᶜv_nudge = similar(Y.c, FT)
#     ᶜinv_τ_wind = similar(Y.c, FT)
#     ᶜinv_τ_scalar = similar(Y.c, FT)
#     ᶜls_subsidence = similar(Y.c, FT)
#     insolation = similar(Fields.level(Y.c.ρ, 1), FT)
#     cos_zenith = similar(Fields.level(Y.c.ρ, 1), FT)

#     (; external_forcing_file, cfsite_number) = external_forcing

#     NC.Dataset(external_forcing_file, "r") do ds

#         function setvar!(cc_field, varname, colidx, zc_gcm, zc_forcing)
#             parent(cc_field[colidx]) .= interp_vertical_prof(
#                 zc_gcm,
#                 zc_forcing,
#                 era5_driven_profile(ds.group[cfsite_number], varname),
#             )
#         end

#         function setvar_subsidence!(
#             cc_field,
#             varname,
#             colidx,
#             zc_gcm,
#             zc_forcing,
#             params,
#         )
#             parent(cc_field[colidx]) .= interp_vertical_prof(
#                 zc_gcm,
#                 zc_forcing,
#                 era5_driven_profile(ds.group[cfsite_number], varname) ./
#                 .-(era5_driven_profile(ds.group[cfsite_number], "rho")) ./
#                 CAP.grav(params),
#             )
#         end

#         function set_insolation!(cc_field)
#             parent(cc_field) .= mean(
#                 ds.group[cfsite_number]["rsdt"][:] ./
#                 ds.group[cfsite_number]["coszen"][:],
#             )
#         end

#         function set_cos_zenith!(cc_field)
#             parent(cc_field) .= ds.group[cfsite_number]["coszen"][1]
#         end

#         zc_forcing = era5_height(ds.group[cfsite_number])
#         Fields.bycolumn(axes(Y.c)) do colidx

#             zc_gcm = Fields.coordinate_field(Y.c).z[colidx]

#             setvar!(ᶜdTdt_hadv, "tntha", colidx, zc_gcm, zc_forcing)
#             setvar!(ᶜdqtdt_hadv, "tnhusha", colidx, zc_gcm, zc_forcing)
#             #setvar!(ᶜdTdt_rad, "tntr", colidx, zc_gcm, zc_forcing)
#             setvar_subsidence!(
#                 ᶜls_subsidence,
#                 "wap",
#                 colidx,
#                 zc_gcm,
#                 zc_forcing,
#                 params,
#             )
#             # GCM states, used for nudging + vertical eddy advection
#             setvar!(ᶜT_nudge, "ta", colidx, zc_gcm, zc_forcing)
#             setvar!(ᶜqt_nudge, "hus", colidx, zc_gcm, zc_forcing)
#             setvar!(ᶜu_nudge, "ua", colidx, zc_gcm, zc_forcing)
#             setvar!(ᶜv_nudge, "va", colidx, zc_gcm, zc_forcing)

#             # vertical eddy advection (Shen et al., 2022; eqn. 9,10)
#             # sum of two terms to give total tendency. First term:
#             setvar!(ᶜdTdt_fluc, "tntva", colidx, zc_gcm, zc_forcing)
#             setvar!(ᶜdqtdt_fluc, "tnhusva", colidx, zc_gcm, zc_forcing)
#             # second term:
#             eddy_vert_fluctuation!(ᶜdTdt_fluc, ᶜT_nudge, ᶜls_subsidence)
#             eddy_vert_fluctuation!(ᶜdqtdt_fluc, ᶜqt_nudge, ᶜls_subsidence)

#             set_insolation!(insolation)
#             set_cos_zenith!(cos_zenith)

#             @. ᶜinv_τ_wind[colidx] = 1 / (6 * 3600)
#             @. ᶜinv_τ_scalar[colidx] = compute_gcm_driven_scalar_inv_τ(zc_gcm)
#         end
#     end

#     return (;
#         ᶜdTdt_fluc,
#         ᶜdqtdt_fluc,
#         ᶜdTdt_hadv,
#         ᶜdqtdt_hadv,
#         #ᶜdTdt_rad,
#         ᶜT_nudge,
#         ᶜqt_nudge,
#         ᶜu_nudge,
#         ᶜv_nudge,
#         ᶜinv_τ_wind,
#         ᶜinv_τ_scalar,
#         ᶜls_subsidence,
#         insolation,
#         cos_zenith,
#     )
# end

function external_forcing_tendency!(
    Yₜ,
    Y,
    p,
    t,
    ::Union{ERA5Forcing, GCMForcing},
)
    # horizontal advection, vertical fluctuation, nudging, subsidence (need to add),
    (; params) = p
    thermo_params = CAP.thermodynamics_params(params)
    (; ᶜspecific, ᶜts, ᶜh_tot) = p.precomputed
    (;
        ᶜdTdt_fluc,
        ᶜdqtdt_fluc,
        ᶜdTdt_hadv,
        ᶜdqtdt_hadv,
        #ᶜdTdt_rad,
        ᶜT_nudge,
        ᶜqt_nudge,
        ᶜu_nudge,
        ᶜv_nudge,
        ᶜls_subsidence,
        ᶜinv_τ_wind,
        ᶜinv_τ_scalar,
    ) = p.external_forcing

    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜuₕ_nudge = p.scratch.ᶜtemp_C12
    @. ᶜuₕ_nudge = C12(Geometry.UVVector(ᶜu_nudge, ᶜv_nudge), ᶜlg)
    @. Yₜ.c.uₕ -= (Y.c.uₕ - ᶜuₕ_nudge) * ᶜinv_τ_wind

    # nudging tendency
    ᶜdTdt_nudging = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_nudging = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt_nudging =
        -(TD.air_temperature(thermo_params, ᶜts) - ᶜT_nudge) * ᶜinv_τ_scalar
    @. ᶜdqtdt_nudging = -(ᶜspecific.q_tot - ᶜqt_nudge) * ᶜinv_τ_scalar

    ᶜdTdt_sum = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_sum = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt_sum = ᶜdTdt_hadv + ᶜdTdt_nudging + ᶜdTdt_fluc
    @. ᶜdqtdt_sum = ᶜdqtdt_hadv + ᶜdqtdt_nudging + ᶜdqtdt_fluc

    T_0 = TD.Parameters.T_0(thermo_params)
    Lv_0 = TD.Parameters.LH_v0(thermo_params)
    cv_v = TD.Parameters.cv_v(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)
    # total energy
    @. Yₜ.c.ρe_tot +=
        Y.c.ρ * (
            TD.cv_m(thermo_params, ᶜts) * ᶜdTdt_sum +
            (
                cv_v * (TD.air_temperature(thermo_params, ᶜts) - T_0) + Lv_0 -
                R_v * T_0
            ) * ᶜdqtdt_sum
        )
    # total specific humidity
    @. Yₜ.c.ρq_tot += Y.c.ρ * ᶜdqtdt_sum

    ## subsidence -->
    ᶠls_subsidence³ = p.scratch.ᶠtemp_CT3
    @. ᶠls_subsidence³ =
        ᶠinterp(ᶜls_subsidence * CT3(unit_basis_vector_data(CT3, ᶜlg)))
    subsidence!(
        Yₜ.c.ρe_tot,
        Y.c.ρ,
        ᶠls_subsidence³,
        ᶜh_tot,
        Val{:first_order}(),
    )
    subsidence!(
        Yₜ.c.ρq_tot,
        Y.c.ρ,
        ᶠls_subsidence³,
        ᶜspecific.q_tot,
        Val{:first_order}(),
    )

    # needed to address top boundary condition for forcings. Otherwise upper portion of domain is anomalously cold
    ρe_tot_top = Fields.level(Yₜ.c.ρe_tot, Spaces.nlevels(axes(Y.c)))
    @. ρe_tot_top = 0.0

    ρq_tot_top = Fields.level(Yₜ.c.ρq_tot, Spaces.nlevels(axes(Y.c)))
    @. ρq_tot_top = 0.0
    # <-- subsidence

    return nothing
end


function external_height(ds, ::ERA5Forcing)
    vec(mean(ds["z"][:], dims = 2))
end

function external_height(ds, ::GCMForcing)
    vec(mean(ds["zg"][:, :], dims = 2))
end

function external_driven_profile_tmean(ds, varname, ::GCMForcing)
    vec(mean(ds[varname][:, :], dims = 2))
end

function external_driven_profile_tmean(ds, varname, ::ERA5Forcing)
    ds[varname][:]
end


external_forcing_cache(Y, atmos::AtmosModel, params) =
    external_forcing_cache(Y, atmos.external_forcing, params)

external_forcing_cache(Y, external_forcing::Nothing, params) = (;)
function external_forcing_cache(
    Y,
    external_forcing::Union{GCMForcing, ERA5Forcing},
    params,
)
    #varname = "z" if external_forcing isa ERA5Forcing else "zg"
    height_tv = TimeVaryingInput(
        axes(Y.c),
        "myfile.nc",
        alpha,
        preprocess_func = (x) -> 1 / x,
    )
    height = similar(Y.c, FT)

    FT = Spaces.undertype(axes(Y.c))
    ᶜdTdt_fluc = similar(Y.c, FT)
end
