#####
##### External forcing (for single column experiments)
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields
import NCDatasets as NC
import Dierckx

function interp_vertical_prof(x, xp, fp)
    spl = Dierckx.Spline1D(xp, fp; k = 1)
    return spl(vec(x))
end

external_forcing_cache(Y, atmos::AtmosModel) =
    external_forcing_cache(Y, atmos.external_forcing)

external_forcing_cache(Y, external_forcing::Nothing) = (;)
function external_forcing_cache(
    Y,
    external_forcing::GCMForcing{DType},
) where {DType}
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
    ᶜτ_wind = similar(Y.c, FT)
    ᶜτ_scalar = similar(Y.c, FT)
    ᶜls_subsidence = similar(Y.c, FT)

    external_forcing_file = external_forcing.external_forcing_file
    imin = 100  # TODO: move into `GCMForcing` (and `parsed_args`)

    NC.Dataset(external_forcing_file, "r") do ds
        function setvar!(cc_field, varname, colidx, zc_gcm, zc_les)
            parent(cc_field[colidx]) .= interp_vertical_prof(
                zc_gcm,
                zc_les,
                gcm_driven_profile_tmean(DType, ds, varname; imin),  # TODO: time-varying tendencies
            )
        end

        function setnudgevar!(cc_field, varname, colidx, zc_gcm, zc_les)
            parent(cc_field[colidx]) .= interp_vertical_prof(
                zc_gcm,
                zc_les,
                gcm_driven_profile(DType, ds, varname)[:, 1],
            )
        end

        zc_les = gcm_driven_reference(DType, ds, "z")[:]
        Fields.bycolumn(axes(Y.c)) do colidx

            zc_gcm = Fields.coordinate_field(Y.c).z[colidx]

            setvar!(ᶜdTdt_fluc, "dtdt_fluc", colidx, zc_gcm, zc_les)
            setvar!(ᶜdqtdt_fluc, "dqtdt_fluc", colidx, zc_gcm, zc_les)
            setvar!(ᶜdTdt_hadv, "dtdt_hadv", colidx, zc_gcm, zc_les)
            setvar!(ᶜdqtdt_hadv, "dqtdt_hadv", colidx, zc_gcm, zc_les)
            setvar!(ᶜdTdt_rad, "dtdt_rad", colidx, zc_gcm, zc_les)
            setvar!(ᶜls_subsidence, "ls_subsidence", colidx, zc_gcm, zc_les)

            setnudgevar!(ᶜT_nudge, "temperature_mean", colidx, zc_gcm, zc_les)
            setnudgevar!(ᶜqt_nudge, "qt_mean", colidx, zc_gcm, zc_les)
            setnudgevar!(ᶜu_nudge, "u_mean", colidx, zc_gcm, zc_les)
            setnudgevar!(ᶜv_nudge, "v_mean", colidx, zc_gcm, zc_les)

            # TODO: make it a function of z for scalar (call function above)
            hr = 3600
            parent(ᶜτ_wind[colidx]) .= 6hr
            parent(ᶜτ_scalar[colidx]) .= 24hr
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
        ᶜτ_wind,
        ᶜτ_scalar,
        ᶜls_subsidence,
    )
end

external_forcing_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) = nothing
function external_forcing_tendency!(Yₜ, Y, p, t, colidx, ::GCMForcing)
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
        ᶜτ_wind,
        ᶜτ_scalar,
    ) = p.external_forcing

    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜuₕ_nudge = p.scratch.ᶜtemp_C12
    @. ᶜuₕ_nudge[colidx] =
        C12(Geometry.UVVector(ᶜu_nudge[colidx], ᶜv_nudge[colidx]), ᶜlg[colidx])
    @. Yₜ.c.uₕ[colidx] -= (Y.c.uₕ[colidx] - ᶜuₕ_nudge[colidx]) / ᶜτ_wind[colidx]

    ᶜdTdt_nudging = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_nudging = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt_nudging[colidx] =
        -(TD.air_temperature(thermo_params, ᶜts[colidx]) - ᶜT_nudge[colidx]) /
        ᶜτ_scalar[colidx]
    @. ᶜdqtdt_nudging[colidx] =
        -(ᶜspecific.q_tot[colidx] - ᶜqt_nudge[colidx]) / ᶜτ_scalar[colidx]

    ᶜdTdt_sum = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_sum = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt_sum[colidx] =
        ᶜdTdt_hadv[colidx] +
        ᶜdTdt_fluc[colidx] +
        ᶜdTdt_rad[colidx] +
        ᶜdTdt_nudging[colidx]
    @. ᶜdqtdt_sum[colidx] =
        ᶜdqtdt_hadv[colidx] + ᶜdqtdt_fluc[colidx] + ᶜdqtdt_nudging[colidx]

    T_0 = TD.Parameters.T_0(thermo_params)
    Lv_0 = TD.Parameters.LH_v0(thermo_params)
    cv_v = TD.Parameters.cv_v(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)
    # total energy
    @. Yₜ.c.ρe_tot[colidx] +=
        Y.c.ρ[colidx] * (
            TD.cv_m(thermo_params, ᶜts[colidx]) * ᶜdTdt_sum[colidx] +
            (
                cv_v * (TD.air_temperature(thermo_params, ᶜts[colidx]) - T_0) +
                Lv_0 - R_v * T_0
            ) * ᶜdqtdt_sum[colidx]
        )
    # total specific humidity
    @. Yₜ.c.ρq_tot[colidx] += Y.c.ρ[colidx] * ᶜdqtdt_sum[colidx]

    ## subsidence -->
    tom(f) = Spaces.level(f, Spaces.nlevels(axes(f)))  # get value at top of the model
    wvec = Geometry.WVector
    RBh = Operators.RightBiasedC2F(;
        top = Operators.SetValue(tom(ᶜh_tot[colidx])),
    )
    RBq = Operators.RightBiasedC2F(;
        top = Operators.SetValue(tom(ᶜspecific.q_tot[colidx])),
    )
    @. Yₜ.c.ρe_tot[colidx] -=
        Y.c.ρ[colidx] *
        ᶜls_subsidence[colidx] *
        ᶜdivᵥ(wvec(RBh(ᶜh_tot[colidx])))  # ρ⋅w⋅∇h_tot
    @. Yₜ.c.ρq_tot[colidx] -=
        Y.c.ρ[colidx] *
        ᶜls_subsidence[colidx] *
        ᶜdivᵥ(wvec(RBq(ᶜspecific.q_tot[colidx])))  # ρ⋅w⋅∇q_tot
    # <-- subsidence

    return nothing
end
