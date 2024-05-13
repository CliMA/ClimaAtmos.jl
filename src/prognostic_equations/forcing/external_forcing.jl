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

external_forcing_cache(Y, atmos::AtmosModel) =
    external_forcing_cache(Y, atmos.external_forcing)

external_forcing_cache(Y, external_forcing::Nothing) = (;)
function external_forcing_cache(Y, external_forcing::GCMForcing)
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

    (; external_forcing_file) = external_forcing
    imin = 100  # TODO: move into `GCMForcing` (and `parsed_args`)

    NC.Dataset(external_forcing_file, "r") do ds
        function setvar!(cc_field, varname, colidx, zc_gcm, zc_les)
            parent(cc_field[colidx]) .= interp_vertical_prof(
                zc_gcm,
                zc_les,
                gcm_driven_profile_tmean(ds, varname; imin),  # TODO: time-varying tendencies
            )
        end

        function setnudgevar!(cc_field, varname, colidx, zc_gcm, zc_les)
            parent(cc_field[colidx]) .= interp_vertical_prof(
                zc_gcm,
                zc_les,
                gcm_driven_profile(ds, varname)[:, 1],
            )
        end

        zc_les = gcm_driven_reference(ds, "z")[:]
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

            @. ᶜinv_τ_wind[colidx] = 1 / (6 * 3600)
            @. ᶜinv_τ_scalar[colidx] = compute_gcm_driven_scalar_inv_τ(zc_gcm)
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
    )
end

external_forcing_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing
function external_forcing_tendency!(Yₜ, Y, p, t, ::GCMForcing)
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

    ᶜdTdt_nudging = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_nudging = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt_nudging =
        -(TD.air_temperature(thermo_params, ᶜts) - ᶜT_nudge) * ᶜinv_τ_scalar
    @. ᶜdqtdt_nudging = -(ᶜspecific.q_tot - ᶜqt_nudge) * ᶜinv_τ_scalar

    ᶜdTdt_sum = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_sum = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt_sum = ᶜdTdt_hadv + ᶜdTdt_fluc + ᶜdTdt_rad + ᶜdTdt_nudging
    @. ᶜdqtdt_sum = ᶜdqtdt_hadv + ᶜdqtdt_fluc + ᶜdqtdt_nudging

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
    tom(f) = Spaces.level(f, Spaces.nlevels(axes(f)))  # get value at top of the model
    wvec = Geometry.WVector
    RBh = Operators.RightBiasedC2F(; top = Operators.SetValue(tom(ᶜh_tot)))
    RBq = Operators.RightBiasedC2F(;
        top = Operators.SetValue(tom(ᶜspecific.q_tot)),
    )
    @. Yₜ.c.ρe_tot -= Y.c.ρ * ᶜls_subsidence * ᶜdivᵥ(wvec(RBh(ᶜh_tot)))  # ρ⋅w⋅∇h_tot
    @. Yₜ.c.ρq_tot -= Y.c.ρ * ᶜls_subsidence * ᶜdivᵥ(wvec(RBq(ᶜspecific.q_tot)))  # ρ⋅w⋅∇q_tot
    # <-- subsidence

    return nothing
end
