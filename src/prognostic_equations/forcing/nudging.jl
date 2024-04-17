#####
##### Nudging/relaxation tendency for GCM-driven SCM
#####

import ClimaCore.Fields as Fields

nudging_cache(Y, atmos::AtmosModel) = nudging_cache(Y, atmos.nudging)

nudging_cache(Y, ::Nothing) = (;)
nudging_tendency!(Yₜ, Y, p, t, colidx, ::Nothing) = nothing

function nudging_cache(Y, ::Nudging)
    FT = Spaces.undertype(axes(Y.c))
    ᶜT_mean = similar(Y.c, FT)
    ᶜq_tot_mean = similar(Y.c, FT)
    ᶜu_mean = similar(Y.c, FT)
    ᶜv_mean = similar(Y.c, FT)
    # TODO: read profiles from LES
    @. ᶜT_mean = 290
    @. ᶜq_tot_mean = 0.010
    @. ᶜu_mean = 5
    @. ᶜv_mean = 0
    # TODO: make it a function of z and add timescale to climaparams
    hr = 3600
    ᶜτ_scalar = 24hr
    ᶜτ_wind = 6hr
    return (; ᶜτ_scalar, ᶜτ_wind, ᶜT_mean, ᶜq_tot_mean, ᶜu_mean, ᶜv_mean)
end

function nudging_tendency!(Yₜ, Y, p, t, colidx, ::Nudging)

    thermo_params = CAP.thermodynamics_params(p.params)
    T_0 = TD.Parameters.T_0(thermo_params)
    Lv_0 = TD.Parameters.LH_v0(thermo_params)
    cv_v = TD.Parameters.cv_v(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)

    (; ᶜu_mean, ᶜv_mean, ᶜτ_wind, ᶜτ_scalar, ᶜT_mean, ᶜq_tot_mean) = p.nudging
    (; ᶜspecific, ᶜts) = p.precomputed
    ᶜlg = Fields.local_geometry_field(Y.c)
    ᶜuₕ_mean = p.scratch.ᶜtemp_C12
    @. ᶜuₕ_mean[colidx] =
        C12(Geometry.UVVector(ᶜu_mean[colidx], ᶜv_mean[colidx]), ᶜlg[colidx])
    @. Yₜ.c.uₕ[colidx] -= (Y.c.uₕ[colidx] - ᶜuₕ_mean[colidx]) / ᶜτ_wind

    ᶜdTdt_nudging = p.scratch.ᶜtemp_scalar
    ᶜdqtdt_nudging = p.scratch.ᶜtemp_scalar_2
    @. ᶜdTdt_nudging[colidx] =
        -(TD.air_temperature(thermo_params, ᶜts[colidx]) - ᶜT_mean[colidx]) /
        ᶜτ_scalar
    @. ᶜdqtdt_nudging[colidx] =
        -(ᶜspecific.q_tot[colidx] - ᶜq_tot_mean[colidx]) / ᶜτ_scalar

    @. Yₜ.c.ρe_tot[colidx] +=
        Y.c.ρ[colidx] * (
            TD.cv_m(thermo_params, ᶜts[colidx]) * ᶜdTdt_nudging[colidx] +
            (
                cv_v * (TD.air_temperature(thermo_params, ᶜts[colidx]) - T_0) +
                Lv_0 - R_v * T_0
            ) * ᶜdqtdt_nudging[colidx]
        )
    @. Yₜ.c.ρq_tot[colidx] += Y.c.ρ[colidx] * ᶜdqtdt_nudging[colidx]
end
