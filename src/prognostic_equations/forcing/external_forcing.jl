#####
##### External forcing (for single column experiments)
#####

import Thermodynamics as TD
import ClimaCore.Spaces as Spaces
import ClimaCore.Fields as Fields

external_forcing_cache(Y, atmos::AtmosModel) =
    external_forcing_cache(Y, atmos.external_forcing)

external_forcing_cache(Y, ::Nothing) = (;)
function external_forcing_cache(Y, ::GCMForcing)
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

    # TODO: read profiles from LES files and add here
    @. ᶜdTdt_fluc = 0
    @. ᶜdqtdt_fluc = 0
    @. ᶜdTdt_hadv = 0
    @. ᶜdqtdt_hadv = 0
    @. ᶜdTdt_rad = 0
    @. ᶜT_nudge = 290
    @. ᶜqt_nudge = FT(0.01)
    @. ᶜu_nudge = -5
    @. ᶜv_nudge = 0
    @. ᶜls_subsidence = 0
    # TODO: make it a function of z and add timescale to climaparams
    hr = 3600
    @. ᶜτ_wind = 6hr
    @. ᶜτ_scalar = 24hr
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
    (; ᶜspecific, ᶜts) = p.precomputed
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
    @. Yₜ.c.ρe_tot[colidx] +=
        Y.c.ρ[colidx] * (
            TD.cv_m(thermo_params, ᶜts[colidx]) * ᶜdTdt_sum[colidx] +
            (
                cv_v * (TD.air_temperature(thermo_params, ᶜts[colidx]) - T_0) +
                Lv_0 - R_v * T_0
            ) * ᶜdqtdt_sum[colidx]
        )
    @. Yₜ.c.ρq_tot[colidx] += Y.c.ρ[colidx] * ᶜdqtdt_sum[colidx]

    return nothing
end
