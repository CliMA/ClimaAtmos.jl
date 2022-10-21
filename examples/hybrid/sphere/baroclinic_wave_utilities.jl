using Statistics: mean
using SurfaceFluxes
using CloudMicrophysics
const SF = SurfaceFluxes
const CCG = ClimaCore.Geometry
import ClimaAtmos.TurbulenceConvection as TC
import ClimaCore.Operators as CCO
const CM = CloudMicrophysics
import ClimaAtmos.Parameters as CAP

include("../staggered_nonhydrostatic_model.jl")
include("./topography.jl")
include("../initial_conditions.jl")

# 0-Moment Microphysics

microphysics_cache(Y, ::Nothing) = NamedTuple()
microphysics_cache(Y, ::Microphysics0Moment) = (
    ᶜS_ρq_tot = similar(Y.c, FT),
    ᶜλ = similar(Y.c, FT),
    ᶜ3d_rain = similar(Y.c, FT),
    ᶜ3d_snow = similar(Y.c, FT),
    col_integrated_rain = similar(ClimaCore.Fields.level(Y.c.ρ, 1), FT),
    col_integrated_snow = similar(ClimaCore.Fields.level(Y.c.ρ, 1), FT),
)

function zero_moment_microphysics_tendency!(Yₜ, Y, p, t, colidx)
    (;
        ᶜts,
        ᶜΦ,
        ᶜT,
        ᶜ3d_rain,
        ᶜ3d_snow,
        ᶜS_ρq_tot,
        ᶜλ,
        col_integrated_rain,
        col_integrated_snow,
        params,
    ) = p # assume ᶜts has been updated
    thermo_params = CAP.thermodynamics_params(params)
    cm_params = CAP.microphysics_params(params)
    @. ᶜS_ρq_tot[colidx] =
        Y.c.ρ[colidx] * CM.Microphysics0M.remove_precipitation(
            cm_params,
            TD.PhasePartition(thermo_params, ᶜts[colidx]),
        )
    @. Yₜ.c.ρq_tot[colidx] += ᶜS_ρq_tot[colidx]
    @. Yₜ.c.ρ[colidx] += ᶜS_ρq_tot[colidx]

    # update precip in cache for coupler's use
    # 3d rain and snow
    @. ᶜT[colidx] = TD.air_temperature(thermo_params, ᶜts[colidx])
    @. ᶜ3d_rain[colidx] =
        ifelse(ᶜT[colidx] >= FT(273.15), ᶜS_ρq_tot[colidx], FT(0))
    @. ᶜ3d_snow[colidx] =
        ifelse(ᶜT[colidx] < FT(273.15), ᶜS_ρq_tot[colidx], FT(0))
    CCO.column_integral_definite!(col_integrated_rain[colidx], ᶜ3d_rain[colidx])
    CCO.column_integral_definite!(col_integrated_snow[colidx], ᶜ3d_snow[colidx])

    @. col_integrated_rain[colidx] =
        col_integrated_rain[colidx] / CAP.ρ_cloud_liq(params)
    @. col_integrated_snow[colidx] =
        col_integrated_snow[colidx] / CAP.ρ_cloud_liq(params)

    # liquid fraction
    @. ᶜλ[colidx] = TD.liquid_fraction(thermo_params, ᶜts[colidx])

    if :ρe_tot in propertynames(Y.c)
        @. Yₜ.c.ρe_tot[colidx] +=
            ᶜS_ρq_tot[colidx] * (
                ᶜλ[colidx] *
                TD.internal_energy_liquid(thermo_params, ᶜts[colidx]) +
                (1 - ᶜλ[colidx]) *
                TD.internal_energy_ice(thermo_params, ᶜts[colidx]) +
                ᶜΦ[colidx]
            )
    elseif :ρe_int in propertynames(Y.c)
        @. Yₜ.c.ρe_int[colidx] +=
            ᶜS_ρq_tot[colidx] * (
                ᶜλ[colidx] *
                TD.internal_energy_liquid(thermo_params, ᶜts[colidx]) +
                (1 - ᶜλ[colidx]) *
                TD.internal_energy_ice(thermo_params, ᶜts[colidx])
            )
    end
    return nothing
end
