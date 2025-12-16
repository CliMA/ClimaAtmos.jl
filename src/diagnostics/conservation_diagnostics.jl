# This file is included in Diagnostics.jl

# Diagnostics for computing mass and energy conservation
# All the diagnostics in this file are scalars, we need to
# turn them into vectors as the diagnostics need to be mutable
# Related issue: https://github.com/CliMA/ClimaDiagnostics.jl/issues/100

###
# Total mass of the air (scalar)
###
add_diagnostic_variable!(short_name = "massa", units = "kg",
    long_name = "Total Mass of the Air",
    compute! = (out, state, cache, time) -> begin
        ρ_total = sum(state.c.ρ)
        isnothing(out) ? (out = [ρ_total]) : (out .= ρ_total)
        return out
    end,
)

###
# Total energy of air (scalar)
###
add_diagnostic_variable!(short_name = "energya", units = "J",
    long_name = "Total Energy of the Air",
    compute! = (out, state, cache, time) -> begin
        ρe_total = sum(state.c.ρe_tot)
        isnothing(out) ? (out = [ρe_total]) : (out .= ρe_total)
        return out
    end,
)

###
# Total water of the air (scalar)
###
compute_watera!(out, state, cache, time) =
    compute_watera!(out, state, cache, time, cache.atmos.moisture_model)
compute_watera!(_, _, _, _, model) = error_diagnostic_variable("watera", model)

function compute_watera!(out, state, _, _, ::Union{EquilMoistModel, NonEquilMoistModel})
    ρq_total = sum(state.c.ρq_tot)
    isnothing(out) ? (out = [ρq_total]) : (out .= ρq_total)
    return out
end

add_diagnostic_variable!(short_name = "watera", units = "kg",
    long_name = "Total Water of the Air",
    compute! = compute_watera!,
)

###
# Total energy of the slab ocean (scalar)
###
compute_energyo!(out, state, cache, time) =
    compute_energyo!(out, state, cache, time, cache.atmos.surface_model)
compute_energyo!(_, _, _, _, model) = error_diagnostic_variable("energyo", model)

function compute_energyo!(out, state, _, _, surface_model::SlabOceanSST)
    sfc_cρh = surface_model.ρ_ocean * surface_model.cp_ocean * surface_model.depth_ocean
    energyo = horizontal_integral_at_boundary(@. lazy(state.sfc.T * sfc_cρh))
    isnothing(out) ? (out = [energyo]) : (out .= energyo)
    return out
end

add_diagnostic_variable!(short_name = "energyo", units = "J",
    long_name = "Total Energy of the Ocean",
    compute! = compute_energyo!,
)

###
# Total water of the slab ocean (scalar)
###
compute_watero!(out, state, cache, time) = compute_watero!(
    out, state, cache, time, cache.atmos.moisture_model, cache.atmos.surface_model,
)
compute_watero!(_, _, _, _, _, _, _, _) = error_diagnostic_variable(
    "Can only compute total water of the ocean with a moist model and with SlabOceanSST",
)

function compute_watero!(out, state, _, _,
    ::Union{EquilMoistModel, NonEquilMoistModel}, ::SlabOceanSST,
)
    watero = horizontal_integral_at_boundary(state.sfc.water)
    isnothing(out) ? (out = [watero]) : (out .= watero)
    return out
end

add_diagnostic_variable!(short_name = "watero", units = "kg",
    long_name = "Total Water of the Ocean",
    compute = compute_watero!,
)
