# This file is included in Diagnostics.jl

# Diagnostics for computing mass and energy conservation
# All the diagnostics in this file are scalars, we need to
# turn them into vectors as the diagnostics need to be mutable
# Related issue: https://github.com/CliMA/ClimaDiagnostics.jl/issues/100

###
# Total mass of the air (scalar)
###
add_diagnostic_variable!(
    short_name = "massa",
    long_name = "Total Mass of the Air",
    units = "kg",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return [sum(state.c.ρ)]
        else
            out .= [sum(state.c.ρ)]
        end
    end,
)

###
# Total energy of air (scalar)
###
add_diagnostic_variable!(
    short_name = "energya",
    long_name = "Total Energy of the Air",
    units = "J",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return [sum(state.c.ρe_tot)]
        else
            out .= [sum(state.c.ρe_tot)]
        end
    end,
)

###
# Total water of the air (scalar)
###
compute_watera!(out, state, cache, time) =
    compute_watera!(out, state, cache, time, cache.atmos.microphysics_model)
compute_watera!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("watera", model)

function compute_watera!(
    out,
    state,
    cache,
    time,
    microphysics_model::T,
) where {T <: MoistMicrophysics}
    if isnothing(out)
        return [sum(state.c.ρq_tot)]
    else
        out .= [sum(state.c.ρq_tot)]
    end
end

add_diagnostic_variable!(
    short_name = "watera",
    long_name = "Total Water of the Air",
    units = "kg",
    compute! = compute_watera!,
)

###
# Total energy of the slab ocean (scalar)
###
compute_energyo!(out, state, cache, time) =
    compute_energyo!(out, state, cache, time, cache.atmos.surface_model)
compute_energyo!(_, _, _, _, model::T) where {T} =
    error_diagnostic_variable("energyo", model)

function compute_energyo!(
    out,
    state,
    cache,
    time,
    surface_model::T,
) where {T <: SlabOceanSST}
    sfc_cρh =
        surface_model.ρ_ocean *
        surface_model.cp_ocean *
        surface_model.depth_ocean
    if isnothing(out)
        return [horizontal_integral_at_boundary(state.sfc.T .* sfc_cρh)]
    else
        out .= [horizontal_integral_at_boundary(state.sfc.T .* sfc_cρh)]
    end
end

add_diagnostic_variable!(
    short_name = "energyo",
    long_name = "Total Energy of the Ocean",
    units = "J",
    compute! = compute_energyo!,
)

###
# Total water of the slab ocean (scalar)
###
compute_watero!(out, state, cache, time) = compute_watero!(
    out,
    state,
    cache,
    time,
    cache.atmos.microphysics_model,
    cache.atmos.surface_model,
)
compute_watero!(
    _,
    _,
    _,
    _,
    microphysics_model::T1,
    surface_model::T2,
) where {T1, T2} = error_diagnostic_variable(
    "Can only compute total water of the ocean with a moist model and with SlabOceanSST",
)

function compute_watero!(
    out,
    state,
    cache,
    time,
    microphysics_model::MoistMicrophysics,
    surface_model::SlabOceanSST,
)
    if isnothing(out)
        return [horizontal_integral_at_boundary(state.sfc.water)]
    else
        out .= [horizontal_integral_at_boundary(state.sfc.water)]
    end
end

add_diagnostic_variable!(
    short_name = "watero",
    long_name = "Total Water of the Ocean",
    units = "kg",
    compute! = compute_watero!,
)
