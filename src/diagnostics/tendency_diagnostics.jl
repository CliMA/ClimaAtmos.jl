###
# Temporary diagnostics for microphysics tendencies
###

add_diagnostic_variable!(
    short_name = "accr_sno_ice",
    long_name = "accr_sno_ice",
    standard_name = "accr_sno_ice",
    units = "1/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.scratch.tmp_accr_sno_ice)
        else
            out .= cache.scratch.tmp_accr_sno_ice
        end
    end,
)

add_diagnostic_variable!(
    short_name = "accr_rai_liq",
    long_name = "accr_rai_liq",
    standard_name = "accr_rai_liq",
    units = "1/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.scratch.tmp_accr_rai_liq)
        else
            out .= cache.scratch.tmp_accr_rai_liq
        end
    end,
)

add_diagnostic_variable!(
    short_name = "acnv_ice_sno",
    long_name = "acnv_ice_sno",
    standard_name = "acnv_ice_sno",
    units = "1/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.scratch.tmp_acnv_ice_sno)
        else
            out .= cache.scratch.tmp_acnv_ice_sno
        end
    end,
)

add_diagnostic_variable!(
    short_name = "acnv_liq_rai",
    long_name = "acnv_liq_rai",
    standard_name = "acnv_liq_rai",
    units = "1/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.scratch.tmp_acnv_liq_rai)
        else
            out .= cache.scratch.tmp_acnv_liq_rai
        end
    end,
)

add_diagnostic_variable!(
    short_name = "accr_sno_liq_sno_part",
    long_name = "accr_sno_liq_sno_part",
    standard_name = "accr_sno_liq_sno_part",
    units = "1/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.scratch.tmp_accr_sno_liq_sno_part)
        else
            out .= cache.scratch.tmp_accr_sno_liq_sno_part
        end
    end,
)

add_diagnostic_variable!(
    short_name = "accr_sno_liq_liq_part",
    long_name = "accr_sno_liq_liq_part",
    standard_name = "accr_sno_liq_liq_part",
    units = "1/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.scratch.tmp_accr_sno_liq_liq_part)
        else
            out .= cache.scratch.tmp_accr_sno_liq_liq_part
        end
    end,
)

add_diagnostic_variable!(
    short_name = "accr_rai_ice_sno_part",
    long_name = "accr_rai_ice_sno_part",
    standard_name = "accr_rai_ice_sno_part",
    units = "1/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.scratch.tmp_accr_rai_ice_sno_part)
        else
            out .= cache.scratch.tmp_accr_rai_ice_sno_part
        end
    end,
)

add_diagnostic_variable!(
    short_name = "accr_rai_ice_rai_part",
    long_name = "accr_rai_ice_rai_part",
    standard_name = "accr_rai_ice_rai_part",
    units = "1/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.scratch.tmp_accr_rai_ice_rai_part)
        else
            out .= cache.scratch.tmp_accr_rai_ice_rai_part
        end
    end,
)

add_diagnostic_variable!(
    short_name = "accr_rai_sno",
    long_name = "accr_rai_sno",
    standard_name = "accr_rai_sno",
    units = "1/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.scratch.tmp_accr_rai_sno)
        else
            out .= cache.scratch.tmp_accr_rai_sno
        end
    end,
)

add_diagnostic_variable!(
    short_name = "evap",
    long_name = "evap",
    standard_name = "evap",
    units = "1/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.scratch.tmp_evap)
        else
            out .= cache.scratch.tmp_evap
        end
    end,
)

add_diagnostic_variable!(
    short_name = "melt",
    long_name = "melt",
    standard_name = "melt",
    units = "1/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.scratch.tmp_melt)
        else
            out .= cache.scratch.tmp_melt
        end
    end,
)

add_diagnostic_variable!(
    short_name = "dep_sub",
    long_name = "dep_sub",
    standard_name = "dep_sub",
    units = "1/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.scratch.tmp_dep_sub)
        else
            out .= cache.scratch.tmp_dep_sub
        end
    end,
)

add_diagnostic_variable!(
    short_name = "w_rai",
    long_name = "w_rai",
    standard_name = "w_rai",
    units = "m/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.precomputed.ᶜwᵣ)
        else
            out .= cache.precomputed.ᶜwᵣ
        end
    end,
)

add_diagnostic_variable!(
    short_name = "w_sno",
    long_name = "w_sno",
    standard_name = "w_sno",
    units = "m/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(abs.(cache.precomputed.ᶜwₛ))
        else
            out .= abs.(cache.precomputed.ᶜwₛ)
        end
    end,
)

add_diagnostic_variable!(
    short_name = "w_liq",
    long_name = "w_liq",
    standard_name = "w_liq",
    units = "m/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.precomputed.ᶜwₗ)
        else
            out .= cache.precomputed.ᶜwₗ
        end
    end,
)

add_diagnostic_variable!(
    short_name = "w_ice",
    long_name = "w_ice",
    standard_name = "w_ice",
    units = "m/s",
    compute! = (out, state, cache, time) -> begin
        if isnothing(out)
            return copy(cache.precomputed.ᶜwᵢ)
        else
            out .= cache.precomputed.ᶜwᵢ
        end
    end,
)
