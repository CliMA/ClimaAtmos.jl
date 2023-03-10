# convective velocity scale
function get_wstar(bflux::FT) where {FT}
    # average depth of the mixed layer (prescribed for now)
    zi = FT(1000)
    return cbrt(max(bflux * zi, 0))
end
