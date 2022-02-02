import TurbulenceConvection
const TC = TurbulenceConvection

function init_1d_rico_column(::Type{FT}, grid::TC.Grid, state) where {FT}
    @unpack z = local_geometry.coordinates
    # velocity
    ρ0_c = TC.center_ref_state(state).ρ0
    uv(local_geometry) = begin
        u0 = -9.9 + 2.0e-3 * z
        v0 = -3.8
        return Geometry.UVVector(u0, v0)
    end
    w(local_geometry) = Geometry.WVector(w0) # w component
    # potential temperature
    ρθ(local_geometry) = begin
        @unpack z = local_geometry.coordinates
        θ = if z <= 740.0
            297.9
        else
            (297.9 + (317.0 - 297.9) / (4000.0 - 740.0) * (z - 740.0))
        end
        return ρ(local_geometry).*θ
    end
    # specific humidity
    ρq_tot(local_geometry) = begin
        @unpack z = local_geometry.coordinates
        q_tot = if z <= 740.0
            (16.0 + (13.8 - 16.0) / 740.0 * z) / 1000.0
        elseif z > 740.0 && z <= 3260.0
            (13.8 + (2.4 - 13.8) / (3260.0 - 740.0) * (z - 740.0)) / 1000.0
        else
            (2.4 + (1.8 - 2.4) / (4000.0 - 3260.0) * (z - 3260.0)) / 1000.0
        end
        return ρ(local_geometry).*q_tot
    end
    return (u = u, v = v, ρθ = ρθ, ρq_tot = ρq_tot)
end
