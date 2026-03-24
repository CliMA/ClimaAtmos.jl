"""
    DryDensityCurrentProfile()

A dry density current (cold bubble) setup. A cosine-shaped negative potential
temperature perturbation is centered at (x=25600, z=2000) m, producing a
negatively buoyant region that drives a density current.

Handles both 2D (XZ) and 3D (XYZ) domains automatically.

## Example
```julia
setup = DryDensityCurrentProfile()
```
"""
struct DryDensityCurrentProfile end

function center_initial_condition(::DryDensityCurrentProfile, local_geometry, params)
    FT = eltype(params)
    grav = CAP.grav(params)
    cp_d = CAP.cp_d(params)
    p_0 = CAP.p_ref_theta(params)
    R_d = CAP.R_d(params)

    ndims = length(propertynames(local_geometry.coordinates))
    (; x, z) = local_geometry.coordinates
    x_c = FT(25600)
    x_r = FT(4000)
    z_c = FT(2000)
    z_r = FT(2000)
    r_c = FT(1)
    θ_b = FT(300)
    θ_c = FT(-15)

    r² = ((x - x_c) / x_r)^2 + ((z - z_c) / z_r)^2
    if ndims == 3
        (; y) = local_geometry.coordinates
        y_r = FT(2000)
        y_c = FT(3200)
        r² += ((y - y_c) / y_r)^2
    end
    θ_p = sqrt(r²) < r_c ? FT(1 / 2) * θ_c * (FT(1) + cospi(sqrt(r²) / r_c)) : FT(0)
    θ = θ_b + θ_p
    π_exn = FT(1) - grav * z / cp_d / θ
    T = π_exn * θ
    p = p_0 * π_exn^(cp_d / R_d)

    return physical_state(; T, p)
end
