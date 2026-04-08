"""
    RisingThermalBubbleProfile()

A rising thermal bubble setup. A cosine-shaped positive potential temperature
perturbation is centered at (x=500, z=350) m, producing a positively buoyant
region that rises.

Handles both 2D (XZ) and 3D (XYZ) domains automatically.

## Example
```julia
setup = RisingThermalBubbleProfile()
```
"""
struct RisingThermalBubbleProfile end

function center_initial_condition(::RisingThermalBubbleProfile, local_geometry, params)
    FT = eltype(params)
    grav = CAP.grav(params)
    cp_d = CAP.cp_d(params)
    p_0 = CAP.p_ref_theta(params)
    R_d = CAP.R_d(params)

    ndims = length(propertynames(local_geometry.coordinates))
    (; x, z) = local_geometry.coordinates
    x_c = FT(500)
    x_r = FT(250)
    z_c = FT(350)
    z_r = FT(250)
    r_c = FT(1)
    θ_b = FT(300)
    θ_c = FT(0.5)

    r² = ((x - x_c) / x_r)^2 + ((z - z_c) / z_r)^2
    if ndims == 3
        (; y) = local_geometry.coordinates
        y_c = FT(500)
        y_r = FT(250)
        r² += ((y - y_c) / y_r)^2
    end
    θ_p = sqrt(r²) < r_c ? FT(1 / 2) * θ_c * (FT(1) + cospi(sqrt(r²) / r_c)) : FT(0)
    θ = θ_b + θ_p
    π_exn = FT(1) - grav * z / cp_d / θ
    T = π_exn * θ
    p = p_0 * π_exn^(cp_d / R_d)

    return physical_state(; T, p)
end
