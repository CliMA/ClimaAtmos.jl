"""
    init_solid_body_rotation(FT, params, model)

    Hydrostatically balanced initial condition for 3D sphere benchmarking.
"""
function init_solid_body_rotation(::Type{FT}, params, model) where {FT}
    # physics parameters
    p_0 = CLIMAParameters.Planet.MSLP(params)
    cv_d = CLIMAParameters.Planet.cv_d(params)
    R_d = CLIMAParameters.Planet.R_d(params)
    T_tri = CLIMAParameters.Planet.T_triple(params)
    grav = CLIMAParameters.Planet.grav(params)

    # initial condition specific parameters
    T_0 = 300.0
    H = R_d * T_0 / grav # scale height

    # function space needed for initialize Fields
    hv_center_space, hv_face_space = make_function_space(model.domain)
    c_coords = Fields.coordinate_field(hv_center_space)
    f_coords = Fields.coordinate_field(hv_face_space)
    z_top = model.domain.height
    n_vert = model.domain.nelements[2]

    # geopotential
    Φ(z) = grav * z
    # pressure
    function pressure(ρ, e, normuvw, z)
        I = e - Φ(z) - normuvw^2 / 2
        T = I / cv_d + T_tri
        return ρ * R_d * T
    end
    # analytical formula: initial conditions for density and total energy
    function init_sbr_thermo(z)
        p = p_0 * exp(-z / H)
        ρ = 1 / R_d / T_0 * p

        e = cv_d * (T_0 - T_tri) + Φ(z)
        ρe = ρ * e

        return (ρ = ρ, ρe = ρe)
    end
    
    # discrete hydrostatic profile
    zc_vec = parent(c_coords.z) |> unique

    N = length(zc_vec)

    var = init_sbr_thermo.(zc_vec)
    ρ = []
    ρe = []
    p = []
    for i in 1:N
        append!(ρ, var[i].ρ)
        append!(ρe, var[i].ρe)
        append!(p, pressure(var[i].ρ, var[i].ρe / var[i].ρ, 0.0, zc_vec[i]))
    end

    ρ_ana = copy(ρ) # keep a copy for analytical ρ which will be used in correction ρe

    function discrete_hydrostatic_balance!(ρ, p, dz, grav)
        for i in 1:(length(ρ) - 1)
            ρ[i + 1] = -ρ[i] - 2 * (p[i + 1] - p[i]) / dz / grav
        end
    end

    discrete_hydrostatic_balance!(ρ, p, z_top / n_vert, grav)
    # now ρ (after correction) and p (computed from analytical relation) are in discrete hydrostatic balance
    # only need to correct ρe without changing ρ and p, i.e., keep ρT unchanged before vs after the correction on ρ 
    ρe = @. ρe + (ρ - ρ_ana) * Φ(zc_vec) - (ρ - ρ_ana) * cv_d * T_tri

    # Note: In princile, ρe = @. cv_d * p /R_d - ρ * cv_d * T_tri + ρ * Φ(zc_vec) should work, 
    #       however, it is not as accurate as the above correction

    # set up initial condition: not discretely balanced; only create a Field as a place holder
    cρ = map(_ -> Geometry.Scalar(0.0), c_coords)
    cρe = map(_ -> Geometry.Scalar(0.0), c_coords)
    
    # put the dicretely balanced ρ and ρe into Yc
    parent(cρ) .= ρ  # Yc.ρ is a VIJFH layout
    parent(cρe) .= ρe

    # initialize velocity: at rest
    uh = map(_ -> Geometry.Covariant12Vector(0.0, 0.0), c_coords)
    w = map(_ -> Geometry.Covariant3Vector(0.0), f_coords)
    
    return (ρ = cρ, ρe_tot = cρe, uh = uh, w = w)
end
