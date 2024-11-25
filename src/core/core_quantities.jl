import ClimaCore.Geometry

function f³(
    coord::Geometry.LatLongZPoint,
    global_geom::Geometry.DeepSphericalGlobalGeometry,
    params,
)
    Ω = CAP.Omega(params)
    CT3(
        CT123(
            Geometry.LocalVector(
                Geometry.Cartesian123Vector(zero(Ω), zero(Ω), 2 * Ω),
                global_geom,
                coord,
            ),
        ),
    )
end

function f¹²(
    coord::Geometry.LatLongZPoint,
    global_geom::Geometry.DeepSphericalGlobalGeometry,
    params,
)
    Ω = CAP.Omega(params)
    CT12(
        CT123(
            Geometry.LocalVector(
                Geometry.Cartesian123Vector(zero(Ω), zero(Ω), 2 * Ω),
                global_geom,
                coord,
            ),
        ),
    )
end

# Shallow sphere
function f³(coord::Geometry.LatLongZPoint, global_geom, params)
    Ω = CAP.Omega(params)
    CT3(Geometry.WVector(2 * Ω * sind(coord.lat)))
end

# Shallow cartesian
function f³(coord, global_geom, params)
    f = CAP.f_plane_coriolis_frequency(params)
    CT3(Geometry.WVector(f))
end

f¹²(coord::Geometry.LatLongZPoint, global_geom, params) =
    error("Not supported for $coord, $global_geom.")
f¹²(coord, global_geom, p::CartesianCoriolisParams) =
    error("Not supported for $coord, $global_geom.")

Φ(g, ᶠz) = g * z
