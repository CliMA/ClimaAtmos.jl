import ClimaCore.Geometry

f³(lg, global_geom, params) = f³(lg, lg.coordinates, global_geom, params)

f¹²(lg, global_geom, params) = f¹²(lg, lg.coordinates, global_geom, params)

function f³(
    lg,
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
            lg,
        ),
        lg,
    )
end

function f¹²(
    lg,
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
            lg,
        ),
        lg,
    )
end

# Shallow sphere
function f³(lg, coord::Geometry.LatLongZPoint, global_geom, params)
    Ω = CAP.Omega(params)
    CT3(Geometry.WVector(2 * Ω * sind(coord.lat), lg), lg)
end

# Shallow cartesian
function f³(lg, coord, global_geom, params)
    f = CAP.f_plane_coriolis_frequency(params)
    CT3(Geometry.WVector(f, lg), lg)
end

f¹²(lg, coord::Geometry.LatLongZPoint, global_geom, params) =
    error("Not supported for $coord, $global_geom.")
f¹²(lg, coord, global_geom, p) =
    error("Not supported for $coord, $global_geom.")

Φ(g, z) = g * z
