import ClimaCore.Geometry

f³(lg, fparams) = f³(lg, lg.coordinates, fparams.global_geom, fparams)

f¹²(lg, fparams) = f¹²(lg, lg.coordinates, fparams.global_geom, fparams)

function f³(
    lg,
    coord::Geometry.LatLongZPoint,
    global_geom::Geometry.DeepSphericalGlobalGeometry,
    params,
)
    (; Ω) = params
    c123 = Geometry.Cartesian123Vector(zero(Ω), zero(Ω), 2 * Ω)
    lv = Geometry.LocalVector(c123, global_geom, coord)
    CT3(CT123(lv, lg), lg)
end

function f¹²(
    lg,
    coord::Geometry.LatLongZPoint,
    global_geom::Geometry.DeepSphericalGlobalGeometry,
    params,
)
    (; Ω) = params
    c123 = Geometry.Cartesian123Vector(zero(Ω), zero(Ω), 2 * Ω)
    lv = Geometry.LocalVector(c123, global_geom, coord)
    CT12(CT123(lv, lg), lg)
end

# Shallow sphere
function f³(lg, coord::Geometry.LatLongZPoint, global_geom, params)
    (; Ω) = params
    CT3(Geometry.WVector(2 * Ω * sind(coord.lat), lg), lg)
end

# Shallow cartesian
function f³(lg, coord, global_geom, params)
    (; f_plane_coriolis_frequency) = params
    CT3(Geometry.WVector(f, lg), lg)
end

f¹²(lg, coord::Geometry.LatLongZPoint, global_geom, params) =
    error("Not supported for $coord, $global_geom.")
f¹²(lg, coord, global_geom, params) =
    error("Not supported for $coord, $global_geom.")

Φ(g, z) = g * z
