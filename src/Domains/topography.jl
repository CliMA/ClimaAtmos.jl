abstract type AbstractTopography end

"""
  CanonicalSurface <: AbstractTopography
Specification when simple surfaces are assembled 
Returns input coordinate (canonical surface for plane/box/sphere).
(e.g. canonical box, sphere)
"""
struct CanonicalSurface <: AbstractTopography end

"""
  WarpedSurface <: AbstractTopography
Specification when surface warp is defined by analytical function 
Returns object with new warped coordinate surface and appropriate
interior adapted mesh.
(e.g. Gaussian mountains)
"""
struct WarpedSurface <: AbstractTopography
    surface_function::Function
    interior_warping::LinearAdaption
end
function AnalyticalTopography(f::Function; interior_warping = LinearAdaption())
    # Assumes user prescribes functions that "morphs" bottom surface
    # based on a smooth, analytical profile. 
    return WarpedSurface(f, interior_warping)
end
