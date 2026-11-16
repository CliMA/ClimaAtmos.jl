#####
##### Sedimentation kinetic energy helper
#####

"""
    sedimentation_kinetic_energy(w, uvw)

Return the specific kinetic energy `|uvw - w ẑ|² / 2` [J/kg] of a species that
falls at terminal velocity `w` [m/s] through air moving with the `UVWVector`
`uvw` [m/s].
"""
@inline sedimentation_kinetic_energy(w, uvw) =
    norm_sqr(Geometry.UVWVector(0, 0, -w) + uvw) / 2
