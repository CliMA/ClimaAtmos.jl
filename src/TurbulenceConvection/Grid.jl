struct Grid{NZ, CS, FS, SC, SF}
    cs::CS
    fs::FS
    zc::SC
    zf::SF
    function Grid(space::CC.Spaces.CenterFiniteDifferenceSpace)
        nz = length(space)
        cs = space
        fs = CC.Spaces.FaceFiniteDifferenceSpace(cs)
        zc = CC.Fields.coordinate_field(cs)
        zf = CC.Fields.coordinate_field(fs)
        CS = typeof(cs)
        FS = typeof(fs)
        SC = typeof(zc)
        SF = typeof(zf)
        return new{nz, CS, FS, SC, SF}(cs, fs, zc, zf)
    end
end

Grid(mesh::CC.Meshes.IntervalMesh) =
    Grid(CC.Spaces.CenterFiniteDifferenceSpace(mesh))

function Grid(Δz::FT, nz::Int) where {FT <: AbstractFloat}
    z₀, z₁ = FT(0), FT(nz * Δz)

    domain = CC.Domains.IntervalDomain(
        CC.Geometry.ZPoint{FT}(z₀),
        CC.Geometry.ZPoint{FT}(z₁),
        boundary_tags = (:bottom, :top),
    )

    mesh = CC.Meshes.IntervalMesh(domain, nelems = nz)
    return Grid(mesh)
end

n_cells(::Grid{NZ}) where {NZ} = NZ

# Index of the first interior cell above the surface
kc_surface(grid::Grid) = Cent(1)
kf_surface(grid::Grid) = CCO.PlusHalf(1)
kc_top_of_atmos(grid::Grid) = Cent(n_cells(grid))
