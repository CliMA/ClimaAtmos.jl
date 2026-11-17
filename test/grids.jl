using Test
import ClimaAtmos as CA
import ClimaCore as CC

@testset "Grid generation" begin
    @testset "SphereGrid" begin
        # Test sphere grid space-filling curve
        grid = CA.SphereGrid(Float64, h_elem = 3)
        mesh = grid.horizontal_grid.topology.mesh
        sfc_elemorder = CC.Topologies.spacefillingcurve(mesh)

        # ClimaCore.jl/test/Topologies/cubedsphere_sfc.jl indices test for spacefillingcurve
        sfc_orderindex = CC.Meshes.linearindices(sfc_elemorder)
        for (order, cartindex) in enumerate(sfc_elemorder)
            @test sfc_orderindex[cartindex] == order
        end

        # The spacefillingcurve order is not the same as the linear order
        linearorder = vec(collect(CC.Meshes.elements(mesh)))
        @test linearorder != sfc_elemorder
    end

    @testset "Vertical stretching" begin
        stretching(grid) = CC.Grids.vertical_topology(grid).mesh.stretch

        uniform_grid = CA.ColumnGrid(Float64)
        @test stretching(uniform_grid) isa CC.Meshes.Uniform

        fine_grid = CA.ColumnGrid(
            Float64;
            z_elem = 63,
            z_max = 60000.0,
            z_stretch = true,
            dz_bottom = 30.0,
        )
        @test stretching(fine_grid) isa
              CC.Meshes.BoundaryLayerHyperbolicTangentStretching{Float64}
        @test stretching(fine_grid).dz_surface == 30.0
        @test stretching(fine_grid).dz_top == 3000.0
        @test stretching(fine_grid).z_pbl == 2200.0

        fine_sphere = CA.SphereGrid(
            Float64;
            h_elem = 3,
            z_elem = 79,
            z_max = 60000.0,
            z_stretch = true,
            dz_bottom = 28.0,
            dz_top = 2500.0,
            z_pbl = 2000.0,
        )
        @test stretching(fine_sphere).dz_top == 2500.0
        @test stretching(fine_sphere).z_pbl == 2000.0

        # The two-scale rule rejects a `z_elem` outside its admissible range:
        # the 10-level default grid lies below it, and 200 levels with
        # `dz_bottom = 30` m up to 45 km lie above it.
        @test_throws ArgumentError CA.ColumnGrid(Float64; z_stretch = true)
        @test_throws ArgumentError CA.ColumnGrid(
            Float64;
            z_elem = 200,
            z_max = 45000.0,
            z_stretch = true,
            dz_bottom = 30.0,
        )
    end

    @testset "BoxGrid" begin
        # use nh_poly=1 for clearer checks of element ordering (1 node per element)
        # Set x_max and y_max to x_elem and y_elem so coordinates align with indices
        x_elem = x_max = 3
        y_elem = y_max = 6
        grid = CA.BoxGrid(Float64; x_elem, y_elem, x_max, y_max, nh_poly = 1)

        # Extract the space-filling curve from the topology's mesh
        mesh = grid.horizontal_grid.topology.mesh
        spacefilling = CC.Topologies.spacefillingcurve(mesh)

        # We can now verify the coordinates directly (add 1 to account for 1-based indexing)
        # `Nh` is the last dimension of the parent array; the number of leading
        # node/field dimensions depends on the ClimaCore data layout, so take
        # the first index of each instead of hard-coding their count.
        first_node(p) = p[ntuple(_ -> 1, ndims(p) - 1)..., :]
        coords_x = first_node(parent(grid.horizontal_grid.local_geometry.coordinates.x))
        coords_y = first_node(parent(grid.horizontal_grid.local_geometry.coordinates.y))
        coords = @. CartesianIndex(Int(coords_x + 1), Int(coords_y + 1))

        @test coords == spacefilling

        # ClimaCore.jl/test/Topologies/rectangle_sfc.jl indices test for spacefillingcurve
        sfc_orderindex = CC.Meshes.linearindices(spacefilling)
        for (order, cartindex) in enumerate(spacefilling)
            @test sfc_orderindex[cartindex] == order
        end

        # The spacefillingcurve order is not the same as the linear order
        linearorder = vec(collect(CC.Meshes.elements(mesh)))
        @test linearorder != spacefilling
    end

    @testset "MultiColumnGrid" begin
        n_columns = 3
        grid = CA.MultiColumnGrid(Float64; n_columns, z_elem = 5)
        (; center_space) = CA.get_spaces(grid)
        @test center_space isa CC.Spaces.MultiColumnFiniteDifferenceSpace
        @test CC.Spaces.ncolumns(center_space) == n_columns

        coords = CC.Fields.coordinate_field(center_space)
        # We assign each point the coordinates of zero for both latitude and
        # longitude
        @test all(iszero, parent(coords.lat))
        @test all(iszero, parent(coords.long))
        column_space = CA.get_spaces(CA.ColumnGrid(Float64; z_elem = 5)).center_space
        @test vec(parent(CC.Fields.column(coords, 1, 1, n_columns).z)) ==
              vec(parent(CC.Fields.coordinate_field(column_space).z))

        @test_throws ErrorException CA.MultiColumnGrid(Float64; n_columns = 0)
    end

    @testset "Vertical mesh from explicit faces (z_faces)" begin
        faces = [0.0, 100.0, 250.0, 500.0, 1000.0]
        mesh = CA.z_faces_mesh(Float32, faces, 1000.0)
        @test [CC.Geometry.component(f, 1) for f in mesh.faces] == Float32.(faces)
        grid = CA.ColumnGrid(
            Float64;
            z_elem = 4,
            z_max = 1000.0,
            z_mesh = CA.z_faces_mesh(Float64, faces, 1000.0),
        )
        (; face_space) = CA.get_spaces(grid)
        @test vec(parent(CC.Fields.coordinate_field(face_space).z)) == faces
        # The faces must rise strictly from 0 to z_max.
        @test_throws ArgumentError CA.z_faces_mesh(Float64, [0.0, 500.0, 900.0], 1000.0)
        @test_throws ArgumentError CA.z_faces_mesh(Float64, [10.0, 500.0, 1000.0], 1000.0)
        @test_throws ArgumentError CA.z_faces_mesh(
            Float64,
            [0.0, 500.0, 500.0, 1000.0],
            1000.0,
        )
        @test_throws ArgumentError CA.z_faces_mesh(Float64, [0.0, 1000.0], 999.0)
        @test_throws ArgumentError CA.z_faces_mesh(Float64, [0.0], 0.0)
        @test_throws ArgumentError CA.z_faces_from_config("no_such_preset", 60000.0)
        # The preset is built for a 60 km top only.
        @test_throws ArgumentError CA.z_faces_from_config("hybrid_0p5km", 55000.0)
        @test_throws ArgumentError CA.hybrid_z_faces(; z_max = 80000.0)
        # Layers down to 0.1 m are allowed, thinner ones are not.
        @test CA.z_faces_mesh(Float64, [0.0, 0.1, 1000.0], 1000.0) isa
              CC.Meshes.IntervalMesh
        @test_throws ArgumentError CA.z_faces_mesh(Float64, [0.0, 0.05, 1000.0], 1000.0)
        @test isnothing(CA.z_faces_from_config(nothing, 60000.0))
        @test CA.z_faces_from_config([0, 1, 2], 2.0) == [0.0, 1.0, 2.0]

        # The hybrid 0.5 km preset
        f = CA.z_faces_from_config("hybrid_0p5km", 60000.0)
        dz = diff(f)
        z_c = (f[1:(end - 1)] .+ f[2:end]) ./ 2
        @test f[1] == 0 && f[end] == 60000.0
        @test all(>(0), dz)
        # Below the shrinking, the faces of the 63-level two-scale grid, exactly.
        base_mesh = CC.CommonGrids.DefaultZMesh(
            Float64; z_min = 0, z_max = 60000.0, z_elem = 63,
            stretch = CC.Meshes.BoundaryLayerHyperbolicTangentStretching{Float64}(
                30.0,
                3000.0,
                2200.0,
            ),
        )
        base = [CC.Geometry.component(x, 1) for x in base_mesh.faces]
        k = findfirst(i -> f[i] != base[i], eachindex(f)) - 1   # last shared face
        @test f[1:k] == base[1:k]
        @test 5000 < f[k] < 11000
        # Uniform 500 m from about 11 to 35 km, reached without a jump: above the
        # inherited 63-level faces (which themselves change by up to 10.6% per level near
        # 3.9 km) and below 35 km, no spacing changes by more than 10% per level.
        strat = (z_c .> 11500) .& (z_c .< 34500)
        @test all(d -> isapprox(d, 500; atol = 1e-6), dz[strat])
        k_strat = findlast(<(34500), z_c)
        @test maximum(abs.(dz[(k + 1):k_strat] ./ dz[k:(k_strat - 1)] .- 1)) <= 0.1 + 1e-9
        k_start = findfirst(d -> isapprox(d, 500; atol = 1e-6), dz)
        @test 10500 < f[k_start] < 11500
        # Above 35 km: growing geometrically (about 11% per level) to about 3 km at the top.
        top = z_c .> 35000
        growth = dz[top][2:end] ./ dz[top][1:(end - 1)]
        @test all(r -> isapprox(r, growth[1]; rtol = 1e-9), growth)
        @test isapprox(growth[1], 1.113; atol = 1e-3)
        @test length(dz) == 106
        @test count(isapprox.(dz, 500; atol = 1e-6)) == 48
        @test 2700 < dz[end] < 3300
        # Through the config key, on a column grid.
        grid = CA.get_grid(
            Dict(
                "config" => "column", "n_columns" => 1, "z_elem" => 10,
                "z_max" => 60000.0,
                "z_stretch" => true, "dz_bottom" => 30.0, "dz_top" => 3000.0,
                "z_pbl" => 2200.0, "z_faces" => "hybrid_0p5km",
            ),
            CA.ClimaAtmosParameters(Float64),
            CA.ClimaComms.context(),
        )
        (; face_space) = CA.get_spaces(grid)
        @test vec(parent(CC.Fields.coordinate_field(face_space).z)) == f
    end

    @testset "Explicit faces equal to a built-in mesh give the same spaces" begin
        # The faces of the built-in uniform and 63-level two-scale meshes, given as an
        # explicit list, must give the same faces, centres and local geometry, bit for bit.
        for FT in (Float32, Float64),
            (z_elem, z_max, z_stretch) in
            ((10, 30000.0, false), (63, 60000.0, true))

            kw = (;
                z_elem,
                z_max,
                z_stretch,
                dz_bottom = 30.0,
                dz_top = 3000.0,
                z_pbl = 2200.0,
            )
            builtin = CA.get_z_mesh(FT; kw...)
            faces = [Float64(CC.Geometry.component(x, 1)) for x in builtin.faces]
            explicit = CA.z_faces_mesh(FT, faces, z_max)
            @test explicit.faces == builtin.faces
            for make in (
                m -> CA.ColumnGrid(FT; z_elem, z_max, z_mesh = m),
                m -> CA.SphereGrid(FT; z_elem, z_max, z_mesh = m, h_elem = 2),
            )
                a = CA.get_spaces(make(builtin))
                b = CA.get_spaces(make(explicit))
                for sp in (:center_space, :face_space)
                    ga = CC.Fields.local_geometry_field(getproperty(a, sp))
                    gb = CC.Fields.local_geometry_field(getproperty(b, sp))
                    @test parent(ga) == parent(gb)
                end
            end
        end
    end

    @testset "Vertical operators on the hybrid faces converge at second order" begin
        # Refine the hybrid faces by cubic Hermite interpolation of z(ξ) at the face
        # index ξ (Catmull-Rom slopes), which keeps the spacing smooth, and compare
        # the vertical operators applied to analytic profiles with the exact values.
        FT = Float64
        Ops = CC.Operators
        f0 = CA.z_faces_from_config("hybrid_0p5km", 60000.0)
        function refine(f, r)
            n = length(f) - 1
            d = [
                i == 1 ? f[2] - f[1] :
                i == n + 1 ? f[n + 1] - f[n] :
                (f[i + 1] - f[i - 1]) / 2 for i in 1:(n + 1)
            ]
            out = [f[1]]
            for i in 1:n, j in 1:r
                t = j / r
                h00, h10 = 2t^3 - 3t^2 + 1, t^3 - 2t^2 + t
                h01, h11 = -2t^3 + 3t^2, t^3 - t^2
                push!(out, h00 * f[i] + h10 * d[i] + h01 * f[i + 1] + h11 * d[i + 1])
            end
            out[end] = f[end]
            return out
        end
        L, H = 20000.0, 7000.0
        prof(z) = sin(2π * z / L) + exp(-z / H)
        dprof(z) = 2π / L * cos(2π * z / L) - exp(-z / H) / H
        iprof(z) = -L / (2π) * cos(2π * z / L) - H * exp(-z / H)
        lin(z) = 3 + 2e-4 * z
        rs = (1, 2, 4, 8)
        errs = Dict{Symbol, Vector{Float64}}()
        trans = Dict{Symbol, Vector{Float64}}()   # 7 to 12 km, the spacing transition
        lin_errs = Dict{Symbol, Float64}()
        for r in rs
            fr = refine(f0, r)
            @test all(>(0), diff(fr)) && fr[1] == 0 && fr[end] == 60000.0
            grid = CA.ColumnGrid(FT; z_elem = length(fr) - 1, z_max = 60000.0,
                z_mesh = CA.z_faces_mesh(FT, fr, 60000.0))
            (; center_space, face_space) = CA.get_spaces(grid)
            ᶜz = CC.Fields.coordinate_field(center_space).z
            ᶠz = CC.Fields.coordinate_field(face_space).z
            zc = vec(parent(ᶜz));
            zf = vec(parent(ᶠz))
            for (g, label) in ((prof, :smooth), (lin, :linear))
                ᶜg = g.(ᶜz);
                ᶠg = g.(ᶠz)
                grad = Ops.GradientC2F(
                    bottom = Ops.SetValue(g(0.0)),
                    top = Ops.SetValue(g(60000.0)),
                )
                div = Ops.DivergenceF2C()
                c2f =
                    Ops.InterpolateC2F(bottom = Ops.Extrapolate(), top = Ops.Extrapolate())
                f2c = Ops.InterpolateF2C()
                ᶠgrad = vec(parent(CC.Geometry.WVector.(grad.(ᶜg))))
                ᶜdiv = vec(parent(div.(CC.Geometry.WVector.(ᶠg))))
                ᶠint = vec(parent(c2f.(ᶜg)))
                ᶜint = vec(parent(f2c.(ᶠg)))
                col = CC.Fields.Field(FT, CC.Spaces.level(face_space, CC.Utilities.half))
                CC.Operators.column_integral_definite!(col, ᶜg)
                dg = label == :smooth ? dprof : (z -> 2e-4)
                ig =
                    label == :smooth ? (iprof(60000.0) - iprof(0.0)) :
                    (3 * 60000.0 + 1e-4 * 60000.0^2)
                inner = 2:(length(zf) - 1)
                e = (
                    grad = abs.(ᶠgrad[inner] .- dg.(zf[inner])),
                    div = abs.(ᶜdiv .- dg.(zc)),
                    c2f = abs.(ᶠint[inner] .- g.(zf[inner])),
                    f2c = abs.(ᶜint .- g.(zc)),
                )
                if label == :smooth
                    for (k, v) in pairs(e)
                        push!(get!(errs, k, Float64[]), maximum(v))
                        win =
                            k in (:grad, :c2f) ? (7000 .< zf[inner] .< 12000) :
                            (7000 .< zc .< 12000)
                        push!(get!(trans, k, Float64[]), maximum(v[win]))
                    end
                    push!(get!(errs, :integral, Float64[]), abs(parent(col)[1] - ig))
                elseif r == 1
                    for (k, v) in pairs(e)
                        lin_errs[k] = maximum(v)
                    end
                    lin_errs[:integral] = abs(parent(col)[1] - ig) / ig
                end
            end
        end
        order(v) = log2(v[end - 1] / v[end])
        for k in (:grad, :div, :c2f, :f2c, :integral)
            @info "hybrid faces, $k: max errors $(errs[k]) (r = $rs), order $(round(order(errs[k]), digits = 2))" *
                  (
                haskey(trans, k) ?
                "; 7-12 km: $(trans[k]), order $(round(order(trans[k]), digits = 2))" : ""
            )
            @test 1.9 < order(errs[k]) < 2.2
            # In 7-12 km the largest face error sits at 8.1 km, where the spacing stops
            # growing and starts shrinking: the refined spacing ratio there tends to 1
            # more slowly, and the face operators (gradient, centre-to-face
            # interpolation) converge at about 1.7-1.85 (1.84 from r = 8 to 16), with
            # errors several times below the column maximum; the centre operators keep 2.
            haskey(trans, k) &&
                @test (k in (:grad, :c2f) ? 1.6 : 1.9) < order(trans[k]) < 2.2
            haskey(trans, k) && @test trans[k][end] < errs[k][end]
        end
        @info "hybrid faces, linear profile (r = 1): $(lin_errs)"
        # Linear profiles: the gradient, the divergence, the face-to-centre interpolation
        # (centres are mid-way between faces) and the midpoint column integral are exact.
        @test lin_errs[:grad] < 1e-12
        @test lin_errs[:div] < 1e-12
        @test lin_errs[:f2c] < 1e-10
        @test lin_errs[:integral] < 1e-12
    end
end
