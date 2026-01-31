using Test
using ClimaAtmos
using ClimaComms
using ClimaCore
using Statistics
import ClimaCore: Fields, Spaces, Domains, Meshes, Topologies, Geometry

# Include the source file directly for unit testing functions
# (Unless it's already included in ClimaAtmos and we can import it? 
#  But these functions are not exported usually. We rely on the source file logic or internal access)
# Since the file is included in `diagnostic.jl` which is in ClimaAtmos, we might access them via ClimaAtmos if exported?
# They are NOT exported.
# So I should include the file again? Or access via ClimaAtmos.Internal?
# They are defined at top level in `src/diagnostics/negative_scalars_diagnostics.jl`.
# So they are in `ClimaAtmos` module.
# Check if `ClimaAtmos` exports them. No.
# So I access them as `ClimaAtmos.compute_min_per_level!`.

# Wait, if I `using ClimaAtmos`, I can access internals via `ClimaAtmos...`.

function test_negative_scalars_diagnostics()
    FT = Float64
    
    # Setup Vertical Domain
    vertdomain = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(10);
        boundary_names = (:bottom, :top),
    )
    vertmesh = Meshes.IntervalMesh(vertdomain, nelems = 2)
    vert_center_space = Spaces.CenterFiniteDifferenceSpace(vertmesh)
    
    # Setup Horizontal Domain
    x_domain = Domains.IntervalDomain(
        Geometry.XPoint{FT}(0),
        Geometry.XPoint{FT}(1);
        periodic = true,
    )
    y_domain = Domains.IntervalDomain(
        Geometry.YPoint{FT}(0),
        Geometry.YPoint{FT}(1);
        periodic = true,
    )
    horzdomain = Domains.RectangleDomain(x_domain, y_domain)
    
    # 2x2 elements, GLL{3} (3x3 nodes per element)
    horzmesh = Meshes.RectilinearMesh(horzdomain, 2, 2) 
    quad = Spaces.Quadratures.GLL{3}() 
    horztopology = Topologies.Topology2D(ClimaComms.SingletonCommsContext(), horzmesh)
    horzspace = Spaces.SpectralElementSpace2D(horztopology, quad)
    
    hv_center_space = Spaces.ExtrudedFiniteDifferenceSpace(horzspace, vert_center_space)
    
    # Create Field
    field = Fields.Field(FT, hv_center_space)
    coords = Fields.coordinate_field(hv_center_space)
    
    # Initialize Field
    # Level 1 (0 to 5m): Constant +5.0
    # Level 2 (5 to 10m): Negative Pattern
    # Use x < 0.6 pattern which we know results in 58.33% nodes.
    # Pattern: -2.0 if x < 0.6 else +2.0.
    
    function set_val(coord)
        if coord.z < 5.0 
            return 5.0
        else 
            return coord.x < 0.6 ? -2.0 : 2.0
        end
    end
    
    field .= set_val.(coords)
    
    # Prepare Outputs
    out_min = Fields.Field(FT, vert_center_space)
    out_neg_mean = Fields.Field(FT, vert_center_space)
    out_neg_frac = Fields.Field(FT, vert_center_space)
    
    # Call Functions (using ClimaAtmos namespace)
    # Using `ClimaAtmos.compute_min_per_level!` etc.
    
    ClimaAtmos.Diagnostics.compute_min_per_level!(out_min, field)
    ClimaAtmos.Diagnostics.compute_negative_mean_per_level!(out_neg_mean, field)
    ClimaAtmos.Diagnostics.compute_negative_fraction_per_level!(out_neg_frac, field)
    
    # Verification
    
    # Level 1
    # Min: 5.0
    # Neg Mean: 0.0 (Implementation returns 0 if no negatives, unlike my previous NaN)
    # Neg Frac: 0.0
    
    min_1 = parent(Fields.level(out_min, 1))[1]
    neg_mean_1 = parent(Fields.level(out_neg_mean, 1))[1]
    neg_frac_1 = parent(Fields.level(out_neg_frac, 1))[1]
    
    @test min_1 ≈ 5.0
    @test neg_mean_1 ≈ 0.0  # Implementation uses min(field, 0), so 5->0. mean(0) = 0.
    @test neg_frac_1 ≈ 0.0
    
    # Level 2
    # Min: -2.0
    # Neg Mean: -2.0 (Negatives are -2.0. Zeros from > 0 do NOT contribute to negative mean?
    #   Wait, user implementation: `clipped = min(field, 0)`. `mean(clipped)`.
    #   If field has -2 and +2.
    #   clipped has -2 and 0.
    #   mean(clipped) includes the ZEROs!
    #   So mean will be smaller (closer to 0) than -2.0.
    #   Previous implementation filtered negatives.
    #   User implementation: `mean(min(field, 0))`.
    #   So it averages over the WHOLE level.
    #   Let's calculate expectation.
    #   Negatives (-2.0) are 58.33% of nodes.
    #   Zeros (from +2.0 which became 0) are 41.67%.
    #   Mean = 0.5833 * (-2.0) + 0.4167 * (0).
    #   Mean = -1.1666...
    #   
    
    # Neg Frac:
    #   `reduce(+, neg_mask) / reduce(+, one)`.
    #   Should be 58.33%. 
    #   Check if `reduce(+, ...)` works as Expected sum.
    #   Implementation: `sum(-sign(min(f, 0)))`. 
    #   If f=-2 -> min=-2 -> sign=-1 -> -sign=1.
    #   If f=2 -> min=0 -> sign=0 -> -sign=0.
    #   So it counts negatives. Confirmed.
    
    min_2 = parent(Fields.level(out_min, 2))[1]
    neg_mean_2 = parent(Fields.level(out_neg_mean, 2))[1]
    neg_frac_2 = parent(Fields.level(out_neg_frac, 2))[1]
    
    expected_frac = 21/36 # 58.333... %?
    # Wait, implementation output is Fraction? Or Percentage?
    # `val = count / total`. IT IS FRACTION (0 to 1).
    # Previous implementation had `* 100`.
    # User implementation: `Fields.level(out′, i) .= reduce(...) / reduce(...)`.
    # NO `* 100`.
    # So expectation is 0.58333.
    
    expected_mean = expected_frac * (-2.0) # Since the rest are 0.
    
    @test min_2 ≈ -2.0
    @test neg_frac_2 ≈ expected_frac
    @test neg_mean_2 ≈ expected_mean

end

test_negative_scalars_diagnostics()
