
using Test
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore as CC
import Thermodynamics as TD

### Common Objects ###
@testset begin
    "Smagorinsky Lilly function"
    ### Boilerplate default integrator objects
    config = CA.AtmosConfig(
        Dict(
            "initial_condition" => "DryDensityCurrentProfile",
            "moist" => "dry",
            "precip_model" => "0M",
            "config" => "box",
            "x_max" => π,
            "y_max" => π,
            "z_max" => 1.0,
            "z_stretch" => false,
            "output_default_diagnostics" => false
        ),
    );
    parsed_args = config.parsed_args;
    simulation = CA.get_simulation(config);
    (; integrator) = simulation;
    Y = integrator.u;
    p = integrator.p;
    params = p.params;
    cm_params = CAP.microphysics_params(params);
    thermo_params = CAP.thermodynamics_params(params);

    FT = eltype(Y)
    ᶜYₜ = Y .* FT(0)
    c_xyz = CC.Fields.coordinate_field(Y.c)
    f_xyz = CC.Fields.coordinate_field(Y.f)
    x = c_xyz.x
    y = c_xyz.y
    z = c_xyz.z

    u = @. sin(x) * cos(y) 
    v = @. sin(y) * cos(x)
    w = @. z

    u_x = @. cos(x) * cos(y)
    v_x = @. -sin(y) * sin(x)
    u_y = @. -sin(x) * sin(y)
    v_y = @. cos(y) * cos(x)
    w_x =  zeros(axes(Y.c))
    w_y =  zeros(axes(Y.c))

    vel = @. CC.Geometry.UVVector(u, v)
    vel = @. CC.Geometry.UVWVector(u,v,w)

    wgrad = CC.Operators.WeakGradient()
    grad = CC.Operators.Gradient()
    div = CC.Operators.Divergence()
    wdiv = CC.Operators.WeakDivergence()

    ∇vel = @. grad(vel)
    ∇velᵀ = similar(∇vel)
    @. ∇velᵀ = CC.Geometry.AxisTensor(CC.Geometry.axes(∇vel), transpose(CC.Geometry.components(∇vel)))
    S = @. FT(1/2) * (∇vel + ∇velᵀ)
    S₃ = @. CC.Geometry.project(CC.Geometry.UVWAxis(), S)
    # Then, if ᶜϵ is known, we can compute 
    # the S₃ complete strain-rate tensor, and then take its 
    # divergence such that we can add it to the tendency terms.
    divS = @. wdiv(S₃)


    ### Component test begins here
end
