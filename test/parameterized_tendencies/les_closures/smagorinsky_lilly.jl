
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
    Yₜ = similar(Y);
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
    z = f_xyz.z

    u = @. sin(x) * cos(y) 
    v = @. sin(y) * cos(x)
    w = @. Geometry.WVector(z ./ maximum(z))

    u_x = @. cos(x) * cos(y)
    v_x = @. -sin(y) * sin(x)
    u_y = @. -sin(x) * sin(y)
    v_y = @. cos(y) * cos(x)
    w_x =  zeros(axes(Y.c))
    w_y =  zeros(axes(Y.c))

    vel = @. CC.Geometry.UVVector(u, v)

    Y.c.ρ .= FT(1)
    Y.c.uₕ .= Geometry.Covariant12Vector(vel)
    Y.f.u₃ .= Geometry.Covariant3Vector(w)

    horizontal_smagorinsky_lilly_tendency(Yₜ, Y, p, t, SmagorinskyLilly(FT(0.2)))

    ### Component test begins here
end
