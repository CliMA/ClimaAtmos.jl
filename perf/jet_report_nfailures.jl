# This script assumes that `integrator` is defined.
import JET
import HDF5, NCDatasets, CUDA
# Suggested in: https://github.com/aviatesk/JET.jl/issues/455
macro n_failures(ex)
    return :(
        let result =
                JET.@report_opt ignored_modules = (HDF5, CUDA, NCDatasets) $(
                    ex
                )
            length(JET.get_reports(result.analyzer, result.result))
        end
    )
end

import ClimaComms
ClimaComms.@import_required_backends
import SciMLBase
import ClimaAtmos as CA
n = Dict()
Y = integrator.u;
du = integrator.u;
p = integrator.p;
t = integrator.t;
Yₜ = similar(Y);
Yₜ_exp = similar(Y);
Yₜ_lim = similar(Y);
ref_Y = similar(Y);
#! format: off
n["step!"]                                       = @n_failures SciMLBase.step!(integrator);
n["horizontal_dynamics_tendency!"]              = @n_failures CA.horizontal_dynamics_tendency!(Yₜ, Y, p, t);
n["horizontal_tracer_advection_tendency!"]       = @n_failures CA.horizontal_tracer_advection_tendency!(Yₜ, Y, p, t);
n["explicit_vertical_advection_tendency!"]       = @n_failures CA.explicit_vertical_advection_tendency!(Yₜ, Y, p, t);
n["hyperdiffusion_tendency!"]                    = @n_failures CA.hyperdiffusion_tendency!(Yₜ_exp, Yₜ_lim, Y, p, t);
n["remaining_tendency!"]                         = @n_failures CA.remaining_tendency!(Yₜ_exp, Yₜ_lim, Y, p, t);
n["additional_tendency!"]                        = @n_failures CA.additional_tendency!(Yₜ, Y, p, t);
n["vertical_diffusion_boundary_layer_tendency!"] = @n_failures CA.vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t);
n["implicit_tendency!"]                          = @n_failures CA.implicit_tendency!(Yₜ, Y, p, t);
n["set_precomputed_quantities!"]                 = @n_failures CA.set_precomputed_quantities!(Y, p, t);
n["limiters_func!"]                              = @n_failures CA.limiters_func!(Y, p, t, ref_Y);
n["update_surface_conditions!"]                  = @n_failures CA.SurfaceConditions.update_surface_conditions!(Y, p, t);
n["dss!"]                                        = @n_failures CA.dss!(Y, p, t);
n["fill_with_nans!"]                             = @n_failures CA.fill_with_nans!(p);
#! format: on

n = filter(x -> x.second ≠ 0, n)
@info "n-jet failures (excluding n=0):"
show(IOContext(stdout, :limit => false), MIME"text/plain"(), n)
println()
