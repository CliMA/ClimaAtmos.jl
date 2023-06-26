# This script assumes that `integrator` is defined.

import JET
# Suggested in: https://github.com/aviatesk/JET.jl/issues/455
macro n_failures(ex)
    return :(
        let result = JET.@report_opt $(ex)
            length(JET.get_reports(result.analyzer, result.result))
        end
    )
end

macro jet_and_allocs(ex)
    return quote
        local n = @n_failures $(ex)
        local _allocs = @allocated $(ex)
        (; nfailures = n, allocs = _allocs)
    end
end

import OrdinaryDiffEq
import ClimaAtmos as CA
n = Dict()
Y = integrator.u;
du = integrator.u;
p = integrator.p;
t = integrator.t;
Yₜ = similar(Y);
ref_Y = similar(Y);
#! format: off
n["dss_callback!"]                               = @jet_and_allocs CA.dss_callback!(integrator);
# n["flux_accumulation!"]                          = @jet_and_allocs CA.flux_accumulation!(integrator);
n["rrtmgp_model_callback!"]                      = @jet_and_allocs CA.rrtmgp_model_callback!(integrator);
n["save_to_disk_func"]                           = @jet_and_allocs CA.save_to_disk_func(integrator);
n["save_restart_func"]                           = @jet_and_allocs CA.save_restart_func(integrator);
n["step!"]                                       = @jet_and_allocs OrdinaryDiffEq.step!(integrator);
n["limited_tendency!"]                           = @jet_and_allocs CA.limited_tendency!(Yₜ, Y, p, t);
n["horizontal_advection_tendency!"]              = @jet_and_allocs CA.horizontal_advection_tendency!(Yₜ, Y, p, t);
n["horizontal_tracer_advection_tendency!"]       = @jet_and_allocs CA.horizontal_tracer_advection_tendency!(Yₜ, Y, p, t);
n["explicit_vertical_advection_tendency!"]       = @jet_and_allocs CA.explicit_vertical_advection_tendency!(Yₜ, Y, p, t);
n["hyperdiffusion_tendency!"]                    = @jet_and_allocs CA.hyperdiffusion_tendency!(Yₜ, Y, p, t);
n["tracer_hyperdiffusion_tendency!"]             = @jet_and_allocs CA.tracer_hyperdiffusion_tendency!(Yₜ, Y, p, t);
n["remaining_tendency!"]                         = @jet_and_allocs CA.remaining_tendency!(Yₜ, Y, p, t);
n["additional_tendency!"]                        = @jet_and_allocs CA.additional_tendency!(Yₜ, Y, p, t);
n["vertical_diffusion_boundary_layer_tendency!"] = @jet_and_allocs CA.vertical_diffusion_boundary_layer_tendency!(Yₜ, Y, p, t);
n["implicit_tendency!"]                          = @jet_and_allocs CA.implicit_tendency!(Yₜ, Y, p, t);
n["set_precomputed_quantities!"]                 = @jet_and_allocs CA.set_precomputed_quantities!(Y, p, t);
n["limiters_func!"]                              = @jet_and_allocs CA.limiters_func!(Y, p, t, ref_Y);
n["update_surface_conditions!"]                  = @jet_and_allocs CA.SurfaceConditions.update_surface_conditions!(Y, p, t);
n["dss!"]                                        = @jet_and_allocs CA.dss!(Y, p, t);
n["fill_with_nans!"]                             = @jet_and_allocs CA.fill_with_nans!(p);
#! format: on

n = filter(x -> x.second.nfailures ≠ 0 || x.second.allocs ≠ 0, n)
@info "n-jet failures (excluding n=0), @allocated:"
show(IOContext(stdout, :limit => false), MIME"text/plain"(), n)
println()
