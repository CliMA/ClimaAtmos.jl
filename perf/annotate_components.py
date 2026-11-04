#!/usr/bin/env python3
"""Temporary (comparison branch, not for merging): mark every top-level method of the
cache, tendency, and Jacobian functions of a ClimaAtmos source tree with an NVTX range
named after the function (`NVTX.@annotate "name"`), so that profiles of PR and main
attribute device time to the same components.
usage: annotate_components.py <ClimaAtmos tree>"""
import os
import re
import sys

COMPONENTS = """
set_precomputed_quantities! set_implicit_precomputed_quantities!
set_implicit_precomputed_quantities_core! set_explicit_precomputed_quantities!
update_implicit_1m_tendency! set_sgs_moments! set_velocity_quantities!
set_velocity_at_surface! set_velocity_at_top! set_enthalpy_and_pressure!
update_implicit_microphysics_cache! set_smagorinsky_lilly_precomputed_quantities!
set_prognostic_edmf_precomputed_quantities_draft!
set_prognostic_edmf_precomputed_quantities_environment!
set_prognostic_edmf_precomputed_quantities_explicit_closures!
set_edonly_precomputed_quantities_env_closures!
update_surface_conditions! set_precipitation_velocities! set_precipitation_surface_fluxes!
set_covariance_cache_and_cloud_fraction! set_covariance_cache! set_buoyancy_gradient_inputs!
set_precip_fraction! set_face_diffusivities! set_horizontal_diffusivities!
set_microphysics_tendency_cache!
implicit_tendency! implicit_vertical_advection_tendency!
vertical_advection_of_water_tendency! microphysics_tendency!
edmfx_sgs_vertical_advection_tendency! edmfx_sgs_diffusive_flux_tendency!
edmfx_entr_detr_tendency! edmfx_sgs_mass_flux_tendency! edmfx_boundary_condition_tendency!
sgs_u₃_implicit_tendency! vertical_diffusion_boundary_layer_tendency!
surface_precipitation_tendency! pressure_work_tendency! zero_velocity_tendency!
vertical_smagorinsky_lilly_tendency! correct_implicit_advection_tendency!
remaining_tendency! horizontal_advection_tendency! horizontal_dynamics_tendency!
horizontal_tracer_advection_tendency! hyperdiffusion_tendency!
prep_hyperdiffusion_tendency! apply_hyperdiffusion_tendency!
prep_tracer_hyperdiffusion_tendency! apply_tracer_hyperdiffusion_tendency!
explicit_vertical_advection_tendency! additional_tendency! viscous_sponge_tendency!
rayleigh_sponge_tendency! surface_flux_tendency! edmfx_tke_tendency!
edmfx_sgs_horizontal_diffusive_flux_tendency! radiation_tendency! amd_tendency!
tracer_nonnegativity_vapor_tendency! external_forcing_tendency! subsidence_tendency!
surface_temp_tendency! horizontal_smagorinsky_lilly_tendency!
horizontal_constant_diffusion_tendency! non_orographic_gravity_wave_apply_tendency!
orographic_gravity_wave_apply_tendency! chemistry_tendency!
update_jacobian! invert_jacobian! initialize_implicit_stage_problem!
update_advection_jacobian! update_sedimentation_jacobian! update_diffusion_jacobian!
update_sgs_advection_jacobian! update_sgs_diffusion_jacobian! update_sgs_entr_detr_jacobian!
update_sgs_boundary_condition_jacobian! update_sgs_massflux_jacobian!
update_microphysics_jacobian!
dss! limiters_func! constrain_state!
""".split()

def main(tree):
    counts = dict.fromkeys(COMPONENTS, 0)
    # A method definition at the top level of a file (long or short form), with or without an
    # existing annotation; calls of these functions are never at the top level of a source file.
    definition = re.compile(r"(NVTX\.@annotate\s+)?(function\s+)?([^\s(]+!)\(")
    for root, _, files in os.walk(os.path.join(tree, "src")):
        for file in files:
            if not file.endswith(".jl"):
                continue
            path = os.path.join(root, file)
            lines = open(path, encoding="utf-8").read().split("\n")
            changed = False
            in_string = False
            for i, line in enumerate(lines):
                if line.count('"""') % 2 == 1:
                    in_string = not in_string
                    continue
                m = None if in_string else definition.match(line)
                if not m or m.group(3) not in counts:
                    continue
                lines[i] = f'NVTX.@annotate "{m.group(3)}" ' + line[len(m.group(1) or ""):]
                counts[m.group(3)] += 1
                changed = True
            if changed:
                # Files of submodules (e.g., SurfaceConditions) do not see the NVTX of ClimaAtmos
                if not any(re.match(r"(import|using) NVTX\b", line) for line in lines):
                    module_line = next((i for i, line in enumerate(lines) if line.startswith("module ")), -1)
                    lines.insert(module_line + 1, "import NVTX")
                open(path, "w", encoding="utf-8").write("\n".join(lines))
    for name, count in counts.items():
        print(f"{count:3d} {name}")


if __name__ == "__main__":
    main(sys.argv[1])
