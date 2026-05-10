# Offline quadrature utilities for saved high-resolution truth profiles.

using NCDatasets
using JLD2
using Interpolations
using Statistics
using ClimaAtmos

using ClimaCore: Geometry, Operators, Fields, Spaces, Meshes, Domains, Topologies
import ClimaComms
import Thermodynamics as TD

function read_highres_truth_profile(ncfile::String)
    NCDataset(ncfile) do ds
        z = ds["z"][:]
        T = ds["T"][:]
        qv = ds["qv"][:]
        p = ds["p"][:]
        q_var_sgs = ds["q_var_sgs"][:]
        T_var_sgs = ds["T_var_sgs"][:]
        corr_Tq = ds["corr_Tq"][:]
        rho = ds["rho"][:]
        q_liq = ds["q_liq"][:]
        q_ice = ds["q_ice"][:]
        theta_li = ds["theta_li"][:]
        z_faces = ds["z_faces"][:]

        return Dict(
            "z" => z, "z_faces" => z_faces,
            "T" => T, "qv" => qv, "p" => p, 
               "q_var_sgs" => q_var_sgs, "T_var_sgs" => T_var_sgs, "corr_Tq" => corr_Tq, "corr" => corr_Tq,
            "q_var_total" => q_var_sgs, "T_var_total" => T_var_sgs, # High-res identity
            "rho" => rho, "q_liq" => q_liq, "q_ice" => q_ice, "theta_li" => theta_li
        )
    end
end

function regrid_truth_profile_linear(z_src, fields::Dict, z_target_centers)
    _interp(zs, fs, zt) = LinearInterpolation(zs, fs, extrapolation_bc=Line())(zt)
    
    T_coarse = _interp(z_src, fields["T"], z_target_centers)
    qv_coarse = _interp(z_src, fields["qv"], z_target_centers)
    p_coarse = _interp(z_src, fields["p"], z_target_centers)
    
    # Regrid physical state and variance components
    prof_r = Dict(
        "z" => z_target_centers, 
        "T" => _interp(z_src, fields["T"], z_target_centers), 
        "qv" => _interp(z_src, fields["qv"], z_target_centers), 
        "p" => _interp(z_src, fields["p"], z_target_centers),
        "rho" => _interp(z_src, fields["rho"], z_target_centers),
        "theta_li" => _interp(z_src, fields["theta_li"], z_target_centers),
        "q_liq" => max.(0.0, _interp(z_src, fields["q_liq"], z_target_centers)),
        "q_ice" => max.(0.0, _interp(z_src, fields["q_ice"], z_target_centers)),
        "q_var_sgs" => max.(0.0, _interp(z_src, fields["q_var_sgs"], z_target_centers)),
        "T_var_sgs" => max.(0.0, _interp(z_src, fields["T_var_sgs"], z_target_centers)),
        "q_var_total" => max.(0.0, _interp(z_src, fields["q_var_total"], z_target_centers)),
        "T_var_total" => max.(0.0, _interp(z_src, fields["T_var_total"], z_target_centers)),
        "corr" => clamp.(_interp(z_src, fields["corr_Tq"], z_target_centers), -1.0, 1.0)
    )

    return prof_r, prof_r["q_var_total"], prof_r["T_var_total"], prof_r["corr"]
end

function regrid_truth_profile_block_average(z_src, fields::Dict, z_target_centers, z_target_faces)
    _interp(zs, fs, zt) = LinearInterpolation(zs, fs, extrapolation_bc=Line())(zt)
    N_coarse = length(z_target_centers)
    N_fine = length(z_src)
    
    T_coarse = zeros(N_coarse)
    qv_coarse = zeros(N_coarse)
    p_coarse = zeros(N_coarse)
    
    qvar_coarse = zeros(N_coarse)
    Tvar_coarse = zeros(N_coarse)
    corr_coarse = zeros(N_coarse)
    sgs_qvar_coarse = zeros(N_coarse)
    sgs_Tvar_coarse = zeros(N_coarse)
    rho_coarse = zeros(N_coarse)
    qliq_coarse = zeros(N_coarse)
    qice_coarse = zeros(N_coarse)
    theta_li_coarse = zeros(N_coarse)

    for i in 1:N_coarse
        z0 = z_target_faces[i]
        z1 = z_target_faces[i+1]
        
        # Find high-res points in this coarse cell
        idx = Int[]
        for j in 1:N_fine
            if z0 <= z_src[j] < z1 || (i == N_coarse && z_src[j] == z1)
                push!(idx, j)
            end
        end
        
        if isempty(idx)
            # Linear fallback
            T_coarse[i] = _interp(z_src, fields["T"], z_target_centers[i])
            qv_coarse[i] = _interp(z_src, fields["qv"], z_target_centers[i])
            p_coarse[i] = _interp(z_src, fields["p"], z_target_centers[i])
            sgs_qvar_coarse[i] = max(0.0, _interp(z_src, fields["q_var_sgs"], z_target_centers[i]))
            sgs_Tvar_coarse[i] = max(0.0, _interp(z_src, fields["T_var_sgs"], z_target_centers[i]))
            qvar_coarse[i] = max(0.0, _interp(z_src, fields["q_var_total"], z_target_centers[i]))
            Tvar_coarse[i] = max(0.0, _interp(z_src, fields["T_var_total"], z_target_centers[i]))
            corr_coarse[i] = clamp(_interp(z_src, fields["corr_Tq"], z_target_centers[i]), -1.0, 1.0)
            rho_coarse[i] = _interp(z_src, fields["rho"], z_target_centers[i])
            qliq_coarse[i] = max(0.0, _interp(z_src, fields["q_liq"], z_target_centers[i]))
            qice_coarse[i] = max(0.0, _interp(z_src, fields["q_ice"], z_target_centers[i]))
            theta_li_coarse[i] = _interp(z_src, fields["theta_li"], z_target_centers[i])
        else
            # Block average
            T_fine = fields["T"][idx]
            qv_fine = fields["qv"][idx]
            p_fine = fields["p"][idx]
            
            T_mean = mean(T_fine)
            qv_mean = mean(qv_fine)
            
            T_coarse[i] = T_mean
            qv_coarse[i] = qv_mean
            p_coarse[i] = mean(p_fine)
            
            # Sub-grid variance = Average of high-res SGS variance + Variance of the high-res mean values
            sgs_qvar = mean(fields["q_var_sgs"][idx])
            res_qvar = mean(qv_fine.^2) - qv_mean^2
            qvar_coarse[i] = sgs_qvar + max(0.0, res_qvar)
            
            sgs_Tvar = mean(fields["T_var_sgs"][idx])
            res_Tvar = mean(T_fine.^2) - T_mean^2
            Tvar_coarse[i] = sgs_Tvar + max(0.0, res_Tvar)
            
            # Covariance
            cov_fine = fields["corr_Tq"][idx] .* sqrt.(fields["q_var_sgs"][idx]) .* sqrt.(fields["T_var_sgs"][idx])
            sgs_cov = mean(cov_fine)
            res_cov = mean(T_fine .* qv_fine) - (T_mean * qv_mean)
            total_cov = sgs_cov + res_cov
            
            denom = sqrt(Tvar_coarse[i]) * sqrt(qvar_coarse[i])
            corr_coarse[i] = denom > 1e-12 ? clamp(total_cov / denom, -1.0, 1.0) : 0.0

            sgs_qvar_coarse[i] = sgs_qvar
            sgs_Tvar_coarse[i] = sgs_Tvar
            rho_coarse[i] = mean(fields["rho"][idx])
            qliq_coarse[i] = max(0.0, mean(fields["q_liq"][idx]))
            qice_coarse[i] = max(0.0, mean(fields["q_ice"][idx]))
            theta_li_coarse[i] = mean(fields["theta_li"][idx])
        end
    end
    
    prof_r = Dict(
        "z" => z_target_centers, 
        "T" => T_coarse, 
        "qv" => qv_coarse, 
        "p" => p_coarse,
        "rho" => rho_coarse,
        "theta_li" => theta_li_coarse,
        "q_liq" => qliq_coarse,
        "q_ice" => qice_coarse,
        "q_var_sgs" => sgs_qvar_coarse,
        "T_var_sgs" => sgs_Tvar_coarse,
        "q_var_total" => qvar_coarse,
        "T_var_total" => Tvar_coarse,
        "corr" => corr_coarse
    )
    
    return prof_r, qvar_coarse, Tvar_coarse, corr_coarse
end

function _convert_profile_eltype(profile::Dict, ::Type{FT}) where {FT}
    converted = Dict{String, Any}()
    for (key, value) in profile
        converted[key] = value isa AbstractVector ? FT.(value) : value
    end
    return converted
end

# ============================================================================
# ClimaCore column space construction
# ============================================================================

"""
    build_column_center_space(z_faces::Vector{FT}) where FT

Construct a ClimaCore `CenterFiniteDifferenceSpace` from face positions,
matching the exact grid geometry ClimaAtmos uses internally.
"""
function build_column_center_space(z_faces::Vector{FT}) where {FT}
    domain = Domains.IntervalDomain(
        Geometry.ZPoint(z_faces[1]),
        Geometry.ZPoint(z_faces[end]);
        boundary_names = (:bottom, :top),
    )
    mesh = Meshes.IntervalMesh(domain, Geometry.ZPoint.(z_faces))
    topology = Topologies.IntervalTopology(ClimaComms.SingletonCommsContext(), mesh)
    return Spaces.CenterFiniteDifferenceSpace(topology)
end

# ============================================================================
# Vertically resolved SGS integration using ClimaCore infrastructure
# ============================================================================

"""
    integrate_condensate_vertically_resolved(
        quad, prof_r, qvar, Tvar, corr, z_faces;
        thermo_params,
    )

Compute condensate profile using the 18-argument `integrate_over_sgs` for
vertically resolved SGS distributions. Mirrors the exact pattern from
`microphysics_cache.jl`: ClimaCore Fields → gradient operators → existing API.

Requires `rho`, `q_liq`, `q_ice` in `prof_r` to compute θ_li exactly.
"""
function integrate_condensate_vertically_resolved(
    quad, prof_r::Dict, qvar, Tvar, corr, z_faces::Vector{FT};
    thermo_params::TD.Parameters.AbstractThermodynamicsParameters{FT},
) where {FT}
    N = length(prof_r["z"])

    # Validate that required fields are present
    for key in ("rho", "q_liq", "q_ice")
        haskey(prof_r, key) || error(
            "Vertically resolved SGS requires '$key' in the regridded profile. " *
            "Re-run the truth profile with the updated diagnostics (rhoa, clw, cli)."
        )
    end

    # Build ClimaCore column space
    center_space = build_column_center_space(z_faces)

    # ClimaCore operators
    C3 = Geometry.Covariant3Vector
    # Use Extrapolate() instead of SetGradient(0) to avoid forcing flat profiles at boundaries,
    # which can break cloud top truncation in coarse cells.
    # Use SetGradient(C3(0)) to match ClimaAtmos.jl/src/utils/abbreviations.jl precisely.
    # This ensures exact parity with p.precomputed.ᶜgradᵥ_θ_liq_ice.
    ᶠgradᵥ = Operators.GradientC2F(
        bottom = Operators.SetGradient(C3(0)),
        top = Operators.SetGradient(C3(0)),
    )
    ᶜleft_bias = Operators.LeftBiasedF2C()
    ᶜright_bias = Operators.RightBiasedF2C()

    # Helper to create a center-space scalar Field from a vector
    function _vec_to_field(vals::Vector, space)
        f = Fields.Field(FT, space)
        parent(f) .= vals
        return f
    end

    # Create center-space Fields from profile vectors
    ᶜT = _vec_to_field(FT.(prof_r["T"]), center_space)
    ᶜq = _vec_to_field(FT.(prof_r["qv"]), center_space)
    ᶜρ = _vec_to_field(FT.(prof_r["rho"]), center_space)
    ᶜq_liq = _vec_to_field(FT.(prof_r["q_liq"]), center_space)
    ᶜq_ice = _vec_to_field(FT.(prof_r["q_ice"]), center_space)
    ᶜqvar = _vec_to_field(FT.(qvar), center_space)
    ᶜTvar = _vec_to_field(FT.(Tvar), center_space)

    # Use the simulation's liquid-ice potential temperature for exact gradient parity.
    ᶜθ_li = _vec_to_field(FT.(prof_r["theta_li"]), center_space)

    # Compute ∂T/∂θ_li exactly (same function ClimaAtmos uses)
    ᶜ∂T_∂θ = @. ClimaAtmos.∂T_∂θ_li(thermo_params, ᶜT, ᶜθ_li, ᶜq_liq, ᶜq_ice, ᶜq, ᶜρ)

    # Half-cell gradients via ClimaCore operators (exact microphysics_cache.jl pattern)
    # This now perfectly matches p.precomputed.ᶜgradᵥ_θ_liq_ice
    ᶜgrad_q_dn = @. ᶜleft_bias(ᶠgradᵥ(ᶜq))
    ᶜgrad_q_up = @. ᶜright_bias(ᶠgradᵥ(ᶜq))
    ᶜgrad_θ_dn = @. ᶜleft_bias(ᶠgradᵥ(ᶜθ_li))
    ᶜgrad_θ_up = @. ᶜright_bias(ᶠgradᵥ(ᶜθ_li))
    
    # Slopes for variances (using SGS variance, not total, to avoid double counting with resolved slopes)
    ᶜqvar_sgs = _vec_to_field(FT.(prof_r["q_var_sgs"]), center_space)
    ᶜTvar_sgs = _vec_to_field(FT.(prof_r["T_var_sgs"]), center_space)
    ᶜgrad_qq_dn = @. ᶜleft_bias(ᶠgradᵥ(ᶜqvar_sgs))
    ᶜgrad_qq_up = @. ᶜright_bias(ᶠgradᵥ(ᶜqvar_sgs))
    ᶜgrad_TT_dn = @. ᶜleft_bias(ᶠgradᵥ(ᶜTvar_sgs))
    ᶜgrad_TT_up = @. ᶜright_bias(ᶠgradᵥ(ᶜTvar_sgs))

    # Geometry fields
    ᶜdz = Fields.Δz_field(center_space)
    ᶜlg = Fields.local_geometry_field(center_space)

    # Loop over levels, calling the existing 18-arg integrate_over_sgs
    condensate_profile = zeros(FT, N)
    for i in 1:N
        μ_T = Fields.level(ᶜT, i)[]
        μ_q = Fields.level(ᶜq, i)[]
        ρ_i = Fields.level(ᶜρ, i)[]
        corr_i = prof_r["corr"][i]
        p_i = prof_r["p"][i]
        
        # Exact evaluator from ClimaAtmos (0,1-moment saturation adjustment)
        # It takes (T_hat, q_tot_hat) and returns a NamedTuple with q_liq, q_ice, etc.
        sa_eval = CA.SaturationAdjustmentEvaluator(thermo_params, ρ_i)
        
        Δz_i = Fields.level(ᶜdz, i)[]
        lg_i = Fields.level(ᶜlg, i)[]
        gq_dn_i = Fields.level(ᶜgrad_q_dn, i)[]
        gq_up_i = Fields.level(ᶜgrad_q_up, i)[]
        gθ_dn_i = Fields.level(ᶜgrad_θ_dn, i)[]
        gθ_up_i = Fields.level(ᶜgrad_θ_up, i)[]
        ∂T∂θ_i = Fields.level(ᶜ∂T_∂θ, i)[]
        gqq_dn_i = Fields.level(ᶜgrad_qq_dn, i)[]
        gqq_up_i = Fields.level(ᶜgrad_qq_up, i)[]
        gTT_dn_i = Fields.level(ᶜgrad_TT_dn, i)[]
        gTT_up_i = Fields.level(ᶜgrad_TT_up, i)[]

        # Condensate evaluator (same physics as the production 0M scheme)
        f_local = (T_hat, q_hat) -> begin
            sa = sa_eval(T_hat, q_hat)
            return sa.q_liq + sa.q_ice
        end
        
        # Use SGS variance for vertically resolved (captures slopes separately),
        # but keep total variance for bivariate fallbacks.
        is_vert = ClimaAtmos._is_vertically_resolved_sgs(quad.dist)
        q′q′_i = is_vert ? prof_r["q_var_sgs"][i] : prof_r["q_var_total"][i]
        T′T′_i = is_vert ? prof_r["T_var_sgs"][i] : prof_r["T_var_total"][i]

        condensate_profile[i] = ClimaAtmos.integrate_over_sgs(
            f_local, quad, μ_q, μ_T, q′q′_i, T′T′_i, corr_i,
            Δz_i, lg_i,
            gq_dn_i, gq_up_i, gθ_dn_i, gθ_up_i, ∂T∂θ_i,
            gqq_dn_i, gqq_up_i, gTT_dn_i, gTT_up_i,
        )
    end
    return condensate_profile
end

# ============================================================================
# Main sweep functions
# ============================================================================

function run_offline_quadrature_sweep(profile::Dict; outdir::String, target_grids::Vector, quadrature_orders::Vector{Int}=[1, 2, 3, 4, 5], sgs_distributions::Vector{String}=String["lognormal"], skip_existing::Bool=true, regrid_method::Symbol=:linear, thermo_params::TD.Parameters.AbstractThermodynamicsParameters)
    FT = eltype(thermo_params)
    fields = _convert_profile_eltype(profile, FT)
    z = fields["z"]

    mkpath(outdir)
    truth_path = joinpath(outdir, "truth.jld2")
    if !skip_existing || !isfile(truth_path)
        dist_obj_truth = ClimaAtmos.get_sgs_distribution(Dict("sgs_distribution" => "lognormal_vertical_profile_full_cubature"))
        quad_truth = ClimaAtmos.SGSQuadrature(FT; quadrature_order=5, distribution=dist_obj_truth)
        
        # Use the real face coordinates from the simulation grid
         cond_truth = integrate_condensate_vertically_resolved(
            quad_truth, fields, fields["q_var_sgs"], fields["T_var_sgs"], fields["corr_Tq"], fields["z_faces"]; 
            thermo_params=thermo_params
        )
        truth_res = Dict("z"=>z, "cond"=>cond_truth)
        @save truth_path truth_res
    end

    for (grid_name, zt, faces_z) in target_grids
        zt = FT.(zt)
        faces_z = FT.(faces_z)
        if regrid_method == :linear
            prof_r, qvar, Tvar, corr = regrid_truth_profile_linear(z, fields, zt)
        elseif regrid_method == :block_average
            prof_r, qvar, Tvar, corr = regrid_truth_profile_block_average(z, fields, zt, faces_z)
        else
            error("Unknown regrid_method: $regrid_method")
        end

        for N in quadrature_orders
            for dist_str in sgs_distributions
                dist_path = joinpath(outdir, string(regrid_method), grid_name, "N_$N", string(dist_str, ".jld2"))
                if skip_existing && isfile(dist_path)
                    continue
                end

                dist_obj = ClimaAtmos.get_sgs_distribution(Dict("sgs_distribution" => dist_str))
                quad = ClimaAtmos.SGSQuadrature(FT; quadrature_order=N, distribution=dist_obj)

                is_vertically_resolved = ClimaAtmos._is_vertically_resolved_sgs(quad.dist)

                if is_vertically_resolved
                    # Vertically resolved path: use 18-arg integrate_over_sgs
                    # with ClimaCore geometry, matching microphysics_cache.jl exactly
                    condensate_profile = integrate_condensate_vertically_resolved(
                        quad, prof_r, qvar, Tvar, corr, faces_z;
                        thermo_params=thermo_params,
                    )
                else
                    # Base distribution path: use 7-arg bivariate integrate_over_sgs
                    condensate_profile = zeros(length(zt))
                    for (i, zlev) in enumerate(zt)
                        μ_T = prof_r["T"][i]
                        μ_q = prof_r["qv"][i]
                        qv_i = qvar[i]
                        Tv_i = Tvar[i]
                        corr_i = corr[i]
                        p_i = prof_r["p"][i]
                        ρ_i = prof_r["rho"][i]
                        
                        # Exact evaluator from ClimaAtmos (0-moment saturation adjustment)
                        sa_eval = CA.SaturationAdjustmentEvaluator(thermo_params, ρ_i)
                        f_local = (T_hat, q_hat) -> begin
                            sa = sa_eval(T_hat, q_hat)
                            return sa.q_liq + sa.q_ice
                        end
                        condensate_profile[i] = ClimaAtmos.integrate_over_sgs(f_local, quad, μ_q, μ_T, qv_i, Tv_i, corr_i)
                    end
                end

                res = Dict("z"=>zt, "cond"=>condensate_profile)
                mkpath(dirname(dist_path))
                @save dist_path res
            end
        end
    end
    return nothing
end

function run_offline_quadrature_from_netcdf!(infile::String; outdir::String, target_grids::Vector, quadrature_orders::Vector{Int}=[1, 2, 3, 4, 5], sgs_distributions::Vector{String}=String["lognormal"], skip_existing::Bool=true, regrid_method::Symbol=:linear, thermo_params::TD.Parameters.AbstractThermodynamicsParameters)
    profile = read_highres_truth_profile(infile)
    return run_offline_quadrature_sweep(profile; outdir, target_grids, quadrature_orders, sgs_distributions, skip_existing, regrid_method, thermo_params)
end
