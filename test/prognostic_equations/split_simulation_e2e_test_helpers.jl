using Test
using Dates
using LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import ClimaCore: Fields, Geometry, Spaces, Topologies
import ClimaDiagnostics as CD
import ClimaTimeSteppers as CTS
const TD = CA.TD

_is_cpu(field) = ClimaComms.device(field) isa ClimaComms.AbstractCPUDevice

function _all_finite(Y)
    ok = all(isfinite, Array(parent(Y.c))) && all(isfinite, Array(parent(Y.f)))
    if hasproperty(Y, :sfc)
        ok &= all(isfinite, Array(parent(Y.sfc)))
    end
    return ok
end

function _column_integral(field)
    return sum(field)
end

function _surface_integral(sfc_field, h_space)
    if h_space isa Spaces.PointSpace
        return only(Array(parent(sfc_field)))
    end
    return sum(Fields.Field(Fields.field_values(sfc_field), h_space))
end

function _last_diag(dict, sn)
    k = only(filter(k -> k == sn || startswith(k, "$(sn)_"), collect(keys(dict))))
    d = dict[k]
    return Array(parent(d[maximum(keys(d))]))
end

@noinline function _allocs_vt(Y, p, Δt, ic)
    CA.step_vertical_scalar_transport!(Y, p, Δt, ic)
    return @allocated CA.step_vertical_scalar_transport!(Y, p, Δt, ic)
end

@noinline function _allocs_cphys(Y, p, t, Δt, ᶠF̃)
    CA.step_column_physics!(Y, p, t, Δt; ᶠF̃)
    return @allocated CA.step_column_physics!(Y, p, t, Δt; ᶠF̃)
end

@noinline function _allocs_pre_mp(Y, p, t)
    CA.refresh_pre_microphysics_thermo!(Y, p, t)
    return @allocated CA.refresh_pre_microphysics_thermo!(Y, p, t)
end

@noinline function _allocs_stage_cache(U, p, t)
    CA.set_precomputed_quantities!(U, p, t)
    return @allocated CA.set_precomputed_quantities!(U, p, t)
end

@noinline function _allocs_smag(U, p, sl)
    CA.set_smagorinsky_lilly_precomputed_quantities!(U, p, sl)
    return @allocated CA.set_smagorinsky_lilly_precomputed_quantities!(U, p, sl)
end

@noinline function _allocs_imp_tendency(Yₜ, Y, p, t)
    CA.implicit_tendency!(Yₜ, Y, p, t)
    return @allocated CA.implicit_tendency!(Yₜ, Y, p, t)
end

function _make_imex_algo(alg_name, constraint = CTS.Unconstrained())
    nm = CTS.NewtonsMethod(;
        max_iters = 1,
        update_j = CTS.UpdateEvery(CTS.NewNewtonIteration),
    )
    return CTS.IMEXAlgorithm(alg_name, nm, constraint)
end

function _seed_moist_column!(Y, p, FT; with_condensates::Bool = true)
    tps = CAP.thermodynamics_params(p.params)
    grav = FT(CAP.grav(p.params))
    R_d = FT(CAP.R_d(p.params))
    zc = Fields.coordinate_field(Y.c).z
    zf = Fields.coordinate_field(Y.f).z
    zmax = FT(Spaces.z_max(axes(zf)))
    lgc = Fields.local_geometry_field(Y.c)
    lgf = Fields.local_geometry_field(Y.f)

    @. Y.c.uₕ = Geometry.Covariant12Vector(Geometry.UVVector(FT(8), FT(-3)), lgc)
    @. Y.f.u₃ = Geometry.Covariant3Vector(
        Geometry.WVector(FT(0.4) * sin(FT(π) * zf / zmax)),
        lgf,
    )
    @. Y.c.ρ = FT(1.15) * exp(-grav * zc / (R_d * FT(275)))
    T_prof = @. FT(296) - FT(6.5e-3) * zc
    q_prof = @. FT(0.015) * exp(-zc / FT(2500)) + FT(5e-4)
    if with_condensates && hasproperty(Y.c, :ρq_lcl)
        ql_prof = @. FT(6e-4) * exp(-((zc - FT(1800)) / FT(600))^2)
        qi_prof = @. FT(2e-4) * exp(-((zc - FT(4200)) / FT(700))^2)
        qr_prof = @. FT(8e-4) * exp(-(zc / FT(1500))^2)
        qs_prof = @. FT(2e-4) * exp(-((zc - FT(3000)) / FT(1000))^2)
        @. Y.c.ρq_tot = Y.c.ρ * (q_prof + ql_prof + qi_prof + qr_prof + qs_prof)
        @. Y.c.ρq_lcl = Y.c.ρ * ql_prof
        @. Y.c.ρq_icl = Y.c.ρ * qi_prof
        @. Y.c.ρq_rai = Y.c.ρ * qr_prof
        @. Y.c.ρq_sno = Y.c.ρ * qs_prof
        e_prof = @. TD.internal_energy(
            tps,
            T_prof,
            q_prof + ql_prof + qi_prof + qr_prof + qs_prof,
            ql_prof + qr_prof,
            qi_prof + qs_prof,
        )
    else
        @. Y.c.ρq_tot = Y.c.ρ * q_prof
        e_prof = @. TD.internal_energy(tps, T_prof, q_prof, FT(0), FT(0))
    end
    if hasproperty(Y.c, :ρn_lcl)
        @. Y.c.ρn_lcl = Y.c.ρ * FT(1e8) * exp(-((zc - FT(1800)) / FT(600))^2)
        @. Y.c.ρn_rai = Y.c.ρ * FT(1e5) * exp(-(zc / FT(1500))^2)
    end
    if hasproperty(Y.c, :ρn_ice)
        @. Y.c.ρn_ice = Y.c.ρ * FT(1e5) * exp(-((zc - FT(4200)) / FT(700))^2)
        @. Y.c.ρq_rim = FT(0.25) * Y.c.ρq_icl
        @. Y.c.ρb_rim = Y.c.ρq_rim / FT(400)
    end
    CA.set_implicit_precomputed_quantities_core!(Y, p, FT(0))
    @. Y.c.ρe_tot = Y.c.ρ * (e_prof + p.precomputed.ᶜK + p.core.ᶜΦ)
    if hasproperty(Y.c, :ρtke)
        @. Y.c.ρtke = Y.c.ρ * FT(0.6)
    end
    if hasproperty(Y.c, :sgsʲs)
        @. Y.c.sgsʲs.:(1).ρa = Y.c.ρ * FT(0.06)
        @. Y.f.sgsʲs.:(1).u₃ = Geometry.Covariant3Vector(
            Geometry.WVector(FT(3.0) * sin(FT(π) * zf / zmax)),
            lgf,
        )
        @. Y.c.sgsʲs.:(1).q_tot = Y.c.ρq_tot / Y.c.ρ + FT(1e-3)
        if hasproperty(Y.c.sgsʲs.:(1), :q_lcl)
            @. Y.c.sgsʲs.:(1).q_lcl = Y.c.ρq_lcl / Y.c.ρ
            @. Y.c.sgsʲs.:(1).q_icl = Y.c.ρq_icl / Y.c.ρ
            @. Y.c.sgsʲs.:(1).q_rai = Y.c.ρq_rai / Y.c.ρ
            @. Y.c.sgsʲs.:(1).q_sno = Y.c.ρq_sno / Y.c.ρ
        end
        CA.set_implicit_precomputed_quantities_core!(Y, p, FT(0))
        @. Y.c.sgsʲs.:(1).mse =
            p.precomputed.ᶜh_tot - p.precomputed.ᶜK + FT(800)
    end
    if hasproperty(Y, :sfc) && hasproperty(Y.sfc, :T)
        @. Y.sfc.T = FT(301)
        @. Y.sfc.water = FT(0)
    end
    CA.dss!(Y, p, FT(0))
    CA.set_precomputed_quantities!(Y, p, FT(0))
    return nothing
end
