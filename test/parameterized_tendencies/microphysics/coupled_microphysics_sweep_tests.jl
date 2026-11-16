#=
Unit and column-shaft tests for the coupled microphysics and sedimentation
top-down sweep (`src/parameterized_tendencies/microphysics/coupled_microphysics_sweep.jl`).
=#

using Test
import ClimaAtmos as CA
import ClimaComms
ClimaComms.@import_required_backends
import ClimaCore:
    Domains,
    Fields,
    Geometry,
    Grids,
    Hypsography,
    Meshes,
    Operators,
    Quadratures,
    Spaces,
    Topologies
import ClimaCore.MatrixFields: @name
import ClimaParams as CP
import CloudMicrophysics as CM
import CloudMicrophysics.Parameters as CMP

const SA = CA.SA
const TD = CA.TD
const CAP = CA.CAP

include("../../test_helpers.jl")

# ============================================================================
# High-precision Taylor reference for 2 × 2 matrix functions
# ============================================================================

function _bigfloat_exp_phi1_phi2(
    D1::Real,
    D2::Real,
    D12::Real,
    D21::Real,
    dt::Real,
)
    M = BigFloat[
        -BigFloat(D1)*BigFloat(dt) BigFloat(D21)*BigFloat(dt)
        BigFloat(D12)*BigFloat(dt) -BigFloat(D2)*BigFloat(dt)
    ]
    I2 = BigFloat[1 0; 0 1]
    # Direct quadrature of `exp(s M)` on `1024`-point Gauss-Legendre-like
    # midpoint rule is slow for stiff `||M|| = 1e5`, so evaluate via Higham
    # scaling for `E` and exact nonsingular linear solves (or Taylor for small
    # norm) in 256-bit `BigFloat`.
    opnorm_M = maximum(sum(abs.(M); dims = 2))
    if opnorm_M <= BigFloat(0.25)
        E = zeros(BigFloat, 2, 2)
        Phi1 = zeros(BigFloat, 2, 2)
        Phi2 = zeros(BigFloat, 2, 2)
        Mk = copy(I2)
        fact = BigFloat(1)
        for k in 0:60
            E .+= Mk ./ fact
            Phi1 .+= Mk ./ (fact * BigFloat(k + 1))
            Phi2 .+= Mk ./ (fact * BigFloat((k + 1) * (k + 2)))
            Mk = Mk * M
            fact *= BigFloat(k + 1)
        end
        return E, Phi1, Phi2
    else
        # Scale down by `2^s` so `||M / 2^s|| <= 0.125`, compute `(E_0, Phi1_0, Phi2_0)`
        # by 50-term Taylor, then double `s` times using exact φ doubling identities:
        #   E(2A)    = E(A) * E(A)
        #   Phi1(2A) = 0.5 * (I + E(A)) * Phi1(A)
        #   Phi2(2A) = 0.25 * (2 * Phi2(A) + Phi1(A) * Phi1(A))
        s = ceil(Int, log2(Float64(opnorm_M) / 0.125))
        Ms = M ./ (BigFloat(2)^s)
        Es = zeros(BigFloat, 2, 2)
        P1s = zeros(BigFloat, 2, 2)
        P2s = zeros(BigFloat, 2, 2)
        Mk = copy(I2)
        fact = BigFloat(1)
        for k in 0:50
            Es .+= Mk ./ fact
            P1s .+= Mk ./ (fact * BigFloat(k + 1))
            P2s .+= Mk ./ (fact * BigFloat((k + 1) * (k + 2)))
            Mk = Mk * Ms
            fact *= BigFloat(k + 1)
        end
        for _ in 1:s
            Es_next = Es * Es
            P1_next = BigFloat(0.5) .* (I2 .+ Es) * P1s
            P2_next = BigFloat(0.25) .* (BigFloat(2) .* P2s .+ P1s * P1s)
            Es = Es_next
            P1s = P1_next
            P2s = P2_next
        end
        return Es, P1s, P2s
    end
end

# ============================================================================
# Lightweight column & extruded state builders for fast multi-Δt testing
# ============================================================================

function _make_column_spaces(
    ::Type{FT};
    z_max = FT(6000),
    nz::Int = 40,
    stretched::Bool = false,
    device = ClimaComms.device(),
) where {FT}
    ctx = ClimaComms.SingletonCommsContext(device)
    vdom = Domains.IntervalDomain(
        Geometry.ZPoint{FT}(0),
        Geometry.ZPoint{FT}(z_max);
        boundary_names = (:bottom, :top),
    )
    vmesh = if stretched
        Meshes.IntervalMesh(
            vdom,
            Meshes.ExponentialStretching(FT(500));
            nelems = nz,
        )
    else
        Meshes.IntervalMesh(vdom; nelems = nz)
    end
    vtopo = Topologies.IntervalTopology(ctx, vmesh)
    vgrid = Grids.FiniteDifferenceGrid(vtopo)
    cspace = Spaces.CenterFiniteDifferenceSpace(vgrid)
    fspace = Spaces.FaceFiniteDifferenceSpace(vgrid)
    return cspace, fspace
end

function _arr(f)
    f_contig = similar(f, eltype(f))
    f_contig .= f
    return vec(Array(parent(f_contig)))
end

function _column_integrals(Y, p)
    FT = eltype(Y.c.ρ)
    sfc_space = axes(p.precomputed.surface_rain_flux)
    int_ρ = Fields.Field(FT, sfc_space)
    int_ρqt = Fields.Field(FT, sfc_space)
    int_ρe = Fields.Field(FT, sfc_space)
    int_abs_ρe = Fields.Field(FT, sfc_space)
    Operators.column_integral_definite!(int_ρ, Y.c.ρ)
    Operators.column_integral_definite!(int_ρqt, Y.c.ρq_tot)
    Operators.column_integral_definite!(int_ρe, Y.c.ρe_tot)
    Operators.column_integral_definite!(int_abs_ρe, abs.(Y.c.ρe_tot))
    return (;
        dry_air = _arr(int_ρ) .- _arr(int_ρqt),
        water = _arr(int_ρqt),
        energy = _arr(int_ρe),
        abs_energy = _arr(int_abs_ρe),
    )
end

function _assert_column_conservation(
    b0,
    b1,
    dt,
    P_rain,
    P_snow,
    P_ener,
    nz::Int;
    Y0 = nothing,
    Y1 = nothing,
    p = nothing,
)
    FT = eltype(b0.energy)
    atol_dry = FT(4 * nz) * eps(FT) .* max.(abs.(b0.dry_air), one(FT))
    atol_wat = FT(4 * nz) * eps(FT) .* max.(abs.(b0.water), FT(1e-6))
    atol_eng = FT(4 * nz) * eps(FT) .* max.(b0.abs_energy, one(FT))
    @test all(abs.(b1.dry_air .- b0.dry_air) .<= atol_dry)
    @test all(
        abs.((b1.water .- b0.water) .- dt .* (P_rain .+ P_snow)) .<= atol_wat,
    )
    @test all(abs.((b1.energy .- b0.energy) .- dt .* P_ener) .<= atol_eng)
    if !isnothing(Y0) && !isnothing(Y1) && !isnothing(p)
        sfc_space = axes(p.precomputed.surface_rain_flux)
        int_dρe = Fields.Field(FT, sfc_space)
        int_dρqt = Fields.Field(FT, sfc_space)
        Operators.column_integral_definite!(int_dρe, Y1.c.ρe_tot .- Y0.c.ρe_tot)
        Operators.column_integral_definite!(
            int_dρqt,
            Y1.c.ρq_tot .- Y0.c.ρq_tot,
        )
        atol_deng = FT(4) * eps(FT) .* max.(b0.abs_energy, one(FT))
        atol_dwat = FT(4) * eps(FT) .* max.(abs.(b0.water), FT(1e-6))
        @test all(abs.(_arr(int_dρe) .- dt .* P_ener) .<= atol_deng)
        @test all(
            abs.(_arr(int_dρqt) .- dt .* (P_rain .+ P_snow)) .<= atol_dwat,
        )
    end
    return nothing
end

function _build_synthetic_1m_state(
    cspace,
    fspace;
    w_mode_rai = CA.Chen2022TerminalVelocity(),
    w_mode_sno = CA.Chen2022TerminalVelocity(),
    fixed_w_rai = nothing,
    fixed_w_sno = nothing,
    with_edmfx::Bool = false,
    with_sgs::Bool = false,
)
    FT = Spaces.undertype(cspace)
    params = if !isnothing(fixed_w_rai) || !isnothing(fixed_w_sno)
        mktemp() do path, io
            if !isnothing(fixed_w_rai)
                println(io, "[fixed_rain_terminal_velocity]")
                println(io, "value = ", Float64(fixed_w_rai))
                println(io, "type = \"float\"")
            end
            if !isnothing(fixed_w_sno)
                println(io, "[fixed_snow_terminal_velocity]")
                println(io, "value = ", Float64(fixed_w_sno))
                println(io, "type = \"float\"")
            end
            close(io)
            CA.ClimaAtmosParameters(CP.create_toml_dict(FT; override_file = path))
        end
    else
        CA.ClimaAtmosParameters(FT)
    end

    tps = CAP.thermodynamics_params(params)
    mp_model = CA.NonEquilibriumMicrophysics1M()
    turbconv_model =
        with_edmfx ?
        CA.PrognosticEDMFX(;
            n_updrafts = 1,
            prognostic_tke = false,
            area_fraction = FT(0.1),
        ) : nothing

    z_c = Fields.coordinate_field(cspace).z
    g = FT(CAP.grav(params))
    ᶜΦ = @. g * z_c
    ᶠlg = Fields.local_geometry_field(fspace)
    ᶠ∂ξ³∂x³ = (lg -> parent(lg.∂ξ∂x)[3, 3]).(ᶠlg)

    Yc = if with_edmfx
        broadcast(
            (args...) -> (;
                ρ = args[1],
                ρe_tot = args[2],
                ρq_tot = args[3],
                ρq_lcl = args[4],
                ρq_icl = args[5],
                ρq_rai = args[6],
                ρq_sno = args[7],
                sgsʲs = ((;
                    ρa = args[8],
                    mse = args[9],
                    q_tot = args[10],
                    q_lcl = args[11],
                    q_icl = args[12],
                    q_rai = args[13],
                    q_sno = args[14],
                ),),
            ),
            ntuple(_ -> zeros(cspace), 14)...,
        )
    else
        broadcast(
            (ρ, ρe_tot, ρq_tot, ρq_lcl, ρq_icl, ρq_rai, ρq_sno) ->
                (; ρ, ρe_tot, ρq_tot, ρq_lcl, ρq_icl, ρq_rai, ρq_sno),
            zeros(cspace),
            zeros(cspace),
            zeros(cspace),
            zeros(cspace),
            zeros(cspace),
            zeros(cspace),
            zeros(cspace),
        )
    end
    Yf = broadcast(
        u3 -> (; u₃ = Geometry.Covariant3Vector(u3)),
        zeros(fspace),
    )
    Y = Fields.FieldVector(; c = Yc, f = Yf)

    # Populate moist hydrostatic-like state.
    ᶜlg = Fields.local_geometry_field(cspace)
    @. Y.c.ρ = FT(1.15) * exp(-z_c / FT(8000))
    @. Y.c.ρq_lcl = FT(0)
    @. Y.c.ρq_icl = FT(0)
    @. Y.c.ρq_rai = FT(0)
    @. Y.c.ρq_sno = FT(0)
    @. Y.c.ρq_tot = Y.c.ρ * FT(0.004) * exp(-z_c / FT(3000))
    @. Y.c.ρe_tot =
        Y.c.ρ * (
            TD.internal_energy(
                tps,
                FT(290) - FT(0.0065) * z_c,
                Y.c.ρq_tot / Y.c.ρ,
                FT(0),
                FT(0),
            ) + ᶜΦ
        )

    sfc_space = Spaces.level(fspace, Operators.half)
    ᶜu = broadcast(
        lg -> Geometry.Covariant123Vector(
            Geometry.UVWVector(zero(lg.J), zero(lg.J), zero(lg.J)),
            lg,
        ),
        ᶜlg,
    )
    pre_base = (;
        ᶜu,
        ᶜwₗ = zeros(cspace),
        ᶜwᵢ = zeros(cspace),
        ᶜwᵣ = zeros(cspace),
        ᶜwₛ = zeros(cspace),
        ᶜwₕhₜ = Geometry.WVector.(zeros(cspace)),
        surface_rain_flux = zeros(sfc_space),
        surface_snow_flux = zeros(sfc_space),
        col_integrated_precip_energy_tendency = zeros(sfc_space),
    )
    sgs_extra = if with_sgs
        m0 = (;
            sigma_S = FT(3e-4),
            λ_lagrange = zero(FT),
            λ_eq = zero(FT),
        )
        (;
            ᶜcloud_fraction = fill(FT(0.35), cspace),
            ᶜprecip_frac = zeros(cspace),
            ᶜT′T′ = fill(FT(0.04), cspace),
            ᶜq′q′ = fill(FT(1e-7), cspace),
            ᶜcorr_Tq = fill(FT(-0.2), cspace),
            ᶜsgs_moments = fill(m0, cspace),
            ᶜα = ones(cspace),
        )
    else
        (;)
    end

    precomputed = if with_edmfx
        ᶜT⁰ = @. FT(290) - FT(0.0065) * z_c
        ᶜp = @. TD.air_pressure(tps, ᶜT⁰, Y.c.ρ, Y.c.ρq_tot / Y.c.ρ, FT(0), FT(0))
        (;
            pre_base...,
            sgs_extra...,
            ᶜρʲs = (copy(Y.c.ρ),),
            ᶜTʲs = (copy(ᶜT⁰),),
            ᶜuʲs = (copy(ᶜu),),
            ᶜwₗʲs = (zeros(cspace),),
            ᶜwᵢʲs = (zeros(cspace),),
            ᶜwᵣʲs = (zeros(cspace),),
            ᶜwₛʲs = (zeros(cspace),),
            ᶜT⁰,
            ᶜp,
            ᶜu⁰ = copy(ᶜu),
        )
    else
        (; pre_base..., sgs_extra...)
    end

    water = (;
        terminal_velocity_liquid = CA.FixedTerminalVelocity(),
        terminal_velocity_ice = CA.FixedTerminalVelocity(),
        terminal_velocity_rain = w_mode_rai,
        terminal_velocity_snow = w_mode_sno,
    )
    sgs_quadrature =
        with_sgs ?
        CA.SGSQuadrature(
            FT;
            quadrature_order = 2,
            distribution = CA.GaussianSGS(),
        ) : CA.GridMeanSGS()
    atmos = (;
        microphysics_model = mp_model,
        turbconv_model,
        sgs_quadrature,
        water,
    )
    core = (; ᶜΦ, ᶠ∂ξ³∂x³)
    p = (; params, atmos, core, precomputed)
    return Y, p
end

function _build_synthetic_0m_edmfx_state(
    cspace,
    fspace;
    params = CA.ClimaAtmosParameters(Spaces.undertype(cspace)),
)
    FT = Spaces.undertype(cspace)
    tps = CAP.thermodynamics_params(params)
    z_c = Fields.coordinate_field(cspace).z
    g = FT(CAP.grav(params))
    ᶜΦ = @. g * z_c
    ᶠlg = Fields.local_geometry_field(fspace)
    ᶠ∂ξ³∂x³ = (lg -> parent(lg.∂ξ∂x)[3, 3]).(ᶠlg)

    Yc = broadcast(
        (ρ, ρe_tot, ρq_tot, ρa, mse, q_tot) -> (;
            ρ,
            ρe_tot,
            ρq_tot,
            sgsʲs = ((; ρa, mse, q_tot),),
        ),
        ntuple(_ -> zeros(cspace), 6)...,
    )
    Yf = broadcast(u3 -> (; u₃ = Geometry.Covariant3Vector(u3)), zeros(fspace))
    Y = Fields.FieldVector(; c = Yc, f = Yf)

    @. Y.c.ρ = FT(1.15) * exp(-z_c / FT(8000))
    ᶜT = @. FT(288) - FT(0.0065) * z_c
    @. Y.c.sgsʲs.:(1).ρa = ifelse(z_c < FT(3000), FT(0.15) * Y.c.ρ, FT(0))
    @. Y.c.sgsʲs.:(1).q_tot = FT(0.018) * exp(-z_c / FT(3000))
    @. Y.c.ρq_tot =
        Y.c.sgsʲs.:(1).ρa * Y.c.sgsʲs.:(1).q_tot +
        (Y.c.ρ - Y.c.sgsʲs.:(1).ρa) * FT(0.012) * exp(-z_c / FT(3000))
    @. Y.c.sgsʲs.:(1).mse =
        TD.moist_static_energy(tps, ᶜT, ᶜΦ, Y.c.sgsʲs.:(1).q_tot, FT(0.003), FT(0))
    @. Y.c.ρe_tot =
        Y.c.ρ *
        (TD.internal_energy(tps, ᶜT, Y.c.ρq_tot / Y.c.ρ, FT(0.002), FT(0)) + ᶜΦ)

    ᶜp = @. TD.air_pressure(tps, ᶜT, Y.c.ρ, Y.c.ρq_tot / Y.c.ρ, FT(0.002), FT(0))
    ᶜq_tot⁰ = @. max(
        FT(0),
        (Y.c.ρq_tot - Y.c.sgsʲs.:(1).ρa * Y.c.sgsʲs.:(1).q_tot) /
        max(eps(FT), Y.c.ρ - Y.c.sgsʲs.:(1).ρa),
    )
    sfc_space = Spaces.level(fspace, Operators.half)
    precomputed = (;
        ᶜρʲs = (copy(Y.c.ρ),),
        ᶜTʲs = (copy(ᶜT),),
        ᶜq_liqʲs = (fill(FT(0.003), cspace),),
        ᶜq_iceʲs = (zeros(cspace),),
        ᶜT⁰ = copy(ᶜT),
        ᶜp,
        ᶜq_tot_nonneg⁰ = ᶜq_tot⁰,
        ᶜq_liq⁰ = fill(FT(0.001), cspace),
        ᶜq_ice⁰ = zeros(cspace),
        surface_rain_flux = zeros(sfc_space),
        surface_snow_flux = zeros(sfc_space),
        col_integrated_precip_energy_tendency = zeros(sfc_space),
        ᶜρ_dq_tot_dt = zeros(cspace),
        ᶜρ_de_tot_dt = zeros(cspace),
        ᶜprecip_evap = broadcast(
            _ -> (; ρ_evap_rai = zero(FT), ρ_evap_sno = zero(FT)),
            zeros(cspace),
        ),
        col_integrated_precip_evap_rai = zeros(sfc_space),
        col_integrated_precip_evap_sno = zeros(sfc_space),
        col_integrated_updraft_precip = zeros(sfc_space),
    )
    atmos = (;
        microphysics_model = CA.EquilibriumMicrophysics0M(),
        turbconv_model = CA.PrognosticEDMFX(;
            n_updrafts = 1,
            prognostic_tke = false,
            area_fraction = FT(0.1),
        ),
        sgs_quadrature = CA.GridMeanSGS(),
    )
    p = (; params, atmos, core = (; ᶜΦ, ᶠ∂ξ³∂x³), precomputed)
    return Y, p
end

function _build_synthetic_2m_state(
    cspace,
    fspace;
    with_p3::Bool = false,
    with_edmfx::Bool = false,
)
    FT = Spaces.undertype(cspace)
    params = CA.ClimaAtmosParameters(FT; implicit_microphysics = false)

    tps = CAP.thermodynamics_params(params)
    z_c = Fields.coordinate_field(cspace).z
    g = FT(CAP.grav(params))
    ᶜΦ = @. g * z_c
    ᶠlg = Fields.local_geometry_field(fspace)
    ᶠ∂ξ³∂x³ = (lg -> parent(lg.∂ξ∂x)[3, 3]).(ᶠlg)
    ᶜlg = Fields.local_geometry_field(cspace)

    Yc = if with_p3
        broadcast(
            (a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12) -> (;
                ρ = a1,
                ρe_tot = a2,
                ρq_tot = a3,
                ρq_lcl = a4,
                ρq_icl = a5,
                ρq_rai = a6,
                ρq_sno = a7,
                ρn_lcl = a8,
                ρn_rai = a9,
                ρn_ice = a10,
                ρq_rim = a11,
                ρb_rim = a12,
            ),
            ntuple(_ -> zeros(cspace), 12)...,
        )
    elseif with_edmfx
        broadcast(
            (a1, a2, a3, a4, a5, a6, a7, a8, a9, b1, b2, b3, b4, b5, b6, b7, b8, b9) ->
                (;
                    ρ = a1,
                    ρe_tot = a2,
                    ρq_tot = a3,
                    ρq_lcl = a4,
                    ρq_icl = a5,
                    ρq_rai = a6,
                    ρq_sno = a7,
                    ρn_lcl = a8,
                    ρn_rai = a9,
                    sgsʲs = ((;
                        ρa = b1,
                        mse = b2,
                        q_tot = b3,
                        q_lcl = b4,
                        q_icl = b5,
                        q_rai = b6,
                        q_sno = b7,
                        n_lcl = b8,
                        n_rai = b9,
                    ),),
                ),
            ntuple(_ -> zeros(cspace), 18)...,
        )
    else
        broadcast(
            (a1, a2, a3, a4, a5, a6, a7, a8, a9) -> (;
                ρ = a1,
                ρe_tot = a2,
                ρq_tot = a3,
                ρq_lcl = a4,
                ρq_icl = a5,
                ρq_rai = a6,
                ρq_sno = a7,
                ρn_lcl = a8,
                ρn_rai = a9,
            ),
            ntuple(_ -> zeros(cspace), 9)...,
        )
    end
    Yf = broadcast(u3 -> (; u₃ = Geometry.Covariant3Vector(u3)), zeros(fspace))
    Y = Fields.FieldVector(; c = Yc, f = Yf)

    @. Y.c.ρ = FT(1.1) * exp(-z_c / FT(8000))
    @. Y.c.ρq_lcl = ifelse(z_c > FT(1500) && z_c < FT(3500), Y.c.ρ * FT(8e-4), FT(0))
    @. Y.c.ρq_rai = ifelse(z_c > FT(1000) && z_c < FT(3000), Y.c.ρ * FT(5e-4), FT(0))
    @. Y.c.ρq_icl = ifelse(z_c > FT(3500), Y.c.ρ * FT(4e-4), FT(0))
    @. Y.c.ρq_sno = FT(0)
    @. Y.c.ρn_lcl = ifelse(Y.c.ρq_lcl > FT(0), Y.c.ρ * FT(1e8), FT(0))
    @. Y.c.ρn_rai = ifelse(Y.c.ρq_rai > FT(0), Y.c.ρ * FT(1e5), FT(0))
    if with_p3
        @. Y.c.ρn_ice = ifelse(Y.c.ρq_icl > FT(0), Y.c.ρ * FT(5e5), FT(0))
        @. Y.c.ρq_rim = FT(0.4) * Y.c.ρq_icl
        @. Y.c.ρb_rim = Y.c.ρq_rim / FT(400)
    end
    @. Y.c.ρq_tot =
        Y.c.ρ * FT(0.008) * exp(-z_c / FT(3000)) +
        Y.c.ρq_lcl +
        Y.c.ρq_icl +
        Y.c.ρq_rai +
        Y.c.ρq_sno

    ᶜT = @. FT(288) - FT(0.0065) * z_c
    @. Y.c.ρe_tot =
        Y.c.ρ * (
            TD.internal_energy(
                tps,
                ᶜT,
                Y.c.ρq_tot / Y.c.ρ,
                (Y.c.ρq_lcl + Y.c.ρq_rai) / Y.c.ρ,
                (Y.c.ρq_icl + Y.c.ρq_sno) / Y.c.ρ,
            ) + ᶜΦ
        )
    ᶜp = @. TD.air_pressure(
        tps,
        ᶜT,
        Y.c.ρ,
        Y.c.ρq_tot / Y.c.ρ,
        (Y.c.ρq_lcl + Y.c.ρq_rai) / Y.c.ρ,
        (Y.c.ρq_icl + Y.c.ρq_sno) / Y.c.ρ,
    )
    ᶜu = broadcast(
        lg -> Geometry.Covariant123Vector(
            Geometry.UVWVector(zero(lg.J), zero(lg.J), zero(lg.J)),
            lg,
        ),
        ᶜlg,
    )
    sfc_space = Spaces.level(fspace, Operators.half)

    if with_edmfx
        @. Y.c.sgsʲs.:(1).ρa = ifelse(z_c < FT(3500), FT(0.1) * Y.c.ρ, FT(0))
        @. Y.c.sgsʲs.:(1).q_lcl = Y.c.ρq_lcl / Y.c.ρ
        @. Y.c.sgsʲs.:(1).q_icl = Y.c.ρq_icl / Y.c.ρ
        @. Y.c.sgsʲs.:(1).q_rai = Y.c.ρq_rai / Y.c.ρ
        @. Y.c.sgsʲs.:(1).q_sno = FT(0)
        @. Y.c.sgsʲs.:(1).q_tot = Y.c.ρq_tot / Y.c.ρ
        @. Y.c.sgsʲs.:(1).n_lcl = Y.c.ρn_lcl / Y.c.ρ
        @. Y.c.sgsʲs.:(1).n_rai = Y.c.ρn_rai / Y.c.ρ
        @. Y.c.sgsʲs.:(1).mse =
            TD.moist_static_energy(
                tps,
                ᶜT,
                ᶜΦ,
                Y.c.sgsʲs.:(1).q_tot,
                Y.c.sgsʲs.:(1).q_lcl + Y.c.sgsʲs.:(1).q_rai,
                Y.c.sgsʲs.:(1).q_icl,
            )
    end

    precomputed = if with_p3
        (;
            ᶜu,
            ᶜp,
            ᶜwₗ = zeros(cspace),
            ᶜwᵢ = zeros(cspace),
            ᶜwᵣ = zeros(cspace),
            ᶜwₛ = zeros(cspace),
            ᶜwₙₗ = zeros(cspace),
            ᶜwₙᵣ = zeros(cspace),
            ᶜwnᵢ = zeros(cspace),
            ᶜwₕhₜ = Geometry.WVector.(zeros(cspace)),
            surface_rain_flux = zeros(sfc_space),
            surface_snow_flux = zeros(sfc_space),
            col_integrated_precip_energy_tendency = zeros(sfc_space),
        )
    elseif with_edmfx
        (;
            ᶜwₗ = zeros(cspace),
            ᶜwᵢ = zeros(cspace),
            ᶜwᵣ = zeros(cspace),
            ᶜwₛ = zeros(cspace),
            ᶜwₙₗ = zeros(cspace),
            ᶜwₙᵣ = zeros(cspace),
            ᶜwₕhₜ = Geometry.WVector.(zeros(cspace)),
            ᶜρʲs = (copy(Y.c.ρ),),
            ᶜTʲs = (copy(ᶜT),),
            ᶜuʲs = (copy(ᶜu),),
            ᶜT⁰ = copy(ᶜT),
            ᶜp,
            ᶜu⁰ = copy(ᶜu),
            surface_rain_flux = zeros(sfc_space),
            surface_snow_flux = zeros(sfc_space),
            col_integrated_precip_energy_tendency = zeros(sfc_space),
        )
    else
        (;
            ᶜu,
            ᶜp,
            ᶜwₗ = zeros(cspace),
            ᶜwᵢ = zeros(cspace),
            ᶜwᵣ = zeros(cspace),
            ᶜwₛ = zeros(cspace),
            ᶜwₙₗ = zeros(cspace),
            ᶜwₙᵣ = zeros(cspace),
            ᶜwₕhₜ = Geometry.WVector.(zeros(cspace)),
            surface_rain_flux = zeros(sfc_space),
            surface_snow_flux = zeros(sfc_space),
            col_integrated_precip_energy_tendency = zeros(sfc_space),
        )
    end

    mp_model =
        with_p3 ? CA.NonEquilibriumMicrophysics2MP3() :
        CA.NonEquilibriumMicrophysics2M()
    turbconv_model =
        with_edmfx ?
        CA.PrognosticEDMFX(;
            n_updrafts = 1,
            prognostic_tke = false,
            area_fraction = FT(0.1),
        ) : nothing
    atmos = (;
        microphysics_model = mp_model,
        turbconv_model,
        sgs_quadrature = CA.GridMeanSGS(),
    )
    core = (; ᶜΦ, ᶠ∂ξ³∂x³)
    p = (; params, atmos, core, precomputed)
    return Y, p
end

function _measure_sweep_allocs(Y, p, dt)
    Y_work = deepcopy(Y)
    p_work = deepcopy(p)
    CA.step_coupled_microphysics_sedimentation!(Y_work, p_work, dt)
    return @allocated CA.step_coupled_microphysics_sedimentation!(
        Y_work,
        p_work,
        dt,
    )
end

_is_cpu_state(Y) = ClimaComms.device(Y.c) isa ClimaComms.AbstractCPUDevice

# ============================================================================
# Test Suite
# ============================================================================

@testset "Coupled Microphysics and Sedimentation Sweep" begin

    @testset "Divided-difference 2 × 2 Metzler matrix functions" begin
        for FT in (Float32, Float64)
            rtol_ref = FT === Float32 ? FT(5e-5) : FT(1e-12)
            atol_ref = FT === Float32 ? FT(1e-6) : FT(1e-14)
            @testset "FT = $FT" begin
                # 1. Exact zero matrix M = 0
                E0, P1_0, P2_0 = CA.metzler_2x2_exp_phi1_phi2(
                    zero(FT),
                    zero(FT),
                    zero(FT),
                    zero(FT),
                    FT(300),
                )
                @test !any(isnan, E0) && !any(isnan, P1_0) && !any(isnan, P2_0)
                @test E0 == SA.SMatrix{2, 2, FT, 4}(1, 0, 0, 1)
                @test P1_0 == SA.SMatrix{2, 2, FT, 4}(1, 0, 0, 1)
                @test P2_0 == SA.SMatrix{2, 2, FT, 4}(FT(0.5), 0, 0, FT(0.5))

                # 2. Defective Jordan block (Δ = 0: D1 == D2, D12 > 0, D21 == 0)
                #    and near-degenerate blocks Δ → 0
                for d in (FT(1e-4), FT(1.0), FT(50.0)),
                    eps_d in (zero(FT), FT(1e-7), FT(1e-4)),
                    (d12, d21) in ((d * FT(0.5), zero(FT)), (FT(1e-6) * d, FT(1e-6) * d))

                    D1 = d * (one(FT) + eps_d)
                    D2 = d
                    dt = FT(2.0)
                    E, P1, P2 = CA.metzler_2x2_exp_phi1_phi2(D1, D2, d12, d21, dt)
                    @test all(isfinite, E) && all(isfinite, P1) && all(isfinite, P2)
                    @test all(>=(zero(FT)), E)
                    @test all(>=(zero(FT)), P1)
                    @test all(>=(zero(FT)), P2)

                    Er, P1r, P2r = _bigfloat_exp_phi1_phi2(D1, D2, d12, d21, dt)
                    @test isapprox(Matrix(E), FT.(Er); rtol = rtol_ref, atol = atol_ref)
                    @test isapprox(Matrix(P1), FT.(P1r); rtol = rtol_ref, atol = atol_ref)
                    @test isapprox(Matrix(P2), FT.(P2r); rtol = rtol_ref, atol = atol_ref)
                end

                # 3. Stiff regime dt * ||M|| = 1e5 and ODE consistency identity
                stiff_cases = (
                    (FT(250), FT(180), FT(120), FT(90), FT(400)),     # ||dt M|| ~ 1e5
                    (FT(1e3), FT(1e-2), FT(500), FT(5e-3), FT(100)),  # multiscale stiff
                    (FT(200), FT(200), FT(200), FT(200), FT(500)),    # singular conservative stiff
                )
                for (D1, D2, D12, D21, dt) in stiff_cases
                    M = SA.SMatrix{2, 2, FT, 4}(-D1, D12, D21, -D2)
                    E, P1, P2 = CA.metzler_2x2_exp_phi1_phi2(M, dt)
                    @test all(isfinite, E) && all(isfinite, P1) && all(isfinite, P2)
                    @test all(>=(zero(FT)), E)
                    @test all(>=(zero(FT)), P1)
                    @test all(>=(zero(FT)), P2)

                    Er, P1r, P2r = _bigfloat_exp_phi1_phi2(D1, D2, D12, D21, dt)
                    @test isapprox(Matrix(E), FT.(Er); rtol = rtol_ref, atol = atol_ref)
                    @test isapprox(Matrix(P1), FT.(P1r); rtol = rtol_ref, atol = atol_ref)
                    @test isapprox(Matrix(P2), FT.(P2r); rtol = rtol_ref, atol = atol_ref)

                    # Exact ODE balance: q(dt) - q0 == dt * (M * q_bar + b),
                    # scaled by backward-error norm dt * ||M|| * ||q_bar|| for singular stiff M.
                    q0 = SA.SVector{2, FT}(FT(1.2e-3), FT(0.7e-3))
                    b = SA.SVector{2, FT}(FT(3e-6), FT(1.5e-6))
                    q_new = E * q0 + dt * (P1 * b)
                    q_bar = P1 * q0 + dt * (P2 * b)
                    lhs = q_new - q0
                    rhs = dt * (M * q_bar + b)
                    cond_scale = max(
                        maximum(abs.(q0)),
                        dt * maximum(abs.(b)),
                        dt * maximum(abs.(M)) * maximum(abs.(q_bar)),
                    )
                    @test isapprox(lhs, rhs; atol = FT(16) * eps(FT) * cond_scale)
                end

                # 4. Exact column-sum conservation when D1 == D12 and D2 == D21
                for scale_dt in (FT(1e-4), FT(1.0), FT(1e3), FT(1e5))
                    D1 = FT(0.7)
                    D2 = FT(0.3)
                    E, P1, P2 = CA.metzler_2x2_exp_phi1_phi2(D1, D2, D1, D2, scale_dt)
                    ones2 = SA.SVector{2, FT}(one(FT), one(FT))
                    @test isapprox(E' * ones2, ones2; rtol = FT(16) * eps(FT))
                    @test isapprox(P1' * ones2, ones2; rtol = FT(16) * eps(FT))
                    @test isapprox(P2' * ones2, FT(0.5) .* ones2; rtol = FT(16) * eps(FT))
                end
            end
        end
    end

    @testset "Residence-limited supply and 2-step Newton inversion" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                dt = FT(300)
                for (q0, S_in) in (
                    (FT(1e-3), zero(FT)),
                    (zero(FT), FT(5e-6)),
                    (FT(5e-4), FT(4e-6)),
                )
                    Q_sup = q0 + S_in * dt
                    for C_sed in (zero(FT), FT(0.2), FT(2.0), FT(15.0), FT(40.0))
                        D_sed = C_sed / dt
                        for frac in (FT(1e-5), FT(1e-2), FT(0.25), FT(0.7), FT(0.98))
                            target = frac * Q_sup
                            D_e = CA.invert_residence_supply(target, q0, S_in, D_sed, dt)
                            @test isfinite(D_e) && D_e >= zero(FT)
                            realized = CA.residence_supply(D_e, q0, S_in, D_sed, dt)
                            @test isapprox(
                                realized,
                                target;
                                rtol = FT(2e-3),
                                atol = FT(10) * eps(FT) * Q_sup,
                            )
                            @test realized <= Q_sup * (one(FT) + FT(10) * eps(FT))
                        end
                        # Oversaturated demand clamps safely below Q_sup
                        D_e_cap =
                            CA.invert_residence_supply(FT(10) * Q_sup, q0, S_in, D_sed, dt)
                        @test isfinite(D_e_cap) && D_e_cap > zero(FT)
                        realized_cap = CA.residence_supply(D_e_cap, q0, S_in, D_sed, dt)
                        @test realized_cap <= Q_sup
                        @test realized_cap >= FT(0.99) * Q_sup
                    end
                end
            end
        end
    end

    @testset "Vapor sinks meet their targets with rain-snow transfer in the step" begin
        for FT in (Float32, Float64)
            @testset "FT = $FT" begin
                z = zero(FT)
                dt = FT(600)
                target = FT(2e-5)
                q_src = FT(3e-4)
                # Warm cell: melting snow feeds rain that evaporates.
                melt = q_src * (1 - exp(-dt / FT(200)))
                for q0_rai in (z, FT(1e-5), FT(1e-4))
                    p_new, _ = CA._solve_1m_precip_step(
                        q0_rai, q_src, z, z, z, z, z, z, z, -target, z, melt,
                        true, one(FT), one(FT), dt,
                    )
                    evap = q0_rai + q_src - sum(p_new)
                    @test target * (1 - FT(1e-3)) <= evap <= FT(1.05) * target
                end
                # Cold cell: rain freezing onto snow feeds snow that sublimates.
                for q0_sno in (z, FT(1e-5))
                    p_new, _ = CA._solve_1m_precip_step(
                        q_src, q0_sno, z, z, z, z, FT(1 / 200), z, z, z,
                        -target, z, false, one(FT), one(FT), dt,
                    )
                    subl = q_src + q0_sno - sum(p_new)
                    @test target * (1 - FT(1e-3)) <= subl <= FT(1.05) * target
                end
            end
        end
    end

    @testset "Warm-air rain-ice gate" begin
        for FT in (Float32, Float64)
            params = CA.ClimaAtmosParameters(FT)
            tps = CAP.thermodynamics_params(params)
            mp = CA._combined_microphysics_1m_params(params)
            T_frz = FT(TD.Parameters.T_freeze(tps))
            ρ = FT(1.0)
            q_v0 = FT(0.003)
            q_icl = FT(5e-4)
            q_rai = FT(1e-3)

            r_cold = CA.coupled_1m_source_terms(
                mp, tps, ρ, T_frz - FT(2), zero(FT), q_v0, zero(FT), q_icl, q_rai,
                zero(FT),
            )
            @test r_cold.S_accr_icl_rai_cold > zero(FT)
            @test r_cold.S_accr_freeze_icl_rai > zero(FT)
            @test r_cold.S_accr_icl_rai_warm == zero(FT)

            for ΔT_warm in (zero(FT), FT(2))
                r_warm = CA.coupled_1m_source_terms(
                    mp, tps, ρ, T_frz + ΔT_warm, zero(FT), q_v0, zero(FT), q_icl, q_rai,
                    zero(FT),
                )
                @test r_warm.S_accr_icl_rai_warm > zero(FT)
                @test r_warm.S_accr_icl_rai_cold == zero(FT)
                @test r_warm.S_accr_freeze_icl_rai == zero(FT)
            end

            # Full cell step at T > T_freeze with snow autoconversion off and
            # q_icl + q_rai > 0, q_sno = 0 routes warm collisions exclusively to rain.
            mp_no_acnv = CMP.Microphysics1MParams(
                CP.create_toml_dict(FT);
                snow_autoconversion = nothing,
            )
            T_w = T_frz + FT(1.5)
            q_sl = TD.q_vap_saturation(tps, T_w, ρ, TD.Liquid())
            q0 = SA.SVector{4, FT}(zero(FT), q_icl, q_rai, zero(FT))
            z4 = zero(SA.SVector{4, FT})
            res = CA.coupled_1m_cell_step(
                mp_no_acnv, tps, ρ, T_w, zero(FT), q_sl + sum(q0), q0, z4, q0, z4,
                FT(120),
            )
            @test res.q_new[4] == zero(FT)
            @test res.q_new[3] > q_rai
        end
    end

    @testset "WBF sign-flip and snow-deposition safety (Δt = 1..1200 s)" begin
        for FT in (Float32, Float64)
            params = CA.ClimaAtmosParameters(FT)
            tps = CAP.thermodynamics_params(params)
            mp = CA._combined_microphysics_1m_params(params)
            ρ = FT(0.85)
            T0 = FT(261.0)
            q_sl = TD.q_vap_saturation(tps, T0, ρ, TD.Liquid())
            q_si = TD.q_vap_saturation(tps, T0, ρ, TD.Ice())
            z4 = zero(SA.SVector{4, FT})

            @testset "FT = $FT: |δ̄_l / δ_l⁰| ≫ 1 with snow deposition" begin
                # Near liquid saturation (tiny δ_l⁰ < 0, positive δ_i⁰ > 0):
                # snow deposition pulls vapor down, amplifying rain evaporation |δ̄_l / δ_l⁰| ≫ 1.
                q_v0 = q_sl - FT(1e-7)
                q0 = SA.SVector{4, FT}(FT(1e-4), zero(FT), FT(1.5e-3), FT(2.5e-3))
                q_tot0 = q_v0 + sum(q0)

                for dt in FT.((1, 10, 60, 300, 600, 1200))
                    res = CA.coupled_1m_cell_step(
                        mp, tps, ρ, T0, zero(FT), q_tot0, q0, z4, q0, z4, dt,
                    )
                    @test all(isfinite, res.q_new) && all(>=(zero(FT)), res.q_new)
                    @test isfinite(res.α_cap) && zero(FT) <= res.α_cap <= one(FT)
                    q_v_new = q_tot0 - sum(res.q_new)
                    @test q_v_new >= -FT(10) * eps(FT) * q_tot0
                    # Snow grows via WBF vapor deposition while supercooled rain/cloud evaporate
                    @test res.q_new[4] > q0[4]
                    @test res.q_new[3] < q0[3]

                    # Post-step vapor stays above ice saturation (up to Clausius-Clapeyron nonlinearity)
                    e_int0 = TD.internal_energy(
                        tps, T0, q_tot0, q0[1] + q0[3], q0[2] + q0[4],
                    )
                    T_new = TD.air_temperature(
                        tps,
                        e_int0,
                        q_tot0,
                        res.q_new[1] + res.q_new[3],
                        res.q_new[2] + res.q_new[4],
                    )
                    q_si_new = TD.q_vap_saturation(tps, T_new, ρ, TD.Ice())
                    @test q_v_new >= q_si_new - FT(8e-5)
                end
            end

            @testset "FT = $FT: δ_i⁰ < 0 < δ̄_i sign flip" begin
                # Initially slightly subsaturated wrt ice (`δ_i⁰ < 0`), with supercooled
                # liquid + rain whose rapid evaporation raises `q_v` above `q_si` mid-step.
                q_v0 = q_si - FT(3e-5)
                @test q_v0 - q_si < zero(FT)
                q0 = SA.SVector{4, FT}(FT(1.5e-3), zero(FT), FT(2.0e-3), FT(8e-4))
                q_tot0 = q_v0 + sum(q0)

                flipped_any = false
                for dt in FT.((1, 10, 60, 300, 600, 1200))
                    res = CA.coupled_1m_cell_step(
                        mp, tps, ρ, T0, zero(FT), q_tot0, q0, z4, q0, z4, dt,
                    )
                    @test all(isfinite, res.q_new) && all(>=(zero(FT)), res.q_new)
                    q_v_new = q_tot0 - sum(res.q_new)
                    @test q_v_new >= zero(FT)
                    if dt >= FT(60)
                        flipped_any |= (res.q_new[4] > q0[4])
                        @test res.q_new[4] >= q0[4]
                    end
                end
                @test flipped_any
            end
        end
    end

    @testset "θ = T - T_freeze = 0 crossing thermal arrest" begin
        for FT in (Float32, Float64)
            params = CA.ClimaAtmosParameters(FT)
            tps = CAP.thermodynamics_params(params)
            mp = CA._combined_microphysics_1m_params(params)
            T_frz = FT(TD.Parameters.T_freeze(tps))
            L_f = TD.latent_heat_fusion(tps, T_frz + FT(2))
            ρ = FT(1.05)
            T0 = T_frz + FT(2)
            q_sl = TD.q_vap_saturation(tps, T0, ρ, TD.Liquid())
            z4 = zero(SA.SVector{4, FT})

            @testset "FT = $FT: simultaneous snow melt + rain evap at Δt = 300 s" begin
                # Subsaturated +2 K layer: rain evaporation + snow melt cross T_freeze mid-step.
                q_v0 = FT(0.45) * q_sl
                q0 = SA.SVector{4, FT}(zero(FT), zero(FT), FT(4e-3), FT(6e-3))
                q_tot0 = q_v0 + sum(q0)
                dt = FT(300)

                res_cv = CA.coupled_1m_cell_step(
                    mp, tps, ρ, T0, zero(FT), q_tot0, q0, z4, q0, z4, dt;
                    is_constant_p = false,
                )
                res_cp = CA.coupled_1m_cell_step(
                    mp, tps, ρ, T0, zero(FT), q_tot0, q0, z4, q0, z4, dt;
                    is_constant_p = true,
                )

                for (res, c_eff) in (
                    (res_cv, TD.cv_m(tps, q_tot0, q0[3], q0[4])),
                    (res_cp, TD.cp_m(tps, q_tot0, q0[3], q0[4])),
                )
                    @test all(isfinite, res.q_new) && all(>=(zero(FT)), res.q_new)
                    @test zero(FT) < res.f_melt < one(FT)
                    # Snow melts partially and never un-melts (`q_sno_new < q_sno0`).
                    Δq_melt = q0[4] - res.q_new[4]
                    @test zero(FT) < Δq_melt < q0[4]
                    # Evaporative cooling took part of the 2 K budget, so melt cooling < 2 K.
                    @test (L_f * Δq_melt) / c_eff < FT(2.0)
                end
                # Isobaric c_pm > isochoric c_vm holds more sensible heat above T_freeze,
                # permitting strictly more snow to melt before arresting at T_freeze.
                @test (q0[4] - res_cp.q_new[4]) > (q0[4] - res_cv.q_new[4])
            end

            @testset "FT = $FT: pure abundant snow melt arrests at T_freeze" begin
                mp_pure_melt = CMP.Microphysics1MParams(
                    CP.create_toml_dict(FT);
                    cloud_liquid_formation = nothing,
                    cloud_ice_formation = nothing,
                    snow_deposition_sublimation = nothing,
                )
                q0 = SA.SVector{4, FT}(zero(FT), zero(FT), zero(FT), FT(2e-2))
                q_tot0 = q_sl + sum(q0)
                e_int0 = TD.internal_energy(tps, T0, q_tot0, zero(FT), q0[4])

                for dt in FT.((60, 300, 600, 1200))
                    res = CA.coupled_1m_cell_step(
                        mp_pure_melt, tps, ρ, T0, zero(FT), q_tot0, q0, z4, q0, z4, dt;
                        is_constant_p = false,
                    )
                    T_post = TD.air_temperature(
                        tps,
                        e_int0,
                        q_tot0,
                        res.q_new[1] + res.q_new[3],
                        res.q_new[2] + res.q_new[4],
                    )
                    @test abs(T_post - T_frz) < FT(0.05)
                    @test res.q_new[4] < q0[4]
                    @test res.q_new[3] > zero(FT)
                end
            end

            @testset "FT = $FT: smooth transition across θ = 0 crossing" begin
                q0 = SA.SVector{4, FT}(zero(FT), zero(FT), FT(3e-3), FT(1.5e-3))
                rh_grid = range(FT(0.35), FT(1.0); length = 25)
                sno_vals = map(rh_grid) do rh
                    qv = rh * q_sl
                    CA.coupled_1m_cell_step(
                        mp, tps, ρ, T0, zero(FT), qv + sum(q0), q0, z4, q0, z4, FT(300),
                    ).q_new[4]
                end
                @test all(isfinite, sno_vals)
                # Monotone more melting as air becomes moister (less evaporative cooling)
                @test all(diff(sno_vals) .<= FT(1e-6))
                @test maximum(abs.(diff(sno_vals))) < FT(3e-4)
            end
        end
    end

    @testset "Empty-cell steady & transient virga shafts (Δt = 1..600 s, C_sed up to 40)" begin
        @testset "Exact Δt-invariance of steady-shaft attenuation F_out = F_in / (1 + D_e Δz / w)" begin
            for FT in (Float32, Float64)
                nz = 12
                dz = FT(150)
                w = FT(10)          # C_sed = w * 600 / 150 = 40 at dt = 600 s
                D_e_true = FT(1e-3) # Fixed linear evaporation rate [1/s]
                F_top = FT(2e-3)    # Incoming rain mass flux [kg m⁻² s⁻¹]
                ρ_col = [FT(1.1) * exp(-FT(k * 150) / FT(8000)) for k in 1:nz]

                F_exact = F_top / (one(FT) + D_e_true * dz / w)^nz
                for dt_s in (1, 10, 60, 300, 600)
                    dt = FT(dt_s)
                    C_sed = w * dt / dz
                    D_sed = w / dz
                    # Integrate initially empty column (`q = 0`) to `t = 2400 s`
                    # (`> 13` shaft transit times; 4 steps at `dt = 600 s`, `C_sed = 40`).
                    q_col = zeros(FT, nz)
                    n_steps = 2400 ÷ dt_s
                    F_bot = zero(FT)
                    for _ in 1:n_steps
                        F_in = F_top
                        for k in nz:-1:1
                            q0_k = q_col[k]
                            S_in = F_in / (ρ_col[k] * dz)
                            target = CA.residence_supply(D_e_true, q0_k, S_in, D_sed, dt)
                            D_e_inv =
                                CA.invert_residence_supply(target, q0_k, S_in, D_sed, dt)
                            E_m, P1, P2 = CA.metzler_2x2_exp_phi1_phi2(
                                D_sed + D_e_inv, zero(FT), zero(FT), zero(FT), dt,
                            )
                            q_col[k] = E_m[1, 1] * q0_k + dt * P1[1, 1] * S_in
                            q_bar = P1[1, 1] * q0_k + dt * P2[1, 1] * S_in
                            q_out = D_sed * q_bar * dt
                            F_in = (ρ_col[k] * dz / dt) * q_out
                        end
                        F_bot = F_in
                    end
                    @test isapprox(F_bot, F_exact; rtol = FT(2e-3))
                    if dt_s == 600
                        @test C_sed ≈ FT(40)
                        # Single-step empty-cell limit at C_sed = 40 already delivers > 70% of steady flux
                        F_1 = F_top
                        for k in nz:-1:1
                            S_in = F_1 / (ρ_col[k] * dz)
                            target =
                                CA.residence_supply(D_e_true, zero(FT), S_in, D_sed, dt)
                            D_e_inv = CA.invert_residence_supply(
                                target,
                                zero(FT),
                                S_in,
                                D_sed,
                                dt,
                            )
                            _, _, P2 = CA.metzler_2x2_exp_phi1_phi2(
                                D_sed + D_e_inv, zero(FT), zero(FT), zero(FT), dt,
                            )
                            F_1 = (ρ_col[k] * dz) * D_sed * P2[1, 1] * S_in * dt
                        end
                        @test F_1 > FT(0.7) * F_exact
                    end
                end
            end
        end

        @testset "Full 1M column virga shaft across Δt ∈ [1, 10, 60, 300, 600] s (C_sed = 40)" begin
            for FT in (Float32, Float64)
                # 20 levels with Δz = 150 m (`z_max = 3000 m`), fixed `w_rai = 10 m/s`
                # so `C_sed = 10 * 600 / 150 = 40` at `Δt = 600 s`.
                cspace, fspace = _make_column_spaces(FT; z_max = FT(3000), nz = 20)
                Y0, p = _build_synthetic_1m_state(
                    cspace,
                    fspace;
                    w_mode_rai = CA.FixedTerminalVelocity(),
                    fixed_w_rai = FT(10),
                )
                tps = CAP.thermodynamics_params(p.params)
                z_c = Fields.coordinate_field(cspace).z
                ᶜΦ = p.core.ᶜΦ
                # Put a rain reservoir in top 4 levels (`z > 2400 m`) above an
                # initially empty (`q_rai = 0`) subsaturated virga shaft below.
                @. Y0.c.ρq_rai = ifelse(z_c > FT(2400), Y0.c.ρ * FT(2e-3), FT(0))
                @. Y0.c.ρq_tot = Y0.c.ρ * FT(0.003) + Y0.c.ρq_rai
                @. Y0.c.ρe_tot =
                    Y0.c.ρ * (
                        TD.internal_energy(
                            tps,
                            FT(285) - FT(0.005) * z_c,
                            Y0.c.ρq_tot / Y0.c.ρ,
                            Y0.c.ρq_rai / Y0.c.ρ,
                            FT(0),
                        ) + ᶜΦ
                    )

                for dt in FT.((1, 10, 60, 300, 600))
                    Y = deepcopy(Y0)
                    b0 = _column_integrals(Y, p)
                    CA.step_coupled_microphysics_sedimentation!(Y, p, dt)
                    b1 = _column_integrals(Y, p)

                    P_rain = _arr(p.precomputed.surface_rain_flux)
                    P_snow = _arr(p.precomputed.surface_snow_flux)
                    P_ener = _arr(p.precomputed.col_integrated_precip_energy_tendency)

                    @test all(P_rain .<= zero(FT))
                    @test all(P_snow .<= zero(FT))
                    if dt >= FT(300)
                        # At dt = 300..600 s (`C_sed = 20..40`), the shaft traverses all 16
                        # empty levels (`2400 m`) in a single step and reaches the ground.
                        @test all(P_rain .< zero(FT))
                        atol_deng = FT(4) * eps(FT) .* max.(b0.abs_energy, one(FT))
                        @test all(abs.(dt .* P_ener) .> FT(5) .* atol_deng)
                    end
                    @test all(>=(zero(FT)), _arr(Y.c.ρq_rai))
                    @test all(>=(zero(FT)), _arr(Y.c.ρq_tot))

                    # Round-off conservation of dry air, total water, and total energy
                    _assert_column_conservation(
                        b0,
                        b1,
                        dt,
                        P_rain,
                        P_snow,
                        P_ener,
                        20;
                        Y0 = Y0,
                        Y1 = Y,
                        p = p,
                    )
                end

                if _is_cpu_state(Y0)
                    @test _measure_sweep_allocs(Y0, p, FT(60)) == 0
                end
            end
        end

        @testset "Donor-cell post-microphysics temperature energy export & ᶜwₕhₜ" begin
            FT = Float64
            cspace, fspace = _make_column_spaces(FT; z_max = FT(2000), nz = 10)
            # 1. Grid-mean rain + snow at k = 1 with horizontal wind u = 6 m/s
            Y, p = _build_synthetic_1m_state(
                cspace,
                fspace;
                w_mode_rai = CA.FixedTerminalVelocity(),
                w_mode_sno = CA.FixedTerminalVelocity(),
                fixed_w_rai = FT(8),
                fixed_w_sno = FT(2),
            )
            tps = CAP.thermodynamics_params(p.params)
            z_c = Fields.coordinate_field(cspace).z
            ᶜΦ = p.core.ᶜΦ
            ᶜlg = Fields.local_geometry_field(cspace)
            u_horiz = Geometry.UVWVector(FT(6), zero(FT), zero(FT))
            p.precomputed.ᶜu .=
                broadcast(lg -> Geometry.Covariant123Vector(u_horiz, lg), ᶜlg)
            @. Y.c.ρq_rai = ifelse(z_c < FT(200), Y.c.ρ * FT(2.5e-3), FT(0))
            @. Y.c.ρq_sno = ifelse(z_c < FT(200), Y.c.ρ * FT(1.5e-3), FT(0))
            @. Y.c.ρq_tot = Y.c.ρ * FT(0.001) + Y.c.ρq_rai + Y.c.ρq_sno
            K_horiz = FT(18) # 0.5 * 6^2
            @. Y.c.ρe_tot =
                Y.c.ρ * (
                    TD.internal_energy(
                        tps,
                        FT(270),
                        Y.c.ρq_tot / Y.c.ρ,
                        Y.c.ρq_rai / Y.c.ρ,
                        Y.c.ρq_sno / Y.c.ρ,
                    ) +
                    K_horiz +
                    ᶜΦ
                )

            dt = FT(15)
            CA.step_coupled_microphysics_sedimentation!(Y, p, dt)

            ρ_1 = _arr(Y.c.ρ)[1]
            ρqt_1 = _arr(Y.c.ρq_tot)[1]
            ρql_1 = _arr(Y.c.ρq_lcl)[1]
            ρqi_1 = _arr(Y.c.ρq_icl)[1]
            ρqr_1 = _arr(Y.c.ρq_rai)[1]
            ρqs_1 = _arr(Y.c.ρq_sno)[1]
            ρe_1 = _arr(Y.c.ρe_tot)[1]
            Φ_1 = _arr(ᶜΦ)[1]
            wr_1 = _arr(p.precomputed.ᶜwᵣ)[1]
            ws_1 = _arr(p.precomputed.ᶜwₛ)[1]

            e_int_1 = ρe_1 / ρ_1 - K_horiz - Φ_1
            T1_post = TD.air_temperature(
                tps,
                e_int_1,
                ρqt_1 / ρ_1,
                (ρql_1 + ρqr_1) / ρ_1,
                (ρqi_1 + ρqs_1) / ρ_1,
            )
            P_rain_1 = _arr(p.precomputed.surface_rain_flux)[1]
            P_snow_1 = _arr(p.precomputed.surface_snow_flux)[1]
            P_ener_1 = _arr(p.precomputed.col_integrated_precip_energy_tendency)[1]
            # Subsaturated over liquid and ice (`q_v = 1e-3 < q_si(270 K)`), so
            # cloud pools stay zero and surface fluxes come purely from rain and snow.
            @test ρql_1 == zero(FT) && ρqi_1 == zero(FT)
            K_rai = FT(0.5) * (FT(6)^2 + (-wr_1)^2)
            K_sno = FT(0.5) * (FT(6)^2 + (-ws_1)^2)
            P_ener_expected =
                P_rain_1 * (TD.internal_energy_liquid(tps, T1_post) + Φ_1 + K_rai) +
                P_snow_1 * (TD.internal_energy_ice(tps, T1_post) + Φ_1 + K_sno)
            @test isapprox(P_ener_1, P_ener_expected; rtol = 1e-12)

            whht_1 = _arr(p.precomputed.ᶜwₕhₜ.components.data.:1)[1]
            @test isapprox(whht_1, -P_ener_1 / ρ_1; rtol = 1e-12)

            # 2. EDMFX warm updraft (`T¹ = T⁰ + 12 K`) with rain only in the
            #    environment: the export uses `T⁰_post`, uncontaminated by the
            #    warm updraft, and the kinetic energy of the environment fall
            #    speed relative to the environment air (`w⁰ = 0.7 m/s`; the
            #    updraft rises at 3 m/s).
            Y_ed, p_ed = _build_synthetic_1m_state(
                cspace,
                fspace;
                w_mode_rai = CA.FixedTerminalVelocity(),
                fixed_w_rai = FT(8),
                with_edmfx = true,
            )
            @. Y_ed.c.sgsʲs.:(1).ρa = FT(0.3) * Y_ed.c.ρ
            @. Y_ed.c.sgsʲs.:(1).q_tot = FT(0.004)
            @. Y_ed.c.sgsʲs.:(1).q_lcl = FT(0)
            @. Y_ed.c.sgsʲs.:(1).q_icl = FT(0)
            @. Y_ed.c.sgsʲs.:(1).q_rai = FT(0)
            @. Y_ed.c.sgsʲs.:(1).q_sno = FT(0)
            @. Y_ed.c.ρq_rai =
                ifelse(z_c < FT(200), (Y_ed.c.ρ - Y_ed.c.sgsʲs.:(1).ρa) * FT(3e-3), FT(0))
            @. Y_ed.c.ρq_tot =
                Y_ed.c.sgsʲs.:(1).ρa * FT(0.004) +
                (Y_ed.c.ρ - Y_ed.c.sgsʲs.:(1).ρa) * FT(0.004) +
                Y_ed.c.ρq_rai
            @. p_ed.precomputed.ᶜT⁰ = FT(280)
            @. p_ed.precomputed.ᶜTʲs.:(1) = FT(292)
            # Environment area far above `42 a_half`, so the environment pool
            # holds all environment rain.
            p_ed = (;
                p_ed...,
                atmos = (;
                    p_ed.atmos...,
                    turbconv_model = CA.PrognosticEDMFX(;
                        n_updrafts = 1,
                        prognostic_tke = false,
                        area_fraction = FT(1e-5),
                    ),
                ),
            )
            uvw_env = Geometry.UVWVector(zero(FT), zero(FT), FT(0.7))
            uvw_up = Geometry.UVWVector(zero(FT), zero(FT), FT(3))
            p_ed.precomputed.ᶜu⁰ .=
                broadcast(lg -> Geometry.Covariant123Vector(uvw_env, lg), ᶜlg)
            p_ed.precomputed.ᶜuʲs.:(1) .=
                broadcast(lg -> Geometry.Covariant123Vector(uvw_up, lg), ᶜlg)
            @. Y_ed.c.sgsʲs.:(1).mse =
                TD.moist_static_energy(tps, FT(292), ᶜΦ, FT(0.004), FT(0), FT(0))
            @. Y_ed.c.ρe_tot =
                Y_ed.c.sgsʲs.:(1).ρa *
                (TD.internal_energy(tps, FT(292), FT(0.004), FT(0), FT(0)) + ᶜΦ) +
                (Y_ed.c.ρ - Y_ed.c.sgsʲs.:(1).ρa) * (
                    TD.internal_energy(
                        tps,
                        FT(280),
                        (Y_ed.c.ρq_tot - Y_ed.c.sgsʲs.:(1).ρa * FT(0.004)) /
                        (Y_ed.c.ρ - Y_ed.c.sgsʲs.:(1).ρa),
                        Y_ed.c.ρq_rai / (Y_ed.c.ρ - Y_ed.c.sgsʲs.:(1).ρa),
                        FT(0),
                    ) + ᶜΦ
                )
            CA.step_coupled_microphysics_sedimentation!(Y_ed, p_ed, dt)
            Pr_ed = _arr(p_ed.precomputed.surface_rain_flux)[1]
            Pe_ed = _arr(p_ed.precomputed.col_integrated_precip_energy_tendency)[1]
            T0_post_ed = _arr(p_ed.precomputed.ᶜT⁰)[1]
            wr_ed = _arr(p_ed.precomputed.ᶜwᵣ)[1]
            @test T0_post_ed < FT(281)
            Pe_env_expected =
                Pr_ed * (
                    TD.internal_energy_liquid(tps, T0_post_ed) +
                    Φ_1 +
                    CA.sedimentation_kinetic_energy(wr_ed, uvw_env)
                )
            @test isapprox(Pe_ed, Pe_env_expected; rtol = 1e-11)
            Pe_up_air =
                Pr_ed * (
                    TD.internal_energy_liquid(tps, T0_post_ed) +
                    Φ_1 +
                    CA.sedimentation_kinetic_energy(wr_ed, uvw_up)
                )
            @test !isapprox(Pe_ed, Pe_up_air; rtol = 1e-6)

            # 3. The export splits into static energy and the kinetic energy of
            #    each species' own fall speed relative to the cell's air.
            q_out = SA.SVector{4, FT}(1e-4, 2e-5, 3e-4, 1e-4)
            w_sed = SA.SVector{4, FT}(0.01, 0.4, 6.0, 1.2)
            uvw = Geometry.UVWVector(FT(4), FT(-2), FT(0.5))
            ρ_old, V, dt_h, Φ_h = FT(1.0), FT(200), FT(60), FT(9800)
            ρ_new = ρ_old - sum(q_out) * ρ_old
            ρq_new = SA.SVector{4, FT}(2e-4, 1e-4, 5e-4, 2e-4) .* ρ_old
            ρqt_new = sum(ρq_new) + FT(8e-3) * ρ_new
            ρe_old =
                ρ_old * (
                    TD.internal_energy(tps, FT(275), FT(0.01), FT(1e-3), FT(4e-4)) +
                    Φ_h +
                    (uvw.u^2 + uvw.v^2 + uvw.w^2) / 2
                )
            T_h, F_e_h, F_K_h = CA._donor_post_mp_temperature_and_energy_out(
                tps, ρ_old, ρe_old, zero(FT), V, dt_h, ρ_new, ρqt_new, ρq_new,
                q_out, w_sed, uvw, Φ_h,
            )
            m_out = ρ_old .* q_out .* (V / dt_h)
            K_s = CA.sedimentation_kinetic_energy.(w_sed, Ref(uvw))
            e_l = TD.internal_energy_liquid(tps, T_h)
            e_i = TD.internal_energy_ice(tps, T_h)
            @test isapprox(F_K_h, sum(m_out .* K_s); rtol = 1e-14)
            @test isapprox(
                F_e_h - F_K_h,
                (m_out[1] + m_out[3]) * (e_l + Φ_h) +
                (m_out[2] + m_out[4]) * (e_i + Φ_h);
                rtol = 1e-14,
            )
            whht_ed_1 = _arr(p_ed.precomputed.ᶜwₕhₜ.components.data.:1)[1]
            @test isapprox(whht_ed_1, -Pe_ed / _arr(Y_ed.c.ρ)[1]; rtol = 1e-12)
        end

        @testset "Transient cloud-top autoconversion & melt-layer rain fall immediately" begin
            FT = Float64
            cspace, fspace = _make_column_spaces(FT; z_max = FT(4000), nz = 20)

            # 1. Cloud-top autoconversion with q_rai = 0 everywhere initially
            Y_cld, p_cld = _build_synthetic_1m_state(cspace, fspace)
            tps = CAP.thermodynamics_params(p_cld.params)
            z_c = Fields.coordinate_field(cspace).z
            ᶜΦ = p_cld.core.ᶜΦ
            @. Y_cld.c.ρq_lcl = ifelse(z_c > FT(3400), Y_cld.c.ρ * FT(3.5e-3), FT(0))
            @. Y_cld.c.ρq_rai = FT(0)
            @. Y_cld.c.ρq_tot = Y_cld.c.ρ * FT(0.008) + Y_cld.c.ρq_lcl
            @. Y_cld.c.ρe_tot =
                Y_cld.c.ρ * (
                    TD.internal_energy(
                        tps,
                        FT(288) - FT(0.004) * z_c,
                        Y_cld.c.ρq_tot / Y_cld.c.ρ,
                        Y_cld.c.ρq_lcl / Y_cld.c.ρ,
                        FT(0),
                    ) + ᶜΦ
                )
            CA.step_coupled_microphysics_sedimentation!(Y_cld, p_cld, FT(300))
            qr_after = _arr(Y_cld.c.ρq_rai)
            wr_after = _arr(p_cld.precomputed.ᶜwᵣ)
            # Newly autoconverted rain activated `w_rai > 0` in the cloud layer and
            # fell immediately into empty levels well below `z = 3400 m`.
            @test wr_after[end] > FT(1.0)
            @test any(qr_after[1:15] .> FT(1e-8))

            # 2. Cold snow falling into an initially empty warm layer (`T > T_freeze`)
            #    melts into rain and falls immediately within a single step.
            Y_mlt, p_mlt = _build_synthetic_1m_state(cspace, fspace)
            @. Y_mlt.c.ρq_sno = ifelse(z_c > FT(2600), Y_mlt.c.ρ * FT(2e-3), FT(0))
            @. Y_mlt.c.ρq_rai = FT(0)
            @. Y_mlt.c.ρq_tot = Y_mlt.c.ρ * FT(0.006) + Y_mlt.c.ρq_sno
            # Lapse rate puts T_freeze (~273.16 K) near z = 2300 m (`T_sfc = 287 K`).
            @. Y_mlt.c.ρe_tot =
                Y_mlt.c.ρ * (
                    TD.internal_energy(
                        tps,
                        FT(287) - FT(0.006) * z_c,
                        Y_mlt.c.ρq_tot / Y_mlt.c.ρ,
                        FT(0),
                        Y_mlt.c.ρq_sno / Y_mlt.c.ρ,
                    ) + ᶜΦ
                )
            b0 = _column_integrals(Y_mlt, p_mlt)
            CA.step_coupled_microphysics_sedimentation!(Y_mlt, p_mlt, FT(300))
            b1 = _column_integrals(Y_mlt, p_mlt)
            qr_mlt = _arr(Y_mlt.c.ρq_rai)
            P_rain = _arr(p_mlt.precomputed.surface_rain_flux)
            P_snow = _arr(p_mlt.precomputed.surface_snow_flux)
            P_ener = _arr(p_mlt.precomputed.col_integrated_precip_energy_tendency)
            @test any(qr_mlt[1:10] .> FT(1e-7))
            @test all(P_rain .< zero(FT))
            _assert_column_conservation(b0, b1, FT(300), P_rain, P_snow, P_ener, 20)
        end
    end

    @testset "Stretched & 3D warped-terrain spherical conservation + zero allocations" begin
        # 1. Stretched 1D column
        c_str, f_str =
            _make_column_spaces(Float64; z_max = 12000.0, nz = 30, stretched = true)
        Y_str, p_str = _build_synthetic_1m_state(c_str, f_str)
        z_str = Fields.coordinate_field(c_str).z
        @. Y_str.c.ρq_rai =
            ifelse(z_str > 2000.0 && z_str < 5000.0, Y_str.c.ρ * 1.5e-3, 0.0)
        @. Y_str.c.ρq_sno =
            ifelse(z_str >= 5000.0 && z_str < 9000.0, Y_str.c.ρ * 1.0e-3, 0.0)
        @. Y_str.c.ρq_tot += Y_str.c.ρq_rai + Y_str.c.ρq_sno
        b0_s = _column_integrals(Y_str, p_str)
        CA.step_coupled_microphysics_sedimentation!(Y_str, p_str, 180.0)
        b1_s = _column_integrals(Y_str, p_str)
        Pr_s = _arr(p_str.precomputed.surface_rain_flux)
        Ps_s = _arr(p_str.precomputed.surface_snow_flux)
        Pe_s = _arr(p_str.precomputed.col_integrated_precip_energy_tendency)
        _assert_column_conservation(b0_s, b1_s, 180.0, Pr_s, Ps_s, Pe_s, 30)
        if _is_cpu_state(Y_str)
            @test _measure_sweep_allocs(Y_str, p_str, 180.0) == 0
        end

        # 2. 3D cubed-sphere extruded space with 2 km Gaussian mountain topography
        topo = get_spherical_extruded_spaces_with_topography(; FT = Float64)
        Y_3d, p_3d = _build_synthetic_1m_state(topo.cent_space, topo.face_space)
        z_3d = Fields.coordinate_field(topo.cent_space).z
        @. Y_3d.c.ρq_rai = ifelse(z_3d < 8000.0, Y_3d.c.ρ * 1.2e-3, 0.0)
        @. Y_3d.c.ρq_sno = ifelse(z_3d >= 8000.0 && z_3d < 15000.0, Y_3d.c.ρ * 8e-4, 0.0)
        @. Y_3d.c.ρq_tot += Y_3d.c.ρq_rai + Y_3d.c.ρq_sno
        b0_3d = _column_integrals(Y_3d, p_3d)
        CA.step_coupled_microphysics_sedimentation!(Y_3d, p_3d, 300.0)
        b1_3d = _column_integrals(Y_3d, p_3d)
        Pr_3d = _arr(p_3d.precomputed.surface_rain_flux)
        Ps_3d = _arr(p_3d.precomputed.surface_snow_flux)
        Pe_3d = _arr(p_3d.precomputed.col_integrated_precip_energy_tendency)
        nz_3d = Spaces.nlevels(topo.cent_space)
        _assert_column_conservation(b0_3d, b1_3d, 300.0, Pr_3d, Ps_3d, Pe_3d, nz_3d)
        if _is_cpu_state(Y_3d)
            @test _measure_sweep_allocs(Y_3d, p_3d, 300.0) == 0
        end
    end

    @testset "PrognosticEDMFX detrainment, entrainment, SGS, 0M, and Δt sweep" begin
        for FT in (Float32, Float64), with_sgs in (false, true)
            cspace, fspace = _make_column_spaces(FT; z_max = FT(4000), nz = 20)
            Y0, p0 = _build_synthetic_1m_state(
                cspace,
                fspace;
                with_edmfx = true,
                with_sgs,
            )
            tps = CAP.thermodynamics_params(p0.params)
            z_c = Fields.coordinate_field(cspace).z
            ᶜΦ = p0.core.ᶜΦ

            # 1. Narrowing updraft (`a¹ = 0.25` above 2000 m, `0.05` in 1000..2000 m,
            #    `0` below 1000 m so ∂_z a¹ > 0) with rain initially only inside updraft.
            a1_prof = @. ifelse(
                z_c > FT(2000),
                FT(0.25),
                ifelse(z_c > FT(1000), FT(0.05), FT(0)),
            )
            @. Y0.c.sgsʲs.:(1).ρa = a1_prof * Y0.c.ρ
            @. Y0.c.sgsʲs.:(1).q_rai = ifelse(z_c > FT(2000), FT(3e-3), FT(0))
            @. Y0.c.sgsʲs.:(1).q_lcl = FT(0)
            @. Y0.c.sgsʲs.:(1).q_icl = FT(0)
            @. Y0.c.sgsʲs.:(1).q_sno = FT(0)
            @. Y0.c.sgsʲs.:(1).q_tot = FT(0.005) + Y0.c.sgsʲs.:(1).q_rai
            @. Y0.c.ρq_rai = Y0.c.sgsʲs.:(1).ρa * Y0.c.sgsʲs.:(1).q_rai
            @. Y0.c.ρq_tot = Y0.c.ρ * FT(0.005) + Y0.c.ρq_rai
            ᶜT = p0.precomputed.ᶜT⁰
            @. Y0.c.sgsʲs.:(1).mse = TD.moist_static_energy(
                tps,
                ᶜT,
                ᶜΦ,
                Y0.c.sgsʲs.:(1).q_tot,
                Y0.c.sgsʲs.:(1).q_rai,
                FT(0),
            )
            @. Y0.c.ρe_tot =
                Y0.c.ρ * (
                    TD.internal_energy(
                        tps,
                        ᶜT,
                        Y0.c.ρq_tot / Y0.c.ρ,
                        Y0.c.ρq_rai / Y0.c.ρ,
                        FT(0),
                    ) + ᶜΦ
                )

            for dt in FT.((1, 60, 180, 600))
                Y = deepcopy(Y0)
                p = deepcopy(p0)
                b0 = _column_integrals(Y, p)
                CA.step_coupled_microphysics_sedimentation!(Y, p, dt)
                b1 = _column_integrals(Y, p)

                P_rain = _arr(p.precomputed.surface_rain_flux)
                P_snow = _arr(p.precomputed.surface_snow_flux)
                P_ener = _arr(p.precomputed.col_integrated_precip_energy_tendency)
                _assert_column_conservation(b0, b1, dt, P_rain, P_snow, P_ener, 20)

                # Updraft state invariants
                ρ_arr = _arr(Y.c.ρ)
                ρa1_arr = _arr(Y.c.sgsʲs.:(1).ρa)
                qt1_arr = _arr(Y.c.sgsʲs.:(1).q_tot)
                ql1_arr = _arr(Y.c.sgsʲs.:(1).q_lcl)
                qi1_arr = _arr(Y.c.sgsʲs.:(1).q_icl)
                qr1_arr = _arr(Y.c.sgsʲs.:(1).q_rai)
                qs1_arr = _arr(Y.c.sgsʲs.:(1).q_sno)
                mse1_arr = _arr(Y.c.sgsʲs.:(1).mse)
                @test all(zero(FT) .<= ρa1_arr .<= ρ_arr)
                @test all(>=(zero(FT)), qt1_arr)
                @test all(>=(zero(FT)), ql1_arr)
                @test all(>=(zero(FT)), qi1_arr)
                @test all(>=(zero(FT)), qr1_arr)
                @test all(>=(zero(FT)), qs1_arr)
                @test all(
                    qt1_arr .+ FT(16) * eps(FT) .>=
                    ql1_arr .+ qi1_arr .+ qr1_arr .+ qs1_arr,
                )
                @test all(isfinite, mse1_arr)

                if dt == FT(180)
                    # Below z = 1000 m (`k = 1..5`), `a¹ == 0`, so detrained updraft rain
                    # appears in the grid-mean/environment (`ρq_rai > 0`) and reaches surface.
                    qr_gs = _arr(Y.c.ρq_rai)
                    @test any(qr_gs[1:5] .> FT(1e-7))
                    @test all(P_rain .< zero(FT))
                    if with_sgs
                        pf_ed = _arr(p.precomputed.ᶜprecip_frac)
                        @test any(pf_ed .> zero(FT))
                        @test all(zero(FT) .<= pf_ed .<= one(FT))
                    end
                end
            end

            if _is_cpu_state(Y0)
                @test _measure_sweep_allocs(Y0, p0, FT(180)) == 0
            end
        end

        # 2. Widening updraft (`∂_z a¹ < 0`: `0` above 2500 m, `0.08` in 1200..2500 m,
        #    `0.25` below 1200 m) with rain falling from the environment aloft:
        #    lateral entrainment transfers environment precipitation into the updraft.
        cspace, fspace = _make_column_spaces(Float64; z_max = 4000.0, nz = 20)
        Yw, pw = _build_synthetic_1m_state(cspace, fspace; with_edmfx = true)
        tps = CAP.thermodynamics_params(pw.params)
        z_c = Fields.coordinate_field(cspace).z
        ᶜΦ = pw.core.ᶜΦ
        a1_wide = @. ifelse(
            z_c > 2500.0,
            0.0,
            ifelse(z_c > 1200.0, 0.08, 0.25),
        )
        @. Yw.c.sgsʲs.:(1).ρa = a1_wide * Yw.c.ρ
        @. Yw.c.sgsʲs.:(1).q_rai = 0.0
        @. Yw.c.sgsʲs.:(1).q_tot = 0.005
        @. Yw.c.ρq_rai = ifelse(z_c > 2500.0, Yw.c.ρ * 2.5e-3, 0.0)
        @. Yw.c.ρq_tot = Yw.c.ρ * 0.005 + Yw.c.ρq_rai
        ᶜT_w = pw.precomputed.ᶜT⁰
        @. Yw.c.sgsʲs.:(1).mse =
            TD.moist_static_energy(tps, ᶜT_w, ᶜΦ, 0.005, 0.0, 0.0)
        @. Yw.c.ρe_tot =
            Yw.c.ρ * (
                TD.internal_energy(
                    tps,
                    ᶜT_w,
                    Yw.c.ρq_tot / Yw.c.ρ,
                    Yw.c.ρq_rai / Yw.c.ρ,
                    0.0,
                ) + ᶜΦ
            )
        b0_w = _column_integrals(Yw, pw)
        CA.step_coupled_microphysics_sedimentation!(Yw, pw, 180.0)
        b1_w = _column_integrals(Yw, pw)
        @test any(_arr(Yw.c.sgsʲs.:(1).q_rai)[1:12] .> 1e-6)
        _assert_column_conservation(
            b0_w,
            b1_w,
            180.0,
            _arr(pw.precomputed.surface_rain_flux),
            _arr(pw.precomputed.surface_snow_flux),
            _arr(pw.precomputed.col_integrated_precip_energy_tendency),
            20,
        )

        # 3. Equilibrium 0M + PrognosticEDMFX
        Y0m, p0m = _build_synthetic_0m_edmfx_state(cspace, fspace)
        ρqt0_0m = copy(_arr(Y0m.c.ρq_tot))
        ρe0_0m = copy(_arr(Y0m.c.ρe_tot))
        b0_0m_ed = _column_integrals(Y0m, p0m)
        CA.step_coupled_microphysics_sedimentation!(Y0m, p0m, 120.0)
        b1_0m_ed = _column_integrals(Y0m, p0m)
        Pr_0m_ed = _arr(p0m.precomputed.surface_rain_flux)
        Ps_0m_ed = _arr(p0m.precomputed.surface_snow_flux)
        Pe_0m_ed = _arr(p0m.precomputed.col_integrated_precip_energy_tendency)
        @test all(Pr_0m_ed .< 0.0)
        @test _arr(p0m.precomputed.ᶜρ_dq_tot_dt) .* 120.0 ≈
              _arr(Y0m.c.ρq_tot) .- ρqt0_0m rtol = 1e-12 atol = 1e-14
        @test _arr(p0m.precomputed.ᶜρ_de_tot_dt) .* 120.0 ≈
              _arr(Y0m.c.ρe_tot) .- ρe0_0m rtol = 1e-12 atol = 1e-10
        @test all(0.0 .<= _arr(Y0m.c.sgsʲs.:(1).ρa) .<= _arr(Y0m.c.ρ))
        @test all(>=(0.0), _arr(Y0m.c.sgsʲs.:(1).q_tot))
        @test all(isfinite, _arr(Y0m.c.sgsʲs.:(1).mse))
        _assert_column_conservation(
            b0_0m_ed,
            b1_0m_ed,
            120.0,
            Pr_0m_ed,
            Ps_0m_ed,
            Pe_0m_ed,
            20,
        )
        if _is_cpu_state(Y0m)
            @test _measure_sweep_allocs(Y0m, p0m, 120.0) == 0
        end

        # 4. Float32 trace updraft (`ρa¹ = 1f-9 < eps(Float32)`): subsaturated
        #    dry trace draft preserves `mse¹` and `q_tot¹` without dilution.
        c32, f32 = _make_column_spaces(Float32; z_max = 4000.0f0, nz = 12)
        Yt1, pt1 = _build_synthetic_1m_state(c32, f32; with_edmfx = true)
        tps32 = CAP.thermodynamics_params(pt1.params)
        @. Yt1.c.ρq_lcl = 0.0f0
        @. Yt1.c.ρq_icl = 0.0f0
        @. Yt1.c.ρq_rai = 0.0f0
        @. Yt1.c.ρq_sno = 0.0f0
        @. Yt1.c.ρq_tot = Yt1.c.ρ * 0.002f0
        @. Yt1.c.sgsʲs.:(1).ρa = 1.0f-9
        @. Yt1.c.sgsʲs.:(1).q_tot = 0.002f0
        @. Yt1.c.sgsʲs.:(1).q_lcl = 0.0f0
        @. Yt1.c.sgsʲs.:(1).q_icl = 0.0f0
        @. Yt1.c.sgsʲs.:(1).q_rai = 0.0f0
        @. Yt1.c.sgsʲs.:(1).q_sno = 0.0f0
        @. Yt1.c.sgsʲs.:(1).mse = TD.moist_static_energy(
            tps32,
            pt1.precomputed.ᶜT⁰,
            pt1.core.ᶜΦ,
            0.002f0,
            0.0f0,
            0.0f0,
        )
        @. pt1.precomputed.ᶜTʲs.:(1) = pt1.precomputed.ᶜT⁰
        mse1_before = copy(_arr(Yt1.c.sgsʲs.:(1).mse))
        CA.step_coupled_microphysics_sedimentation!(Yt1, pt1, 120.0f0)
        @test isapprox(
            _arr(Yt1.c.sgsʲs.:(1).mse),
            mse1_before;
            rtol = 1.0f-5,
        )
        @test isapprox(
            _arr(Yt1.c.sgsʲs.:(1).q_tot),
            fill(0.002f0, size(mse1_before));
            rtol = 1.0f-5,
        )

        Yt2, pt2 = _build_synthetic_2m_state(c32, f32; with_edmfx = true)
        @. Yt2.c.ρq_lcl = 0.0f0
        @. Yt2.c.ρq_icl = 0.0f0
        @. Yt2.c.ρq_rai = 0.0f0
        @. Yt2.c.ρq_sno = 0.0f0
        @. Yt2.c.ρn_lcl = 0.0f0
        @. Yt2.c.ρn_rai = 0.0f0
        @. Yt2.c.ρq_tot = Yt2.c.ρ * 0.002f0
        @. Yt2.c.sgsʲs.:(1).ρa = 1.0f-9
        @. Yt2.c.sgsʲs.:(1).q_tot = 0.002f0
        @. Yt2.c.sgsʲs.:(1).q_lcl = 0.0f0
        @. Yt2.c.sgsʲs.:(1).q_icl = 0.0f0
        @. Yt2.c.sgsʲs.:(1).q_rai = 0.0f0
        @. Yt2.c.sgsʲs.:(1).q_sno = 0.0f0
        @. Yt2.c.sgsʲs.:(1).n_lcl = 0.0f0
        @. Yt2.c.sgsʲs.:(1).n_rai = 0.0f0
        @. Yt2.c.sgsʲs.:(1).mse = TD.moist_static_energy(
            tps32,
            pt2.precomputed.ᶜT⁰,
            pt2.core.ᶜΦ,
            0.002f0,
            0.0f0,
            0.0f0,
        )
        @. pt2.precomputed.ᶜTʲs.:(1) = pt2.precomputed.ᶜT⁰
        mse2_before = copy(_arr(Yt2.c.sgsʲs.:(1).mse))
        CA.step_coupled_microphysics_sedimentation!(
            Yt2,
            pt2,
            120.0f0;
            dt_sub_2m = 30.0f0,
        )
        @test isapprox(
            _arr(Yt2.c.sgsʲs.:(1).mse),
            mse2_before;
            rtol = 1.0f-5,
        )
    end

    @testset "Unified dispatch: 2M, 2MP3, 0M, and full AtmosSimulation + SGS quadrature" begin
        # 1. Subcycled donor-implicit 2M (grid-mean & PrognosticEDMFX)
        for with_edmfx in (false, true)
            cspace, fspace = _make_column_spaces(Float64; z_max = 6000.0, nz = 20)
            Y2_base, p2_base = _build_synthetic_2m_state(cspace, fspace; with_edmfx)
            for dt in (1.0, 60.0, 300.0, 600.0)
                Y2 = deepcopy(Y2_base)
                p2 = deepcopy(p2_base)
                b0 = _column_integrals(Y2, p2)
                CA.step_coupled_microphysics_sedimentation!(Y2, p2, dt; dt_sub_2m = 30.0)
                b1 = _column_integrals(Y2, p2)

                Pr = _arr(p2.precomputed.surface_rain_flux)
                Ps = _arr(p2.precomputed.surface_snow_flux)
                Pe = _arr(p2.precomputed.col_integrated_precip_energy_tendency)

                _assert_column_conservation(b0, b1, dt, Pr, Ps, Pe, 20)
                @test all(>=(0.0), _arr(Y2.c.ρn_lcl))
                @test all(>=(0.0), _arr(Y2.c.ρn_rai))
                @test any(>(0.0), _arr(p2.precomputed.ᶜwₙₗ))
                @test any(>(0.0), _arr(p2.precomputed.ᶜwₙᵣ))
                if with_edmfx
                    @test all(0.0 .<= _arr(Y2.c.sgsʲs.:(1).ρa) .<= _arr(Y2.c.ρ))
                    @test all(>=(0.0), _arr(Y2.c.sgsʲs.:(1).q_tot))
                    @test all(isfinite, _arr(Y2.c.sgsʲs.:(1).mse))
                end
            end
            if _is_cpu_state(Y2_base)
                @test _measure_sweep_allocs(Y2_base, p2_base, 120.0) == 0
            end
        end

        # 2MP3 is unsupported, with or without PrognosticEDMFX or SGS quadrature
        cspace_p3, fspace_p3 = _make_column_spaces(Float64; z_max = 6000.0, nz = 10)
        Yp3, pp3 = _build_synthetic_2m_state(cspace_p3, fspace_p3; with_p3 = true)
        edmfx_p3 = CA.PrognosticEDMFX(;
            n_updrafts = 1,
            prognostic_tke = false,
            area_fraction = 0.1,
        )
        for atmos_p3 in (
            pp3.atmos,
            (; pp3.atmos..., turbconv_model = edmfx_p3),
            (; pp3.atmos..., sgs_quadrature = CA.SGSQuadrature(Float64)),
        )
            Yp3_case = deepcopy(Yp3)
            ρq_icl_before = copy(_arr(Yp3_case.c.ρq_icl))
            @test_throws ArgumentError CA.step_coupled_microphysics_sedimentation!(
                Yp3_case,
                (; pp3..., atmos = atmos_p3),
                60.0,
            )
            @test _arr(Yp3_case.c.ρq_icl) == ρq_icl_before
        end

        # 2M rejects SGSQuadrature
        Y2_q, p2_q = _build_synthetic_2m_state(cspace_p3, fspace_p3)
        p2_q_sgs = (;
            p2_q...,
            atmos = (; p2_q.atmos..., sgs_quadrature = CA.SGSQuadrature(Float64)),
        )
        @test_throws ErrorException CA.step_coupled_microphysics_sedimentation!(
            Y2_q,
            p2_q_sgs,
            60.0,
        )

        # Verify 2M aerosol activation produces droplet number sources when `w > 0`
        for with_edmfx in (false, true)
            cspace_a, fspace_a = _make_column_spaces(Float64; z_max = 6000.0, nz = 20)
            Y_base_a, p_base_a = _build_synthetic_2m_state(
                cspace_a,
                fspace_a;
                with_p3 = false,
                with_edmfx,
            )
            fill!(Y_base_a.c.ρn_lcl, 0.0)
            z_a = Fields.coordinate_field(cspace_a).z
            ᶜT_a = @. 288.0 - 0.0065 * z_a
            tps_a = CAP.thermodynamics_params(p_base_a.params)
            ᶜq_sat_a = @. TD.q_vap_saturation(
                tps_a,
                ᶜT_a,
                Y_base_a.c.ρ,
                TD.Liquid(),
            )
            @. Y_base_a.c.ρq_tot =
                Y_base_a.c.ρ * (ᶜq_sat_a * 1.0005) +
                Y_base_a.c.ρq_lcl +
                Y_base_a.c.ρq_icl +
                Y_base_a.c.ρq_rai +
                Y_base_a.c.ρq_sno
            @. Y_base_a.c.ρe_tot =
                Y_base_a.c.ρ * (
                    TD.internal_energy(
                        tps_a,
                        ᶜT_a,
                        Y_base_a.c.ρq_tot / Y_base_a.c.ρ,
                        (Y_base_a.c.ρq_lcl + Y_base_a.c.ρq_rai) / Y_base_a.c.ρ,
                        (Y_base_a.c.ρq_icl + Y_base_a.c.ρq_sno) / Y_base_a.c.ρ,
                    ) + p_base_a.core.ᶜΦ
                )
            ᶜlg_a = Fields.local_geometry_field(cspace_a)
            ᶜu_up = broadcast(
                lg -> Geometry.Covariant123Vector(
                    Geometry.UVWVector(0.0, 0.0, 2.0),
                    lg,
                ),
                ᶜlg_a,
            )
            prescribed_aerosols_field = broadcast(
                _ -> (; SSLT01 = 1e-8, SO4 = 1e-8),
                zeros(cspace_a),
            )
            scratch_a = (;
                ᶜtemp_scalar = zeros(cspace_a),
                ᶜtemp_scalar_2 = zeros(cspace_a),
                ᶜtemp_scalar_3 = zeros(cspace_a),
            )
            if with_edmfx
                fill!(Y_base_a.c.sgsʲs.:(1).n_lcl, 0.0)
                @. Y_base_a.c.sgsʲs.:(1).q_tot = Y_base_a.c.ρq_tot / Y_base_a.c.ρ
                @. p_base_a.precomputed.ᶜuʲs.:(1) = ᶜu_up
                @. p_base_a.precomputed.ᶜu⁰ = ᶜu_up
            else
                @. p_base_a.precomputed.ᶜu = ᶜu_up
            end
            Y_no_aero = deepcopy(Y_base_a)
            p_no_aero = deepcopy(p_base_a)
            CA.step_coupled_microphysics_sedimentation!(
                Y_no_aero,
                p_no_aero,
                10.0;
                dt_sub_2m = 10.0,
            )
            Y_aero = deepcopy(Y_base_a)
            p_aero = (;
                deepcopy(p_base_a)...,
                tracers = (; prescribed_aerosols_field),
                scratch = scratch_a,
            )
            CA.step_coupled_microphysics_sedimentation!(
                Y_aero,
                p_aero,
                10.0;
                dt_sub_2m = 10.0,
            )
            @test any(_arr(Y_aero.c.ρn_lcl) .> _arr(Y_no_aero.c.ρn_lcl)) ||
                  maximum(_arr(Y_aero.c.ρn_lcl) .- _arr(Y_no_aero.c.ρn_lcl)) > 0.0
            if with_edmfx
                @test maximum(
                    _arr(Y_aero.c.sgsʲs.:(1).n_lcl) .-
                    _arr(Y_no_aero.c.sgsʲs.:(1).n_lcl),
                ) > 0.0
            end
        end

        # 2. Full `AtmosSimulation` integration: 0M and 1M + SGS quadrature + `ᶜprecip_frac`
        mktempdir() do tmp_out
            cfg_0m = CA.AtmosConfig(
                Dict(
                    "initial_condition" => "DYCOMS_RF02",
                    "microphysics_model" => "0M",
                    "config" => "column",
                    "z_stretch" => false,
                    "output_default_diagnostics" => false,
                    "output_dir" => joinpath(tmp_out, "sim_0m"),
                );
                job_id = "sweep_test_0m",
            )
            sim_0m = generate_test_simulation(cfg_0m)
            nz_0m = Spaces.nlevels(axes(sim_0m.Y.c.ρ))
            b0_0m = _column_integrals(sim_0m.Y, sim_0m.p)
            CA.step_coupled_microphysics_sedimentation!(sim_0m.Y, sim_0m.p, Float32(120))
            b1_0m = _column_integrals(sim_0m.Y, sim_0m.p)
            Pr_0m = _arr(sim_0m.p.precomputed.surface_rain_flux)
            Ps_0m = _arr(sim_0m.p.precomputed.surface_snow_flux)
            Pe_0m = _arr(sim_0m.p.precomputed.col_integrated_precip_energy_tendency)
            _assert_column_conservation(b0_0m, b1_0m, 120.0f0, Pr_0m, Ps_0m, Pe_0m, nz_0m)
            if _is_cpu_state(sim_0m.Y)
                @test _measure_sweep_allocs(sim_0m.Y, sim_0m.p, Float32(120)) == 0
            end

            cfg_1m_sgs = CA.AtmosConfig(
                Dict(
                    "initial_condition" => "PrecipitatingColumn",
                    "microphysics_model" => "1M",
                    "config" => "column",
                    "z_stretch" => false,
                    "use_sgs_quadrature" => true,
                    "sgs_distribution" => "gaussian",
                    "quadrature_order" => 2,
                    "output_default_diagnostics" => false,
                    "output_dir" => joinpath(tmp_out, "sim_1m_sgs"),
                );
                job_id = "sweep_test_1m_sgs",
            )
            sim_1m = generate_test_simulation(cfg_1m_sgs)
            nz_1m = Spaces.nlevels(axes(sim_1m.Y.c.ρ))
            # Seed top-half cloud fraction and precipitation so in-sweep `ᶜprecip_frac`
            # exercises maximum-random overlap decay on `q̄_sed`.
            @. sim_1m.p.precomputed.ᶜcloud_fraction = Float32(0.4)
            @. sim_1m.p.precomputed.ᶜprecip_frac = Float32(0)
            b0_1m = _column_integrals(sim_1m.Y, sim_1m.p)
            CA.step_coupled_microphysics_sedimentation!(sim_1m.Y, sim_1m.p, Float32(120))
            b1_1m = _column_integrals(sim_1m.Y, sim_1m.p)
            Pr_1m = _arr(sim_1m.p.precomputed.surface_rain_flux)
            Ps_1m = _arr(sim_1m.p.precomputed.surface_snow_flux)
            Pe_1m = _arr(sim_1m.p.precomputed.col_integrated_precip_energy_tendency)
            pf_after = _arr(sim_1m.p.precomputed.ᶜprecip_frac)
            @test any(pf_after .> 0.0f0)
            @test all(0.0f0 .<= pf_after .<= 1.0f0)
            _assert_column_conservation(b0_1m, b1_1m, 120.0f0, Pr_1m, Ps_1m, Pe_1m, nz_1m)
            if _is_cpu_state(sim_1m.Y)
                @test _measure_sweep_allocs(sim_1m.Y, sim_1m.p, Float32(120)) == 0
            end
        end
    end

    @testset "Equation parity with the tendency formulation" begin
        @testset "2M number adjustment enters once, through BMT" begin
            for FT in (Float32, Float64)
                params = CA.ClimaAtmosParameters(FT)
                tps = CAP.thermodynamics_params(params)
                cm2p = CAP.microphysics_2m_params(params)
                sb = cm2p.warm_rain.seifert_beheng
                numadj_lcl = (;
                    τ = sb.numadj.τ,
                    x_min = sb.pdf_c.xc_min,
                    x_max = sb.pdf_c.xc_max,
                )
                ρ = FT(1)
                T = FT(285)
                q_lcl = FT(1e-6)
                # Mean droplet mass 100 times below `x_min`, so the relaxation
                # toward the mass limits dominates the number tendency.
                n_lcl = FT(100) * q_lcl / FT(sb.pdf_c.xc_min)
                q_tot = TD.q_vap_saturation(tps, T, ρ, TD.Liquid()) + q_lcl
                bmt = CM.BulkMicrophysicsTendencies.bulk_microphysics_tendencies(
                    CM.BulkMicrophysicsTendencies.Microphysics2Moment(),
                    cm2p,
                    tps,
                    ρ,
                    T,
                    q_tot,
                    q_lcl,
                    n_lcl,
                    zero(FT),
                    zero(FT),
                )
                dn_adj = CM.Microphysics2M.number_tendency_from_mass_limits(
                    numadj_lcl,
                    q_lcl,
                    n_lcl,
                )
                @test dn_adj < zero(FT)
                @test isapprox(bmt.dn_lcl_dt, dn_adj; rtol = FT(1e-2))
                for dt in FT.((1, 30, 300))
                    n_new, n_out = CA._donor_implicit_number_step(
                        n_lcl,
                        zero(FT),
                        bmt.dn_lcl_dt,
                        zero(FT),
                        dt,
                    )
                    # One backward-Euler relaxation with the BMT rate.
                    n_once = n_lcl / (one(FT) - bmt.dn_lcl_dt / n_lcl * dt)
                    @test isapprox(n_new, n_once; rtol = FT(8) * eps(FT))
                    @test n_out == zero(FT)
                end
            end
        end

        @testset "1M source terms match BMT process rates" begin
            BMT = CM.BulkMicrophysicsTendencies
            same_rate_names = (
                (:S_vap_lcl, :S_phase_change_vap_lcl),
                (:S_vap_icl, :S_phase_change_vap_icl),
                (:S_acnv_lcl_rai, :S_acnv_lcl_rai),
                (:S_acnv_icl_sno, :S_acnv_icl_sno),
                (:S_accr_lcl_rai, :S_accr_lcl_rai),
                (:S_accr_lcl_sno_cold, :S_accr_lcl_sno_cold),
                (:S_accr_lcl_sno_warm, :S_accr_lcl_sno_warm),
                (:S_accr_melt_lcl_sno, :S_accr_melt_lcl_sno),
                (:S_accr_icl_sno, :S_accr_icl_sno),
                (:S_accr_rai_sno_cold, :S_accr_rai_sno_cold),
                (:S_accr_melt_rai_sno, :S_accr_melt_rai_sno),
                (:S_vap_rai, :S_phase_change_vap_rai),
                (:S_vap_sno, :S_phase_change_vap_sno),
                (:S_melt_icl_lcl, :S_melt_icl_lcl),
                (:S_melt_sno_rai, :S_melt_sno_rai),
                (:S_freeze_lcl_icl, :S_freeze_lcl_icl),
            )
            for FT in (Float32, Float64)
                params = CA.ClimaAtmosParameters(FT)
                tps = CAP.thermodynamics_params(params)
                mp = CA._combined_microphysics_1m_params(params)
                T_frz = FT(TD.Parameters.T_freeze(tps))
                rtol = FT === Float64 ? FT(1e-10) : FT(1e-4)
                atol = FT === Float64 ? FT(1e-20) : FT(1e-12)
                ρ = FT(0.9)
                w = FT(0.5)
                species_cases = (
                    FT.((2e-4, 1e-4, 5e-4, 3e-4)),
                    FT.((2e-4, 0, 0, 0)),
                    FT.((0, 1e-4, 5e-4, 0)),
                    FT.((0, 0, 5e-4, 3e-4)),
                    FT.((2e-4, 0, 0, 3e-4)),
                )
                n_warm = 0
                for ΔT in FT.((-25, -5, -0.5, 0, 3)),
                    RH in FT.((0.7, 1.03)),
                    (q_lcl, q_icl, q_rai, q_sno) in species_cases

                    T = T_frz + ΔT
                    q_v = RH * TD.q_vap_saturation(tps, T, ρ, TD.Liquid())
                    q_tot = q_v + q_lcl + q_icl + q_rai + q_sno
                    r = CA.coupled_1m_source_terms(
                        mp, tps, ρ, T, w, q_v, q_lcl, q_icl, q_rai, q_sno,
                    )
                    b = BMT.bulk_microphysics_tendencies(
                        BMT.InstantaneousVerbose(),
                        BMT.Microphysics1Moment(),
                        mp, tps, ρ, T, w, q_tot, q_lcl, q_icl, q_rai, q_sno,
                    )
                    for (r_name, b_name) in same_rate_names
                        @test isapprox(
                            getproperty(r, r_name),
                            getproperty(b, b_name);
                            rtol,
                            atol,
                        )
                    end
                    # Warm-air gate: CloudMicrophysics routes every cloud ice +
                    # rain collision to snow and freezes rain; at
                    # `T ≥ T_freeze` the sweep sends the collected ice to rain
                    # and freezes none.
                    S_icl_rai = r.S_accr_icl_rai_cold + r.S_accr_icl_rai_warm
                    @test isapprox(S_icl_rai, b.S_accr_icl_rai; rtol, atol)
                    if T >= T_frz
                        n_warm += 1
                        @test r.S_accr_icl_rai_cold == zero(FT)
                        @test r.S_accr_freeze_icl_rai == zero(FT)
                    else
                        @test r.S_accr_icl_rai_warm == zero(FT)
                        @test isapprox(
                            r.S_accr_freeze_icl_rai,
                            b.S_accr_freeze_icl_rai;
                            rtol,
                            atol,
                        )
                    end
                end
                @test n_warm > 0
            end
        end

        @testset "Latent-heating limiter ΔT is the exact phase-change ΔT" begin
            FT = Float64
            params = CA.ClimaAtmosParameters(FT)
            tps = CAP.thermodynamics_params(params)
            P = TD.Parameters
            T0 = FT(268)
            q_tot = FT(0.012)
            q_liq = FT(2e-3)
            q_ice = FT(1e-3)
            e_v = TD.internal_energy_vapor(tps, T0)
            e_l = TD.internal_energy_liquid(tps, T0)
            e_i = TD.internal_energy_ice(tps, T0)
            # Condensation, deposition, freezing/riming, melting, and mixtures.
            for (Δl, Δi) in (
                (1e-3, 0.0),
                (0.0, 8e-4),
                (-6e-4, 6e-4),
                (5e-4, -5e-4),
                (-1e-3, 0.0),
                (7e-4, 4e-4),
            )
                # Fixed e_int and ρ (grid-mean cell step).
                e_int = TD.internal_energy(tps, T0, q_tot, q_liq, q_ice)
                T_v = TD.air_temperature(tps, e_int, q_tot, q_liq + Δl, q_ice + Δi)
                ΔT_v = CA._phase_change_ΔT(
                    e_v - e_l,
                    e_v - e_i,
                    TD.cv_m(tps, q_tot, q_liq, q_ice),
                    FT(P.cv_l(tps) - P.cv_v(tps)),
                    FT(P.cv_i(tps) - P.cv_v(tps)),
                    Δl,
                    Δi,
                )
                @test isapprox(ΔT_v, T_v - T0; rtol = 1e-10, atol = 1e-12)
                # Fixed h and p (EDMF subdomain steps).
                h = TD.enthalpy(e_int, TD.gas_constant_air(tps, q_tot, q_liq, q_ice), T0)
                T_p = TD.air_temperature(
                    tps,
                    TD.ph(),
                    h,
                    q_tot,
                    q_liq + Δl,
                    q_ice + Δi,
                )
                ΔT_p = CA._phase_change_ΔT(
                    TD.latent_heat_vapor(tps, T0),
                    TD.latent_heat_sublim(tps, T0),
                    TD.cp_m(tps, q_tot, q_liq, q_ice),
                    FT(P.cp_l(tps) - P.cp_v(tps)),
                    FT(P.cp_i(tps) - P.cp_v(tps)),
                    Δl,
                    Δi,
                )
                @test isapprox(ΔT_p, T_p - T0; rtol = 1e-10, atol = 1e-12)
                # Freezing and riming warm, melting cools.
                if Δl == -Δi
                    @test sign(ΔT_v) == sign(Δi)
                    @test sign(ΔT_p) == sign(Δi)
                end
            end
        end

        @testset "Sweep environment from specific_env_value" begin
            FT = Float64
            ρ = FT(1.1)
            ρq = FT(1.1e-3)
            for a_half in FT.((1e-5, 0.1)), a1 in FT.((0.1, 0.8))
                tc = CA.PrognosticEDMFX(;
                    n_updrafts = 1,
                    prognostic_tke = false,
                    area_fraction = a_half,
                )
                ρa1 = a1 * ρ
                ρa0 = ρ - ρa1
                for q1 in FT.((0, 5e-4, 1e-3))
                    sgs = ((; ρa = ρa1, q_rai = q1),)
                    χ⁰ = CA.specific_env_value(@name(q_rai), ρq, ρ, sgs, ρa0, tc)
                    χ⁰_sweep = CA._sweep_env_value(
                        @name(q_rai), ρq, ρ, sgs, ρa0, tc, ρq, ρa1, q1, ρa0,
                    )
                    # Realizable state outside the `specific` blend: the sweep
                    # sees the transport environment.
                    if a_half < 1e-3
                        @test χ⁰_sweep == max(zero(FT), χ⁰)
                    else
                        @test χ⁰_sweep <= max(zero(FT), χ⁰)
                    end
                end
                # Updraft pool larger than the grid mean, or negative updraft
                # tracer: the environment is bounded by the residual of the
                # non-negative updraft pool.
                for (q1_raw, q1_pool) in ((FT(2e-3), ρq / ρa1), (FT(-1e-3), FT(0)))
                    sgs = ((; ρa = ρa1, q_rai = q1_raw),)
                    χ⁰_sweep = CA._sweep_env_value(
                        @name(q_rai), ρq, ρ, sgs, ρa0, tc, ρq, ρa1, q1_pool, ρa0,
                    )
                    @test χ⁰_sweep >= zero(FT)
                    @test ρa0 * χ⁰_sweep + ρa1 * q1_pool <= ρq * (1 + 4 * eps(FT))
                end
            end
            # The `specific` blend applies at small environment area.
            tc_blend = CA.PrognosticEDMFX(;
                n_updrafts = 1,
                prognostic_tke = false,
                area_fraction = FT(0.1),
            )
            ρa1 = FT(0.85) * ρ
            q1 = FT(2e-4)
            sgs = ((; ρa = ρa1, q_rai = q1),)
            χ⁰_blend = CA._sweep_env_value(
                @name(q_rai), ρq, ρ, sgs, ρ - ρa1, tc_blend, ρq, ρa1, q1, ρ - ρa1,
            )
            @test χ⁰_blend < (ρq - ρa1 * q1) / (ρ - ρa1)

            # A column sweep in the blend regime conserves mass, water, and
            # energy.
            cspace, fspace = _make_column_spaces(FT; z_max = FT(4000), nz = 16)
            Yb, pb = _build_synthetic_1m_state(cspace, fspace; with_edmfx = true)
            tps = CAP.thermodynamics_params(pb.params)
            z_c = Fields.coordinate_field(cspace).z
            @. Yb.c.sgsʲs.:(1).ρa = FT(0.85) * Yb.c.ρ
            @. Yb.c.sgsʲs.:(1).q_rai = ifelse(z_c > FT(2000), FT(2e-3), FT(0))
            @. Yb.c.sgsʲs.:(1).q_tot = FT(0.004) + Yb.c.sgsʲs.:(1).q_rai
            @. Yb.c.ρq_rai = Yb.c.ρ * ifelse(z_c > FT(1000), FT(1.5e-3), FT(0))
            @. Yb.c.ρq_tot = Yb.c.ρ * FT(0.004) + Yb.c.ρq_rai
            @. Yb.c.sgsʲs.:(1).mse = TD.moist_static_energy(
                tps,
                pb.precomputed.ᶜT⁰,
                pb.core.ᶜΦ,
                Yb.c.sgsʲs.:(1).q_tot,
                Yb.c.sgsʲs.:(1).q_rai,
                FT(0),
            )
            b0 = _column_integrals(Yb, pb)
            CA.step_coupled_microphysics_sedimentation!(Yb, pb, FT(120))
            b1 = _column_integrals(Yb, pb)
            _assert_column_conservation(
                b0,
                b1,
                FT(120),
                _arr(pb.precomputed.surface_rain_flux),
                _arr(pb.precomputed.surface_snow_flux),
                _arr(pb.precomputed.col_integrated_precip_energy_tendency),
                16,
            )
            @test all(>=(zero(FT)), _arr(Yb.c.ρq_rai))
        end

        # Updraft column that widens downward at 2500 m below environment rain
        # (1M) or cloud ice (2M); the environment carries horizontal wind `U`
        # and the matching kinetic energy.
        function _widening_updraft_state(nm, U; a1_low = 0.08)
            FT = Float64
            cspace, fspace = _make_column_spaces(FT; z_max = FT(4000), nz = 20)
            Y, p =
                nm == :one_moment ?
                _build_synthetic_1m_state(cspace, fspace; with_edmfx = true) :
                _build_synthetic_2m_state(cspace, fspace; with_edmfx = true)
            tps = CAP.thermodynamics_params(p.params)
            z_c = Fields.coordinate_field(cspace).z
            ᶜΦ = p.core.ᶜΦ
            ᶜT = copy(p.precomputed.ᶜT⁰)
            sgs = Y.c.sgsʲs.:(1)
            @. sgs.ρa = ifelse(z_c > FT(2500), FT(0), FT(a1_low) * Y.c.ρ)
            @. sgs.q_lcl = FT(0)
            @. sgs.q_icl = FT(0)
            @. sgs.q_rai = FT(0)
            @. sgs.q_sno = FT(0)
            @. sgs.q_tot = FT(0.004)
            @. Y.c.ρq_lcl = FT(0)
            @. Y.c.ρq_sno = FT(0)
            if nm == :one_moment
                @. Y.c.ρq_icl = FT(0)
                @. Y.c.ρq_rai = ifelse(z_c > FT(2500), Y.c.ρ * FT(2.5e-3), FT(0))
            else
                @. Y.c.ρq_rai = FT(0)
                @. Y.c.ρn_lcl = FT(0)
                @. Y.c.ρn_rai = FT(0)
                @. sgs.n_lcl = FT(0)
                @. sgs.n_rai = FT(0)
                @. Y.c.ρq_icl = ifelse(z_c > FT(2500), Y.c.ρ * FT(1e-3), FT(0))
            end
            @. Y.c.ρq_tot = Y.c.ρ * FT(0.004) + Y.c.ρq_rai + Y.c.ρq_icl
            @. sgs.mse = TD.moist_static_energy(tps, ᶜT, ᶜΦ, FT(0.004), FT(0), FT(0))
            @. p.precomputed.ᶜTʲs.:(1) = ᶜT
            ᶜlg = Fields.local_geometry_field(cspace)
            u_env = Geometry.UVWVector(FT(U), zero(FT), zero(FT))
            p.precomputed.ᶜu⁰ .=
                broadcast(lg -> Geometry.Covariant123Vector(u_env, lg), ᶜlg)
            @. Y.c.ρe_tot =
                Y.c.ρ * (
                    TD.internal_energy(
                        tps,
                        ᶜT,
                        Y.c.ρq_tot / Y.c.ρ,
                        Y.c.ρq_rai / Y.c.ρ,
                        Y.c.ρq_icl / Y.c.ρ,
                    ) + ᶜΦ
                ) + (Y.c.ρ - sgs.ρa) * FT(U)^2 / 2
            k_entr = findlast(z -> z <= 2500, _arr(z_c))
            return Y, p, k_entr
        end

        @testset "Updraft sedimentation exchange ($nm)" for nm in
                                                            (:one_moment, :two_moment)
            run!(Y, p) =
                nm == :one_moment ?
                CA.step_coupled_microphysics_sedimentation!(Y, p, 120.0) :
                CA.step_coupled_microphysics_sedimentation!(
                    Y,
                    p,
                    120.0;
                    dt_sub_2m = 120.0,
                )
            sed_name = nm == :one_moment ? :q_rai : :q_icl
            # Entrainment of environment condensate for any updraft area,
            # including areas below `min_area`.
            a_min = CAP.min_area(CAP.turbconv_params(CA.ClimaAtmosParameters(Float64)))
            for a1_low in (0.08, 0.2 * a_min)
                Y, p, k_entr = _widening_updraft_state(nm, 0.0; a1_low)
                b0 = _column_integrals(Y, p)
                run!(Y, p)
                b1 = _column_integrals(Y, p)
                q_sed1 = _arr(getproperty(Y.c.sgsʲs.:(1), sed_name))
                @test q_sed1[k_entr] > 0
                _assert_column_conservation(
                    b0,
                    b1,
                    120.0,
                    _arr(p.precomputed.surface_rain_flux),
                    _arr(p.precomputed.surface_snow_flux),
                    _arr(p.precomputed.col_integrated_precip_energy_tendency),
                    20,
                )
            end
            # The entrained condensate brings `h + Φ` into the updraft `mse`;
            # its kinetic energy, here raised by environment wind, stays in the
            # grid-mean energy.
            Y_calm, p_calm, k_entr = _widening_updraft_state(nm, 0.0)
            Y_wind, p_wind, _ = _widening_updraft_state(nm, 30.0)
            run!(Y_calm, p_calm)
            b0_wind = _column_integrals(Y_wind, p_wind)
            run!(Y_wind, p_wind)
            b1_wind = _column_integrals(Y_wind, p_wind)
            mse_calm = _arr(Y_calm.c.sgsʲs.:(1).mse)[k_entr]
            mse_wind = _arr(Y_wind.c.sgsʲs.:(1).mse)[k_entr]
            @test _arr(getproperty(Y_wind.c.sgsʲs.:(1), sed_name))[k_entr] > 0
            @test isapprox(mse_wind, mse_calm; rtol = 1e-13)
            _assert_column_conservation(
                b0_wind,
                b1_wind,
                120.0,
                _arr(p_wind.precomputed.surface_rain_flux),
                _arr(p_wind.precomputed.surface_snow_flux),
                _arr(p_wind.precomputed.col_integrated_precip_energy_tendency),
                20,
            )
        end

        @testset "0M EDMFX rain/snow split by subdomain temperature" begin
            cspace, fspace = _make_column_spaces(Float64; z_max = 4000.0, nz = 16)
            for (T_env, T_up) in ((290.0, 255.0), (255.0, 290.0))
                Y, p = _build_synthetic_0m_edmfx_state(cspace, fspace)
                tps = CAP.thermodynamics_params(p.params)
                # Only the updraft holds condensate above the removal threshold.
                @. p.precomputed.ᶜT⁰ = T_env
                @. p.precomputed.ᶜTʲs.:(1) = T_up
                @. p.precomputed.ᶜq_liq⁰ = 0.0
                @. p.precomputed.ᶜq_ice⁰ = 0.0
                @. p.precomputed.ᶜq_liqʲs.:(1) = ifelse(T_up > 273.15, 0.006, 0.0)
                @. p.precomputed.ᶜq_iceʲs.:(1) = ifelse(T_up > 273.15, 0.0, 0.006)
                b0 = _column_integrals(Y, p)
                CA.step_coupled_microphysics_sedimentation!(Y, p, 120.0)
                b1 = _column_integrals(Y, p)
                P_rain = _arr(p.precomputed.surface_rain_flux)
                P_snow = _arr(p.precomputed.surface_snow_flux)
                P_ener = _arr(p.precomputed.col_integrated_precip_energy_tendency)
                if T_up > 273.15
                    @test all(P_rain .< 0.0) && all(P_snow .== 0.0)
                else
                    @test all(P_snow .< 0.0) && all(P_rain .== 0.0)
                end
                _assert_column_conservation(b0, b1, 120.0, P_rain, P_snow, P_ener, 16)
            end
        end

        @testset "0M EDMFX updraft precipitation timescale ratio" begin
            cspace, fspace = _make_column_spaces(Float64; z_max = 4000.0, nz = 16)
            ratio_params(r) = CA.ClimaAtmosParameters(
                CP.create_toml_dict(
                    Float64;
                    override_file = Dict(
                        "precipitation_timescale_updraft_ratio" =>
                            Dict("value" => r, "type" => "float"),
                    ),
                ),
            )
            # Condensate in one subdomain at a time, above the threshold.
            function run_ratio(r, updraft::Bool)
                Y, p = _build_synthetic_0m_edmfx_state(
                    cspace, fspace; params = ratio_params(r),
                )
                q_up, q_env = updraft ? (0.006, 0.0) : (0.0, 0.006)
                @. p.precomputed.ᶜq_liqʲs.:(1) = q_up
                @. p.precomputed.ᶜq_liq⁰ = q_env
                b0 = _column_integrals(Y, p)
                CA.step_coupled_microphysics_sedimentation!(Y, p, 120.0)
                b1 = _column_integrals(Y, p)
                P_rain = _arr(p.precomputed.surface_rain_flux)
                P_snow = _arr(p.precomputed.surface_snow_flux)
                P_ener = _arr(p.precomputed.col_integrated_precip_energy_tendency)
                _assert_column_conservation(b0, b1, 120.0, P_rain, P_snow, P_ener, 16)
                P_up = _arr(p.precomputed.col_integrated_updraft_precip)
                return (; P = P_rain .+ P_snow, P_up)
            end
            up1, up_half = run_ratio(1.0, true), run_ratio(0.5, true)
            env1, env_half = run_ratio(1.0, false), run_ratio(0.5, false)
            # Updraft condensate precipitates twice as fast with half the
            # timescale; all of it is updraft precipitation.
            @test all(up1.P .< 0)
            @test up_half.P ≈ 2 .* up1.P rtol = 1e-12
            @test up1.P_up ≈ up1.P rtol = 1e-12
            @test up_half.P_up ≈ up_half.P rtol = 1e-12
            # The environment keeps the grid-mean timescale.
            @test all(env1.P .< 0)
            @test env_half.P == env1.P
            @test all(iszero, env1.P_up)
        end
    end

    @testset "Passive draft tracer mass conservation across EDMFX sweeps" begin
        function _with_passive_draft_tracer(Y)
            FT = Spaces.undertype(axes(Y.c))
            cspace = axes(Y.c)
            z_c = Fields.coordinate_field(cspace).z
            q_init = @. FT(1e-3) * (FT(1) + FT(0.5) * sin(z_c / FT(1500)))
            ρq_init = @. Y.c.ρ * q_init
            sgs1_new = broadcast(
                (s, q) -> merge(s, (; q_gas_A = q)),
                Y.c.sgsʲs.:(1),
                q_init,
            )
            Yc_new = broadcast(
                (c, s1, ρq) -> merge(c, (; sgsʲs = (s1,), ρq_gas_A = ρq)),
                Y.c,
                sgs1_new,
                ρq_init,
            )
            return Fields.FieldVector(; c = Yc_new, f = deepcopy(Y.f))
        end

        cspace, fspace = _make_column_spaces(Float64; z_max = 6000.0, nz = 20)

        # 0M EDMFX
        Y0_base, p0 = _build_synthetic_0m_edmfx_state(cspace, fspace)
        @. p0.precomputed.ᶜq_liqʲs.:(1) = 0.006
        Y0 = _with_passive_draft_tracer(Y0_base)

        # 1M EDMFX (seed nonzero rain/cloud in updraft so ρa1 changes)
        Y1_base, p1 = _build_synthetic_1m_state(
            cspace,
            fspace;
            with_edmfx = true,
            fixed_w_rai = -5.0,
            fixed_w_sno = -2.0,
        )
        z1 = Fields.coordinate_field(cspace).z
        @. Y1_base.c.sgsʲs.:(1).ρa = 0.15 * Y1_base.c.ρ
        @. Y1_base.c.sgsʲs.:(1).q_rai = ifelse(z1 > 1500.0, 2e-3, 0.0)
        @. Y1_base.c.sgsʲs.:(1).q_tot = 0.006 + Y1_base.c.sgsʲs.:(1).q_rai
        @. Y1_base.c.sgsʲs.:(1).mse = TD.moist_static_energy(
            CAP.thermodynamics_params(p1.params),
            p1.precomputed.ᶜT⁰,
            p1.core.ᶜΦ,
            Y1_base.c.sgsʲs.:(1).q_tot,
            Y1_base.c.sgsʲs.:(1).q_rai,
            0.0,
        )
        @. Y1_base.c.ρq_rai = Y1_base.c.ρ * 1e-3
        @. Y1_base.c.ρq_tot += Y1_base.c.ρq_rai
        Y1 = _with_passive_draft_tracer(Y1_base)

        # 2M EDMFX
        Y2_base, p2 = _build_synthetic_2m_state(cspace, fspace; with_edmfx = true)
        Y2 = _with_passive_draft_tracer(Y2_base)

        for (label, Y_init, p_case, dt, dt_sub_2m) in (
            ("0M", Y0, p0, 120.0, nothing),
            ("1M", Y1, p1, 120.0, nothing),
            ("2M", Y2, p2, 120.0, 30.0),
        )
            @testset "$label" begin
                Y_step = deepcopy(Y_init)
                p_step = deepcopy(p_case)
                ρa_before = _arr(Y_step.c.sgsʲs.:(1).ρa)
                q_before = _arr(Y_step.c.sgsʲs.:(1).q_gas_A)
                ρq_gs_before = _arr(Y_step.c.ρq_gas_A)

                if isnothing(dt_sub_2m)
                    CA.step_coupled_microphysics_sedimentation!(
                        Y_step,
                        p_step,
                        dt,
                    )
                else
                    CA.step_coupled_microphysics_sedimentation!(
                        Y_step,
                        p_step,
                        dt;
                        dt_sub_2m,
                    )
                end

                ρa_after = _arr(Y_step.c.sgsʲs.:(1).ρa)
                q_after = _arr(Y_step.c.sgsʲs.:(1).q_gas_A)
                ρq_gs_after = _arr(Y_step.c.ρq_gas_A)

                @test any(abs.(ρa_after .- ρa_before) .> 1e-12)
                @test ρa_after .* q_after ≈ ρa_before .* q_before rtol = 1e-12
                @test ρq_gs_after == ρq_gs_before
                if isnothing(dt_sub_2m) && _is_cpu_state(Y_init)
                    @test _measure_sweep_allocs(Y_init, p_case, dt) == 0
                end
            end
        end
    end

    if ClimaComms.device() isa ClimaComms.CUDADevice
        @testset "CUDA device kernel compilation & conservation smoke test" begin
            gpu_dev = ClimaComms.CUDADevice()
            for FT in (Float32, Float64), with_edmfx in (false, true)
                c_gpu, f_gpu = _make_column_spaces(
                    FT;
                    z_max = FT(4000),
                    nz = 16,
                    device = gpu_dev,
                )
                Yg, pg = _build_synthetic_1m_state(
                    c_gpu,
                    f_gpu;
                    with_edmfx,
                    with_sgs = true,
                )
                zg = Fields.coordinate_field(c_gpu).z
                @. Yg.c.ρq_rai =
                    ifelse(zg > FT(1500), Yg.c.ρ * FT(2e-3), FT(0))
                @. Yg.c.ρq_sno =
                    ifelse(zg > FT(2500), Yg.c.ρ * FT(1e-3), FT(0))
                @. Yg.c.ρq_tot += Yg.c.ρq_rai + Yg.c.ρq_sno
                if with_edmfx
                    @. Yg.c.sgsʲs.:(1).ρa = FT(0.15) * Yg.c.ρ
                    @. Yg.c.sgsʲs.:(1).q_rai = FT(1.5e-3)
                    @. Yg.c.sgsʲs.:(1).q_tot = FT(0.006)
                    @. Yg.c.sgsʲs.:(1).mse = TD.moist_static_energy(
                        CAP.thermodynamics_params(pg.params),
                        pg.precomputed.ᶜT⁰,
                        pg.core.ᶜΦ,
                        FT(0.006),
                        FT(1.5e-3),
                        FT(0),
                    )
                end
                b0_g = _column_integrals(Yg, pg)
                CA.step_coupled_microphysics_sedimentation!(Yg, pg, FT(180))
                b1_g = _column_integrals(Yg, pg)
                Pr_g = _arr(pg.precomputed.surface_rain_flux)
                Ps_g = _arr(pg.precomputed.surface_snow_flux)
                Pe_g =
                    _arr(pg.precomputed.col_integrated_precip_energy_tendency)
                _assert_column_conservation(
                    b0_g,
                    b1_g,
                    FT(180),
                    Pr_g,
                    Ps_g,
                    Pe_g,
                    16,
                )
            end

            for with_edmfx in (false, true)
                c_gpu, f_gpu = _make_column_spaces(
                    Float32;
                    z_max = 6000.0f0,
                    nz = 12,
                    device = gpu_dev,
                )
                Y2g, p2g = _build_synthetic_2m_state(c_gpu, f_gpu; with_edmfx)
                b0_2g = _column_integrals(Y2g, p2g)
                CA.step_coupled_microphysics_sedimentation!(
                    Y2g,
                    p2g,
                    120.0f0;
                    dt_sub_2m = 30.0f0,
                )
                b1_2g = _column_integrals(Y2g, p2g)
                _assert_column_conservation(
                    b0_2g,
                    b1_2g,
                    120.0f0,
                    _arr(p2g.precomputed.surface_rain_flux),
                    _arr(p2g.precomputed.surface_snow_flux),
                    _arr(p2g.precomputed.col_integrated_precip_energy_tendency),
                    12,
                )
            end
        end
    end
end
