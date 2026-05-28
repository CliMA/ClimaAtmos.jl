#####
##### ML-based tendency correction trained offline against ERA5.
#####
##### The correction is loaded from a BSON file produced by the offline
##### training pipeline (Lux CNN/UNet). A periodic callback evaluates the
##### model and caches tendency corrections; `ml_correction_apply_tendency!`
##### adds them to `Yₜ` every timestep.
#####
##### Heavy data (Lux parameters, state, normalization statistics) is stored
##### in `p.ml_correction` — NOT in `AtmosModel` — so that `AtmosModel`
##### remains `isbitstype`-safe for GPU kernels. `MLCorrectionDirect` and
##### `MLCorrectionFlux` are empty marker structs used only for dispatch.
#####

import ClimaCore.Fields as Fields
import ClimaCore.Spaces as Spaces
import Thermodynamics as TD
import Lux
import BSON

# ─────────────────────────────────────────────────────────────────────────────
# Module-level staging area
# ─────────────────────────────────────────────────────────────────────────────
#
# `get_ml_correction_model` (model_getters.jl) loads the BSON and deposits the
# heavy data here via `set_ml_correction_data!`. `ml_correction_cache` picks it
# up during cache construction and clears the Ref so no stale references linger.

const _ML_CORRECTION_PENDING_DATA = Ref{Any}(nothing)

"""
    set_ml_correction_data!(data)

Stage heavy ML correction data (NamedTuple) so that `ml_correction_cache`
can retrieve it during cache construction.
"""
set_ml_correction_data!(data) = (_ML_CORRECTION_PENDING_DATA[] = data)

# ─────────────────────────────────────────────────────────────────────────────
# Cache allocation
# ─────────────────────────────────────────────────────────────────────────────

ml_correction_cache(Y, atmos::AtmosModel) =
    ml_correction_cache(Y, atmos.ml_correction_model)

ml_correction_cache(Y, ::Nothing) = (;)

function ml_correction_cache(Y, ::MLCorrectionModel)
    data = _ML_CORRECTION_PENDING_DATA[]
    _ML_CORRECTION_PENDING_DATA[] = nothing
    isnothing(data) && error(
        "ML correction model enabled but no data was staged. " *
        "Did get_ml_correction_model run before cache construction?",
    )

    FT = Spaces.undertype(axes(Y.c))
    ᶜρe_tot_correction = similar(Y.c, FT)
    ᶜρq_tot_correction = similar(Y.c, FT)
    ᶜρe_tot_correction .= FT(0)
    ᶜρq_tot_correction .= FT(0)

    return (;
        ᶜρe_tot_correction,
        ᶜρq_tot_correction,
        ps = data.ps,
        st = data.st,
        x_norm_mean = data.x_norm_mean,
        x_norm_std = data.x_norm_std,
        y_norm_mean = data.y_norm_mean,
        y_norm_std = data.y_norm_std,
        nz = data.nz,
        n_features = data.n_features,
        target_vars = data.target_vars,
        arch = data.arch,
        relaxation = data.relaxation,
        z_max = data.z_max,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Tendency application (every timestep)
# ─────────────────────────────────────────────────────────────────────────────

ml_correction_apply_tendency!(Yₜ, Y, p, t, ::Nothing) = nothing

function ml_correction_apply_tendency!(Yₜ, Y, p, t, ::MLCorrectionModel)
    (; ᶜρe_tot_correction, ᶜρq_tot_correction) = p.ml_correction
    @. Yₜ.c.ρe_tot += ᶜρe_tot_correction
    if hasproperty(Yₜ.c, :ρq_tot)
        @. Yₜ.c.ρq_tot += ᶜρq_tot_correction
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Callback: evaluate the ML model and update cached corrections
# ─────────────────────────────────────────────────────────────────────────────

"""
    ml_correction_compute_tendency!(Y, p, ::MLCorrectionDirect)

Evaluate the ML tendency correction model on the current atmospheric state
and store the resulting corrections in `p.ml_correction`.
"""
function ml_correction_compute_tendency!(Y, p, ::MLCorrectionDirect)
    FT = Spaces.undertype(axes(Y.c))
    thermo_params = CAP.thermodynamics_params(p.params)
    mc = p.ml_correction

    (; ᶜρe_tot_correction, ᶜρq_tot_correction) = mc
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed

    nz_model = mc.nz
    nz_sim = Spaces.nlevels(axes(Y.c))

    T_arr = Array(Fields.field2array(ᶜT))
    q_arr = Array(Fields.field2array(ᶜq_tot_nonneg))
    ρ_arr = Array(Fields.field2array(Y.c.ρ))
    q_liq_arr = Array(Fields.field2array(ᶜq_liq))
    q_ice_arr = Array(Fields.field2array(ᶜq_ice))
    p_arr = Array(Fields.field2array(p.precomputed.ᶜp))

    has_tke = hasproperty(Y.c, :ρtke)
    if has_tke
        tke_arr = Array(Fields.field2array(@. specific(Y.c.ρtke, Y.c.ρ)))
    end

    ncols = size(T_arr, 1)
    nz_use = min(nz_model, nz_sim)

    n_feat = mc.n_features
    input_dim = n_feat * nz_model
    nv = mc.target_vars == :both ? 2 : 1
    output_dim = nv * nz_model

    x_mean = mc.x_norm_mean
    x_std = mc.x_norm_std
    y_mean = mc.y_norm_mean
    y_std = mc.y_norm_std
    relaxation = FT(mc.relaxation)

    st_test = Lux.testmode(mc.st)

    model_lux =
        _build_lux_model(mc.arch, mc.ps, nz_model, nv, n_feat, true)

    X_buf = zeros(Float32, input_dim, 1)

    T_0 = TD.Parameters.T_0(thermo_params)
    Lv_0 = TD.Parameters.LH_v0(thermo_params)
    cv_v = TD.Parameters.cv_v(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)

    z_arr = Array(Fields.field2array(Fields.coordinate_field(axes(Y.c)).z))
    z_max = FT(mc.z_max)

    ρe_corr_cpu = zeros(FT, ncols, nz_sim)
    ρq_corr_cpu = zeros(FT, ncols, nz_sim)

    for col in 1:ncols
        for k in 1:nz_use
            X_buf[(0 * nz_model) + k, 1] = Float32(T_arr[col, k])
            X_buf[(1 * nz_model) + k, 1] = Float32(q_arr[col, k])
            if n_feat >= 3
                X_buf[(2 * nz_model) + k, 1] =
                    Float32(log(max(Float32(p_arr[col, k]), 1.0f0)))
            end
            if n_feat >= 4
                X_buf[(3 * nz_model) + k, 1] =
                    has_tke ? Float32(tke_arr[col, k]) : 0.0f0
            end
        end

        for i in 1:input_dim
            X_buf[i, 1] = (X_buf[i, 1] - x_mean[i]) / x_std[i]
        end

        pred, _ = Lux.apply(model_lux, X_buf, mc.ps, st_test)

        for i in 1:output_dim
            pred[i, 1] = pred[i, 1] * y_std[i] + y_mean[i]
        end

        for k in 1:nz_use
            if FT(z_arr[col, k]) > z_max
                continue
            end

            ρ_k = FT(ρ_arr[col, k])
            T_k = FT(T_arr[col, k])
            q_tot_k = FT(q_arr[col, k])
            q_liq_k = FT(q_liq_arr[col, k])
            q_ice_k = FT(q_ice_arr[col, k])

            dTdt = FT(0)
            dqdt = FT(0)

            if mc.target_vars == :T || mc.target_vars == :both
                dTdt = FT(pred[k, 1]) * relaxation
            end
            if mc.target_vars == :q
                dqdt = FT(pred[k, 1]) * relaxation
            elseif mc.target_vars == :both
                dqdt = FT(pred[nz_model + k, 1]) * relaxation
            end

            dTdt = clamp(dTdt, FT(-0.01), FT(0.01))
            dqdt = clamp(dqdt, FT(-1e-5), FT(1e-5))

            cv_m = TD.cv_m(thermo_params, q_tot_k, q_liq_k, q_ice_k)
            ρe_corr_cpu[col, k] = ρ_k * (
                cv_m * dTdt +
                (cv_v * (T_k - T_0) + Lv_0 - R_v * T_0) * dqdt
            )
            ρq_corr_cpu[col, k] = ρ_k * dqdt
        end
    end

    copyto!(Fields.field2array(ᶜρe_tot_correction), ρe_corr_cpu)
    copyto!(Fields.field2array(ᶜρq_tot_correction), ρq_corr_cpu)

    return nothing
end

function ml_correction_compute_tendency!(Y, p, ::MLCorrectionFlux)
    FT = Spaces.undertype(axes(Y.c))
    thermo_params = CAP.thermodynamics_params(p.params)
    mc = p.ml_correction

    (; ᶜρe_tot_correction, ᶜρq_tot_correction) = mc
    (; ᶜT, ᶜq_tot_nonneg, ᶜq_liq, ᶜq_ice) = p.precomputed

    nz_model = mc.nz
    nz_sim = Spaces.nlevels(axes(Y.c))
    nz_use = min(nz_model, nz_sim)

    T_arr = Array(Fields.field2array(ᶜT))
    q_arr = Array(Fields.field2array(ᶜq_tot_nonneg))
    ρ_arr = Array(Fields.field2array(Y.c.ρ))
    q_liq_arr = Array(Fields.field2array(ᶜq_liq))
    q_ice_arr = Array(Fields.field2array(ᶜq_ice))
    p_arr = Array(Fields.field2array(p.precomputed.ᶜp))

    has_tke = hasproperty(Y.c, :ρtke)
    if has_tke
        tke_arr = Array(Fields.field2array(@. specific(Y.c.ρtke, Y.c.ρ)))
    end

    ncols = size(T_arr, 1)
    n_feat = mc.n_features
    input_dim = n_feat * nz_model
    nv = mc.target_vars == :both ? 2 : 1
    nf = nz_model - 1
    output_dim = nv * nf

    x_mean = mc.x_norm_mean
    x_std = mc.x_norm_std
    y_mean = mc.y_norm_mean
    y_std = mc.y_norm_std
    relaxation = FT(mc.relaxation)

    st_test = Lux.testmode(mc.st)
    model_lux =
        _build_lux_model(mc.arch, mc.ps, nz_model, nv, n_feat, false)

    X_buf = zeros(Float32, input_dim, 1)

    z_arr = Array(Fields.field2array(Fields.coordinate_field(axes(Y.c)).z))
    z_max = FT(mc.z_max)

    T_0 = TD.Parameters.T_0(thermo_params)
    Lv_0 = TD.Parameters.LH_v0(thermo_params)
    cv_v = TD.Parameters.cv_v(thermo_params)
    R_v = TD.Parameters.R_v(thermo_params)

    ρe_corr_cpu = zeros(FT, ncols, nz_sim)
    ρq_corr_cpu = zeros(FT, ncols, nz_sim)

    for col in 1:ncols
        for k in 1:nz_use
            X_buf[(0 * nz_model) + k, 1] = Float32(T_arr[col, k])
            X_buf[(1 * nz_model) + k, 1] = Float32(q_arr[col, k])
            if n_feat >= 3
                X_buf[(2 * nz_model) + k, 1] =
                    Float32(log(max(Float32(p_arr[col, k]), 1.0f0)))
            end
            if n_feat >= 4
                X_buf[(3 * nz_model) + k, 1] =
                    has_tke ? Float32(tke_arr[col, k]) : 0.0f0
            end
        end

        for i in 1:input_dim
            X_buf[i, 1] = (X_buf[i, 1] - x_mean[i]) / x_std[i]
        end

        pred, _ = Lux.apply(model_lux, X_buf, mc.ps, st_test)

        for i in 1:output_dim
            pred[i, 1] = pred[i, 1] * y_std[i] + y_mean[i]
        end

        for v in 1:nv
            F_int = @view pred[((v - 1) * nf + 1):(v * nf), 1]
            for k in 1:nz_use
                if FT(z_arr[col, k]) > z_max
                    continue
                end

                dz_k = if k < nz_use
                    FT(z_arr[col, k + 1] - z_arr[col, k])
                else
                    FT(z_arr[col, k] - z_arr[col, k - 1])
                end
                dz_k = max(dz_k, FT(1))

                F_above = k < nf ? FT(F_int[k]) : FT(0)
                F_below = k > 1 ? FT(F_int[k - 1]) : FT(0)
                tendency = -(F_above - F_below) / dz_k * relaxation

                ρ_k = FT(ρ_arr[col, k])
                T_k = FT(T_arr[col, k])
                q_tot_k = FT(q_arr[col, k])
                q_liq_k = FT(q_liq_arr[col, k])
                q_ice_k = FT(q_ice_arr[col, k])

                tendency = clamp(tendency, FT(-0.01), FT(0.01))

                if (v == 1 && mc.target_vars != :q)
                    cv_m = TD.cv_m(thermo_params, q_tot_k, q_liq_k, q_ice_k)
                    ρe_corr_cpu[col, k] += ρ_k * cv_m * tendency
                elseif (v == 1 && mc.target_vars == :q) ||
                       (v == 2 && mc.target_vars == :both)
                    dqdt = clamp(tendency, FT(-1e-5), FT(1e-5))
                    ρe_corr_cpu[col, k] += ρ_k *
                        (cv_v * (T_k - T_0) + Lv_0 - R_v * T_0) * dqdt
                    ρq_corr_cpu[col, k] += ρ_k * dqdt
                end
            end
        end
    end

    copyto!(Fields.field2array(ᶜρe_tot_correction), ρe_corr_cpu)
    copyto!(Fields.field2array(ᶜρq_tot_correction), ρq_corr_cpu)

    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Lux model reconstruction from saved architecture metadata
# ─────────────────────────────────────────────────────────────────────────────
#
# These builders must exactly reproduce the architectures in the training
# script (train_flux_correction.jl). Hyperparameters like channel count and
# kernel size are inferred from the saved parameter shapes so the code stays
# in sync even if training defaults change.

"""
    _build_lux_model(arch, ps, nz, nv, n_feat, direct)

Reconstruct the Lux model architecture matching the training script.
`ps` is used to infer hyperparameters (channel count, kernel size) from
the saved parameter shapes. `direct` selects the output format: `true`
for tendency mode (`nv*nz` outputs), `false` for flux mode
(`nv*(nz-1)` interior face fluxes).
"""
function _build_lux_model(
    arch::Symbol, ps, nz::Int, nv::Int, n_feat::Int, direct::Bool,
)
    if arch == :cnn
        ch = size(ps.conv1.weight, 3)
        ks = size(ps.conv1.weight, 1)
        return _build_cnn(n_feat, nz, nv, ch, ks, direct)
    elseif arch == :unet
        ch = size(ps.enc1.layer_1.weight, 3)
        return _build_unet(n_feat, nz, nv, ch, direct)
    elseif arch == :mlp
        hidden = size(ps.layer_1.weight, 1)
        return _build_mlp(n_feat * nz, nz, nv, hidden, direct)
    else
        error("Unknown ML correction architecture: $arch")
    end
end

function _build_cnn(
    n_feat::Int, nz::Int, nv::Int, ch::Int, ks::Int, direct::Bool,
)
    nf = nz - 1
    dilations = [1, 1, 2, 4, 1, 1]
    hp = ks ÷ 2

    Lux.@compact(
        conv1 = Lux.Conv((ks,), n_feat => ch, Lux.gelu;
            pad = hp * dilations[1], dilation = dilations[1]),
        conv2 = Lux.Conv((ks,), ch => ch, Lux.gelu;
            pad = hp * dilations[2], dilation = dilations[2]),
        drop1 = Lux.Dropout(0.15f0),
        conv3 = Lux.Conv((ks,), ch => 2ch, Lux.gelu;
            pad = hp * dilations[3], dilation = dilations[3]),
        conv4 = Lux.Conv((ks,), 2ch => 2ch, Lux.gelu;
            pad = hp * dilations[4], dilation = dilations[4]),
        drop2 = Lux.Dropout(0.15f0),
        conv5 = Lux.Conv((ks,), 2ch => ch, Lux.gelu;
            pad = hp * dilations[5], dilation = dilations[5]),
        conv6 = Lux.Conv((ks,), ch => ch, Lux.gelu;
            pad = hp * dilations[6], dilation = dilations[6]),
        out_conv = Lux.Conv((1,), ch => nv),
    ) do x
        B = size(x, 2)
        h = reshape(x, nz, n_feat, B)
        h = conv1(h)
        h = conv2(h)
        h = drop1(h)
        h = conv3(h)
        h = conv4(h)
        h = drop2(h)
        h = conv5(h)
        h = conv6(h)
        out = out_conv(h)
        if direct
            @return vcat([reshape(out[:, v:v, :], nz, B) for v in 1:nv]...)
        else
            faces = (out[1:nf, :, :] .+ out[2:nz, :, :]) ./ 2.0f0
            @return vcat(
                [reshape(faces[:, v:v, :], nf, B) for v in 1:nv]...,
            )
        end
    end
end

function _build_unet(
    n_feat::Int, nz::Int, nv::Int, ch::Int, direct::Bool,
)
    nf = nz - 1
    nz_pad = nz % 4 == 0 ? nz : nz + (4 - nz % 4)

    Lux.@compact(
        enc1 = Lux.Chain(
            Lux.Conv((3,), n_feat => ch, Lux.gelu; pad = 1),
            Lux.Conv((3,), ch => ch, Lux.gelu; pad = 1),
        ),
        enc2 = Lux.Chain(
            Lux.Conv((3,), ch => 2ch, Lux.gelu; pad = 1),
            Lux.Conv((3,), 2ch => 2ch, Lux.gelu; pad = 1),
        ),
        bneck = Lux.Chain(
            Lux.Conv((3,), 2ch => 4ch, Lux.gelu; pad = 1),
            Lux.Conv((3,), 4ch => 4ch, Lux.gelu; pad = 1),
        ),
        up2 = Lux.ConvTranspose((2,), 4ch => 2ch; stride = 2),
        dec2 = Lux.Chain(
            Lux.Conv((3,), 4ch => 2ch, Lux.gelu; pad = 1),
            Lux.Conv((3,), 2ch => 2ch, Lux.gelu; pad = 1),
        ),
        up1 = Lux.ConvTranspose((2,), 2ch => ch; stride = 2),
        dec1 = Lux.Chain(
            Lux.Conv((3,), 2ch => ch, Lux.gelu; pad = 1),
            Lux.Conv((3,), ch => ch, Lux.gelu; pad = 1),
        ),
        out_conv = Lux.Conv((1,), ch => nv),
        pool = Lux.MaxPool((2,)),
    ) do x
        B = size(x, 2)
        x3d = reshape(x, nz, n_feat, B)
        if nz_pad > nz
            x3d = cat(x3d, x3d[1:(nz_pad - nz), :, :] .* 0.0f0; dims = 1)
        end
        e1 = enc1(x3d)
        e2 = enc2(pool(e1))
        b = bneck(pool(e2))
        d2 = dec2(cat(up2(b), e2; dims = 2))
        d1 = dec1(cat(up1(d2), e1; dims = 2))
        out = out_conv(d1)
        c = out[1:nz, :, :]
        if direct
            @return vcat([reshape(c[:, v:v, :], nz, B) for v in 1:nv]...)
        else
            faces = (c[1:nf, :, :] .+ c[2:nz, :, :]) ./ 2.0f0
            @return vcat(
                [reshape(faces[:, v:v, :], nf, B) for v in 1:nv]...,
            )
        end
    end
end

function _build_mlp(
    input_dim::Int, nz::Int, nv::Int, hidden::Int, direct::Bool,
)
    nf = nz - 1
    output_dim = direct ? nv * nz : nv * nf
    return Lux.Chain(
        Lux.Dense(input_dim => hidden, Lux.gelu),
        Lux.Dense(hidden => hidden, Lux.gelu),
        Lux.Dense(hidden => hidden, Lux.gelu),
        Lux.Dense(hidden => output_dim),
    )
end
