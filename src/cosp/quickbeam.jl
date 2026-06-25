#=
Quickbeam radar reflectivity simulator — subcolumn kernel.

Port of COSPv2.0 src/simulator/quickbeam/quickbeam.F90 to Julia.
Original authors: J. Haynes, R. Marchand, A. Bodas-Salcedo (2005–2011)
COSPv2 adaptation: D. Swales (2015)

Scope of this file:
  RadarConfig          — radar simulator configuration struct
  quickbeam_subcolumn  — attenuated/non-attenuated dBZ from pre-computed optics

Not included yet:
  quickbeam_optics  — Mie LUT computation of z_vol, kr_vol from hydrometeor profiles
                      (will interface with CloudMicrophysics.jl)
  quickbeam_column  — CFAD statistics and CloudSat precipitation classification
=#

"""
    RadarConfig{FT}

Configuration for the Quickbeam radar subcolumn simulator.

# Fields
- `freq`: radar frequency (GHz)
- `k2`: dielectric factor |K|²; 0.93 for liquid water at 94 GHz
- `use_gas_abs`: gas absorption mode:
    - `0` = no gas absorption
    - `1` = compute independently for every profile
    - `2` = compute for profile 1 only, copy to all other profiles
- `radar_at_layer_one`: `true` for spaceborne radar (layer 1 = TOA, attenuation
  accumulates downward); `false` for surface-based radar (attenuation accumulates upward)
"""
@kwdef struct RadarConfig{FT}
    freq::FT
    k2::FT
    use_gas_abs::Int = 1
    radar_at_layer_one::Bool = true
end

"""
    quickbeam_subcolumn(rcfg, hgt_matrix, z_vol, kr_vol, g_vol)

Compute attenuated (`dBZe`) and non-attenuated (`Ze_non`) effective radar
reflectivity for each subcolumn profile and range gate.

Faithfully ports the attenuation accumulation scheme of COSPv2.0
`quickbeam_subcolumn` (quickbeam.F90 lines 67–181):
- Hydrometeor attenuation: trapezoidal path integral without the leading 0.5,
  which assigns a full half-layer from each side of each cell boundary.
- Gas attenuation: standard trapezoid rule (with 0.5 prefactor).
- Bootstrap approximation is used at the first one or two gates from the radar
  where no previous gate is available for the running sum.

# Arguments
- `rcfg`: `RadarConfig` instance
- `hgt_matrix`: height at gate mid-points, `(nprof, ngate)`, km
- `z_vol`: effective reflectivity factor, `(nprof, ngate)`, mm⁶ m⁻³
- `kr_vol`: hydrometeor two-way attenuation coefficient, `(nprof, ngate)`, dB km⁻¹
- `g_vol`: gaseous two-way attenuation coefficient, `(nprof, ngate)`, dB km⁻¹

# Returns
`(dBZe, Ze_non)`, each `(nprof, ngate)`, dBZ.
Gates where `z_vol ≤ 0` are set to `cloudsat_undef(FT)`.
"""
function quickbeam_subcolumn(
    rcfg::RadarConfig{FT},
    hgt_matrix::AbstractMatrix{FT},
    z_vol::AbstractMatrix{FT},
    kr_vol::AbstractMatrix{FT},
    g_vol::AbstractMatrix{FT},
) where {FT}

    nprof, ngate = size(hgt_matrix)

    dBZe     = fill(cloudsat_undef(FT), nprof, ngate)
    Ze_non   = fill(cloudsat_undef(FT), nprof, ngate)
    a_to_vol = zeros(FT, nprof, ngate)  # cumulative hydrometeor attenuation, dB
    g_to_vol = zeros(FT, nprof, ngate)  # cumulative gas attenuation, dB

    # Loop from the radar toward the scene.
    # Spaceborne (radar_at_layer_one=true): d_gate=+1, layer 1 → ngate (downward).
    # Surface-based (radar_at_layer_one=false): d_gate=-1, layer ngate → 1 (upward).
    d_gate       = rcfg.radar_at_layer_one ? 1 : -1
    gate_indices = rcfg.radar_at_layer_one ? (1:ngate) : (ngate:-1:1)

    for k in gate_indices
        for pr in 1:nprof

            # Hydrometeor path attenuation to gate k.
            # The accumulation uses (kr[k-1] + kr[k]) * Δh — no leading 0.5.
            # This is equivalent to summing a full half-cell from each adjacent gate.
            # A single-layer bootstrap estimate is used for the first two gates.
            if d_gate == 1
                a_to_vol[pr, k] =
                    if k > 2
                        a_to_vol[pr, k-1] +
                            (kr_vol[pr, k-1] + kr_vol[pr, k]) *
                            (hgt_matrix[pr, k-1] - hgt_matrix[pr, k])
                    else  # k == 1 or 2: bootstrap
                        kr_vol[pr, k] * (hgt_matrix[pr, k] - hgt_matrix[pr, k+1])
                    end
            else
                a_to_vol[pr, k] =
                    if k < ngate
                        a_to_vol[pr, k+1] +
                            (kr_vol[pr, k+1] + kr_vol[pr, k]) *
                            (hgt_matrix[pr, k+1] - hgt_matrix[pr, k])
                    else  # k == ngate: bootstrap
                        kr_vol[pr, k] * (hgt_matrix[pr, k] - hgt_matrix[pr, k-1])
                    end
            end

            # Gas path attenuation to gate k (standard trapezoid, 0.5 prefactor).
            # Mode 2: compute only for pr==1, then copy to all other profiles.
            # The inner pr loop (1:nprof) ensures pr==1 is ready before pr>1 reads it.
            if rcfg.use_gas_abs == 1 || (rcfg.use_gas_abs == 2 && pr == 1)
                if d_gate == 1
                    g_to_vol[pr, k] =
                        if k > 1
                            g_to_vol[pr, k-1] +
                                FT(0.5) * (g_vol[pr, k-1] + g_vol[pr, k]) *
                                (hgt_matrix[pr, k-1] - hgt_matrix[pr, k])
                        else
                            FT(0.5) * g_vol[pr, k] *
                                (hgt_matrix[pr, k] - hgt_matrix[pr, k+1])
                        end
                else
                    g_to_vol[pr, k] =
                        if k < ngate
                            g_to_vol[pr, k+1] +
                                FT(0.5) * (g_vol[pr, k+1] + g_vol[pr, k]) *
                                (hgt_matrix[pr, k+1] - hgt_matrix[pr, k])
                        else
                            FT(0.5) * g_vol[pr, k] *
                                (hgt_matrix[pr, k] - hgt_matrix[pr, k-1])
                        end
                end
            elseif rcfg.use_gas_abs == 2
                g_to_vol[pr, k] = g_to_vol[1, k]
            end
            # use_gas_abs == 0: g_to_vol stays at zero
        end
    end

    # Convert reflectivity factor to dBZ and subtract total path attenuation.
    for k in 1:ngate, pr in 1:nprof
        if z_vol[pr, k] > 0
            Ze_non[pr, k] = 10 * log10(z_vol[pr, k])
            dBZe[pr, k]   = Ze_non[pr, k] - a_to_vol[pr, k] - g_to_vol[pr, k]
        end
    end

    return dBZe, Ze_non
end

# Sentinel for undefined/missing reflectivity. Matches COSPv2 R_UNDEF = -1.0e30.
cloudsat_undef(::Type{FT}) where FT = FT(-1.0e30)
