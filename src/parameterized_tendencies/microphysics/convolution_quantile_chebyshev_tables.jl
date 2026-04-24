# Precomputed Chebyshev coefficients for uniform–Gaussian convolution quantiles u/L
# in mapped log10(η), same pattern as [`gauss_hermite`](@ref): static branches, no mutable cache.
# Coefficients are checked into this module; any regeneration must keep `NCOEFF` / `DEG` / `CHEB_CONV_MAX_N_GL` consistent.

import StaticArrays as SA

"""`(log10(η_min), log10(η_max))` for the offline fit (Float64). Runtime quantiles use `log10(FT(1e-4))`, `log10(FT(1e2))` so τ stays in `FT`."""
const CHEB_CONV_ETA_LOG10_RANGE = (log10(1.0e-4), log10(1.0e2))

"""Number of Chebyshev coefficients (degree 12); must match the offline fit degree used to generate the tables."""
const NCOEFF_CONV_CHEB = 13

"""Largest Gauss–Legendre order with embedded convolution Chebyshev tables."""
const CHEB_CONV_MAX_N_GL = 5

"""
    chebyshev_convolution_coeffs(FT, N_GL, i_node) -> SVector{13,FT}

Precomputed Chebyshev coefficients in τ for `u/L` at GL node `i_node` for order `N_GL`.

Errors if the pair is unsupported (same contract as [`gauss_legendre_01`](@ref): only precomputed orders are valid).

Truncated series in ``\\tau``: at degree 12 the offline fit reports worst **~6e-3** absolute error on ``u/L`` vs dense truth on the generator grid; random checks vs Brent quantiles are typically **O(1e-3)** (see tests). Runtime uses **no** iteration after evaluation.

Coefficients are static `SVector` data in this module.
"""
@inline function chebyshev_convolution_coeffs(::Type{FT}, N_GL::Int, i_node::Int) where {FT}
    if N_GL == 1 && i_node == 1
        return SA.SVector{13,FT}(
            -2.3805845006626114e-12,
            -4.6782086711290994e-12,
            -4.107888137311789e-12,
            -3.491402198132139e-12,
            -2.5782229718129547e-12,
            -1.8469386160533864e-12,
            -9.967788859047197e-13,
            -4.791151025136716e-13,
            1.0858898653174829e-13,
            3.0738315912344385e-13,
            6.244997669960938e-13,
            5.018576826143209e-13,
            6.130168465165683e-13,
        )
    elseif N_GL == 2 && i_node == 1
        return SA.SVector{13,FT}(
            -12.56699086774821,
            -22.772855292790865,
            -18.16359826031846,
            -12.470628908798506,
            -7.403164104665263,
            -3.845370453644011,
            -1.7861296694214317,
            -0.7555319691206123,
            -0.2872773656897147,
            -0.09241872861734415,
            -0.02678410936805057,
            -0.010244497482634023,
            -0.005386635674202868,
        )
    elseif N_GL == 2 && i_node == 2
        return SA.SVector{13,FT}(
            12.56699086774807,
            22.772855292790588,
            18.163598260318217,
            12.470628908798298,
            7.403164104665105,
            3.845370453643895,
            1.7861296694213584,
            0.7555319691205663,
            0.2872773656896999,
            0.09241872861734327,
            0.026784109368069176,
            0.010244497482652139,
            0.005386635674232589,
        )
    elseif N_GL == 3 && i_node == 1
        return SA.SVector{13,FT}(
            -18.973796768303476,
            -34.461023949314146,
            -27.464066291925388,
            -18.844425562499907,
            -11.189886932077727,
            -5.818761311024186,
            -2.7026031135415285,
            -1.1391319438383365,
            -0.43199379283929257,
            -0.14132867445701058,
            -0.04188439885749127,
            -0.014211094784145594,
            -0.006319417327552703,
        )
    elseif N_GL == 3 && i_node == 2
        return SA.SVector{13,FT}(
            -2.3805845006626114e-12,
            -4.6782086711290994e-12,
            -4.107888137311789e-12,
            -3.491402198132139e-12,
            -2.5782229718129547e-12,
            -1.8469386160533864e-12,
            -9.967788859047197e-13,
            -4.791151025136716e-13,
            1.0858898653174829e-13,
            3.0738315912344385e-13,
            6.244997669960938e-13,
            5.018576826143209e-13,
            6.130168465165683e-13,
        )
    elseif N_GL == 3 && i_node == 3
        return SA.SVector{13,FT}(
            18.973796768303963,
            34.46102394931507,
            27.46406629192626,
            18.844425562500657,
            11.189886932078362,
            5.818761311024679,
            2.7026031135419046,
            1.1391319438386,
            0.43199379283947253,
            0.1413286744571215,
            0.04188439885755112,
            0.01421109478416971,
            0.006319417327548163,
        )
    elseif N_GL == 4 && i_node == 1
        return SA.SVector{13,FT}(
            -23.142260546241243,
            -42.09882492320049,
            -33.531725582177764,
            -22.99804349935593,
            -13.659454591876527,
            -7.108184515640419,
            -3.300937034771686,
            -1.3880102867260367,
            -0.5259918646731174,
            -0.17391961837861364,
            -0.051974085510902955,
            -0.01621860638402591,
            -0.006739342552934485,
        )
    elseif N_GL == 4 && i_node == 2
        return SA.SVector{13,FT}(
            -6.900537734883718,
            -12.485915511154051,
            -9.963835498029402,
            -6.843708144559285,
            -4.062233583412799,
            -2.1084133948410106,
            -0.9792488446551204,
            -0.4151818449711016,
            -0.15830495993332755,
            -0.0503496370449294,
            -0.014275093794676596,
            -0.005844014999890269,
            -0.003534428570024688,
        )
    elseif N_GL == 4 && i_node == 3
        return SA.SVector{13,FT}(
            6.900537734883833,
            12.485915511154268,
            9.963835498029628,
            6.843708144559498,
            4.062233583413018,
            2.1084133948412105,
            0.9792488446553254,
            0.41518184497127997,
            0.15830495993350982,
            0.05034963704507741,
            0.014275093794828088,
            0.00584401499999295,
            0.003534428570130526,
        )
    elseif N_GL == 4 && i_node == 4
        return SA.SVector{13,FT}(
            23.142260546240813,
            42.098824923199615,
            33.53172558217704,
            22.998043499355322,
            13.659454591876155,
            7.1081845156401915,
            3.3009370347716924,
            1.3880102867261426,
            0.5259918646733883,
            0.17391961837888706,
            0.051974085511257664,
            0.01621860638427827,
            0.006739342553206576,
        )
    elseif N_GL == 5 && i_node == 1
        return SA.SVector{13,FT}(
            -26.182016723993577,
            -47.68263260014541,
            -37.963215828161864,
            -26.02979757975879,
            -15.462986486206363,
            -8.050748719892715,
            -3.7379682305852318,
            -1.56930145569808,
            -0.5946540220945674,
            -0.19794761073903658,
            -0.05933117371222849,
            -0.017574891431339886,
            -0.007121968377584024,
        )
    elseif N_GL == 5 && i_node == 2
        return SA.SVector{13,FT}(
            -11.54258031650533,
            -20.909868246757142,
            -16.679507688566467,
            -11.452690055961375,
            -6.798662065491836,
            -3.530814074159141,
            -1.6400049516574964,
            -0.6940659849824423,
            -0.2640440325572323,
            -0.08473924789043223,
            -0.024456261575320257,
            -0.009499282440557205,
            -0.005138259372653321,
        )
    elseif N_GL == 5 && i_node == 3
        return SA.SVector{13,FT}(
            -2.3805845006626114e-12,
            -4.6782086711290994e-12,
            -4.107888137311789e-12,
            -3.491402198132139e-12,
            -2.5782229718129547e-12,
            -1.8469386160533864e-12,
            -9.967788859047197e-13,
            -4.791151025136716e-13,
            1.0858898653174829e-13,
            3.0738315912344385e-13,
            6.244997669960938e-13,
            5.018576826143209e-13,
            6.130168465165683e-13,
        )
    elseif N_GL == 5 && i_node == 4
        return SA.SVector{13,FT}(
            11.542580316505182,
            20.90986824675685,
            16.679507688566215,
            11.452690055961146,
            6.798662065491659,
            3.530814074158994,
            1.640004951657394,
            0.6940659849823614,
            0.2640440325571822,
            0.0847392478903944,
            0.024456261575303288,
            0.009499282440546484,
            0.005138259372655324,
        )
    elseif N_GL == 5 && i_node == 5
        return SA.SVector{13,FT}(
            26.18201672399303,
            47.68263260014433,
            37.96321582816099,
            26.02979757975809,
            15.462986486205963,
            8.050748719892512,
            3.7379682305852926,
            1.56930145569825,
            0.5946540220949066,
            0.19794761073937245,
            0.05933117371264576,
            0.017574891431640177,
            0.0071219683779048856,
        )
    else
        error(
            "chebyshev_convolution_coeffs: no precomputed Chebyshev coefficients for " *
            "(N_GL = $N_GL, i_node = $i_node). Supported: 1 ≤ N_GL ≤ $CHEB_CONV_MAX_N_GL and " *
            "1 ≤ i_node ≤ N_GL. Extend `gen_convolution_chebyshev_tables.jl` and regenerate tables, " *
            "or choose `ConvolutionQuantilesBracketed` (Brent) or `ConvolutionQuantilesHalley` (one Halley step).",
        )
    end
end

@inline function chebyshev_evaluate(coeffs::SA.SVector{N, T}, τ::FT) where {N, T, FT}
    acc = zero(FT)
    T_prev2 = one(FT)
    T_prev1 = τ
    acc += FT(coeffs[1]) * T_prev2
    N >= 2 && (acc += FT(coeffs[2]) * T_prev1)
    for j in 3:N
        Tj = FT(2) * τ * T_prev1 - T_prev2
        acc += FT(coeffs[j]) * Tj
        T_prev2, T_prev1 = T_prev1, Tj
    end
    return acc
end
