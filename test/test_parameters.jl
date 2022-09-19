using Test

import CLIMAParameters as CP

include("../tc_driver/generate_namelist.jl")
using .NameList
using TOML
"""
Helper function for comparing the namelist and toml below
"""
function get_lowest_keys_from_namelist(namelist, prepend_key = "")
    lowest_keys = String[]
    for (key, val) in namelist
        new_prepend_key =
            isempty(prepend_key) ? key : string(prepend_key, "-", key)
        if val isa Dict
            append!(
                lowest_keys,
                get_lowest_keys_from_namelist(val, new_prepend_key),
            )
        else
            push!(lowest_keys, new_prepend_key)
        end
    end
    return lowest_keys
end

@testset "Parameters" begin
    FT = Float64

    # Test Invalid Chars
    greek_namelist = Dict([("π", "pi")])
    NameList.namelist_to_toml_file(greek_namelist, tempname())

    test_namelist =
        Dict("A" => "a", "B" => "1", "C" => Dict("A" => "a", "B" => "1"))

    test_toml_dict = Dict(
        "A" => Dict("value" => "a", "type" => "string", "alias" => "A"),
        "B" => Dict("value" => "1", "type" => "string", "alias" => "B"),
        "C-B" => Dict("value" => "1", "type" => "string", "alias" => "B"),
        "C-A" => Dict("value" => "a", "type" => "string", "alias" => "A"),
    )

    @test NameList.namelist_to_toml_dict(test_namelist) == test_toml_dict

    @testset "All Namelist Cases" begin
        case_names = [
            "Soares",
            "Nieuwstadt",
            "Bomex",
            "life_cycle_Tan2018",
            "Rico",
            "TRMM_LBA",
            "ARM_SGP",
            "GATE_III",
            "DYCOMS_RF01",
            "DYCOMS_RF02",
            "GABLS",
            "LES_driven_SCM",
        ]

        for case_name in case_names
            nl = NameList.default_namelist(case_name, write = false)
            namelist_keys = get_lowest_keys_from_namelist(nl)
            fname = tempname()
            # fname = string(case_name, ".toml")
            NameList.namelist_to_toml_file(nl, fname)
            toml_dict = CP.create_toml_dict(
                FT,
                override_file = fname,
                dict_type = "alias",
            )
            for key in namelist_keys
                # This is bad practice but seems necessary for tests
                # Skip tuple cases
                if !(nl[key] isa Tuple)
                    @test toml_dict.data[key]["value"] == nl[key]
                end
            end
        end
    end
end
