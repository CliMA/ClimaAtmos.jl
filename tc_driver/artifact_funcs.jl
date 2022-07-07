#! format: off

import ArtifactWrappers
const AW = ArtifactWrappers

function pycles_output_dataset_folder(lazy_download = true)
    PyCLES_output_dataset = AW.ArtifactWrapper(
        @__DIR__,
        lazy_download,
        "PyCLES_output",
        AW.ArtifactFile[
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/johlutwhohvr66wn38cdo7a6rluvz708.nc", filename = "Rico.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/zraeiftuzlgmykzhppqwrym2upqsiwyb.nc", filename = "Gabls.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/toyvhbwmow3nz5bfa145m5fmcb2qbfuz.nc", filename = "DYCOMS_RF01.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/dgie1774uw5ot8mmrmp46nauhb3ervgp.nc", filename = "DYCOMS_RF02.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/ivo4751camlph6u3k68ftmb1dl4z7uox.nc", filename = "TRMM_LBA.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/4osqp0jpt4cny8fq2ukimgfnyi787vsy.nc", filename = "ARM_SGP.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/jci8l11qetlioab4cxf5myr1r492prk6.nc", filename = "Bomex.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/pzuu6ii99by2s356ij69v5cb615200jq.nc", filename = "Soares.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/7upt639siyc2umon8gs6qsjiqavof5cq.nc", filename = "Nieuwstadt.nc",),
        ],
    )
    return AW.get_data_folder(PyCLES_output_dataset)
end

function scampy_output_dataset_folder(lazy_download = true)
    SCAMPy_output_dataset = AW.ArtifactWrapper(
        @__DIR__,
        lazy_download,
        "SCAMPy_output",
        AW.ArtifactFile[
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/1dzpydqiagjvzfpyv9lbic3atvca93hl.nc", filename = "Rico.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/agkycoum93bd6xjduyaeo3oqg6asoru5.nc", filename = "GABLS.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/fqpq1q74uxfh1e8018hwhqogw1htn2lq.nc", filename = "DYCOMS_RF01.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/wevi0rqiwo6sgkqdhcddr72u5ylt0tqp.nc", filename = "TRMM_LBA.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/fis6n0g9x9lts70m0zmve5ullqnw0pzq.nc", filename = "ARM_SGP.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/t6qq6plt2oxcmmy40r1szgokahykqggp.nc", filename = "Bomex.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/wp8k4m7ta1hs0c6e4j2fpsp3kj05wdip.nc", filename = "Soares.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/vbuxzg85scwy9mg0ziiuzy6fimo0e389.nc", filename = "Nieuwstadt.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/72t6fr1gq10tg3jjputtp35nfzex0o4k.nc", filename = "DryBubble.nc",),
        AW.ArtifactFile(url = "https://caltech.box.com/shared/static/7axeussneeg8g3k0ndvagsn0pkmbij3e.nc", filename = "life_cycle_Tan2018.nc",),
        ],
    )
    return AW.get_data_folder(SCAMPy_output_dataset)
end

function les_driven_scm_data_folder(lazy_download = true)
    LESDrivenSCM_output_dataset = AW.ArtifactWrapper(
        @__DIR__,
        lazy_download,
        "LESDrivenSCM_output_dataset",
        AW.ArtifactFile[
            AW.ArtifactFile(url = "https://caltech.box.com/shared/static/0hnf7nkttueraaqf9tpkqsx38gjqx41p.nc", filename = "Stats.cfsite23_HadGEM2-A_amip_2004-2008.07.nc",),
        ],
    )
    return AW.get_data_folder(LESDrivenSCM_output_dataset)
end

#! format: on
