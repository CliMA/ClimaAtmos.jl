import ClimaCore
import ClimaCore: Domains, Spaces, Topologies

# To allow for backwards compatibility of ClimaCore:
if pkgversion(ClimaCore) < v"0.14.18"
    """
        z_max(::ClimaCore.Spaces.AbstractSpace)

    The domain maximum along the z-direction.
    """
    function z_max end

    z_max(domain::Domains.IntervalDomain) = domain.coord_max.z
    function z_max(space::Spaces.AbstractSpace)
        mesh = Topologies.mesh(Spaces.vertical_topology(space))
        domain = Topologies.domain(mesh)
        return z_max(domain)
    end

else
    z_max(s::Spaces.AbstractSpace) = Spaces.z_max(s)
end
