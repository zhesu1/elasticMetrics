import numpy as np
import torch
from ShapePackages.OneFormMaps import f_to_df


def get_idty_S2(Num_phi, Num_theta):
    phi, theta = torch.linspace(0, np.pi, Num_phi), torch.linspace(0, 2 * np.pi, Num_theta)
    PHI, THETA = torch.meshgrid([phi, theta])
    return torch.stack(spherical_to_Cartesian(THETA, PHI))


def spherical_to_Cartesian(theta, phi):
    return torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)


def Cartesian_to_spherical(x, y, z):  # x: ...*m*n
    return (torch.atan2(y, x) + 2 * np.pi) % (2 * np.pi), torch.acos(z)


# my interpolation function
def compose_gamma(f, gamma):  # f: ...*m*n  gamma: 2*m*n...

    f = f.permute(f.dim()-2, f.dim()-1, *range(f.dim()-2))  # change the size of f to m*n*...
    phi, theta = torch.linspace(0, np.pi, f.size(0)), torch.linspace(0, 2 * np.pi, f.size(1))
    PHI, THETA = torch.meshgrid([phi, theta])
    Ind_gamma = torch.stack((torch.floor((f.size(0)-1)*gamma[1]/np.pi).long(),
                             torch.floor((f.size(1)-1)*gamma[0]/(2*np.pi)).long()))

    # fix the boundary points
    Ind_gamma11 = (Ind_gamma[1]+1) % f.size(1) + Ind_gamma[1] - Ind_gamma[1] % (f.size(1)-1)
    Ind_gamma01 = (Ind_gamma[0]+1) % f.size(0) + Ind_gamma[0] - Ind_gamma[0] % (f.size(0)-1)

    # use the bilinear interpolation method
    F00 = f[Ind_gamma[0], Ind_gamma[1]].permute(*range(2, f.dim()), 0, 1)  # change the size to ...*m*n
    F01 = f[Ind_gamma[0], Ind_gamma11].permute(*range(2, f.dim()), 0, 1)
    F10 = f[Ind_gamma01, Ind_gamma[1]].permute(*range(2, f.dim()), 0, 1)
    F11 = f[Ind_gamma01, Ind_gamma11].permute(*range(2, f.dim()), 0, 1)

    C = (gamma[0] - THETA[Ind_gamma[0], Ind_gamma[1]])*(f.size(1)-1)/(2*np.pi)
    D = (gamma[1] - PHI[Ind_gamma[0], Ind_gamma[1]])*(f.size(0)-1)/np.pi

    F0 = F00 + (F01 - F00)*C
    F1 = F10 + (F11 - F10)*C
    return F0 + (F1 - F0)*D

