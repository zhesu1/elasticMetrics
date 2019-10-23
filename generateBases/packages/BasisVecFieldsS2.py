import numpy as np
import torch
from scipy.special import sph_harm


# integrate func over S2
def integral_over_s2(func):
    # ---input: func: ...k*s*m*n
    # ---output: the integral of func: ...k*s

    m, n = func.shape[-2:]
    phi = torch.linspace(0, np.pi, m)
    theta = torch.linspace(0, 2 * np.pi, n)
    PHI, THETA = torch.meshgrid([phi, theta])
    F = func * torch.sin(PHI)
    return torch_trapz_2d(torch.einsum("...ij->ij...", [F]), np.pi / (m - 1), 2 * np.pi / (n - 1))


def torch_trapz_2d(func, dphi, dtheta):  # func: m*n*3*2*...
    # trapz function in the first 2D slice
    int_1 = torch.sum(func[0:func.size(0) - 1], 0) + torch.sum(func[1:func.size(0)], 0)
    int_fun = torch.sum(int_1[0:int_1.size(0) - 1], 0) + torch.sum(int_1[1:int_1.size(0)], 0)
    return int_fun * dphi * dtheta / 4


def torch_gradient(F, dtheta, dphi):
    dF_phi, dF_theta = torch.zeros(F.size()), torch.zeros(F.size())
    dF_theta[:, 0], dF_theta[:, F.size(1) - 1] = (F[:, 1] - F[:, 0]) / dtheta, (
                F[:, F.size(1) - 1] - F[:, F.size(1) - 2]) / dtheta
    dF_theta[:, 1: F.size(1) - 1] = (F[:, 2: F.size(1)] - F[:, 0:F.size(1) - 2]) / (2 * dtheta)

    dF_phi[0], dF_phi[F.size(0) - 1] = (F[1] - F[0]) / dphi, (F[F.size(0) - 1] - F[F.size(0) - 2]) / dphi
    dF_phi[1: F.size(0) - 1] = (F[2: F.size(0)] - F[0:F.size(0) - 2]) / (2 * dphi)
    return dF_theta, dF_phi


def gradient_map(f):
    m, n = f.size()
    dm, dn = np.pi / (m - 1), 2 * np.pi / (n - 1)
    F = torch.zeros(m + 2, n + 2)
    F[1: m + 1, 1: n + 1] = f
    F[0, 1: int((n + 1) / 2) + 1], F[0, int((n + 1) / 2): n + 1] \
        = f[1, int((n + 1) / 2) - 1: n], f[1, 0: int((n + 1) / 2)]
    F[m + 1, 1: int((n + 1) / 2) + 1], F[m + 1, int((n + 1) / 2): n + 1] \
        = f[m - 2, int((n + 1) / 2) - 1: n], f[m - 2, 0: int((n + 1) / 2)]
    F[:, 0], F[:, n + 1] = F[:, n - 1], F[:, 2]
    [X0, Y0] = torch_gradient(F, dn, dm)
    return X0[1:m + 1, 1:n + 1], Y0[1:m + 1, 1:n + 1]


#  get the spherical harmonic function of degree L and order M, where -L<=M<=L
def spherical_harmonic_function_ml(L,M,Num_phi,Num_theta):
    PHI, THETA = torch.meshgrid([torch.linspace(0, np.pi, Num_phi), torch.linspace(0, 2*np.pi, Num_theta)])
    phi, theta = PHI.numpy(), THETA.numpy()
    if M<0:
        spharmonics = np.sqrt(2)*(-1)**M*sph_harm(abs(M), L, theta, phi).imag  # spharmonics = sph_harm(m, l, theta, phi)
    elif M == 0:
        spharmonics = sph_harm(0, L, theta, phi).real
    elif M>0:
        spharmonics = np.sqrt(2)*(-1)**M*sph_harm(M, L, theta, phi).real
    return torch.tensor(spharmonics)


# generate a spherical harmonic function basis with the maximal degree max_deg and of length (max_deg+1)**2
def spherical_harmonic_basis(max_deg, Num_phi, Num_theta):
    S_basis = torch.zeros((max_deg+1)**2, Num_phi, Num_theta)
    for L in range(max_deg+1):
        for M in range(-L, L+1):
            S_basis[L*(L+1)+M] = spherical_harmonic_function_ml(L,M,Num_phi,Num_theta)
    return S_basis


def f_to_df_1dim(f):  # f: m*n
    m, n = f.shape[-2:]
    PHI, THETA = torch.meshgrid([torch.linspace(0, np.pi, m), torch.linspace(0, 2 * np.pi, n)])
    df = torch.zeros(2, m, n)
    Xf10, df[1] = gradient_map(f)  # 1/sin(phi)d/dtheta, d/dphi

    df[0] = Xf10 / torch.sin(PHI)
    df[0, [0, m - 1], :] = 0  # when phi is 0 and pi, df/dtheta = 0: singularity
    return df


# get a truncated basis for the space of exact one-forms with the maximal degree max_deg
def get_basis_vecFields_S2(max_deg, Num_phi, Num_theta):
    shb = spherical_harmonic_basis(max_deg, Num_phi, Num_theta)[1:]

    X = torch.zeros(shb.size(0), 2, Num_phi, Num_theta)
    for i in range(shb.size(0)):
        X[i] = f_to_df_1dim(shb[i])
        X[i] = X[i]/integral_over_s2(torch.einsum("ijk,ijk->jk", [X[i], X[i]])).sqrt()
    X1 = torch.flip(X, [1])  # rotate the vector fields by pi/2
    X1[:, 1] = -X1[:, 1]
    return torch.stack((X, X1))  # size 2 * L* 2 * m * n


def BasisSph_to_BasisCar(B):  # B: 2 * L * 2 * m * n...two halfs of the basis
    Num_phi, Num_theta = B.shape[-2:]
    phi, theta = torch.linspace(0, np.pi, Num_phi), torch.linspace(0, 2*np.pi, Num_theta)
    PHI, THETA = torch.meshgrid([phi, theta])
    B1_vfieldCar = torch.stack((-torch.sin(THETA), torch.cos(THETA), torch.zeros(B.shape[-2:])))
    B2_vfieldCar = torch.stack((torch.cos(PHI)*torch.cos(THETA), torch.cos(PHI)*torch.sin(THETA), -torch.sin(PHI)))
    B_v_Car = torch.einsum("...jk,sjk->...sjk",[B[:,:,0], B1_vfieldCar]) \
        + torch.einsum("...jk,sjk->...sjk",[B[:,:,1], B2_vfieldCar])
    # fix the poles
    V = B_v_Car.permute(3,4,0,1,2)
    V[0] = torch.ones(*V.shape[1:])*torch.sum(V[1],0)/V.size(1)
    V[V.size(0)-1] =torch.ones(*V.shape[1:])*torch.sum(V[V.size(0)-2],0)/V.size(1)
    return V.permute(3,2,4,0,1)  # size L* 2 * 3 * m * n


def spherical_to_Cartesian(theta, phi):
    return torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)


def Cartesian_to_spherical(x, y, z): # x: ...*m*n
    return (torch.atan2(y, x) + 2 * np.pi) % (2 * np.pi), torch.acos(z)

