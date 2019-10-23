import torch
import numpy as np
from scipy.special import sph_harm
from math import pi


#  get the spherical harmonic function of degree L and order M, where -L<=M<=L
def spherical_harmonic_function_ml(L, M, Num_phi, Num_theta):
    PHI, THETA = torch.meshgrid([torch.linspace(0, np.pi, Num_phi), torch.linspace(0, 2 * np.pi, Num_theta)])
    phi, theta = PHI.numpy(), THETA.numpy()
    if M < 0:
        spharmonics = np.sqrt(2) * (-1) ** M * sph_harm(abs(M), L, theta, phi).imag
    elif M == 0:
        spharmonics = sph_harm(0, L, theta, phi).real
    elif M > 0:
        spharmonics = np.sqrt(2) * (-1) ** M * sph_harm(M, L, theta, phi).real
    return torch.tensor(spharmonics)


# generate a spherical harmonic function basis with the maximal degree max_deg and of length (max_deg+1)**2
def spherical_harmonic_basis(max_deg, Num_phi, Num_theta):
    S_basis = torch.zeros((max_deg + 1) ** 2, Num_phi, Num_theta)
    for L in range(max_deg + 1):
        for M in range(-L, L + 1):
            S_basis[L * (L + 1) + M] = spherical_harmonic_function_ml(L, M, Num_phi, Num_theta)
    return S_basis


# get a truncated basis for the space of exact one-forms with the maximal degree max_deg
def get_basis_of_exact_one_forms(max_deg, Num_phi, Num_theta):
    shb = spherical_harmonic_basis(max_deg, Num_phi, Num_theta)[1:]
    F_3dim = torch.zeros(3 * shb.size(0), 3, Num_phi, Num_theta)
    B = 3 * torch.arange(1, shb.size(0) + 1)
    F_3dim[B - 1, 2], F_3dim[B - 2, 1], F_3dim[B - 3, 0] = shb, shb, shb

    basis = torch.zeros(3 * shb.size(0), 3, 2, Num_phi, Num_theta)
    for i in range(3 * shb.size(0)):
        basis[i] = f_to_df(F_3dim[i])
    n_b_1forms = torch.zeros(basis.size())
    n_b_sph = torch.zeros(F_3dim.size())
    for i in range(basis.size(0)):
        n_b_1forms[i] = basis[i] / (Euclidean_inner_prod(basis[i], basis[i])).sqrt()
        n_b_sph[i] = F_3dim[i] / (Euclidean_inner_prod(basis[i], basis[i])).sqrt()
    return n_b_sph, n_b_1forms


def Euclidean_inner_prod(a, b):
    prod = torch.einsum("...ismn,...ismn->...mn", [a, b])
    return integral_over_s2(prod)


def f_to_df(f):
    m, n = f.shape[-2:]
    PHI, THETA = torch.meshgrid([torch.linspace(0, pi, m), torch.linspace(0, 2 * pi, n)])
    df = torch.zeros(3, 2, m, n)
    Xf10, df[0, 1] = gradient_map(f[0])  # 1/sin(phi)d/dtheta, d/dphi
    Xf20, df[1, 1] = gradient_map(f[1])
    Xf30, df[2, 1] = gradient_map(f[2])

    # adding 1e-7 is for automatic differentiation
    df[0, 0], df[1, 0], df[2, 0] = Xf10/(1e-7 + torch.sin(PHI)), Xf20/(1e-7 + torch.sin(PHI)), Xf30/(
                1e-7 + torch.sin(PHI))
    df[:, 0, [0, m - 1], :] = 0  # when phi is 0 and pi, df/dtheta = 0: singularity
    return df


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
    dm, dn = pi / (m - 1), 2 * pi / (n - 1)
    F = torch.zeros(m + 2, n + 2)
    F[1: m + 1, 1: n + 1] = f
    F[0, 1: int((n + 1) / 2) + 1], F[0, int((n + 1) / 2): n + 1] \
        = f[1, int((n + 1) / 2) - 1: n], f[1, 0: int((n + 1) / 2)]
    F[m + 1, 1: int((n + 1) / 2) + 1], F[m + 1, int((n + 1) / 2): n + 1] \
        = f[m - 2, int((n + 1) / 2) - 1: n], f[m - 2, 0: int((n + 1) / 2)]
    F[:, 0], F[:, n + 1] = F[:, n - 1], F[:, 2]
    [X0, Y0] = torch_gradient(F, dn, dm)
    return X0[1:m + 1, 1:n + 1], Y0[1:m + 1, 1:n + 1]


def integral_over_s2(func):
    # ---input: func: ...k*s*m*n
    # ---output: the integral of func: ...k*s

    m, n = func.shape[-2:]
    phi = torch.linspace(0, pi, m)
    theta = torch.linspace(0, 2 * pi, n)
    PHI, THETA = torch.meshgrid([phi, theta])
    F = func * torch.sin(PHI)
    return torch_trapz_2d(torch.einsum("...ij->ij...", [F]), pi / (m - 1), 2 * pi / (n - 1))


def torch_trapz_2d(func, dphi, dtheta):  # func: m*n*3*2*...
    # trapz function in the first 2D slice
    int_1 = torch.sum(func[0:func.size(0) - 1], 0) + torch.sum(func[1:func.size(0)], 0)
    int_fun = torch.sum(int_1[0:int_1.size(0) - 1], 0) + torch.sum(int_1[1:int_1.size(0)], 0)
    return int_fun * dphi * dtheta / 4
