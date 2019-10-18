import numpy as np
import torch
from math import pi


# integrate over S2
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


def torch_expm(X):
    ''' 
    input: X = [X1, X2, X3]
    output: the matrix exponential of the anti-symmetric matrix of the form AMX = 
    [  0, X1, X2
     -X1,  0, X3
     -X2, x3,  0]
    '''
    MX = torch.zeros(3,3)
    MX[0, 1], MX[0,2], MX[1,2] = X[0], X[1], X[2]
    AMX = MX - MX.t()
    
    XN = torch.norm(X) + 1e-8
    return torch.eye(3) + torch.sin(XN)*AMX/XN + (1-torch.cos(XN))*AMX@AMX/XN**2


def center_surf(f):
    m, n = f.shape[-2:]
    PHI, THETA = torch.meshgrid([torch.linspace(0, pi, m), torch.linspace(0, 2 * pi, n)])
    df = torch.zeros(3, 2, m, n)
    Xf10, df[0, 1] = gradient_map(f[0])  # 1/sin(phi)d/dtheta, d/dphi
    Xf20, df[1, 1] = gradient_map(f[1])
    Xf30, df[2, 1] = gradient_map(f[2])

    df[0, 0], df[1, 0], df[2, 0] = Xf10 / torch.sin(PHI), Xf20 / torch.sin(PHI), Xf30 / torch.sin(PHI)
    df[:, 0, [0, m - 1], :] = 0  # when phi is 0 and pi, df/dtheta = 0: singularity

    n_f = torch.cross(df[:, 0], df[:, 1], dim=0)
    Norm_n = torch.einsum("imn,imn->mn", [n_f, n_f]).sqrt()
    M = integral_over_s2(Norm_n)
    
    x_c, y_c, z_c \
        = integral_over_s2(f[0]*Norm_n)/M, integral_over_s2(f[1]*Norm_n)/M, integral_over_s2(f[2]*Norm_n)/M
    f_c = torch.zeros(f.size())
    f_c[0] = f[0] - x_c
    f_c[1] = f[1] - y_c
    f_c[2] = f[2] - z_c
    return f_c


# rescale the surface such that its area is 1
def rescale_surf(f):
    df = f_to_df(f)
    n_f = torch.cross(df[:, 0], df[:, 1], dim=0)
    Norm_n = torch.einsum("imn,imn->mn", [n_f, n_f]).sqrt()
    
    return f/integral_over_s2(Norm_n).sqrt()

