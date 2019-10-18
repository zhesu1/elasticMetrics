import scipy.io as sio
import torch
import numpy as np
import scipy.optimize as optimize
from ShapePackages.OneFormMaps import gradient_map, torch_expm, integral_over_s2
from ShapePackages.RegistrationFunc import get_idty_S2, Cartesian_to_spherical, compose_gamma


def initialize_over_diffs_main_SRNF(f1, f2, MaxDegVecFS2 = 2, Max_ite = 50):
    
    idty = get_idty_S2(*f2.shape[-2:])

    f1_gamma, L, f1_barh, E = initialize_over_paraSO3_SRNF(f1, f2, idty)
    f, Energyrepa = initialize_over_diffs_SRNF(f1_gamma, f2, idty, MaxDegVecFS2, Max_ite)
    return f, Energyrepa, f1_gamma, L, f1_barh, E
    
    
def initialize_over_diffs_SRNF(f1, f2, idty, MaxDegVecFS2, Max_ite):
    
    # load the basis for tangent vector fields on S2
    mat_vecF = sio.loadmat('Data/basis_vecFieldsS2_deg25_{0}_{1}.mat'.format(*f1.shape[-2:]))
    
    N_basis_vec = (MaxDegVecFS2 + 1) ** 2 - 1  # half the number of basis for the vector fields on S2
    Basis0_vec = torch.from_numpy(mat_vecF['Basis'])[: N_basis_vec].float()
    Basis_vecFields = torch.cat((Basis0_vec[:, 0], Basis0_vec[:, 1]))  # get a basis of the tangent fields on S2
    
    X = torch.zeros(Basis_vecFields.size(0))
    
    Energyrepa = []
    def Opts(X):
        X = torch.from_numpy(X).float()
        X.requires_grad_()
        y = E_L2_SRNFS(X, f1, f2, idty, Basis_vecFields)
        y.backward()
        return np.double(y.data.numpy()), np.double(X.grad.data.numpy())

    def printx00(x):
        Energyrepa.append(Opts(x)[0])

    res_reg = optimize.minimize(Opts, X, method='BFGS',
                                jac=True, callback=printx00, options={'gtol': 1e-02, 'maxiter': Max_ite, 'disp': False})  # True

    gamma = idty + torch.einsum("i, ijkl->jkl", [torch.from_numpy(res_reg.x).float(), Basis_vecFields])
    gamma = gamma / torch.einsum("ijk, ijk->jk", [gamma, gamma]).sqrt()  # project to S2
    # get the spherical coordinate representation
    gammaSph = torch.stack((Cartesian_to_spherical(gamma[0], gamma[1], gamma[2])))
        
    f = compose_gamma(f1, gammaSph)
        
    return f, Energyrepa


def E_L2_SRNFS(X, f1, f2, idty, Basis_vecFields):
    
    gamma = idty + torch.einsum("i, ijkl->jkl", [X, Basis_vecFields])
    gamma = gamma / torch.einsum("ijk, ijk->jk", [gamma, gamma]).sqrt()  # project to S2

    # get the spherical coordinate representation
    gammaSph = torch.stack((Cartesian_to_spherical(gamma[0] + 1e-7, gamma[1], (1 - 1e-7) * gamma[2])))
    f1_gamma = compose_gamma(f1, gammaSph)

    q1_gamma = f_to_q(f1_gamma)
    q2= f_to_q(f2)
    Diff = q2 - q1_gamma
    return integral_over_s2(torch.einsum("imn,imn->mn", [Diff, Diff]))


def initialize_over_paraSO3_SRNF(f1, f2, idty):
    
    # load the elements in the icosahedral group
    XIco_mat = sio.loadmat('Data/skewIcosahedral.mat')
    XIco = torch.from_numpy(XIco_mat['X']).float()
    
    q2 = f_to_q(f2)
    
    EIco = torch.zeros(60)
    f1_gammaIco = torch.zeros(60, *f1.size())
    q1_gammaIco = torch.zeros(60, *f1.size())
    for i in range(60):
        RIco = torch.einsum("ij,jmn->imn", [torch_expm(XIco[i]), idty])
        gammaIco = torch.stack((Cartesian_to_spherical(RIco[0] + 1e-7, RIco[1], (1 - 1e-7) * RIco[2])))
        f1_gammaIco[i] = compose_gamma(f1, gammaIco)
        q1_gammaIco[i] = f_to_q(f1_gammaIco[i])
    
        EIco[i] = integral_over_s2(torch.einsum("imn,imn->mn", [q2 - q1_gammaIco[i], q2 - q1_gammaIco[i]]))

    # get the index of the smallest value
    Ind = np.argmin(EIco)
    
    X = XIco[Ind] 
    
    L2_ESO3 = []
    def opt(X):
        X = torch.from_numpy(X).float().requires_grad_()
        R = torch.einsum("ij,jmn->imn", [torch_expm(X), idty])
        
        gamma = torch.stack((Cartesian_to_spherical(R[0] + 1e-7, R[1], (1 - 1e-7) * R[2])))
        f1_gamma = compose_gamma(f1, gamma)
        
        q1_gamma = f_to_q(f1_gamma)
        q2 = f_to_q(f2)
        
        y = integral_over_s2(torch.einsum("imn,imn->mn", [q2 - q1_gamma,q2 - q1_gamma]))
        y.backward()
        return np.double(y.data.numpy()), np.double(X.grad.data.numpy())
    
    def printx(x):
        L2_ESO3.append(opt(x)[0])
    
    res = optimize.minimize(opt, X, method='BFGS',
                                jac=True, callback=printx, options={'gtol': 1e-02, 'disp': False})  # True

    X_opt = torch.from_numpy(res.x).float()
    R_opt = torch.einsum("ij,jmn->imn", [torch_expm(X_opt), idty])
    gamma_opt = torch.stack((Cartesian_to_spherical(R_opt[0] + 1e-7, R_opt[1], (1 - 1e-7) * R_opt[2])))
    f1_gamma = compose_gamma(f1, gamma_opt)
    return  f1_gamma, L2_ESO3, f1_gammaIco[Ind], EIco


# compute the SRNF representation of a surface f
def f_to_q(f):
    m, n = f.shape[-2:]
    PHI, THETA = torch.meshgrid([torch.linspace(0, np.pi, m), torch.linspace(0, 2 * np.pi, n)])
    df = torch.zeros(3, 2, m, n)
    Xf10, df[0, 1] = gradient_map(f[0])  # 1/sin(phi)d/dtheta, d/dphi
    Xf20, df[1, 1] = gradient_map(f[1])
    Xf30, df[2, 1] = gradient_map(f[2])

    df[0, 0], df[1, 0], df[2, 0] = Xf10/(1e-7 + torch.sin(PHI)), Xf20/(1e-7 + torch.sin(PHI)), Xf30/(
                1e-7 + torch.sin(PHI))
    df[:, 0, [0, m - 1], :] = 0  # when phi is 0 or pi, df/dtheta = 0

    n_f = torch.cross(df[:, 0], df[:, 1], dim=0)
    Norm_n = (torch.einsum("imn,imn->mn", [n_f, n_f])+ 1e-7).sqrt()

    inv_sqrt_norm_n = 1/ Norm_n.sqrt()
    q = torch.einsum("imn,mn->imn",[n_f, inv_sqrt_norm_n])
    return q
