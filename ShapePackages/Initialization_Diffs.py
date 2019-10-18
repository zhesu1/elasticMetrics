import scipy.io as sio
import torch
import numpy as np
import scipy.optimize as optimize
from ShapePackages.OneFormRieMetric import length_dfunc_Imm, riemann_metric
from ShapePackages.OneFormMaps import torch_expm, integral_over_s2, f_to_df
from ShapePackages.RegistrationFunc import get_idty_S2, compose_gamma, Cartesian_to_spherical


def initialize_over_diffs_main(f1, f2, a, b, c, d, MaxDegVecFS2 = 5, Max_ite = 500):
    
    idty = get_idty_S2(*f2.shape[-2:])
    
    # initialize over the reparametrizations of rotations
    f1_gamma, L, f1_barh, length_linear = initialize_over_paraSO3(f1, f2, idty, a, b, c, d)
    f, Energyrepa = Initialization_overDiff(f1_gamma, f2, idty, a, b, c, d, MaxDegVecFS2, Max_ite)
    return f, Energyrepa, f1_gamma, L, f1_barh, length_linear


def Initialization_overDiff(f1, f2, idty, a, b, c, d, MaxDegVecFS2, Max_ite):
    
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
        y = energy0(X, f1, f2, idty, Basis_vecFields, a, b, c, d)
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


# rescale the surface such that its area is 1
def rescale_surf(f):
    df = f_to_df(f)
    n_f = torch.cross(df[:, 0], df[:, 1], dim=0)
    Norm_n = torch.einsum("imn,imn->mn", [n_f, n_f]).sqrt()
    
    return f/integral_over_s2(Norm_n).sqrt()


# initialize over reparametrizations of rotations
def initialize_over_paraSO3(f1, f2, idty, a, b, c, d):
    
    Tstps = 2
    
    # load the elements in the icosahedral group
    XIco_mat = sio.loadmat('Data/skewIcosahedral.mat')
    XIco = torch.from_numpy(XIco_mat['X']).float()
    
    df2 = f_to_df(f2)
    Time_points = torch.arange(Tstps, out=torch.FloatTensor())
    
    linear_path = torch.zeros(60, Tstps, 3, 2, *f1.shape[-2:])
    f1_gammaIco = torch.zeros(60, *f1.size())
    df1_gammaIco = torch.zeros(60, 3, 2, *f1.shape[-2:])
    for i in range(60):
        RIco = torch.einsum("ij,jmn->imn", [torch_expm(XIco[i]), idty])
        gammaIco = torch.stack((Cartesian_to_spherical(RIco[0] + 1e-7, RIco[1], (1 - 1e-7) * RIco[2])))
        f1_gammaIco[i] = compose_gamma(f1, gammaIco)
        df1_gammaIco[i] = f_to_df(f1_gammaIco[i])
        linear_path[i] = df1_gammaIco[i] + torch.einsum(
            "ijmn,t->tijmn", [df2 - df1_gammaIco[i], Time_points]) / (Tstps - 1)  # df1 + (df2-df1)t

    length_linear = torch.zeros(60)
    for i in range(60):
        length_linear[i] = length_dfunc_Imm(linear_path[i], a, b, c, d)

    # get the index of the smallest value
    Ind = np.argmin(length_linear)
    
    X = XIco[Ind]
    
    L = []
    def opt(X):
        X = torch.from_numpy(X).float().requires_grad_()
        R = torch.einsum("ij,jmn->imn", [torch_expm(X), idty])
        
        gamma = torch.stack((Cartesian_to_spherical(R[0] + 1e-7, R[1], (1 - 1e-7) * R[2])))
        f1_gamma = compose_gamma(f1, gamma)
        
        df1_gamma = f_to_df(f1_gamma)
        df2 = f_to_df(f2)
        
        Time_points = torch.arange(Tstps, out=torch.FloatTensor())
        lin_path = df1_gamma + torch.einsum("ijmn,t->tijmn",
                                            [df2 - df1_gamma, Time_points]) / (Tstps - 1)  # df1 + (df2-df1)t
        y = length_dfunc_Imm(lin_path, a, b, c, d)
        y.backward()
        return np.double(y.data.numpy()), np.double(X.grad.data.numpy())
    
    def printx(x):
        L.append(opt(x)[0])
    
    res = optimize.minimize(opt, X, method='BFGS',
                                jac=True, callback=printx, options={'gtol': 1e-02, 'disp': False})  # True

    X_opt = torch.from_numpy(res.x).float()
    R_opt = torch.einsum("ij,jmn->imn", [torch_expm(X_opt), idty])
    gamma_opt = torch.stack((Cartesian_to_spherical(R_opt[0] + 1e-7, R_opt[1], (1 - 1e-7) * R_opt[2])))
    f1_gamma = compose_gamma(f1, gamma_opt)
    return  f1_gamma, L, f1_gammaIco[Ind], length_linear



def energy0(X, f1, f2, idty, Basis_vecFields, a, b, c, d):

    gamma = idty + torch.einsum("i, ijkl->jkl", [X, Basis_vecFields])
    gamma = gamma / torch.einsum("ijk, ijk->jk", [gamma, gamma]).sqrt()  # project to S2

    # get the spherical coordinate representation
    gammaSph = torch.stack((Cartesian_to_spherical(gamma[0] + 1e-7, gamma[1], (1 - 1e-7) * gamma[2])))
    f1_gamma = compose_gamma(f1, gammaSph)

    df1_gamma = f_to_df(f1_gamma)
    Diff_12 = f_to_df(f2 - f1_gamma)
    E = riemann_metric(df1_gamma, Diff_12, Diff_12, a, b, c, d)  # /dT
    return E



