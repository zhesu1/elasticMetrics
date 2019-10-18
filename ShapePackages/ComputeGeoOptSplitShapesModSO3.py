from ShapePackages.OneFormMaps import f_to_df, torch_expm
from ShapePackages.OneFormRieMetric import riemann_metric
from ShapePackages.RegistrationFunc import get_idty_S2, Cartesian_to_spherical, compose_gamma
from ShapePackages.MultiRes import reduce_resolution
import scipy.optimize as optimize
import torch
import numpy as np
import scipy.io as sio


def get_optCoe_shapes_split(f_f1, Coe_x, RX, f1, f2, Basis_vecFields, 
                            Basis_Sph, Basis_1forms, a, b, c, d, Tstps, 
                            Max_ite, side,*, multires = False):
    
    Num_basis = Basis_Sph.size(0)
    
    # define the function to be optimized
    CoeRX = torch.cat((Coe_x.flatten(), RX))
    
    def Opts_Imm(CoeRX):
        Coe_x = torch.from_numpy(CoeRX[:-3]).float().view(Num_basis, -1).requires_grad_()
        RX = torch.from_numpy(CoeRX[-3:]).float().requires_grad_()
        
        y = energy_func_para_split_modSO3(f_f1, Coe_x, RX, f1, f2, Basis_1forms, a, b, c, d, Tstps)
        y.backward()
        
        CoeRX_grad = torch.cat((Coe_x.grad.flatten(), RX.grad))
        return np.double(y.data.numpy()), np.double(CoeRX_grad.data.numpy())

    # define callback to show the details for each iteration in the optimization process
    def printx(x):
        Energy.append(Opts_Imm(x)[0])
        
    Energy = []
    
    if multires == True: 
        
        # load the bases for surfaces and exact 1forms
        mat_basis = sio.loadmat('Data/basis_exact_1forms_deg25_25_49.mat')
        Basis_1forms_low = torch.tensor(mat_basis['Basis'])[: Num_basis]
        
        m, n = 25, 49 # round(f1.shape[-2] / 2), int((f1.shape[-1] - 1)/2)
        f_f1_low = reduce_resolution(f_f1,m,n)
        f1_low = reduce_resolution(f1,m,n)
        f2_low = reduce_resolution(f2,m,n)
        
        def Opts_0(CoeRX):
            Coe_x = torch.from_numpy(CoeRX[:-3]).float().view(Num_basis, -1).requires_grad_()
            RX = torch.from_numpy(CoeRX[-3:]).float().requires_grad_()
        
            y0 = energy_func_para_split_modSO3(f_f1_low, Coe_x, RX, f1_low, f2_low, Basis_1forms_low, a, b, c, d, Tstps)
            y0.backward()
            
            CoeRX_grad = torch.cat((Coe_x.grad.flatten(), RX.grad))
            return np.double(y0.data.numpy()), np.double(CoeRX_grad.data.numpy())
        
        def printx0(x):
            Energy.append(Opts_0(x)[0])
        
        
        res0 = optimize.minimize(Opts_0, CoeRX, method='BFGS', jac=True, callback=printx0,
                                 options={'gtol': 1e-02, 'disp': False, 'maxiter': Max_ite})
        
        Coe_x_opt = torch.from_numpy(res0.x[:-3]).float().view(Num_basis, -1)
        RX_opt = torch.from_numpy(res0.x[-3:]).float()
        
    else:
        # set callback=printx to return the energy change in the optimization process, otherwise use None
        res = optimize.minimize(Opts_Imm, CoeRX, method='BFGS', jac=True, callback=printx,
                            options={'gtol': 1e-02, 'disp': False, 'maxiter': Max_ite})
        # print(res.fun)
        Coe_x_opt = torch.from_numpy(res.x[:-3]).float().view(Num_basis, -1)
        RX_opt = torch.from_numpy(res.x[-3:]).float()
    
    
    # rotate f_f2 using R
    R = torch_expm(RX)
    Rf2 = torch.einsum("ij,jmn->imn",[R,f2]) # Rotate f_f2
    
    c_f = torch.zeros(2, 3, *f1.shape[-2:])
    c_f[0] = f_f1
    
    # compute the second and the last second discrete linear curve points
    lin_f = f1 + torch.einsum("imn,t->timn", [Rf2 - f1, torch.tensor([1.])]) / (Tstps - 1) # 1*3*Num_phi_Num_theta
    c_f[1] = lin_f[0] + torch.einsum("limn,l->imn", [Basis_Sph, Coe_x_opt[:,0]])
    
    idty = get_idty_S2(*Basis_vecFields.shape[-2:])

#     if side == 0:
#         # 1 only update the first boundary surface
    c_f[0] = compute_optimal_reg_1st(c_f, idty, Basis_vecFields, a, b, c, d)
#     elif side == 1:
#         # 2 update both boundary surfaces
#         c_f[0], c_f[-1] = compute_optimal_reg_both(c_f, idty, Basis_vecFields, a, b, c, d)
        
    return c_f[0], Coe_x_opt, RX_opt, Energy


# the energy function for the curve obtained by perturbing curve0 with Coe_x*Basis_1forms in space of surfaces
def energy_func_para_split_modSO3(f_f1, Coe_x, RX, f1, f2, Basis_1forms, a, b, c, d, Tstps):
    
    # rotate f_f2 using R
    R = torch_expm(RX)
    Rf2 = torch.einsum("ij,jmn->imn",[R,f2]) # Rotate f_f2
    
    df1 = f_to_df(f1)
    dRf2 = f_to_df(Rf2)

    Time_points = torch.arange(Tstps, out=torch.FloatTensor())
    # find the linear path between df1 and df2
    curve0 = df1 + torch.einsum("ijmn,t->tijmn", [dRf2 - df1, Time_points]) / (Tstps - 1)  # df1 + (df2-df1)t
    
    # calculate the new path between the new endpoints using the optimal coefficient matrix
    curve = torch.zeros(curve0.size())
    curve[0], curve[-1] = f_to_df(f_f1), curve0[-1]
    curve[1: Tstps-1] = curve0[1: Tstps-1] + torch.einsum("lijmn,lt->tijmn", [Basis_1forms, Coe_x])
    
    d_curve = (curve[1: Tstps] - curve[0: Tstps-1])*(Tstps-1)
    E = torch.zeros(Tstps-1)
    for i in range(Tstps-1):
        E[i] = riemann_metric(curve[i], d_curve[i], d_curve[i], a, b, c, d)
    return torch.sum(E)/(Tstps-1)


# the energy of the line segment between the first two points in a curve
def energy_reg(X, c_f, idty, Basis_vecFields, a, b, c, d):
    f1, f2 = c_f[0], c_f[1]
    gamma = idty + torch.einsum("i, ijkl->jkl", [X, Basis_vecFields])
    gamma = gamma / torch.einsum("ijk, ijk->jk", [gamma, gamma]).sqrt()  # project to S2

    # get the spherical coordinate representation
    gammaSph = torch.stack((Cartesian_to_spherical(gamma[0] + 1e-7, gamma[1], (1 - 1e-7) * gamma[2])))
    f1_gamma = compose_gamma(f1, gammaSph)

    df1_gamma = f_to_df(f1_gamma)
    Diff_12 = f_to_df(f2 - f1_gamma)
    E = riemann_metric(df1_gamma, Diff_12, Diff_12, a, b, c, d)  # /dT
    return E


# # the energy of the line segment between the last two points in a curve
# def energy_reg1(X, c_f, idty, Basis_vecFields, a, b, c, d):
#     f1, f2 = c_f[0], c_f[1]
#     gamma = idty + torch.einsum("i, ijkl->jkl", [X, Basis_vecFields])
#     gamma = gamma / torch.einsum("ijk, ijk->jk", [gamma, gamma]).sqrt()  # project to S2

#     # get the spherical coordinate representation
#     gammaSph = torch.stack((Cartesian_to_spherical(gamma[0] + 1e-7, gamma[1], (1 - 1e-7) * gamma[2])))
#     f2_gamma = compose_gamma(f2, gammaSph)

#     df1 = f_to_df(f1)
#     Diff_12 = f_to_df(f2_gamma - f1)
#     E = riemann_metric(df1, Diff_12, Diff_12, a, b, c, d)  # /dT
#     return E


# only update the first boundary surfaces
def compute_optimal_reg_1st(c_f, idty, Basis_vecFields, a, b, c, d):
    X = torch.zeros(Basis_vecFields.size(0))

    def Opts(X):
        X = torch.from_numpy(X).float()
        X.requires_grad_()
        y = energy_reg(X, c_f[:2], idty, Basis_vecFields, a, b, c, d)
        y.backward()
        return np.double(y.data.numpy()), np.double(X.grad.data.numpy())

    res_reg = optimize.minimize(Opts, X, method='BFGS',
                                jac=True, callback=None, options={'gtol': 1e-02, 'disp': False})  # True

    gamma = idty + torch.einsum("i, ijkl->jkl", [torch.from_numpy(res_reg.x).float(), Basis_vecFields])
    gamma = gamma / torch.einsum("ijk, ijk->jk", [gamma, gamma]).sqrt()  # project to S2
    # get the spherical coordinate representation
    gammaSph = torch.stack((Cartesian_to_spherical(gamma[0], gamma[1], gamma[2])))

    f1_gamma = compose_gamma(c_f[0], gammaSph)
    return f1_gamma


# update both boundary surfaces
# def compute_optimal_reg_both(c_f, idty, Basis_vecFields, a, b, c, d):
#     X = torch.zeros(Basis_vecFields.size(0))
#     X1 = torch.zeros(Basis_vecFields.size(0))

#     def Opts(X):
#         X = torch.from_numpy(X).float()
#         X.requires_grad_()
#         y = energy_reg(X, c_f[:2], idty, Basis_vecFields, a, b, c, d)
#         y.backward()
#         return np.double(y.data.numpy()), np.double(X.grad.data.numpy())

#     def Opts1(X):
#         X = torch.from_numpy(X).float()
#         X.requires_grad_()
#         y = energy_reg1(X, c_f[-2:], idty, Basis_vecFields, a, b, c, d)
#         y.backward()
#         return np.double(y.data.numpy()), np.double(X.grad.data.numpy())

#     res_reg = optimize.minimize(Opts, X, method='BFGS',
#                                 jac=True, callback=None, options={'gtol': 1e-02, 'disp': False})  # True

#     res_reg1 = optimize.minimize(Opts1, X1, method='BFGS',
#                                  jac=True, callback=None, options={'gtol': 1e-02, 'disp': False})  # True

#     gamma = idty + torch.einsum("i, ijkl->jkl", [torch.from_numpy(res_reg.x).float(), Basis_vecFields])
#     gamma = gamma / torch.einsum("ijk, ijk->jk", [gamma, gamma]).sqrt()  # project to S2
#     # get the spherical coordinate representation
#     gammaSph = torch.stack((Cartesian_to_spherical(gamma[0], gamma[1], gamma[2])))
#     f1_gamma = compose_gamma(c_f[0], gammaSph)

#     gamma1 = idty + torch.einsum("i, ijkl->jkl", [torch.from_numpy(res_reg1.x).float(), Basis_vecFields])
#     gamma1 = gamma1 / torch.einsum("ijk, ijk->jk", [gamma1, gamma1]).sqrt()  # project to S2
#     # get the spherical coordinate representation
#     gammaSph1 = torch.stack((Cartesian_to_spherical(gamma1[0], gamma1[1], gamma1[2])))
#     f2_gamma = compose_gamma(c_f[-1], gammaSph1)

#     return f1_gamma, f2_gamma
