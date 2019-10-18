from ShapePackages.OneFormMaps import f_to_df, torch_expm
from ShapePackages.OneFormRieMetric import riemann_metric
from ShapePackages.RegistrationFunc import compose_gamma, get_idty_S2, Cartesian_to_spherical
import numpy as np
import torch
import scipy.optimize as optimize


# define the combined energy function
def energy_fun_combined_modSO3(f, X, Coe_x, RX, f1, f2, Tstps, Basis_vecFields, Basis_1forms, a, b, c, d):
    # f1 and f2 are the original boundary surfaces

    # calculate R(f2)
    R = torch_expm(RX)
    Rf2 = torch.einsum("ij,jmn->imn",[R,f2]) # Rotate f2
    
    df1, dRf2 = f_to_df(f1), f_to_df(Rf2)

    # compute the linear path between df1 and df2
    Time_points = torch.arange(Tstps, out=torch.FloatTensor())
    curve0 = df1 + torch.einsum("ijmn,t->tijmn", [dRf2 - df1, Time_points]) / (Tstps - 1)  # df1 + (df2-df1)t

    # get the identity map from S2 to S2
    idty = get_idty_S2(*Basis_vecFields.shape[-2:])

    # calculate f(gamma)
    gamma = idty + torch.einsum("i, ijkl->jkl", [X, Basis_vecFields])
    gamma = gamma / torch.einsum("ijk, ijk->jk", [gamma, gamma]).sqrt()  # project to S2
    # get the spherical coordinate representation
    gammaSph = torch.stack((Cartesian_to_spherical(gamma[0] + 1e-7, gamma[1], (1 - 1e-7) * gamma[2])))
    f_gamma = compose_gamma(f, gammaSph)
    
    # get a new curve that goes through all possibilities
    curve = torch.zeros(curve0.size())

    curve[0], curve[curve0.size(0) - 1] = f_to_df(f_gamma), curve0[-1]
    curve[1: curve0.size(0) - 1] = curve0[1: curve0.size(0) - 1] + torch.einsum("lijmn,lt->tijmn", [Basis_1forms, Coe_x])
    d_curve = (curve[1: curve0.size(0)] - curve[0: curve0.size(0) - 1]) * (curve0.size(0) - 1)

    E = torch.zeros(curve0.size(0) - 1)
    for i in range(curve0.size(0) - 1):
        E[i] = riemann_metric(curve[i], d_curve[i], d_curve[i], a, b, c, d)
    return torch.sum(E) / (curve0.size(0) - 1)


def get_optCoe_shapes_combined(f, Coe_x, RX, f1, f2, Basis_vecFields, Basis_1forms, a, b, c, d, Tstps, Max_ite):

    Num_basis = Basis_1forms.size(0)

    X = torch.zeros(Basis_vecFields.size(0))
    XX = torch.cat((X, Coe_x.flatten(), RX))

    # define the function to be optimized
    def Opts0(XX):
        X = torch.from_numpy(XX[:Basis_vecFields.size(0)]).float().requires_grad_()
        Coe_x = torch.from_numpy(XX[Basis_vecFields.size(0):len(XX)-3]).float()
        Coe_x = Coe_x.view(Num_basis, -1).requires_grad_()
        RX = torch.from_numpy(XX[-3:]).float().requires_grad_()

        y = energy_fun_combined_modSO3(f, X, Coe_x, RX, f1, f2, Tstps, Basis_vecFields, Basis_1forms, a, b, c, d)
        y.backward()

        XX_grad = torch.cat((X.grad, Coe_x.grad.flatten(), RX))
        return np.double(y.data.numpy()), np.double(XX_grad.data.numpy())

    def printx(x):
        Energy.append(Opts0(x)[0])

    Energy = []
    res0 = optimize.minimize(Opts0, XX, method='BFGS', jac=True, callback=printx,
                             options={'gtol': 1e-02, 'disp': False, 'maxiter': Max_ite})  # True
    #     print(res0.fun)
    X_opt = torch.from_numpy(res0.x[:Basis_vecFields.size(0)]).float()
    Coe_opt = torch.from_numpy(res0.x[Basis_vecFields.size(0):len(XX)-3]).float().view(Num_basis, -1)
    RX_opt = torch.from_numpy(res0.x[-3:]).float()

    return X_opt, Coe_opt, RX_opt, Energy

