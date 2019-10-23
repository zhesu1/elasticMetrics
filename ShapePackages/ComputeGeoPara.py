from ShapePackages.OneFormRieMetric import riemann_metric
from ShapePackages.OneFormMaps import f_to_df, center_surf
import numpy as np
import torch
import scipy.optimize as optimize
import scipy.io as sio


# the main program for calculating a geodesic in the space of immersions
# inputs: f1, f2...two surfaces of size 3*num_phi*num_theta, where phi denotes the polar angle and theta denotes the azimuthal angle.
#                  (To use the code, num_theta must be odd !!!)
#         MaxDegHarmSurf...the maximal degree of spherical harmonics for the basis of the space of immersions
#         Cmetric = a,b,c,d...weights of the split metric:
#    		    a weights the term that measures the change in metric, 
#    		    b weights the term that measures the change in volume density and 
#    		    c weights the term that measures the change in normal direction
#    		    d weights the term that measures the change in local reparametrization
#         Tpts...the number of time points
#         Max_ite...the maximum number of iteration in the optimization process
# outputs: geo_f...the geodesic in the space of immersions between f1 and f2
#          AllEnergy...the energies of curves in the optimization process
#
def compute_geodesic_main(f1, f2, *, MaxDegHarmSurf, Cmetric=(), Tpts, Max_ite=300):

    # load the bases for surfaces and exact 1forms
    mat_basis = sio.loadmat('Bases/basis_exact_1forms_deg25_{0}_{1}.mat'.format(*f1.shape[-2:]))

    Num_basis = ((MaxDegHarmSurf + 1) ** 2 - 1) * 3  # the number of basis for 1forms
    Basis_Sph = torch.from_numpy(mat_basis['Basis_Sph'])[: Num_basis].float()
    Basis_1forms = torch.from_numpy(mat_basis['Basis'])[: Num_basis].float()

    # center the surfaces
    f1, f2 = center_surf(f1), center_surf(f2)

    # get the coefficients for the split metric
    a, b, c, d = Cmetric
        
    geo_f, AllEnergy = compute_geodesic_para(f1, f2, a, b, c, d, Basis_Sph, Basis_1forms, Tpts, Max_ite)

    return geo_f, AllEnergy


def compute_geodesic_para(f1, f2, a, b, c, d, Basis_Sph, Basis_1forms, Tpts, Max_ite):
    
    Coe_x = torch.zeros(Basis_Sph.size(0), Tpts - 2)
    coe_x_opt, Energy = compute_optCoe_Imm(Coe_x, f1, f2, a, b, c, d, Basis_1forms, Tpts, Max_ite)
        
    Time_points = torch.arange(Tpts, out=torch.FloatTensor())
    
    # find the linear path between f1 and f2
    lin_f = f1 + torch.einsum("imn,t->timn", [f2 - f1, Time_points]) / (Tpts - 1)
    
    geo_f = torch.zeros(Tpts, 3, *f1.shape[-2:])
    geo_f[0], geo_f[-1] = f1, f2
    geo_f[1: lin_f.size(0) - 1] = lin_f[1: lin_f.size(0) - 1] + torch.einsum("limn,lt->timn", [Basis_Sph, coe_x_opt])
    
    return geo_f, Energy


def compute_optCoe_Imm(Coe_x, f1, f2, a, b, c, d, Basis_1forms, Tpts, Max_ite):

    Num_basis = Basis_1forms.size(0)

    df1 = f_to_df(f1)
    df2 = f_to_df(f2)

    Time_points = torch.arange(Tpts, out=torch.FloatTensor())

    # find the linear path between df1 and df2
    curve0 = df1 + torch.einsum("ijmn,t->tijmn", [df2 - df1, Time_points]) / (Tpts - 1)  # df1 + (df2-df1)t

    # define the function to be optimized
    def Opts_Imm(Coe_x):
        Coe_x = torch.from_numpy(Coe_x).float()
        Coe_x = Coe_x.view(Num_basis, -1)
        Coe_x.requires_grad_()
        y = energy_func_Imm(Coe_x, curve0, Basis_1forms, a, b, c, d)
        y.backward()
        return np.double(y.data.numpy()), np.double(Coe_x.grad.data.numpy().flatten())

    # define callback to show the details for each iteration in the optimization process
    def printx(x):
        AllEnergy.append(Opts_Imm(x)[0])
        
    AllEnergy = []
    res = optimize.minimize(Opts_Imm, Coe_x.flatten(), method='BFGS', jac=True, callback=printx,
                            options={'gtol': 1e-02, 'disp': False, 'maxiter': Max_ite})
    
    coe_x_opt = torch.from_numpy(res.x).float().view(Num_basis, -1)

    return coe_x_opt, AllEnergy


# the energy function for the curve obtained by perturbing curve0 with Coe_x*Basis in the space of immersions
def energy_func_Imm(Coe_x, curve0, Basis, a, b, c, d):
    curve = torch.zeros(curve0.size())
    curve[0], curve[curve0.size(0)-1] = curve0[0], curve0[curve0.size(0)-1]
    curve[1: curve0.size(0)-1] = curve0[1: curve0.size(0)-1] + torch.einsum("lijmn,lt->tijmn", [Basis, Coe_x])
    d_curve = (curve[1: curve0.size(0)] - curve[0: curve0.size(0)-1])*(curve0.size(0)-1)
    E = torch.zeros(curve0.size(0)-1)
    for i in range(curve0.size(0)-1):
        E[i] = riemann_metric(curve[i], d_curve[i], d_curve[i], a, b, c, d)
    return torch.sum(E)/(curve0.size(0)-1)

