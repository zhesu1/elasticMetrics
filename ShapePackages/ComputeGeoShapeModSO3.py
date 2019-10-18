from ShapePackages.OneFormMaps import center_surf, torch_expm
from ShapePackages.ComputeGeoOptSplitShapesModSO3 import *
from ShapePackages.ComputeGeoOptCombinedShapesModSO3 import *

from ShapePackages.MultiRes import up_sample
from ShapePackages.InitializationModSO3 import initialize_overSO3

from ShapePackages.Initialization_Diffs_SRNF import initialize_over_diffs_main_SRNF
from ShapePackages.Initialization_Diffs import initialize_over_diffs_main
import scipy.io as sio


# the main program for calculating a geodesic in the space of unparametrized surfaces
# inputs: f1, f2...two surfaces of size 3*num_phi*num_theta, where phi denotes the polar angle and theta denotes the azimuthal angle.
#                  (To use the code, num_theta must be odd !!!)
#         MaxDegVecFS2...the maximal degree of spherical harmonics for the basis of vector fields on S2
#         MaxDegHarmSurf...the maximal degree of spherical harmonics for the basis of the space of immersions
#         Cmetric = a,b,c,d...weights of the split metric:
#    		    a weights the term that measures the change in metric, 
#    		    b weights the term that measures the change in volume density and 
#    		    c weights the term that measures the change in normal direction
#    		    d weights the term that measures the change in local reparametrization
#         Tpts...the number of time points
#         method...split or combined
#         Max_ite...the maximum number of iterations for the whole optimization process and the maximal number of iterations for each 
#                   optimization 
#         **kwargs... use multires or not
# outputs: geo_f...the geodesic in the space of immersions between f1 and f2
#          AllEnergy...the energies of curves in the optimization process
#
def compute_geodesic_shape_main(f1, f2, *, MaxDegVecFS2, MaxDegHarmSurf,
                           Cmetric=(), Tpts, method='split', maxiter=(10, 30), **kwargs):
    
    f1, f2 = center_surf(f1), center_surf(f2)
    
    # initialize over SO3
    f2, L = initialize_overSO3(f1, f2)
    
    # initialize over Diffs
    f, Energyrepa1, f1_gamma1, ESO31, f1_barh1, EIco1 = initialize_over_diffs_main_SRNF(f1, f2, 2, 50)
    
    if Cmetric != (0, 1/2, 1, 0):
        a,b,c,d = Cmetric
        f, Energyrepa, f1_gamma, L, f1_barh, length_linear = initialize_over_diffs_main(f, f2, a, b, c, d, 3, 500)
        
    geo, AllEnergy = compute_geodesic_unparaModSO3(f, f2, Cmetric = Cmetric,
                                             Tpts = Tpts,  # the number of time points of the geodesic
                                             MaxDegHarmSurf = MaxDegHarmSurf,
                                             MaxDegVecFS2 = MaxDegHarmSurf,
                                             method = method,
                                             maxiter = maxiter, **kwargs)
    return geo, AllEnergy




def compute_geodesic_unparaModSO3(f1, f2, *, MaxDegVecFS2, MaxDegHarmSurf,
                           Cmetric=(), Tpts, method='split', maxiter=(10, 30), **kwargs):

    # load the bases for surfaces and exact 1forms
    mat_basis = sio.loadmat('Data/basis_exact_1forms_deg25_{0}_{1}.mat'.format(*f1.shape[-2:]))

    Num_basis = ((MaxDegHarmSurf + 1) ** 2 - 1) * 3  # the number of basis for 1forms
    Basis_Sph = torch.from_numpy(mat_basis['Basis_Sph'])[: Num_basis].float()
    Basis_1forms = torch.from_numpy(mat_basis['Basis'])[: Num_basis].float()

    # load the basis for tangent vector fields on S2
    mat_vecF = sio.loadmat('Data/basis_vecFieldsS2_deg25_{0}_{1}.mat'.format(*f1.shape[-2:]))

    N_basis_vec = (MaxDegVecFS2 + 1) ** 2 - 1  # half the number of basis for the vector fields on S2
    Basis0_vec = torch.from_numpy(mat_vecF['Basis'])[: N_basis_vec].float()
    Basis_vecFields = torch.cat((Basis0_vec[:, 0], Basis0_vec[:, 1]))  # get a basis of the tangent fields on S2

    # get the coefficients for thw split metric
    a, b, c, d = Cmetric
    
    # compute the geodesic 
    EnergyAll = []
    
    # set the number of iteration for the whole algorithm
    N_ite, Max_ite_in = maxiter
    
    f1_new = f1

    if method == 'split':
        
        f_f1, f_f2 = f1_new, f2
         
        side = 0 
        
        Tpts0 = Tpts
        
        if kwargs.get('multires'):
            multires = True
            
            # multresolution in time
            Tpts_low = 5   
            Tpts0 = Tpts_low
            
        else: 
            multires = False
        
        Coe_x = torch.zeros(Basis_Sph.size(0), Tpts0 - 2)
        RX = torch.zeros(3)
        
        for i in range(N_ite):
            
            if i == N_ite-1:
                Max_ite_in = 100
                if Tpts0 != Tpts:
                    Tpts0 = Tpts
                    Coe_x = up_sample(Coe_x, Tpts - 2)
                
            if i > round(int(N_ite/2)):
                multires = False
     
            f_f1, Coe_x, RX, Energy = get_optCoe_shapes_split(f_f1, Coe_x, RX, f1_new, f2, Basis_vecFields, 
                                                                Basis_Sph, Basis_1forms, a, b, c, d, Tpts0,
                                                                Max_ite_in, side, **{'multires': multires})
            EnergyAll.append(Energy)
        
        # rotate f2 using R
        R = torch_expm(RX)
        Rf2 = torch.einsum("ij,jmn->imn",[R,f2]) # Rotate f_f2
    
        Time_points = torch.arange(Tpts, out=torch.FloatTensor())
        lin_f = f1_new + torch.einsum("imn,t->timn", [Rf2-f1_new, Time_points])/(Tpts-1)
        
        # perturbe the linear path using the optimal coefficients
        geo_f = torch.zeros(Tpts, 3, *f1.shape[-2:])
        geo_f[0], geo_f[-1] = f_f1, Rf2
        geo_f[1: Tpts - 1] = lin_f[1: Tpts - 1] + torch.einsum("limn,lt->timn", [Basis_Sph, Coe_x])
        
    elif method == 'combined':
        
        f = f1_new
        Coe_x = torch.zeros(Basis_Sph.size(0), Tpts - 2)
        RX = torch.zeros(3)

        idty = get_idty_S2(*Basis_vecFields.shape[-2:])
        
        for i in range(N_ite):
            
            if i == N_ite-1:
                Max_ite_in = 100
    
            X_new, Coe_x, RX, Energy = get_optCoe_shapes_combined(f, Coe_x, RX, f1_new, f2, Basis_vecFields,
                                                               Basis_1forms, a, b, c, d, Tpts, Max_ite_in)
    
            # update f
            gamma = idty + torch.einsum("i, ijkl->jkl", [X_new, Basis_vecFields])
            gamma = gamma/torch.einsum("ijk, ijk->jk", [gamma, gamma]).sqrt()  # project to S2
            # get the spherical coordinate representation
            gammaSph = torch.stack((Cartesian_to_spherical(gamma[0], gamma[1], gamma[2])))

            f = compose_gamma(f, gammaSph)
            EnergyAll.append(Energy)
        
        # get R(f2)
        
        R = torch_expm(RX)
        Rf2 = torch.einsum("ij,jmn->imn",[R,f2])
        
        Time_points = torch.arange(Tpts, out=torch.FloatTensor())
        lin_f = f1_new + torch.einsum("imn,t->timn", [Rf2-f1_new, Time_points])/(Tpts-1)
        
        # perturbe the linear path using the optimal coefficients
        geo_f = torch.zeros(Tpts, 3, *f1.shape[-2:])
        geo_f[0], geo_f[Tpts - 1] = f, lin_f[-1]
        geo_f[1: Tpts - 1] = lin_f[1: Tpts - 1] + torch.einsum("limn,lt->timn", [Basis_Sph, Coe_x])
        
    
    EnergyAll0 = [item for sublist in EnergyAll for item in sublist]
    return geo_f, EnergyAll0
        


