from ShapePackages.OneFormMaps import center_surf
from ShapePackages.ComputeGeoOptSplitShapes import *
from ShapePackages.ComputeGeoOptCombinedShapes import *
from ShapePackages.MultiRes import up_sample
from ShapePackages.Initialization_Diffs import initialize_over_diffs_main
import scipy.io as sio
   

def compute_geodesic_shape_main(f1, f2, *, MaxDegVecFS2, MaxDegHarmSurf,
                           Cmetric=(), Tstps, method='split', maxiter=(10, 30), **kwargs):
    
    f, f2 = center_surf(f1), center_surf(f2) 
    
    a,b,c,d = Cmetric
    f, Energyrepa, f1_gamma, L, f1_barh, length_linear = initialize_over_diffs_main(f, f2, a, b, c, d, 3, 100)
        
    geo, EnergyAll0 = compute_geodesic_unpara(f, f2, Cmetric = Cmetric,
                                             Tstps = Tstps,  # the number of time points of the geodesic
                                             MaxDegHarmSurf = MaxDegHarmSurf,
                                             MaxDegVecFS2 = MaxDegHarmSurf,
                                             method = method,
                                             maxiter = maxiter, **kwargs)
    return geo, EnergyAll0


def compute_geodesic_unpara(f1, f2, *, MaxDegVecFS2, MaxDegHarmSurf,
                           Cmetric=(), Tstps, method='split', maxiter=(10, 30), **kwargs):

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

    # get the coefficients for the split metric
    a, b, c, d = Cmetric
    
    # initial f1
    f1_new = f1
    
    # compute the geodesic 
    EnergyAll = []
    
    # set the number of iterations
    N_ite, Max_ite_in = maxiter

    if method == 'split':
        
        f_f1, f_f2 = f1_new, f2
        
        # determine the reparametrization on one side (0) or both sides (1)
        side = 0  # 0 or 1
        
        Tstps0 = Tstps
        
        if kwargs.get('multires'):
            multires = True
            
            # multresolution in time
            Tstps_low = 10   
            Tstps0 = Tstps_low
            
        else: 
            multires = False
        
        Coe_x = torch.zeros(Basis_Sph.size(0), Tstps0 - 2)
        
        for i in range(N_ite):
            
            if i == N_ite-1:
                Max_ite_in = 100
                if Tstps0 != Tstps:
                    Tstps0 = Tstps
                    Coe_x = up_sample(Coe_x, Tstps - 2)
                
            if i > round(int(N_ite/2)):
                multires = False
     
            f_f1, f_f2, Coe_x, Energy = get_optCoe_shapes_split(f_f1, f_f2, Coe_x, f1_new, f2, Basis_vecFields, 
                                                                Basis_Sph, Basis_1forms, a, b, c, d, Tstps0,
                                                                Max_ite_in, side, **{'multires': multires})
            EnergyAll.append(Energy)
        
        # get the geodesic
        Time_points = torch.arange(Tstps, out=torch.FloatTensor())
        lin_f = f1_new + torch.einsum("imn,t->timn", [f2-f1_new, Time_points])/(Tstps-1)
        
        # perturbe the linear path using the optimal coefficients
        geo_f = torch.zeros(Tstps, 3, *f1.shape[-2:])
        geo_f[0], geo_f[-1] = f_f1, f_f2
        geo_f[1: Tstps - 1] = lin_f[1: Tstps - 1] + torch.einsum("limn,lt->timn", [Basis_Sph, Coe_x])
        
    elif method == 'combined':
        
        f = f1_new
        Coe_x = torch.zeros(Basis_Sph.size(0), Tstps - 2)

        idty = get_idty_S2(*Basis_vecFields.shape[-2:])
        
        for i in range(N_ite):
            
            if i == N_ite-1:
                Max_ite_in = 100
    
            X_new, Coe_x, Energy = get_optCoe_shapes_combined(f, Coe_x, f1_new, f2, Basis_vecFields,
                                                               Basis_1forms, a, b, c, d, Tstps, Max_ite_in)
    
            # update f
            gamma = idty + torch.einsum("i, ijkl->jkl", [X_new, Basis_vecFields])
            gamma = gamma/torch.einsum("ijk, ijk->jk", [gamma, gamma]).sqrt()  # project to S2
            # get the spherical coordinate representation
            gammaSph = torch.stack((Cartesian_to_spherical(gamma[0], gamma[1], gamma[2])))

            f = compose_gamma(f, gammaSph)
            EnergyAll.append(Energy)
        
        # get the geodesic
        Time_points = torch.arange(Tstps, out=torch.FloatTensor())
        lin_f = f1_new + torch.einsum("imn,t->timn", [f2-f1_new, Time_points])/(Tstps-1)

        # perturbe the linear path using the optimal coefficients
        geo_f = torch.zeros(Tstps, 3, *f1.shape[-2:])
        geo_f[0], geo_f[Tstps - 1] = f, f2
        geo_f[1: Tstps - 1] = lin_f[1: Tstps - 1] + torch.einsum("limn,lt->timn", [Basis_Sph, Coe_x])
    
    EnergyAll0 = [item for sublist in EnergyAll for item in sublist]
    return geo_f, EnergyAll0
        


